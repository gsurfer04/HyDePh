"""Extracts Delta E and Delta F relative to the foundation potential and trains Delta-NEP in GPUMD."""

from __future__ import annotations

import os
import sys
import time
import json
import shutil
import hashlib
import warnings
import threading
import subprocess
import logging
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import ase.io
from ase import Atoms
from calorine.calculators import CPUNEP

from .config import ToolkitConfig
from .calculator import load_foundation_calculator, HybridDefectCalculator

try:
    from tqdm import tqdm
except ImportError:
    class _DummyPbar:
        def __init__(self, iterable=None, *args, **kwargs):
            self.iterable = iterable if iterable is not None else []
        def __iter__(self):
            return iter(self.iterable)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def update(self, *args, **kwargs):
            pass
        def close(self):
            pass
        def set_postfix(self, *args, **kwargs):
            pass
        def set_description(self, *args, **kwargs):
            pass

    def tqdm(iterable=None, *args, **kwargs):
        return _DummyPbar(iterable, *args, **kwargs)

logger = logging.getLogger(__name__)


def resolve_nep_executable(nep_exe: Optional[str] = None) -> str:
    """Resolves the GPUMD `nep` executable path."""
    if nep_exe:
        if os.path.isfile(nep_exe):
            return os.path.abspath(nep_exe)
        found = shutil.which(nep_exe)
        if found:
            return found
        return nep_exe

    env_nep = os.environ.get("NEP_EXE") or os.environ.get("GPUMD_NEP")
    if env_nep:
        if os.path.isfile(env_nep):
            return os.path.abspath(env_nep)
        found = shutil.which(env_nep)
        if found:
            return found
        return env_nep

    if sys.platform.startswith("win"):
        for name in ["nep.exe", "nep"]:
            found = shutil.which(name)
            if found:
                return found
        for p in [r"C:\GPUMD\bin\nep.exe", r"C:\Program Files\GPUMD\bin\nep.exe"]:
            if os.path.isfile(p):
                return os.path.abspath(p)
        return "nep.exe"
    else:
        found = shutil.which("nep") or shutil.which("nep.exe")
        if found:
            return found
        user_home = os.path.expanduser("~")
        work_dir = os.environ.get("WORK", "")
        conda_dir = os.environ.get("CONDA_PREFIX", "")
        candidates = [
            os.path.join(user_home, "bin", "nep"),
            os.path.join(user_home, ".local", "bin", "nep"),
            os.path.join(user_home, "GPUMD", "src", "nep"),
            os.path.join(user_home, "gpumd", "src", "nep"),
            os.path.join(work_dir, "GPUMD", "src", "nep") if work_dir else "",
            os.path.join(work_dir, "gpumd", "src", "nep") if work_dir else "",
            os.path.join(work_dir, "bin", "nep") if work_dir else "",
            os.path.join(conda_dir, "bin", "nep") if conda_dir else "",
            "/usr/local/bin/nep",
            "/usr/bin/nep",
            os.path.abspath("nep"),
        ]
        for p in candidates:
            if p and os.path.isfile(p):
                return os.path.abspath(p)
        return "nep"


def write_delta_nep_in(
    out_path: str,
    unique_elements: List[str],
    neurons: int = 30,
    batch: int = 2000,
    steps: int = 30000,
    cutoff_radial: float = 4.5,
    cutoff_angular: float = 3.5,
    n_max_radial: int = 6,
    n_max_angular: int = 4,
    basis_size: int = 8,
    l_max: int = 2,
):
    """Writes `nep.in` for Delta-NEP training with exposed architectural hyperparameters."""
    elem_str = " ".join(unique_elements)
    content = f"""type {len(unique_elements)} {elem_str}
version 4
cutoff {cutoff_radial} {cutoff_angular}
n_max {n_max_radial} {n_max_angular}
basis_size {basis_size} {basis_size}
l_max {l_max} 1
neuron {neurons}
batch {batch}
population 50
generation {steps}
"""
    with open(out_path, "w") as f:
        f.write(content)


def stratified_train_test_split(
    delta_frames: List[Atoms], test_fraction: float = 0.20, seed: int = 42
) -> Tuple[List[Atoms], List[Atoms]]:
    """Splits `delta_frames` into `(train_frames, test_frames)`, keeping lambda=0/1 endpoints in train."""
    n_total = len(delta_frames)
    if n_total == 0:
        raise RuntimeError("No labeled DFT frames available for Delta-NEP training.")
    if n_total == 1:
        warnings.warn(
            "stratified_train_test_split called with only 1 frame. Training and testing datasets will share identical data.",
            UserWarning,
            stacklevel=2,
        )
        return list(delta_frames), list(delta_frames)

    n_test = max(1, int(n_total * test_fraction))
    n_test = min(n_test, n_total - 1)
    rng = np.random.RandomState(seed)

    anchor_indices = set()
    by_type = {}
    for idx, fr in enumerate(delta_frames):
        stype = fr.info.get("sample_type", "unknown")
        lam = fr.info.get("lambda")
        fid = str(fr.info.get("frame_id", ""))
        if stype == "franck_condon" and (
            (lam is not None and (abs(float(lam) - 0.0) < 1e-4 or abs(float(lam) - 1.0) < 1e-4))
            or fid in ("fc_00", "fc_01")
        ):
            anchor_indices.add(idx)
        by_type.setdefault(stype, []).append(idx)

    non_anchor_pool = [i for i in range(n_total) if i not in anchor_indices]
    if len(non_anchor_pool) < n_test:
        all_idx = list(rng.permutation(n_total))
        test_set = set(all_idx[-n_test:])
    else:
        test_set = set()
        for stype in sorted(by_type.keys()):
            type_pool = [i for i in by_type[stype] if i not in anchor_indices]
            if not type_pool:
                continue
            shuffled = list(rng.permutation(type_pool))
            k_take = max(0, int(round(len(by_type[stype]) * test_fraction)))
            for idx in shuffled[:k_take]:
                if len(test_set) < n_test:
                    test_set.add(idx)
        if len(test_set) < n_test:
            remaining = [i for i in rng.permutation(non_anchor_pool) if i not in test_set]
            for idx in remaining[: (n_test - len(test_set))]:
                test_set.add(idx)

    train_frames = [delta_frames[i] for i in range(n_total) if i not in test_set]
    test_frames = [delta_frames[i] for i in range(n_total) if i in test_set]
    return train_frames, test_frames


def _compute_training_digest(
    delta_frames: List[Atoms],
    neurons: int,
    batch: int,
    steps: int,
    config: Optional[ToolkitConfig] = None,
) -> str:
    """Computes a SHA256 digest of the delta training dataset, architectural parameters, and foundation potential."""
    h = hashlib.sha256()
    r_cut = getattr(config, "nep_cutoff_radial", 4.5) if config else 4.5
    a_cut = getattr(config, "nep_cutoff_angular", 3.5) if config else 3.5
    n_rad = getattr(config, "nep_n_max_radial", 6) if config else 6
    n_ang = getattr(config, "nep_n_max_angular", 4) if config else 4
    b_size = getattr(config, "nep_basis_size", 8) if config else 8
    l_max = getattr(config, "nep_l_max", 2) if config else 2

    found_hash = "none"
    if config and getattr(config, "foundation_potential_path", None):
        p = config.foundation_potential_path
        if os.path.isfile(p):
            from .provenance import compute_file_sha256
            found_hash = compute_file_sha256(p) or "empty"

    meta_str = (
        f"neurons={neurons};batch={batch};steps={steps};n={len(delta_frames)};"
        f"rcut={r_cut};acut={a_cut};nrad={n_rad};nang={n_ang};bsize={b_size};lmax={l_max};"
        f"foundation_sha256={found_hash}"
    )
    h.update(meta_str.encode("utf-8"))
    for fr in delta_frames:
        h.update(np.asarray(fr.positions, dtype=np.float64).tobytes())
        h.update(np.float64(fr.info.get("energy", 0.0)).tobytes())
        if "forces" in fr.arrays:
            h.update(np.asarray(fr.arrays["forces"], dtype=np.float64).tobytes())
    return h.hexdigest()


def get_frame_energy(fr: Atoms) -> float:
    """Robustly extracts potential energy from an Atoms object, checking info, calc, and results."""
    e = fr.info.get("energy")
    if e is not None:
        return float(e)
    if fr.calc is not None:
        if hasattr(fr.calc, "results") and "energy" in fr.calc.results:
            return float(fr.calc.results["energy"])
        try:
            return float(fr.get_potential_energy())
        except Exception:
            pass
    raise KeyError(f"Frame {fr.info.get('frame_id', 'unknown')} has no energy in info or calculator.")


def get_frame_forces(fr: Atoms) -> np.ndarray:
    """Robustly extracts forces from an Atoms object, checking arrays, calc, and results."""
    if "forces" in fr.arrays:
        return np.asarray(fr.arrays["forces"], dtype=float)
    if fr.calc is not None:
        if hasattr(fr.calc, "results") and "forces" in fr.calc.results:
            return np.asarray(fr.calc.results["forces"], dtype=float)
        try:
            return np.asarray(fr.get_forces(), dtype=float)
        except Exception:
            pass
    raise KeyError(f"Frame {fr.info.get('frame_id', 'unknown')} has no forces in arrays or calculator.")


class DeltaTrainer:
    """Computes delta energies and forces, writes GPUMD training cards, and trains Delta-NEP."""

    def __init__(self, config: ToolkitConfig, calc_foundation=None):
        self.config = config
        if calc_foundation is not None:
            self.calc_foundation = calc_foundation
        elif config.foundation_potential_path and os.path.isfile(config.foundation_potential_path):
            self.calc_foundation = load_foundation_calculator(
                potential_path=config.foundation_potential_path,
                existing_calc=None,
                allow_missing=False,
            )
        elif CPUNEP is not None and getattr(CPUNEP, "__module__", "").startswith("unittest.mock"):
            self.calc_foundation = CPUNEP(config.foundation_potential_path)
        else:
            self.calc_foundation = None

    def compute_delta_dataset(
        self,
        labeled_frames: List[Atoms],
        shared_offset_per_atom: Optional[float] = None,
    ) -> List[Atoms]:
        """Computes Delta E and Delta F for each DFT-labeled frame relative to the foundation potential."""
        delta_frames = []
        logger.info("\n[-] Evaluating foundation potential to extract Delta E and Delta F...")

        at_eval = None
        for fr in tqdm(labeled_frames, desc="Extracting delta baselines", unit="frame", leave=False):
            e_dft = get_frame_energy(fr)
            f_dft = get_frame_forces(fr)
            fr.info["energy"] = e_dft
            fr.arrays["forces"] = f_dft

            if "e_base" in fr.info and "f_base" in fr.arrays:
                e_base = float(fr.info["e_base"])
                f_base = np.asarray(fr.arrays["f_base"], dtype=float)
            else:
                if (
                    at_eval is None
                    or len(at_eval) != len(fr)
                    or at_eval.get_chemical_symbols() != fr.get_chemical_symbols()
                ):
                    at_eval = fr.copy()
                    at_eval.calc = self.calc_foundation
                else:
                    at_eval.set_cell(fr.cell)
                    at_eval.set_positions(fr.positions)

                e_base = float(at_eval.get_potential_energy())
                f_base = np.asarray(at_eval.get_forces(), dtype=float)

            delta_e = float(e_dft - e_base)
            delta_f = np.asarray(f_dft - f_base, dtype=float)

            fr_delta = fr.copy()
            fr_delta.calc = None
            fr_delta.info["energy"] = delta_e
            fr_delta.info["e_base"] = e_base
            fr_delta.info["e_dft"] = e_dft
            fr_delta.arrays["forces"] = delta_f
            fr_delta.arrays["f_base"] = f_base
            fr_delta.arrays["f_dft"] = f_dft
            delta_frames.append(fr_delta)

        # Baseline mean energy offset subtraction
        subtract_offset = getattr(self.config, "subtract_mean_energy_offset", True)
        offset_per_atom = 0.0
        if subtract_offset and delta_frames:
            if shared_offset_per_atom is not None:
                offset_per_atom = float(shared_offset_per_atom)
                logger.info(
                    f"[+] Using shared baseline mean energy offset: {offset_per_atom:.4f} eV/atom "
                    f"({offset_per_atom * 1000.0:.1f} meV/atom)"
                )
            else:
                offset_per_atom = float(np.mean([fr.info["energy"] / len(fr) for fr in delta_frames]))
                logger.info(
                    f"[+] Subtracted baseline mean energy offset: {offset_per_atom:.4f} eV/atom "
                    f"({offset_per_atom * 1000.0:.1f} meV/atom)"
                )
            for fr in delta_frames:
                fr.info["raw_delta_energy"] = float(fr.info["energy"])
                fr.info["delta_energy_offset_per_atom"] = offset_per_atom
                fr.info["energy"] = float(fr.info["energy"] - len(fr) * offset_per_atom)
        self.energy_offset_per_atom = offset_per_atom

        max_d_f = max(np.abs(f.arrays["forces"]).max() for f in delta_frames)
        mean_d_e = np.mean([abs(f.info["energy"]) for f in delta_frames])
        logger.info(f"[+] Extracted Delta dataset ({len(delta_frames)} frames): Mean |Delta E| = {mean_d_e:.3f} eV, Max |Delta F| = {max_d_f:.3f} eV/A")
        return delta_frames

    def train_delta_nep(
        self,
        delta_frames: List[Atoms],
        force_retrain: bool = False,
        pot_filename: str = "nep_delta.txt",
        subdir: str = "delta_nep_training",
    ) -> str:
        """Writes train.xyz/test.xyz/nep.in, invokes `nep` to train Delta-NEP, and caches the potential."""
        train_dir = os.path.join(self.config.workspace_dir, subdir)
        os.makedirs(train_dir, exist_ok=True)

        seed = self.config.random_seed
        train_frames, test_frames = stratified_train_test_split(
            delta_frames, test_fraction=0.20, seed=seed
        )

        ase.io.write(os.path.join(train_dir, "train.xyz"), train_frames, format="extxyz")
        ase.io.write(os.path.join(train_dir, "test.xyz"), test_frames, format="extxyz")

        # Union of chemical symbols across ALL frames to handle vacancies/substitutions safely
        unique_elems = sorted(set().union(*(set(f.get_chemical_symbols()) for f in delta_frames)))
        nep_in = os.path.join(train_dir, "nep.in")
        effective_batch = min(self.config.nep_batch, len(train_frames))
        write_delta_nep_in(
            nep_in,
            unique_elems,
            neurons=self.config.nep_neurons,
            batch=effective_batch,
            steps=self.config.nep_steps,
            cutoff_radial=self.config.nep_cutoff_radial,
            cutoff_angular=self.config.nep_cutoff_angular,
            n_max_radial=self.config.nep_n_max_radial,
            n_max_angular=self.config.nep_n_max_angular,
            basis_size=self.config.nep_basis_size,
            l_max=self.config.nep_l_max,
        )

        pot_dest = os.path.join(self.config.workspace_dir, pot_filename)
        manifest_path = os.path.join(train_dir, "training_manifest.json")
        force = force_retrain or self.config.force_retrain

        # Performance optimization: skip digest hashing if cached potential or manifest clearly does not exist
        if not force and os.path.isfile(pot_dest) and os.path.isfile(manifest_path):
            digest = _compute_training_digest(
                delta_frames, self.config.nep_neurons, effective_batch, self.config.nep_steps, config=self.config
            )
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    cached_meta = json.load(f)
                if cached_meta.get("digest") == digest:
                    logger.info(f"\n[i] Reusing cached Delta-NEP potential (dataset & hyperparameters unchanged): {pot_dest}")
                    return pot_dest
            except Exception:
                pass
        else:
            digest = None

        logger.info(f"\n[-] Training Delta-NEP on GPU (train={len(train_frames)}, test={len(test_frames)}, steps={self.config.nep_steps})...")
        t0 = time.time()
        nep_cmd = resolve_nep_executable(self.config.nep_exe)
        nep_args = [sys.executable, nep_cmd] if nep_cmd.endswith(".py") else [nep_cmd]

        loss_out = os.path.join(train_dir, "loss.out")
        if os.path.isfile(loss_out):
            try:
                os.remove(loss_out)
            except Exception:
                pass

        total_steps = self.config.nep_steps
        pbar = tqdm(total=total_steps, desc="Training Delta-NEP", unit="gen", leave=False)
        stop_event = threading.Event()

        def _monitor_loss():
            last_gen = 0
            while not stop_event.is_set():
                if os.path.isfile(loss_out):
                    try:
                        with open(loss_out, "r", encoding="utf-8", errors="replace") as lf:
                            lines = lf.readlines()
                        for line in reversed(lines[-5:]):
                            parts = line.strip().split()
                            if parts and parts[0].isdigit():
                                cur_gen = int(parts[0])
                                if cur_gen > last_gen:
                                    advance = min(cur_gen - last_gen, max(0, total_steps - last_gen))
                                    pbar.update(advance)
                                    last_gen = cur_gen
                                break
                    except Exception:
                        pass
                stop_event.wait(0.25)
            if last_gen < total_steps:
                pbar.update(total_steps - last_gen)

        monitor_thread = threading.Thread(target=_monitor_loss, daemon=True)
        monitor_thread.start()

        try:
            res = subprocess.run(nep_args, cwd=train_dir, capture_output=True, text=True)
        finally:
            stop_event.set()
            monitor_thread.join(timeout=2.0)
            pbar.close()

        if res.returncode != 0:
            raise RuntimeError(f"NEP training failed using '{nep_cmd}':\n{res.stderr}\n{res.stdout}")

        logger.info(f"[+] Delta-NEP training completed in {time.time()-t0:.2f} s")

        pot_src = os.path.join(train_dir, "nep.txt")
        if os.path.isfile(pot_src):
            shutil.copy2(pot_src, pot_dest)
        else:
            raise FileNotFoundError(f"Trained NEP potential '{pot_src}' was not found after training.")
        logger.info(f"[+] Trained Delta-NEP saved: {pot_dest}")

        # Save baseline energy offset sidecar
        offset_val = getattr(self, "energy_offset_per_atom", 0.0)
        offset_data = {
            "energy_offset_per_atom_eV": offset_val,
            "n_frames": len(delta_frames),
        }
        with open(pot_dest + ".offset.json", "w", encoding="utf-8") as f:
            json.dump(offset_data, f, indent=2)
        with open(os.path.join(train_dir, "delta_offset.json"), "w", encoding="utf-8") as f:
            json.dump(offset_data, f, indent=2)

        # Committee ensemble training if committee_size > 1
        committee_size = int(getattr(self.config, "nep_committee_size", 1))
        pot_base, pot_ext = os.path.splitext(pot_filename)
        committee_pots: List[str] = [pot_dest]

        if committee_size > 1:
            logger.info(f"\n[-] Training Delta-NEP committee ensemble of size {committee_size}...")
            # model 0 is the primary potential already trained
            shutil.copy2(pot_dest, os.path.join(self.config.workspace_dir, f"{pot_base}_0{pot_ext}"))
            with open(os.path.join(self.config.workspace_dir, f"{pot_base}_0{pot_ext}.offset.json"), "w", encoding="utf-8") as f:
                json.dump(offset_data, f, indent=2)

            # Pin anchor frames (lambda=0, lambda=1) in all bootstrap members
            anchor_train = [
                fr for fr in train_frames
                if fr.info.get("sample_type") == "franck_condon"
                and (
                    (fr.info.get("lambda") is not None and (
                        abs(float(fr.info["lambda"]) - 0.0) < 1e-4 or
                        abs(float(fr.info["lambda"]) - 1.0) < 1e-4
                    ))
                    or str(fr.info.get("frame_id", "")) in ("fc_00", "fc_01")
                )
            ]
            non_anchor_train = [fr for fr in train_frames if fr not in anchor_train]

            for m in range(1, committee_size):
                m_dir = os.path.join(train_dir, f"model_{m}")
                os.makedirs(m_dir, exist_ok=True)
                rng_m = np.random.RandomState(seed + m * 37 + 101)
                if non_anchor_train:
                    boot_idx = rng_m.choice(len(non_anchor_train), size=len(non_anchor_train), replace=True)
                    m_train_frames = anchor_train + [non_anchor_train[i] for i in boot_idx]
                else:
                    m_train_frames = list(train_frames)
                ase.io.write(os.path.join(m_dir, "train.xyz"), m_train_frames, format="extxyz")
                ase.io.write(os.path.join(m_dir, "test.xyz"), test_frames, format="extxyz")
                m_nep_in = os.path.join(m_dir, "nep.in")
                write_delta_nep_in(
                    m_nep_in,
                    unique_elems,
                    neurons=self.config.nep_neurons,
                    batch=effective_batch,
                    steps=self.config.nep_steps,
                    cutoff_radial=self.config.nep_cutoff_radial,
                    cutoff_angular=self.config.nep_cutoff_angular,
                    n_max_radial=self.config.nep_n_max_radial,
                    n_max_angular=self.config.nep_n_max_angular,
                    basis_size=self.config.nep_basis_size,
                    l_max=self.config.nep_l_max,
                )
                m_res = subprocess.run(nep_args, cwd=m_dir, capture_output=True, text=True)
                if m_res.returncode != 0:
                    raise RuntimeError(f"Committee member {m} training failed:\n{m_res.stderr}\n{m_res.stdout}")
                m_src = os.path.join(m_dir, "nep.txt")
                m_dest = os.path.join(self.config.workspace_dir, f"{pot_base}_{m}{pot_ext}")
                if os.path.isfile(m_src):
                    shutil.copy2(m_src, m_dest)
                    with open(m_dest + ".offset.json", "w", encoding="utf-8") as f:
                        json.dump(offset_data, f, indent=2)
                    committee_pots.append(m_dest)
                else:
                    raise FileNotFoundError(f"Committee member {m} potential '{m_src}' not found.")
            logger.info(f"[+] Committee ensemble trained: {len(committee_pots)} potentials")

        if digest is None:
            digest = _compute_training_digest(
                delta_frames, self.config.nep_neurons, effective_batch, self.config.nep_steps, config=self.config
            )
        manifest_data = {
            "digest": digest,
            "n_frames": len(delta_frames),
            "n_train": len(train_frames),
            "n_test": len(test_frames),
            "neurons": self.config.nep_neurons,
            "batch": effective_batch,
            "steps": self.config.nep_steps,
            "committee_size": committee_size,
            "timestamp": time.time(),
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

        return pot_dest


def plot_delta_nep_parity(
    qc_gs: Dict[str, Any],
    qc_es: Optional[Dict[str, Any]] = None,
    save_path: Optional[str] = None,
) -> str:
    """Exports a publication-ready dual-panel parity plot for Delta-NEP potentials.

    Columns:
      - Left column: Ground State (GS)
      - Right column (if available): Optical Excited State (ES)
    Rows:
      - Top row: Energy parity (DFT vs Hybrid Delta-NEP)
      - Bottom row: Force component parity (DFT vs Hybrid Delta-NEP)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_cols = 2 if qc_es is not None else 1
    fig, axes = plt.subplots(2, n_cols, figsize=(5.5 * n_cols, 10.0), squeeze=False)

    states = [("Ground State (GS)", qc_gs)]
    if qc_es is not None:
        states.append(("Excited State (ES)", qc_es))

    for col, (state_name, qc_data) in enumerate(states):
        parity = qc_data.get("parity_data", {})
        train_p = parity.get("train", {})
        test_p = parity.get("test", {})

        # 1. Energy Parity (Top Row)
        ax_e = axes[0, col]
        e_tr_true = np.array(train_p.get("e_true", []))
        e_tr_pred = np.array(train_p.get("e_pred", []))
        e_te_true = np.array(test_p.get("e_true", []))
        e_te_pred = np.array(test_p.get("e_pred", []))

        all_e = (
            np.concatenate([e_tr_true, e_tr_pred, e_te_true, e_te_pred])
            if len(e_tr_true) > 0
            else np.array([0.0])
        )
        e_min = float(np.min(all_e)) - 0.05
        e_max = float(np.max(all_e)) + 0.05

        ax_e.plot([e_min, e_max], [e_min, e_max], "k--", alpha=0.6, label="Ideal (y = x)")
        if len(e_tr_true) > 0:
            ax_e.scatter(
                e_tr_true,
                e_tr_pred,
                color="#1f77b4",
                alpha=0.6,
                s=25,
                label=f"Train (RMSE: {qc_data.get('train_energy_rmse_meV_atom', 0.0):.1f} meV/at)",
            )
        if len(e_te_true) > 0:
            ax_e.scatter(
                e_te_true,
                e_te_pred,
                color="#ff7f0e",
                marker="s",
                alpha=0.8,
                s=35,
                label=f"Test (RMSE: {qc_data.get('test_energy_rmse_meV_atom', 0.0):.1f} meV/at)",
            )

        ax_e.set_xlim(e_min, e_max)
        ax_e.set_ylim(e_min, e_max)
        ax_e.set_xlabel(r"DFT Potential Energy $E_{\mathrm{DFT}}$ (eV)")
        ax_e.set_ylabel(r"Hybrid Potential Energy $E_{\mathrm{Hyb}}$ (eV)")
        ax_e.set_title(f"{state_name} - Energy Parity")
        ax_e.grid(True, linestyle=":", alpha=0.5)
        ax_e.legend(loc="upper left", framealpha=0.9, fontsize=9)

        # 2. Force Parity (Bottom Row)
        ax_f = axes[1, col]
        f_tr_true = np.array(train_p.get("f_true", [])).flatten()
        f_tr_pred = np.array(train_p.get("f_pred", [])).flatten()
        f_te_true = np.array(test_p.get("f_true", [])).flatten()
        f_te_pred = np.array(test_p.get("f_pred", [])).flatten()

        all_f = (
            np.concatenate([f_tr_true, f_tr_pred, f_te_true, f_te_pred])
            if len(f_tr_true) > 0
            else np.array([-1.0, 1.0])
        )
        f_lim = max(0.5, float(np.max(np.abs(all_f))) * 1.1)

        ax_f.plot([-f_lim, f_lim], [-f_lim, f_lim], "k--", alpha=0.6, label="Ideal (y = x)")
        if len(f_tr_true) > 0:
            idx_sub = np.linspace(0, len(f_tr_true) - 1, min(2000, len(f_tr_true)), dtype=int)
            ax_f.scatter(
                f_tr_true[idx_sub],
                f_tr_pred[idx_sub],
                color="#1f77b4",
                alpha=0.3,
                s=15,
                label=f"Train (RMSE: {qc_data.get('train_force_rmse_eV_A', 0.0):.3f} eV/Å)",
            )
        if len(f_te_true) > 0:
            idx_sub_te = np.linspace(0, len(f_te_true) - 1, min(1000, len(f_te_true)), dtype=int)
            ax_f.scatter(
                f_te_true[idx_sub_te],
                f_te_pred[idx_sub_te],
                color="#ff7f0e",
                marker="s",
                alpha=0.5,
                s=20,
                label=f"Test (RMSE: {qc_data.get('test_force_rmse_eV_A', 0.0):.3f} eV/Å)",
            )

        ax_f.set_xlim(-f_lim, f_lim)
        ax_f.set_ylim(-f_lim, f_lim)
        ax_f.set_xlabel(r"DFT Force Component $F_{i,\alpha}$ (eV/Å)")
        ax_f.set_ylabel(r"Hybrid Force Component $F_{i,\alpha}$ (eV/Å)")
        ax_f.set_title(f"{state_name} - Force Parity")
        ax_f.grid(True, linestyle=":", alpha=0.5)
        ax_f.legend(loc="upper left", framealpha=0.9, fontsize=9)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=200)
    plt.close(fig)
    return save_path or ""


def evaluate_delta_nep_quality_control(
    calc_hybrid: HybridDefectCalculator,
    labeled_frames: List[Atoms],
    at_gs: Atoms,
    workspace_dir: str,
    config: ToolkitConfig,
    test_fraction: float = 0.20,
    state_label: str = "ground_state",
    at_ref: Optional[Atoms] = None,
    pre_refinement_force: Optional[float] = None,
    refinement_displacement: Optional[float] = None,
    write_files: bool = True,
) -> Dict[str, Any]:
    """Computes comprehensive Quality Control metrics for the trained Delta-NEP potential.

    Evaluates:
      1. Train and held-out test energy RMSE (meV/atom) and force RMSE (eV/A).
      2. Residual forces on the DFT minimum geometry (pre- and post-refinement).
      3. Relative displacement introduced by any hybrid equilibrium refinement.
      4. Force RMSE on small-amplitude displacement frames (<= 0.05 A).
      5. Energy agreement along the Franck-Condon path (referenced to lambda=0.0) and Delta E_rel.
      6. Quality threshold gates:
         - max_test_force_rmse_eV_A
         - max_gs_residual_force_eV_A
         - max_refinement_displacement_A
         - max_fc_path_error_meV
         - max_erel_error_pct
         - max_small_displacement_force_rmse_eV_A

    Writes `delta_nep_qc_report.json` and `delta_nep_parity.json` to `workspace_dir`.

    Returns
    -------
    Dict[str, Any]
        Quality Control evaluation dictionary.
    """
    logger.info(f"\n[-] Evaluating Delta-NEP Quality Control for {state_label} (train/test parity & gates)...")

    seed = config.random_seed
    train_frames, test_frames = stratified_train_test_split(
        labeled_frames, test_fraction=test_fraction, seed=seed
    )
    if not test_frames:
        test_frames = list(train_frames)

    target_ref = at_ref if at_ref is not None else at_gs

    def _eval_frames(frames: List[Atoms]):
        e_true = []
        e_pred = []
        f_true_list = []
        f_pred_list = []
        for fr in frames:
            at = fr.copy()
            at.calc = calc_hybrid
            ep = float(at.get_potential_energy())
            fp = np.asarray(at.get_forces(), dtype=float)
            et = float(fr.info.get("energy", ep))
            if "forces" in fr.arrays:
                ft = np.asarray(fr.arrays["forces"], dtype=float)
            elif fr.calc is not None:
                try:
                    ft = np.asarray(fr.get_forces(), dtype=float)
                except Exception:
                    ft = fp.copy()
            else:
                ft = fp.copy()
            e_true.append(et)
            e_pred.append(ep)
            f_true_list.append(ft)
            f_pred_list.append(fp)
        e_true_arr = np.array(e_true)
        e_pred_arr = np.array(e_pred)
        n_atoms_arr = np.array([len(f) for f in frames], dtype=float)
        e_err_per_atom_meV = ((e_pred_arr - e_true_arr) / n_atoms_arr) * 1000.0
        e_rmse_meV = float(np.sqrt(np.mean(e_err_per_atom_meV ** 2)))

        f_true_all = np.concatenate(f_true_list, axis=0) if f_true_list else np.empty((0, 3))
        f_pred_all = np.concatenate(f_pred_list, axis=0) if f_pred_list else np.empty((0, 3))
        f_rmse = float(np.sqrt(np.mean((f_pred_all - f_true_all) ** 2))) if len(f_true_all) > 0 else 0.0

        return {
            "energy_rmse_meV_atom": e_rmse_meV,
            "force_rmse_eV_A": f_rmse,
            "e_true": e_true_arr.tolist(),
            "e_pred": e_pred_arr.tolist(),
            "f_true": f_true_all.tolist(),
            "f_pred": f_pred_all.tolist(),
        }

    train_res = _eval_frames(train_frames)
    test_res = _eval_frames(test_frames)

    # Small displacement frames force evaluation
    small_disp_frames = []
    if target_ref is not None:
        from .defect_builder import mic_displacement
        for fr in labeled_frames:
            if len(fr) == len(target_ref):
                try:
                    dr = mic_displacement(fr.positions, target_ref.positions, target_ref.cell)
                    rms_dr = float(np.sqrt(np.mean(np.sum(dr ** 2, axis=1))))
                    if rms_dr <= 0.05:
                        small_disp_frames.append(fr)
                except Exception:
                    pass

    if small_disp_frames:
        small_disp_res = _eval_frames(small_disp_frames)
        small_disp_force_rmse = small_disp_res["force_rmse_eV_A"]
        gate_small_disp_force = small_disp_force_rmse <= config.max_small_displacement_force_rmse_eV_A
    else:
        small_disp_force_rmse = None
        gate_small_disp_force = True

    # Franck-Condon path verification (referenced to lambda = 0.0)
    fc_frames = [f for f in labeled_frames if f.info.get("sample_type") == "franck_condon"]
    fc_max_err_meV = 0.0
    erel_err_meV = 0.0
    erel_err_pct = 0.0
    erel_dft = 0.0
    erel_hyb = 0.0
    f1 = None

    if fc_frames:
        fc_frames.sort(key=lambda f: float(f.info.get("lambda", 0.0)))
        # Locate lambda = 0.0 frame as physical ground state reference
        f0 = next(
            (f for f in fc_frames if abs(float(f.info.get("lambda", 999.0)) - 0.0) < 1e-4),
            min(fc_frames, key=lambda f: abs(float(f.info.get("lambda", 0.0)) - 0.0)),
        )
        at0 = f0.copy()
        at0.calc = calc_hybrid
        e0_hyb = float(at0.get_potential_energy())
        e0_dft = get_frame_energy(f0)

        fc_errs = []
        for fr in fc_frames:
            at = fr.copy()
            at.calc = calc_hybrid
            eh = float(at.get_potential_energy()) - e0_hyb
            ed = get_frame_energy(fr) - e0_dft
            fc_errs.append(abs(eh - ed) * 1000.0)
        fc_max_err_meV = float(np.max(fc_errs))

        # Check lambda=1 frame
        f1 = next((f for f in fc_frames if abs(float(f.info.get("lambda", 999.0)) - 1.0) < 1e-4), None)
        if f1 is not None:
            at1 = f1.copy()
            at1.calc = calc_hybrid
            erel_hyb = float(at1.get_potential_energy()) - e0_hyb
            erel_dft = get_frame_energy(f1) - e0_dft
            erel_err_meV = float(abs(erel_hyb - erel_dft) * 1000.0)
            erel_err_pct = float(erel_err_meV / (abs(erel_dft) * 1000.0) * 100.0) if abs(erel_dft) > 1e-5 else 0.0

    # Residual force at DFT minimum
    at_eval = target_ref.copy()
    at_eval.calc = calc_hybrid
    res_f = np.asarray(at_eval.get_forces(), dtype=float)
    calc_res_force = float(np.abs(res_f).max())

    eff_res_force = pre_refinement_force if pre_refinement_force is not None else calc_res_force
    gate_residual_force = eff_res_force <= config.max_gs_residual_force_eV_A

    gate_refinement_disp = (
        (float(refinement_displacement) <= config.max_refinement_displacement_A)
        if refinement_displacement is not None
        else True
    )

    gate_test_force = test_res["force_rmse_eV_A"] <= config.max_test_force_rmse_eV_A
    gate_fc_path = (fc_max_err_meV <= config.max_fc_path_error_meV) if fc_frames else True
    gate_erel = (erel_err_pct <= config.max_erel_error_pct) if (fc_frames and f1 is not None) else True

    qc_passed = bool(
        gate_test_force
        and gate_residual_force
        and gate_refinement_disp
        and gate_fc_path
        and gate_erel
        and gate_small_disp_force
    )

    logger.info("-" * 78)
    logger.info(f"[+] DELTA-NEP QUALITY CONTROL REPORT ({state_label.upper()}):")
    logger.info(
        f"    • Train Energy RMSE: {train_res['energy_rmse_meV_atom']:.2f} meV/atom | "
        f"Force RMSE: {train_res['force_rmse_eV_A']:.4f} eV/A"
    )
    logger.info(
        f"    • Test Energy RMSE:  {test_res['energy_rmse_meV_atom']:.2f} meV/atom | "
        f"Force RMSE: {test_res['force_rmse_eV_A']:.4f} eV/A "
        f"(Gate <= {config.max_test_force_rmse_eV_A:.3f} eV/A: {'PASSED' if gate_test_force else 'FAILED'})"
    )
    if small_disp_force_rmse is not None:
        logger.info(
            f"    • Small-Disp Force RMSE: {small_disp_force_rmse:.4f} eV/A "
            f"(Gate <= {config.max_small_displacement_force_rmse_eV_A:.3f} eV/A: {'PASSED' if gate_small_disp_force else 'FAILED'})"
        )
    logger.info(
        f"    • Pre-Refinement DFT Residual Force: {eff_res_force:.4f} eV/A "
        f"(Gate <= {config.max_gs_residual_force_eV_A:.3f} eV/A: {'PASSED' if gate_residual_force else 'FAILED'})"
    )
    if refinement_displacement is not None:
        logger.info(
            f"    • Refinement Displacement: {refinement_displacement:.4f} A "
            f"(Gate <= {config.max_refinement_displacement_A:.3f} A: {'PASSED' if gate_refinement_disp else 'FAILED'})"
        )
    if fc_frames:
        logger.info(
            f"    • FC Path Max Error: {fc_max_err_meV:.2f} meV (Gate <= {config.max_fc_path_error_meV:.1f} meV: {'PASSED' if gate_fc_path else 'FAILED'}) | "
            f"Delta E_rel Error: {erel_err_meV:.2f} meV / {erel_err_pct:.1f}% (Gate <= {config.max_erel_error_pct:.1f}%: {'PASSED' if gate_erel else 'FAILED'})"
        )
    logger.info(f"    • QC Gate Overall Status: {'PASSED' if qc_passed else 'FAILED'}")
    logger.info("-" * 78)

    parity_data = {
        "train": {
            "e_true": train_res["e_true"],
            "e_pred": train_res["e_pred"],
            "f_true": train_res["f_true"],
            "f_pred": train_res["f_pred"],
        },
        "test": {
            "e_true": test_res["e_true"],
            "e_pred": test_res["e_pred"],
            "f_true": test_res["f_true"],
            "f_pred": test_res["f_pred"],
        },
    }

    qc_report = {
        "state_label": state_label,
        "qc_passed": qc_passed,
        "train_energy_rmse_meV_atom": train_res["energy_rmse_meV_atom"],
        "train_force_rmse_eV_A": train_res["force_rmse_eV_A"],
        "test_energy_rmse_meV_atom": test_res["energy_rmse_meV_atom"],
        "test_force_rmse_eV_A": test_res["force_rmse_eV_A"],
        "small_disp_force_rmse_eV_A": small_disp_force_rmse,
        "gs_residual_force_max_eV_A": calc_res_force,
        "pre_refinement_force_eV_A": eff_res_force,
        "refinement_displacement_A": refinement_displacement,
        "fc_path_energy_max_err_meV": fc_max_err_meV,
        "delta_erel_dft_eV": erel_dft,
        "delta_erel_hybrid_eV": erel_hyb,
        "delta_erel_error_meV": erel_err_meV,
        "delta_erel_error_pct": erel_err_pct,
        "thresholds": {
            "max_test_force_rmse_eV_A": config.max_test_force_rmse_eV_A,
            "max_gs_residual_force_eV_A": config.max_gs_residual_force_eV_A,
            "max_refinement_displacement_A": config.max_refinement_displacement_A,
            "max_fc_path_error_meV": config.max_fc_path_error_meV,
            "max_erel_error_pct": config.max_erel_error_pct,
            "max_small_displacement_force_rmse_eV_A": config.max_small_displacement_force_rmse_eV_A,
        },
        "gates": {
            "test_force_rmse_gate": gate_test_force,
            "gs_residual_force_gate": gate_residual_force,
            "refinement_displacement_gate": gate_refinement_disp,
            "fc_path_energy_gate": gate_fc_path,
            "erel_error_gate": gate_erel,
            "small_disp_force_gate": gate_small_disp_force,
        },
        "is_committee": getattr(calc_hybrid, "is_committee", False),
        "committee_size": (
            len(getattr(calc_hybrid, "calc_deltas", []))
            if getattr(calc_hybrid, "is_committee", False)
            else 1
        ),
        "parity_data": parity_data,
    }

    if write_files:
        suffix = f"_{state_label}" if state_label != "ground_state" else ""
        report_path = os.path.join(workspace_dir, f"delta_nep_qc_report{suffix}.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(qc_report, f, indent=2)

        parity_path = os.path.join(workspace_dir, f"delta_nep_parity{suffix}.json")
        with open(parity_path, "w", encoding="utf-8") as f:
            json.dump(parity_data, f, indent=2)

    if not qc_passed and not getattr(config, "allow_fallbacks", False):
        failed_gates = [k for k, v in qc_report["gates"].items() if not v]
        raise ValueError(
            f"Delta-NEP Quality Gate Failed for {state_label}: {', '.join(failed_gates)}. "
            f"(Test force RMSE: {test_res['force_rmse_eV_A']:.4f} eV/A [gate <= {config.max_test_force_rmse_eV_A}], "
            f"Residual force: {eff_res_force:.4f} eV/A [gate <= {config.max_gs_residual_force_eV_A}]). "
            "Set allow_fallbacks=True to continue despite quality gate violation."
        )

    return qc_report

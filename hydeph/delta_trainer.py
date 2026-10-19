"""Extracts Delta E and Delta F relative to the foundation potential and trains Delta-NEP in GPUMD."""

import os
import sys
import time
import json
import shutil
import hashlib
import subprocess
import numpy as np
import ase.io
from ase import Atoms
from calorine.calculators import CPUNEP


def resolve_nep_executable(nep_exe=None):
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
        for p in ["/usr/local/bin/nep", "/usr/bin/nep"]:
            if os.path.isfile(p):
                return p
        return "/usr/local/bin/nep"


def write_delta_nep_in(out_path, unique_elements, neurons=30, batch=2000, steps=30000):
    """Writes `nep.in` for Delta-NEP training."""
    elem_str = " ".join(unique_elements)
    content = f"""type {len(unique_elements)} {elem_str}
version 4
cutoff 4.5 3.5
n_max 6 4
basis_size 8 8
l_max 2 1
neuron {neurons}
batch {batch}
population 50
generation {steps}
"""
    with open(out_path, "w") as f:
        f.write(content)


def stratified_train_test_split(delta_frames, test_fraction=0.20, seed=42):
    """Splits `delta_frames` into `(train_frames, test_frames)`, keeping lambda=0/1 endpoints in train."""
    n_total = len(delta_frames)
    if n_total == 0:
        raise RuntimeError("No labeled DFT frames available for Delta-NEP training.")
    if n_total == 1:
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
        all_idx = list( rng.permutation(n_total) )
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


def _compute_training_digest(delta_frames, neurons, batch, steps) -> str:
    """Computes a SHA256 digest of the delta training dataset and NEP hyperparameters."""
    h = hashlib.sha256()
    h.update(f"neurons={neurons};batch={batch};steps={steps};n={len(delta_frames)}".encode("utf-8"))
    for fr in delta_frames:
        h.update(np.asarray(fr.positions, dtype=np.float64).tobytes())
        h.update(np.float64(fr.info.get("energy", 0.0)).tobytes())
        if "forces" in fr.arrays:
            h.update(np.asarray(fr.arrays["forces"], dtype=np.float64).tobytes())
    return h.hexdigest()


class DeltaTrainer:
    def __init__(self, config, calc_foundation=None):
        self.config = config
        if calc_foundation is not None:
            self.calc_foundation = calc_foundation
        elif getattr(config, "foundation_potential_path", None):
            try:
                self.calc_foundation = CPUNEP(config.foundation_potential_path)
            except Exception:
                self.calc_foundation = None
        else:
            self.calc_foundation = None

    def compute_delta_dataset(self, labeled_frames):
        """Computes Delta E and Delta F for each DFT-labeled frame relative to the foundation potential."""
        delta_frames = []
        print("\n[-] Evaluating foundation potential (nep89) to extract Delta E and Delta F...")

        at_eval = None
        for fr in labeled_frames:
            e_dft = fr.info["energy"]
            f_dft = fr.arrays["forces"]

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
                f_base = np.asarray(at_eval.get_forces(), dtype=float).copy()
                fr.info["e_base"] = e_base
                fr.arrays["f_base"] = f_base

            fr_delta = fr.copy()
            fr_delta.info["energy"] = e_dft - e_base
            fr_delta.arrays["forces"] = f_dft - f_base
            delta_frames.append(fr_delta)

        print(f"[+] Computed Delta properties for {len(delta_frames)} frames.")
        return delta_frames

    def train_delta_nep(self, delta_frames, test_fraction=0.20, force_retrain=False):
        """Writes stratified train/test splits and runs GPUMD `nep` training (reusing cached model if unchanged)."""
        train_dir = os.path.join(self.config.workspace_dir, "delta_nep_training")
        os.makedirs(train_dir, exist_ok=True)

        seed = getattr(self.config, "random_seed", 42)
        train_frames, test_frames = stratified_train_test_split(
            delta_frames, test_fraction=test_fraction, seed=seed
        )

        ase.io.write(os.path.join(train_dir, "train.xyz"), train_frames, format="extxyz")
        ase.io.write(os.path.join(train_dir, "test.xyz"), test_frames, format="extxyz")

        unique_elems = sorted(list(set(delta_frames[0].get_chemical_symbols())))
        nep_in = os.path.join(train_dir, "nep.in")
        effective_batch = min(self.config.nep_batch, len(train_frames))
        write_delta_nep_in(
            nep_in,
            unique_elems,
            neurons=self.config.nep_neurons,
            batch=effective_batch,
            steps=self.config.nep_steps,
        )

        pot_dest = os.path.join(self.config.workspace_dir, "nep_delta.txt")
        manifest_path = os.path.join(train_dir, "training_manifest.json")
        digest = _compute_training_digest(
            delta_frames, self.config.nep_neurons, effective_batch, self.config.nep_steps
        )
        force = force_retrain or getattr(self.config, "force_retrain", False)

        if not force and os.path.isfile(pot_dest) and os.path.isfile(manifest_path):
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    cached_meta = json.load(f)
                if cached_meta.get("digest") == digest:
                    print(f"\n[i] Reusing cached Delta-NEP potential (dataset & hyperparameters unchanged): {pot_dest}")
                    return pot_dest
            except Exception:
                pass

        print(f"\n[-] Training Delta-NEP on GPU (train={len(train_frames)}, test={len(test_frames)}, steps={self.config.nep_steps})...")
        t0 = time.time()
        nep_cmd = resolve_nep_executable(getattr(self.config, "nep_exe", None))
        res = subprocess.run([nep_cmd], cwd=train_dir, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"NEP training failed using '{nep_cmd}':\n{res.stderr}\n{res.stdout}")

        print(f"[+] Delta-NEP training completed in {time.time()-t0:.2f} s")

        pot_src = os.path.join(train_dir, "nep.txt")
        if os.path.isfile(pot_src):
            shutil.copy2(pot_src, pot_dest)
        else:
            raise FileNotFoundError(f"Trained NEP potential '{pot_src}' was not found after training.")
        print(f"[+] Trained Delta-NEP saved: {pot_dest}")

        try:
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump({"digest": digest, "n_train": len(train_frames), "n_test": len(test_frames)}, f, indent=2)
        except Exception:
            pass

        loss_path = os.path.join(train_dir, "loss.out")
        if os.path.isfile(loss_path):
            d_loss = np.loadtxt(loss_path)
            if len(d_loss.shape) > 1 and len(d_loss) > 0:
                final_loss = d_loss[-1]
                print(f"    - Final Loss: Total={final_loss[0]:.4e}, Energy={final_loss[1]:.4e}, Force={final_loss[2]:.4e}")

        return pot_dest

"""Quantum ESPRESSO calculation orchestrator with caching and multithreaded execution."""

from __future__ import annotations

import os
import glob
import time
import shutil
import threading
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Tuple, Dict, Any, Union
import numpy as np
import ase.io
from ase import Atoms

from ..config import (
    ToolkitConfig,
    inspect_upf_metadata,
    resolve_pseudo_path,
)
from .io import (
    is_windows,
    resolve_pw_executable,
    build_qe_command,
    build_delta_scf_occupations,
    write_qe_input,
    check_pw_in_matches_atoms,
)
from .parsers import (
    parse_qe_output,
    parse_qe_relaxation_output,
    parse_qe_band_gap,
    resolve_delta_scf_spec,
    find_frame_calc_dir,
    extract_and_attach_adiabatic_energetics,
)
from .slurm import (
    format_slurm_module_lines,
    run_monitored_qe_subprocess,
)

logger = logging.getLogger(__name__)

_EXCLUDED_ATOMS_INFO_GAP_KEYS = frozenset({"level_diagram", "u_bands", "d_bands", "bands"})


def _attach_gap_info(atoms: Atoms, gap_info: Optional[Dict[str, Any]]) -> None:
    """Attaches scalar band-gap metadata to ``atoms.info``, omitting multiline diagrams and band lists."""
    if not gap_info:
        return
    for k, v in gap_info.items():
        if k in _EXCLUDED_ATOMS_INFO_GAP_KEYS:
            continue
        if isinstance(v, str) and ("\n" in v or "\r" in v):
            continue
        atoms.info[f"dft_{k}"] = v


class QERunner:
    """Manages Quantum ESPRESSO calculations with parallel execution and smart resumption."""

    def __init__(
        self,
        config: ToolkitConfig,
        n_procs: Optional[int] = None,
        pw_exe: Optional[str] = None,
        mpi_command: Optional[str] = None,
        mpi_args: Optional[List[str]] = None,
        n_workers: Optional[int] = None,
        timeout_s: Optional[float] = None,
    ):
        self.config = config
        self.n_procs = int(n_procs if n_procs is not None else config.n_procs_qe)
        self._pw_exe_override = pw_exe
        self._resolved_pw_exe: Optional[str] = None
        self.mpi_command = mpi_command or config.mpi_command
        self.mpi_args = mpi_args or config.mpi_args
        self.n_workers = int(n_workers if n_workers is not None else config.n_workers_qe)
        self.timeout_s = timeout_s if timeout_s is not None else config.qe_timeout_s
        self.last_failed_entries: List[Dict[str, Any]] = []

    @property
    def pw_exe(self) -> str:
        """Resolves the Quantum ESPRESSO pw executable path lazily on first access."""
        if self._resolved_pw_exe is None:
            raw = self._pw_exe_override or self.config.pw_exe
            self._resolved_pw_exe = resolve_pw_executable(raw)
        return self._resolved_pw_exe

    @pw_exe.setter
    def pw_exe(self, val: Optional[str]) -> None:
        self._resolved_pw_exe = val
        self._pw_exe_override = val

    def _resolve_omp_threads(self, n_workers_override: Optional[int] = None) -> int:
        """Determines OpenMP threads per MPI rank so all CPU cores are utilized without oversubscription."""
        if getattr(self.config, "omp_num_threads", None) is not None:
            return max(1, int(self.config.omp_num_threads))
        total_cores = os.cpu_count() or 1
        eff_workers = max(1, int(n_workers_override if n_workers_override is not None else self.n_workers))
        active_ranks = max(1, self.n_procs * eff_workers)
        return max(1, total_cores // active_ranks)

    def label_single_frame(
        self,
        atoms: Atoms,
        calc_dir: str,
        retry: bool = False,
        timeout_s: Optional[float] = None,
        occupations_mode: Optional[str] = None,
        constrained_occupations: Optional[Union[List[List[float]], Tuple[List[float], ...]]] = None,
        worker_idx: Optional[int] = None,
    ) -> Atoms:
        """Runs or reuses a cached single-point QE calculation for `atoms` in `calc_dir`."""
        os.makedirs(calc_dir, exist_ok=True)
        pw_in = os.path.join(calc_dir, "pw.in")
        pw_out = os.path.join(calc_dir, "pw.out")
        eff_timeout = timeout_s if timeout_s is not None else self.timeout_s

        if os.path.isfile(pw_out):
            try:
                with open(pw_out, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
            except Exception:
                content = ""
            if "JOB DONE." in content and "convergence NOT achieved" not in content:
                if check_pw_in_matches_atoms(pw_in, atoms):
                    e_ev, f_ev_a = parse_qe_output(pw_out)
                    if e_ev is not None and f_ev_a is not None and len(f_ev_a) == len(atoms):
                        logger.info(f"      [Cache hit] Reusing converged calculation in {calc_dir}")
                        at_labeled = atoms.copy()
                        at_labeled.calc = None
                        at_labeled.info["energy"] = e_ev
                        at_labeled.arrays["forces"] = f_ev_a
                        _attach_gap_info(at_labeled, parse_qe_band_gap(pw_out))
                        return at_labeled
                else:
                    logger.info(f"      [Cache mismatch] Atomic positions in {pw_in} differ from frame. Recomputing...")
            else:
                tmp_dir = os.path.join(calc_dir, "tmp")
                if os.path.isdir(tmp_dir):
                    try:
                        for item in os.listdir(tmp_dir):
                            item_p = os.path.join(tmp_dir, item)
                            if os.path.isfile(item_p) or os.path.islink(item_p):
                                os.unlink(item_p)
                            elif os.path.isdir(item_p):
                                shutil.rmtree(item_p, ignore_errors=True)
                    except Exception:
                        pass
                logger.info(f"      [Resuming] Incomplete calculation in {calc_dir}. Cleaned scratch tmp/ and restarting...")

        write_qe_input(
            atoms,
            pw_in,
            self.config,
            retry=retry,
            occupations_mode=occupations_mode,
            constrained_occupations=constrained_occupations,
        )
        os.makedirs(os.path.join(calc_dir, "tmp"), exist_ok=True)

        cmd_args = build_qe_command(
            self.pw_exe,
            pw_in="pw.in",
            n_procs=self.n_procs,
            mpi_command=self.mpi_command,
            mpi_args=self.mpi_args,
        )

        env = os.environ.copy()
        omp_threads = str(self._resolve_omp_threads())
        env.update({
            "OMP_NUM_THREADS": omp_threads,
            "MKL_NUM_THREADS": omp_threads,
            "OPENBLAS_NUM_THREADS": omp_threads,
            "VECLIB_MAXIMUM_THREADS": omp_threads,
            "NUMEXPR_NUM_THREADS": "1",
        })
        if worker_idx is not None and getattr(self.config, "gpu_pinning", False):
            try:
                from .hpc.slurm_dispatcher import get_pinned_worker_env
                gpus = getattr(self.config, "slurm_gpus_per_node", 4)
                env = get_pinned_worker_env(worker_idx, gpus_per_node=gpus, cpus_per_worker=int(omp_threads), base_env=env)
            except Exception as exc:
                logger.debug(f"Could not apply worker GPU pinning: {exc}")

        if os.path.isabs(self.pw_exe):
            pw_dir = os.path.dirname(self.pw_exe)
            if pw_dir:
                env["PATH"] = pw_dir + os.pathsep + env.get("PATH", "")

        run_monitored_qe_subprocess(
            cmd_args=cmd_args,
            calc_dir=calc_dir,
            pw_out=pw_out,
            env=env,
            timeout=eff_timeout,
            mode="scf",
            allow_fallbacks=getattr(self.config, "allow_fallbacks", False),
        )

        e_ev, f_ev_a = parse_qe_output(pw_out)
        if e_ev is not None and f_ev_a is not None and len(f_ev_a) == len(atoms):
            try:
                from .hpc.slurm_dispatcher import mark_frame_done
                mark_frame_done(calc_dir, {"energy_eV": e_ev})
            except Exception:
                pass
        if e_ev is None or f_ev_a is None or len(f_ev_a) != len(atoms):
            tail_msg = ""
            if os.path.isfile(pw_out):
                try:
                    with open(pw_out, "r", encoding="utf-8", errors="replace") as f:
                        tail_msg = "".join(f.readlines()[-20:])
                except Exception:
                    pass
            win_msg = ""
            if is_windows():
                win_msg = (
                    "\n[Windows Notice] If Quantum ESPRESSO failed to run, please verify:"
                    "\n  1. Precompiled Windows QE binaries (e.g. from https://github.com/QMatSuite/quantum-espresso-windows-exe) "
                    "are unzipped and pw_exe / bin/ is specified or on PATH."
                    "\n  2. Microsoft MPI is installed (provides mpiexec.exe)."
                    f"\n  3. Resolved pw_exe: '{self.pw_exe}'"
                    f"\n  4. Execution command was: {' '.join(str(c) for c in cmd_args)}"
                    "\n  * Tip: For massive speedups on NVIDIA GPUs, running HyDePh inside WSL2 with GPU-accelerated Quantum ESPRESSO is strongly recommended."
                )
            raise RuntimeError(f"QE calculation failed in {calc_dir}.\n{tail_msg}{win_msg}")

        at_labeled = atoms.copy()
        at_labeled.calc = None
        at_labeled.info["energy"] = e_ev
        at_labeled.arrays["forces"] = f_ev_a
        _attach_gap_info(at_labeled, parse_qe_band_gap(pw_out))

        # Clean scratch tmp/ directory to prevent disk bloat
        tmp_dir = os.path.join(calc_dir, "tmp")
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)

        return at_labeled

    def run_relaxation(
        self,
        atoms: Atoms,
        calc_dir: str,
        tot_magnetization: Optional[int] = None,
        nspin: Optional[int] = None,
        timeout_s: Optional[float] = None,
        force_rerun: bool = False,
        occupations_mode: Optional[str] = None,
        constrained_occupations: Optional[Union[List[List[float]], Tuple[List[float], ...]]] = None,
        calculation: Optional[str] = None,
    ) -> Atoms:
        """Executes or reuses a Quantum ESPRESSO BFGS geometry relaxation calculation.
        
        Parameters
        ----------
        atoms : Atoms
            Initial geometry to optimize.
        calc_dir : str
            Working directory for the QE relaxation.
        tot_magnetization : int, optional
            Target spin magnetization (2S).
        nspin : int, optional
            1 for unpolarized, 2 for spin-polarized.
        timeout_s : float, optional
            Calculation timeout in seconds.
        force_rerun : bool
            If True, re-runs relaxation even if converged `pw.out` exists.
            
        Returns
        -------
        Atoms
            Relaxed geometry with `energy` in `info`.
        """
        os.makedirs(calc_dir, exist_ok=True)
        pw_in = os.path.join(calc_dir, "pw.in")
        pw_out = os.path.join(calc_dir, "pw.out")
        eff_timeout = timeout_s if timeout_s is not None else self.timeout_s

        if not force_rerun and os.path.isfile(pw_out):
            at_rel, e_ev = parse_qe_relaxation_output(pw_out, atoms, require_converged=True)
            if at_rel is not None:
                matches = check_pw_in_matches_atoms(pw_in, atoms, config=self.config) if os.path.isfile(pw_in) else True
                if matches or (len(at_rel) == len(atoms) and at_rel.get_chemical_symbols() == atoms.get_chemical_symbols()):
                    logger.info(f"      [Cache hit] Reusing converged QE relaxation in {calc_dir}")
                    return at_rel

        # Clean scratch tmp/
        tmp_dir = os.path.join(calc_dir, "tmp")
        if os.path.isdir(tmp_dir):
            try:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass
        os.makedirs(tmp_dir, exist_ok=True)

        calc_mode = calculation or getattr(self.config, "relaxation_calculation", "relax")

        write_qe_input(
            atoms,
            pw_in,
            self.config,
            calculation=calc_mode,
            tot_magnetization=tot_magnetization,
            nspin=nspin,
            occupations_mode=occupations_mode,
            constrained_occupations=constrained_occupations,
        )

        cmd_args = build_qe_command(
            self.pw_exe,
            pw_in="pw.in",
            n_procs=self.n_procs,
            mpi_command=self.mpi_command,
            mpi_args=self.mpi_args,
        )

        env = os.environ.copy()
        omp_threads = str(self._resolve_omp_threads())
        env.update({
            "OMP_NUM_THREADS": omp_threads,
            "MKL_NUM_THREADS": omp_threads,
            "OPENBLAS_NUM_THREADS": omp_threads,
            "VECLIB_MAXIMUM_THREADS": omp_threads,
            "NUMEXPR_NUM_THREADS": "1",
        })
        if os.path.isabs(self.pw_exe):
            pw_dir = os.path.dirname(self.pw_exe)
            if pw_dir:
                env["PATH"] = pw_dir + os.pathsep + env.get("PATH", "")

        calc_label = "QE cell & ionic vc-relax" if calc_mode == "vc-relax" else "QE BFGS relaxation"
        logger.info(f"      [pw relax] Starting {calc_label} in {calc_dir} (tot_magnetization={tot_magnetization})...")
        t_start = time.time()

        at_fallback = run_monitored_qe_subprocess(
            cmd_args=cmd_args,
            calc_dir=calc_dir,
            pw_out=pw_out,
            env=env,
            timeout=eff_timeout,
            mode=calc_mode,
            allow_fallbacks=getattr(self.config, "allow_fallbacks", False),
            atoms_template=atoms,
        )
        if at_fallback is not None:
            return at_fallback

        at_rel, e_ev = parse_qe_relaxation_output(pw_out, atoms)
        if at_rel is None:
            tail_msg = ""
            if os.path.isfile(pw_out):
                try:
                    with open(pw_out, "r", encoding="utf-8", errors="replace") as f:
                        tail_msg = "".join(f.readlines()[-25:])
                except Exception:
                    pass
            win_msg = ""
            if is_windows():
                win_msg = (
                    "\n[Windows Notice] If Quantum ESPRESSO failed to run, please verify:"
                    "\n  1. Precompiled Windows QE binaries are unzipped and pw_exe / bin/ is specified or on PATH."
                    "\n  2. Microsoft MPI is installed (provides mpiexec.exe)."
                    f"\n  3. Resolved pw_exe: '{self.pw_exe}'"
                    f"\n  4. Execution command was: {' '.join(str(c) for c in cmd_args)}"
                )
            raise RuntimeError(f"QE relaxation failed in {calc_dir}.\n{tail_msg}{win_msg}")

        dur = time.time() - t_start
        logger.info(f"      [pw relax] Completed in {dur:.1f}s | Energy = {e_ev:.4f} eV" if e_ev else f"      [pw relax] Completed in {dur:.1f}s")
        if at_rel is not None:
            at_rel.calc = None
            # Clean scratch tmp/ directory to prevent disk bloat
            tmp_dir = os.path.join(calc_dir, "tmp")
            if os.path.isdir(tmp_dir):
                shutil.rmtree(tmp_dir, ignore_errors=True)
        return at_rel

    def _obtain_replacement_entry(
        self, stype, unused_by_type, frames, dft_dir, sampler, at_gs, type_total_counts, next_cand_idx
    ):
        """Returns the next unused candidate entry of `stype`, or generates a new frame if none remain."""
        pool = unused_by_type.get(stype, [])
        if pool:
            return pool.pop(0), next_cand_idx

        if sampler is None:
            from ..frame_sampler import FrameSampler
            sampler = FrameSampler(self.config)

        base_atoms = at_gs if at_gs is not None else frames[0]
        type_idx = type_total_counts.get(stype, 0)
        type_total_counts[stype] = type_idx + 1
        cand_idx = next_cand_idx
        next_cand_idx += 1

        new_fr = sampler.generate_replacement_frame(
            stype, base_atoms, type_index=type_idx, candidate_index=cand_idx
        )
        new_fr.calc = None
        frames.append(new_fr)
        cand_xyz = os.path.join(self.config.workspace_dir, "candidate_frames.xyz")
        if os.path.isfile(cand_xyz):
            try:
                for f in frames:
                    f.calc = None
                tmp_cand = cand_xyz + ".tmp"
                ase.io.write(tmp_cand, frames, format="extxyz")
                os.replace(tmp_cand, cand_xyz)
            except Exception:
                pass

        frame_id = new_fr.info["frame_id"]
        cdir = os.path.join(dft_dir, f"frame_{cand_idx:03d}_{frame_id}")
        entry = {
            "index": cand_idx,
            "frame_id": frame_id,
            "sample_type": stype,
            "atoms": new_fr,
            "calc_dir": cdir,
        }
        return entry, next_cand_idx

    def label_fc_excited_state(
        self,
        fc_frames: List[Atoms],
        at_gs: Atoms,
        calc_dir_base: Optional[str] = None,
        constrained_occupations: Optional[Tuple[List[float], List[float]]] = None,
        n_workers: Optional[int] = None,
    ) -> List[Atoms]:
        """Backward-compatible alias for label_excited_state_frames."""
        return self.label_excited_state_frames(
            frames=fc_frames,
            at_gs=at_gs,
            calc_dir_base=calc_dir_base,
            constrained_occupations=constrained_occupations,
            n_workers=n_workers,
        )

    def label_excited_state_frames(
        self,
        frames: List[Atoms],
        at_gs: Atoms,
        calc_dir_base: Optional[str] = None,
        constrained_occupations: Optional[Tuple[List[float], List[float]]] = None,
        n_workers: Optional[int] = None,
    ) -> List[Atoms]:
        """Labels candidate frames with Delta-SCF constrained occupations for the optical excited state."""
        calc_base = calc_dir_base or self.config.workspace_dir
        os.makedirs(calc_base, exist_ok=True)
        dft_es_dir = os.path.join(calc_base, "dft_calculations_es")
        os.makedirs(dft_es_dir, exist_ok=True)
        out_es_xyz = os.path.join(calc_base, "labeled_frames_es.xyz")

        if constrained_occupations is None:
            gs_candidates = [
                os.path.join(calc_base, "qe_gs_relax", "pw.out"),
                os.path.join(calc_base, "dft_calculations", "frame_000_fc_00", "pw.out"),
            ]
            gs_out = next((p for p in gs_candidates if os.path.isfile(p)), None)
            spec = resolve_delta_scf_spec(at_gs, self.config, gs_pw_out_path=gs_out, workspace_dir=calc_base)
            constrained_occupations = (spec["occupations_up"], spec["occupations_down"])

        eff_workers = max(1, int(n_workers if n_workers is not None else self.n_workers))
        omp_threads = self._resolve_omp_threads(eff_workers)
        logger.info(
            f"[-] Optical Excited-State DFT Labeling (Delta-SCF, np={self.n_procs}, omp={omp_threads}, workers={eff_workers})..."
        )
        logger.info(f"    Target excited-state pool: {len(frames)} frames")

        # Discover existing converged runs or resolve directories
        frame_entries = []
        for i, fr in enumerate(frames):
            cand_idx = fr.info.get("candidate_index", i)
            frame_id = fr.info.get("frame_id") or fr.info.get("label") or f"frame_{cand_idx:02d}"
            stype = fr.info.get("sample_type", "franck_condon")

            canonical_dir = os.path.join(dft_es_dir, f"frame_es_{cand_idx:03d}_{frame_id}")
            candidate_dirs = [
                canonical_dir,
                os.path.join(calc_base, f"frame_es_{cand_idx:03d}_{frame_id}"),
                os.path.join(calc_base, f"frame_es_{i:03d}_{frame_id}"),
                os.path.join(dft_es_dir, f"frame_{cand_idx:03d}_{frame_id}"),
                os.path.join(calc_base, f"frame_es_{frame_id}"),
            ]
            if frame_id.startswith("fc_"):
                try:
                    fc_idx = int(frame_id.split("_")[1])
                    candidate_dirs.append(os.path.join(calc_base, f"frame_es_{fc_idx:03d}_{frame_id}"))
                except ValueError:
                    pass
            chosen_dir = canonical_dir
            is_done = False
            for cdir in candidate_dirs:
                if os.path.isdir(cdir):
                    pw_out = os.path.join(cdir, "pw.out")
                    pw_in = os.path.join(cdir, "pw.in")
                    if os.path.isfile(pw_out):
                        try:
                            with open(pw_out, "r", encoding="utf-8", errors="replace") as f:
                                content = f.read()
                            if (
                                "JOB DONE." in content
                                and "convergence NOT achieved" not in content
                                and check_pw_in_matches_atoms(pw_in, fr, config=self.config)
                            ):
                                e_ev, f_ev_a = parse_qe_output(pw_out)
                                if e_ev is not None and f_ev_a is not None and len(f_ev_a) == len(fr):
                                    chosen_dir = cdir
                                    is_done = True
                                    break
                        except Exception:
                            pass
            frame_entries.append({
                "index": i,
                "cand_idx": cand_idx,
                "frame_id": frame_id,
                "sample_type": stype,
                "atoms": fr,
                "calc_dir": chosen_dir,
                "is_done": is_done,
            })

        completed = [e for e in frame_entries if e["is_done"]]
        pending = [e for e in frame_entries if not e["is_done"]]

        if completed:
            logger.info(f"    [i] Already converged: {len(completed)} ES frames")
        if pending:
            logger.info(f"    [i] Pending / uncalculated: {len(pending)} ES frames")
        else:
            logger.info(f"    [i] All {len(frames)} ES candidate frames are already converged!")

        def _enrich_es_frame(at_lab: Atoms, src_fr: Atoms, calc_dir: str):
            at_lab.info["electronic_state"] = "excited"
            at_lab.info["sample_type"] = src_fr.info.get("sample_type", "franck_condon")
            at_lab.info["qe_status"] = "converged"
            if "lambda" in src_fr.info:
                at_lab.info["lambda"] = src_fr.info["lambda"]
            if "candidate_index" in src_fr.info:
                at_lab.info["candidate_index"] = src_fr.info["candidate_index"]
            if "frame_id" in src_fr.info:
                at_lab.info["frame_id"] = src_fr.info["frame_id"]

            # Vertical excitation energy Delta E_vert(R) = E_ES(R) - E_GS(R)
            if "energy" in src_fr.info and "energy" in at_lab.info:
                delta_e = float(at_lab.info["energy"]) - float(src_fr.info["energy"])
                at_lab.info["delta_e_vert_eV"] = delta_e

            # Parse hole/electron Kohn-Sham gap from ES pw.out
            pw_out_es = os.path.join(calc_dir, "pw.out")
            if os.path.isfile(pw_out_es):
                try:
                    gap_res = parse_qe_band_gap(pw_out_es, target_spin_channel=getattr(self.config, "delta_scf_spin_channel", None))
                    if gap_res and gap_res.get("gap_eV") is not None:
                        at_lab.info["hole_electron_gap_eV"] = float(gap_res["gap_eV"])
                except Exception:
                    pass

        labeled_dict = {}
        for e in completed:
            at = self.label_single_frame(
                e["atoms"],
                e["calc_dir"],
                occupations_mode="from_input",
                constrained_occupations=constrained_occupations,
            )
            _enrich_es_frame(at, e["atoms"], e["calc_dir"])
            labeled_dict[e["index"]] = at

        # Write initial checkpoint if any were completed
        if labeled_dict:
            current_frames = [labeled_dict[k] for k in sorted(labeled_dict.keys())]
            for f in current_frames:
                f.calc = None
            tmp_xyz = out_es_xyz + ".tmp"
            ase.io.write(tmp_xyz, current_frames, format="extxyz")
            os.replace(tmp_xyz, out_es_xyz)

        state_lock = threading.Lock()
        retry_queue = []

        def _run_single_es(step_num, entry):
            idx = entry["index"]
            fr = entry["atoms"]
            cdir = entry["calc_dir"]
            label = entry["frame_id"]
            stype = entry["sample_type"]

            t0 = time.time()
            worker_id = (step_num - 1) % eff_workers
            logger.info(f"\n    ---> [ES {step_num}/{len(pending)}] Calculating {label} ({stype}) in {os.path.basename(cdir)} (worker {worker_id})...")
            try:
                at_labeled = self.label_single_frame(
                    fr,
                    cdir,
                    retry=False,
                    occupations_mode="from_input",
                    constrained_occupations=constrained_occupations,
                    worker_idx=worker_id,
                )
                _enrich_es_frame(at_labeled, fr, cdir)
                dt = time.time() - t0
                with state_lock:
                    labeled_dict[idx] = at_labeled
                    current_frames = [labeled_dict[k] for k in sorted(labeled_dict.keys())]
                    for f in current_frames:
                        f.calc = None
                    tmp_xyz = out_es_xyz + ".tmp"
                    ase.io.write(tmp_xyz, current_frames, format="extxyz")
                    os.replace(tmp_xyz, out_es_xyz)
                logger.info(f"    [+] ES {label} labeled | E = {at_labeled.info['energy']:.4f} eV | Max F = {np.abs(at_labeled.arrays['forces']).max():.4f} eV/A ({dt:.1f} s)")
            except Exception as err:
                with state_lock:
                    entry["attempts"] = 1
                    entry["last_error"] = str(err)
                    retry_queue.append(entry)
                    logger.warning(f"    [!] ES {label} failed on attempt 1: {err}. Tagged for retry.")

        if pending:
            if eff_workers > 1 and len(pending) > 1:
                with ThreadPoolExecutor(max_workers=eff_workers) as pool:
                    futures = [
                        pool.submit(_run_single_es, step_num, entry)
                        for step_num, entry in enumerate(pending, 1)
                    ]
                    for fut in futures:
                        fut.result()
            else:
                for step_num, entry in enumerate(pending, 1):
                    _run_single_es(step_num, entry)

        if retry_queue:
            logger.info(f"\n[-] Retrying {len(retry_queue)} tagged ES calculation(s)...")
            for retry_num, entry in enumerate(retry_queue, 1):
                idx = entry["index"]
                fr = entry["atoms"]
                cdir = entry["calc_dir"]
                label = entry["frame_id"]
                stype = entry["sample_type"]
                logger.info(f"    ---> [Retry {retry_num}/{len(retry_queue)}] {label} ({stype}) with conservative mixing (beta=0.2, electron_maxstep=150)...")
                try:
                    at_labeled = self.label_single_frame(
                        fr,
                        cdir,
                        retry=True,
                        occupations_mode="from_input",
                        constrained_occupations=constrained_occupations,
                    )
                    _enrich_es_frame(at_labeled, fr, cdir)
                    with state_lock:
                        labeled_dict[idx] = at_labeled
                        current_frames = [labeled_dict[k] for k in sorted(labeled_dict.keys())]
                        for f in current_frames:
                            f.calc = None
                        tmp_xyz = out_es_xyz + ".tmp"
                        ase.io.write(tmp_xyz, current_frames, format="extxyz")
                        os.replace(tmp_xyz, out_es_xyz)
                    logger.info(f"    [+] ES {label} converged on retry!")
                except Exception as err:
                    is_anchor = (stype == "franck_condon" and (
                        (fr.info.get("lambda") is not None and (abs(float(fr.info["lambda"]) - 0.0) < 1e-4 or abs(float(fr.info["lambda"]) - 1.0) < 1e-4))
                        or label in ("fc_00", "fc_01")
                    ))
                    if is_anchor:
                        raise RuntimeError(
                            f"Critical optical anchor frame {label} failed Delta-SCF convergence on retry: {err}"
                        ) from err
                    logger.error(f"    [-] ES frame {label} ({stype}) permanently failed: {err}. Skipping non-anchor frame.")

        labeled_es = [labeled_dict[k] for k in sorted(labeled_dict.keys())]

        # Check Delta E_vert continuity across frames
        d_verts = [
            (i, at.info.get("frame_id", f"frame_{i}"), float(at.info["delta_e_vert_eV"]))
            for i, at in enumerate(labeled_es)
            if "delta_e_vert_eV" in at.info
        ]
        if d_verts:
            arr_dv = np.array([x[2] for x in d_verts])
            mean_dv = float(np.mean(arr_dv))
            std_dv = float(np.std(arr_dv))
            thresh = max(3.0 * std_dv, 0.15)
            logger.info(
                f"    [i] Vertical excitation energy distribution (Delta E_vert, {len(d_verts)} frames): "
                f"mean = {mean_dv:.3f} eV, std = {std_dv:.3f} eV [range {np.min(arr_dv):.3f} .. {np.max(arr_dv):.3f} eV]"
            )
            for idx, fid, val in d_verts:
                is_outlier = False
                if val <= 0.0:
                    logger.warning(
                        f"    [!] ALERT: ES frame {fid} has negative or zero vertical excitation energy "
                        f"(Delta E_vert = {val:.4f} eV <= 0)! Indicates electronic root inversion or convergence failure."
                    )
                    is_outlier = True
                elif abs(val - mean_dv) > thresh:
                    logger.warning(
                        f"    [!] Warning: ES frame {fid} vertical excitation energy (Delta E_vert = {val:.4f} eV) "
                        f"deviates by > 3sigma from ensemble mean ({mean_dv:.3f} +/- {std_dv:.3f} eV). "
                        f"Possible Delta-SCF state-crossing or root-flipping outlier."
                    )
                    is_outlier = True
                if is_outlier:
                    labeled_es[idx].info["delta_scf_state_crossing_outlier"] = True

        return labeled_es

    def label_all_frames(
        self,
        frames: List[Atoms],
        max_frames: Optional[int] = None,
        max_new_frames: Optional[int] = None,
        replace_failed_frames: Optional[bool] = None,
        sampler=None,
        at_gs: Optional[Atoms] = None,
        n_workers: Optional[int] = None,
        force_resample: bool = False,
    ) -> List[Atoms]:
        """Labels candidate frames with DFT, retrying failed runs and optionally replacing failed rattle/MD frames."""
        dft_dir = os.path.join(self.config.workspace_dir, "dft_calculations")
        os.makedirs(dft_dir, exist_ok=True)
        out_xyz = os.path.join(self.config.workspace_dir, "labeled_dft_frames.xyz")
        eff_workers = max(1, int(n_workers if n_workers is not None else self.n_workers))

        completed_entries = []
        pending_entries = []
        type_total_counts = {}
        max_cand_idx = -1

        for idx, fr in enumerate(frames):
            cand_idx = fr.info.get("candidate_index", idx)
            max_cand_idx = max(max_cand_idx, cand_idx)
            frame_id = fr.info.get("frame_id") or fr.info.get("label") or f"frame_{cand_idx:02d}"
            stype = fr.info.get("sample_type", "unknown")
            type_total_counts[stype] = type_total_counts.get(stype, 0) + 1
            cdir, is_done = find_frame_calc_dir(dft_dir, cand_idx, frame_id, fr, config=self.config)
            entry = {
                "index": cand_idx,
                "frame_id": frame_id,
                "sample_type": stype,
                "atoms": fr,
                "calc_dir": cdir,
            }
            if is_done and not force_resample:
                completed_entries.append(entry)
            else:
                pending_entries.append(entry)

        next_cand_idx = max_cand_idx + 1

        omp_threads = self._resolve_omp_threads(eff_workers)
        logger.info(
            f"\n[-] High-Fidelity DFT Labeling (Quantum ESPRESSO, np={self.n_procs}, omp={omp_threads}, workers={eff_workers})..."
        )
        logger.info(f"    Total candidate pool: {len(frames)} frames")

        type_completed = {}
        for e in completed_entries:
            st = e["sample_type"]
            type_completed[st] = type_completed.get(st, 0) + 1

        type_pending = {}
        for e in pending_entries:
            st = e["sample_type"]
            type_pending[st] = type_pending.get(st, 0) + 1

        if completed_entries:
            comp_str = ", ".join(f"{k}: {v}" for k, v in sorted(type_completed.items()))
            logger.info(f"    [i] Already converged: {len(completed_entries)} frames ({comp_str})")
        if pending_entries:
            pend_str = ", ".join(f"{k}: {v}" for k, v in sorted(type_pending.items()))
            logger.info(f"    [i] Pending / uncalculated: {len(pending_entries)} frames ({pend_str})")
        else:
            logger.info(f"    [i] All {len(frames)} candidate frames are already converged!")

        resume_mode = self.config.dft_resume_mode
        dft_target = max_frames if max_frames is not None else self.config.max_dft_frames
        new_limit = max_new_frames if max_new_frames is not None else self.config.max_new_dft_frames
        do_retry = self.config.retry_failed_dft
        do_replace = replace_failed_frames if replace_failed_frames is not None else self.config.replace_failed_frames

        def _is_anchor_frame(entry) -> bool:
            if entry.get("sample_type") != "franck_condon":
                return False
            fr = entry.get("atoms")
            if fr is None:
                return False
            lam = fr.info.get("lambda")
            if lam is not None:
                return abs(float(lam)) < 1e-4 or abs(float(lam) - 1.0) < 1e-4
            fid = entry.get("frame_id", "")
            return fid in ("fc_00", "fc_01")

        def _frame_sort_key(entry):
            # Prioritize anchor frames (priority 0) over non-anchor frames (priority 1)
            anchor_prio = 0 if _is_anchor_frame(entry) else 1
            return (anchor_prio, type_completed.get(entry["sample_type"], 0), entry["index"])

        to_calculate = []
        if not pending_entries:
            to_calculate = []
        elif resume_mode == "max_new" or (dft_target is None and new_limit is not None):
            budget = new_limit if new_limit is not None else len(pending_entries)
            sorted_pending = sorted(pending_entries, key=_frame_sort_key)
            to_calculate = sorted_pending[:budget]
        elif dft_target is not None:
            n_done = len(completed_entries)
            if n_done >= dft_target:
                logger.info(f"    [i] Target max_dft_frames={dft_target} already satisfied by {n_done} converged frames.")
                logger.info("        Reusing all converged frames without duplicate DFT execution.")
                to_calculate = []
            else:
                needed = dft_target - n_done
                if new_limit is not None:
                    needed = min(needed, new_limit)
                sorted_pending = sorted(pending_entries, key=_frame_sort_key)
                to_calculate = sorted_pending[:needed]
                calc_str = ", ".join(f"{e['frame_id']} ({e['sample_type']})" for e in to_calculate)
                logger.info(f"    [i] Need {needed} additional frames to reach target {dft_target}. Selected: {calc_str}")
        else:
            to_calculate = sorted(pending_entries, key=_frame_sort_key)

        total_budget = len(completed_entries) + len(to_calculate)
        if total_budget < 20:
            logger.warning(
                f"[!] Warning: Total DFT frame budget ({total_budget}) is fewer than 20 frames. "
                "A minimum of 20-40 frames is strongly recommended for reliable Delta-NEP training."
            )

        selected_indices = {e["index"] for e in to_calculate}
        unused_by_type = {}
        for e in pending_entries:
            if e["index"] not in selected_indices:
                unused_by_type.setdefault(e["sample_type"], []).append(e)

        labeled_dict = {}
        for entry in completed_entries:
            at = self.label_single_frame(entry["atoms"], entry["calc_dir"])
            at.info["qe_status"] = "converged"
            labeled_dict[entry["index"]] = at

        retry_queue = []
        failed_entries = []
        state_lock = threading.Lock()

        def _run_primary_entry(step_num, entry):
            idx = entry["index"]
            fr = entry["atoms"]
            cdir = entry["calc_dir"]
            label = entry["frame_id"]
            stype = entry["sample_type"]

            t0 = time.time()
            worker_id = (step_num - 1) % eff_workers
            logger.info(f"\n    ---> [{step_num}/{len(to_calculate)}] Calculating {label} ({stype}) in {os.path.basename(cdir)} (worker {worker_id})...")
            try:
                labeled_fr = self.label_single_frame(fr, cdir, retry=False, worker_idx=worker_id)
                labeled_fr.info["qe_status"] = "converged"
                dt = time.time() - t0
                with state_lock:
                    labeled_dict[idx] = labeled_fr
                    type_completed[stype] = type_completed.get(stype, 0) + 1
                    current_frames = [labeled_dict[k] for k in sorted(labeled_dict.keys())]
                    for f in current_frames:
                        f.calc = None
                    tmp_xyz = out_xyz + ".tmp"
                    ase.io.write(tmp_xyz, current_frames, format="extxyz")
                    os.replace(tmp_xyz, out_xyz)
                logger.info(f"    [+] {label} labeled | E = {labeled_fr.info['energy']:.4f} eV | Max F = {np.abs(labeled_fr.arrays['forces']).max():.4f} eV/A ({dt:.1f} s)")
            except RuntimeError as err:
                with state_lock:
                    if do_retry:
                        fr.info["qe_status"] = "retry_pending"
                        entry["status"] = "retry_pending"
                        entry["attempts"] = 1
                        entry["last_error"] = str(err)
                        retry_queue.append(entry)
                        logger.warning(f"    [!] {label} ({stype}) failed on attempt 1. Tagged for retry after remaining frames.")
                    else:
                        fr.info["qe_status"] = "failed"
                        entry["status"] = "failed"
                        entry["attempts"] = 1
                        entry["last_error"] = str(err)
                        failed_entries.append(entry)
                        logger.warning(f"    [!] {label} ({stype}) failed and retry is disabled. Ignoring frame.")

        if to_calculate:
            logger.info(f"\n[-] Executing {len(to_calculate)} DFT calculation(s)...")
            if eff_workers > 1 and len(to_calculate) > 1:
                with ThreadPoolExecutor(max_workers=eff_workers) as pool:
                    futures = [
                        pool.submit(_run_primary_entry, step_num, entry)
                        for step_num, entry in enumerate(to_calculate, 1)
                    ]
                    for fut in futures:
                        fut.result()
                retry_queue.sort(key=lambda e: e["index"])
            else:
                for step_num, entry in enumerate(to_calculate, 1):
                    _run_primary_entry(step_num, entry)

        if retry_queue:
            logger.info(f"\n[-] Retrying {len(retry_queue)} tagged DFT calculation(s)...")
            for retry_num, entry in enumerate(retry_queue, 1):
                idx = entry["index"]
                fr = entry["atoms"]
                cdir = entry["calc_dir"]
                label = entry["frame_id"]
                stype = entry["sample_type"]

                t0 = time.time()
                logger.info(f"\n    ---> [Retry {retry_num}/{len(retry_queue)}] Re-running {label} ({stype}) in {os.path.basename(cdir)}...")
                try:
                    labeled_fr = self.label_single_frame(fr, cdir, retry=True)
                    labeled_fr.info["qe_status"] = "converged_on_retry"
                    entry["status"] = "converged_on_retry"
                    entry["attempts"] = 2
                    labeled_dict[idx] = labeled_fr
                    dt = time.time() - t0
                    logger.info(f"    [+] {label} converged on retry | E = {labeled_fr.info['energy']:.4f} eV | Max F = {np.abs(labeled_fr.arrays['forces']).max():.4f} eV/A ({dt:.1f} s)")
                    type_completed[stype] = type_completed.get(stype, 0) + 1
                    current_frames = [labeled_dict[k] for k in sorted(labeled_dict.keys())]
                    for f in current_frames:
                        f.calc = None
                    tmp_xyz = out_xyz + ".tmp"
                    ase.io.write(tmp_xyz, current_frames, format="extxyz")
                    os.replace(tmp_xyz, out_xyz)
                except RuntimeError as err:
                    fr.info["qe_status"] = "failed"
                    entry["status"] = "failed"
                    entry["attempts"] = 2
                    entry["last_error"] = str(err)
                    failed_entries.append(entry)
                    logger.warning(f"    [!] {label} ({stype}) failed a second time.")

                    if do_replace and stype in ("core_rattle", "foundation_md_fps"):
                        repl_entry, next_cand_idx = self._obtain_replacement_entry(
                            stype, unused_by_type, frames, dft_dir, sampler, at_gs, type_total_counts, next_cand_idx
                        )
                        r_idx = repl_entry["index"]
                        r_fr = repl_entry["atoms"]
                        r_cdir = repl_entry["calc_dir"]
                        r_label = repl_entry["frame_id"]
                        logger.info(f"    [->] Taking replacement frame {r_label} ({stype}) in {os.path.basename(r_cdir)} to fulfill quota...")
                        try:
                            try:
                                r_labeled = self.label_single_frame(r_fr, r_cdir, retry=False)
                            except RuntimeError:
                                logger.warning(f"    [!] Replacement {r_label} ({stype}) failed on attempt 1; retrying...")
                                r_labeled = self.label_single_frame(r_fr, r_cdir, retry=True)
                            r_labeled.info["qe_status"] = "converged"
                            labeled_dict[r_idx] = r_labeled
                            type_completed[stype] = type_completed.get(stype, 0) + 1
                            current_frames = [labeled_dict[k] for k in sorted(labeled_dict.keys())]
                            for f in current_frames:
                                f.calc = None
                            tmp_xyz = out_xyz + ".tmp"
                            ase.io.write(tmp_xyz, current_frames, format="extxyz")
                            os.replace(tmp_xyz, out_xyz)
                            logger.info(f"    [+] Replacement {r_label} ({stype}) converged | E = {r_labeled.info['energy']:.4f} eV")
                        except RuntimeError as repl_err:
                            r_fr.info["qe_status"] = "failed"
                            repl_entry["status"] = "failed"
                            repl_entry["last_error"] = str(repl_err)
                            failed_entries.append(repl_entry)
                            logger.warning(f"    [!] Replacement {r_label} ({stype}) also failed. Ignoring slot and continuing.")
                    else:
                        logger.info(f"    [i] Ignoring failed calculation {label} ({stype}) and continuing with remaining frames.")

        self.last_failed_entries = failed_entries
        final_labeled = [labeled_dict[k] for k in sorted(labeled_dict.keys())]
        if final_labeled:
            for f in final_labeled:
                f.calc = None
            tmp_xyz = out_xyz + ".tmp"
            ase.io.write(tmp_xyz, final_labeled, format="extxyz")
            os.replace(tmp_xyz, out_xyz)

        if failed_entries:
            fail_str = ", ".join(f"{e['frame_id']} ({e['sample_type']})" for e in failed_entries)
            logger.warning(f"\n[!] Ignored {len(failed_entries)} failed DFT calculation(s): {fail_str}")

        if not final_labeled and (to_calculate or completed_entries):
            raise RuntimeError("All QE calculations failed; no converged frames available for training.")

        final_types = {}
        for fr in final_labeled:
            st = fr.info.get("sample_type", "unknown")
            final_types[st] = final_types.get(st, 0) + 1
        summary_str = ", ".join(f"{k}: {v}" for k, v in sorted(final_types.items()))
        logger.info(f"\n[+] DFT Labeling Complete: {len(final_labeled)} total frames ({summary_str}) -> {out_xyz}")
        return final_labeled

    def _get_slurm_modules(self) -> List[str]:
        """Resolves environment modules configured for cluster SLURM scripts in exact user-defined order."""
        mods = getattr(self.config, "slurm_modules", None)
        if mods:
            if isinstance(mods, str):
                if "," in mods:
                    return [m.strip() for m in mods.split(",") if m.strip()]
                return [m.strip() for m in mods.split() if m.strip()]
            return [str(m).strip() for m in mods if str(m).strip()]
        return []

    def prepare_relaxation_inputs(
        self,
        defect_sc: Atoms,
        init_es: Optional[Atoms] = None,
        calc_dir_base: Optional[str] = None,
        write_slurm_scripts: bool = True,
    ) -> Dict[str, Any]:
        """Prepares Quantum ESPRESSO relaxation and SCF input files and standalone SLURM batch scripts.

        Enables fully decoupled HPC execution for Stage 1:
        - Writes Ground-State geometry relaxation input deck (`qe_gs_relax/pw.in`) and SCF deck (`qe_gs_scf/pw.in`).
        - Writes Excited-State Delta-SCF geometry relaxation input deck (`qe_es_relax/pw.in`) when parameters
          are specified, with automated Jahn-Teller symmetry-breaking distortion applied if orbital degeneracy is detected.
        - Generates standalone SLURM batch scripts (`submit_relaxations_pure_qe.slurm`) ready to submit to compute nodes.
        - The user can copy the workspace to the cluster, execute relaxations with zero Python required on compute nodes,
          and seamlessly resume HyDePh once results return.

        Parameters
        ----------
        defect_sc : Atoms
            Defect supercell geometry to relax.
        init_es : Atoms, optional
            Initial excited-state geometry (e.g. with Jahn-Teller pairing distortion applied).
        calc_dir_base : str, optional
            Base workspace directory (defaults to config.workspace_dir).
        write_slurm_scripts : bool
            Whether to write standalone SLURM runner scripts (default True).

        Returns
        -------
        Dict[str, Any]
            Paths to generated input files and SLURM runner scripts.
        """
        calc_base = calc_dir_base or self.config.workspace_dir
        os.makedirs(calc_base, exist_ok=True)

        # 1. Ground State Relaxation input (qe_gs_relax/pw.in)
        gs_dir = os.path.join(calc_base, "qe_gs_relax")
        os.makedirs(gs_dir, exist_ok=True)
        gs_pw_in = os.path.join(gs_dir, "pw.in")
        tot_mag = int(self.config.spin_multiplicity - 1) if self.config.spin_polarized else 0
        nspin = 2 if (self.config.spin_polarized or tot_mag > 0) else 1

        relax_calc = getattr(self.config, "relaxation_calculation", "relax")
        write_qe_input(
            defect_sc,
            gs_pw_in,
            self.config,
            calculation=relax_calc,
            tot_magnetization=tot_mag,
            nspin=nspin,
        )
        logger.info(f"[+] Prepared Ground-State relaxation input: {gs_pw_in}")

        # 2. Ground State SCF single point input (qe_gs_scf/pw.in)
        gs_scf_dir = os.path.join(calc_base, "qe_gs_scf")
        os.makedirs(gs_scf_dir, exist_ok=True)
        gs_scf_in = os.path.join(gs_scf_dir, "pw.in")
        write_qe_input(
            defect_sc,
            gs_scf_in,
            self.config,
            calculation="scf",
            tot_magnetization=tot_mag,
            nspin=nspin,
        )
        logger.info(f"[+] Prepared Ground-State SCF input: {gs_scf_in}")

        # 3. Excited State Relaxation input (qe_es_relax/pw.in) if delta_scf
        es_dir = os.path.join(calc_base, "qe_es_relax")
        es_pw_in = os.path.join(es_dir, "pw.in")
        es_prepared = False

        if getattr(self.config, "excited_state_method", "delta_scf").lower() == "delta_scf":
            os.makedirs(es_dir, exist_ok=True)
            gs_out = os.path.join(gs_dir, "pw.out")
            spec = resolve_delta_scf_spec(
                defect_sc,
                self.config,
                gs_pw_out_path=gs_out if os.path.isfile(gs_out) else None,
                workspace_dir=calc_base,
            )
            occ_up = spec["occupations_up"]
            occ_dw = spec["occupations_down"]
            tot_mag = spec["tot_magnetization"]
            start_es = init_es if init_es is not None else defect_sc
            write_qe_input(
                start_es,
                es_pw_in,
                self.config,
                calculation=relax_calc,
                tot_magnetization=tot_mag,
                nspin=2,
                occupations_mode="from_input",
                constrained_occupations=(occ_up, occ_dw),
            )
            es_prepared = True
            logger.info(f"[+] Prepared Excited-State Delta-SCF relaxation input: {es_pw_in}")

        slurm_path = None
        if write_slurm_scripts:
            mods = self._get_slurm_modules()
            if not mods:
                logger.warning(
                    "[!] Warning: No cluster environment modules specified ('modules' or 'slurm_modules' is empty in config). "
                    "Cluster jobs may fail to find Quantum ESPRESSO binaries unless loaded in the shell environment."
                )
            acct = (
                getattr(self.config, "slurm_account", None)
                or getattr(self.config, "hpc_account", None)
                or os.environ.get("CINECA_ACCOUNT", os.environ.get("SLURM_ACCOUNT", "cin_staff"))
            )
            part = getattr(self.config, "slurm_partition", None) or "boost_usr_prod"
            relax_qos = getattr(self.config, "slurm_relax_qos", None) or getattr(self.config, "slurm_qos", None)
            qos_line = f"#SBATCH --qos={relax_qos}\n" if relax_qos else ""
            nodes = getattr(self.config, "slurm_nodes", 1) or 1
            relax_gpus = getattr(self.config, "slurm_relax_gpus", None)
            if relax_gpus is None:
                relax_gpus = 1  # 1 GPU for fast queue scheduling
            cpus_per_task = getattr(self.config, "slurm_cpus_per_task", 8) or 8
            relax_time = (
                getattr(self.config, "slurm_relax_time", None)
                or getattr(self.config, "slurm_time", None)
                or "01:00:00"
            )
            module_block = format_slurm_module_lines(mods)

            slurm_content = f"""#!/bin/bash
# ==============================================================================
#  HyDePh Decoupled HPC Relaxation Launcher (Pure Quantum ESPRESSO)
#  Ground-State and Excited-State (Delta-SCF) Geometry Optimizations
#
#  Usage:
#    sbatch submit_relaxations_pure_qe.slurm
#  Fast debug queue (starts immediately if <=30 min on Leonardo):
#    sbatch --qos=boost_qos_dbg --time=00:30:00 submit_relaxations_pure_qe.slurm
# ==============================================================================
#SBATCH --job-name=hydeph_relax
#SBATCH --account={acct}
#SBATCH --partition={part}
{qos_line}#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node={relax_gpus}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --gres=gpu:{relax_gpus}
#SBATCH --time={relax_time}
#SBATCH --output=hydeph_relax_%j.out
#SBATCH --error=hydeph_relax_%j.err

set -e

WORK_DIR="${{SLURM_SUBMIT_DIR:-$PWD}}"
cd "$WORK_DIR"

echo "=============================================================================="
echo " [HPC Relaxation] Running on $(hostname) at $(date)"
echo " Job ID: $SLURM_JOB_ID | Submit Dir: $WORK_DIR"
echo "=============================================================================="

{module_block}

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Locate workspace directory containing relaxation folders
WORKSPACE_DIR=""
for cand in "$WORK_DIR" "$WORK_DIR/output" "$WORK_DIR/hydeph_run" "$WORK_DIR/../output" "$WORK_DIR/../hydeph_run" "$WORK_DIR/.."; do
    if [ -d "$cand/qe_gs_relax" ]; then
        WORKSPACE_DIR="$(cd "$cand" && pwd)"
        break
    fi
done

if [ -z "$WORKSPACE_DIR" ]; then
    echo "[!] Error: qe_gs_relax directory not found relative to $WORK_DIR!"
    exit 1
fi

echo "[+] Using workspace directory: $WORKSPACE_DIR"
cd "$WORKSPACE_DIR"

# Resolve pseudopotentials directory
PSEUDO_DIR=""
for cand in "$WORKSPACE_DIR/pseudo" "$WORKSPACE_DIR/../pseudo" "$WORKSPACE_DIR/../../pseudo" "$WORK_DIR/pseudo" "$WORK_DIR/../pseudo" "$WORK_DIR/../../pseudo"; do
    if [ -d "$cand" ]; then
        PSEUDO_DIR="$(cd "$cand" && pwd)"
        break
    fi
done
if [ -n "$PSEUDO_DIR" ]; then
    for sub in qe_gs_relax qe_es_relax qe_gs_scf; do
        if [ -d "$WORKSPACE_DIR/$sub" ] && [ ! -e "$WORKSPACE_DIR/$sub/pseudo" ]; then
            ln -sfn "$PSEUDO_DIR" "$WORKSPACE_DIR/$sub/pseudo"
        fi
    done
fi

# 1. Run Ground State Relaxation
if [ -f "$WORKSPACE_DIR/qe_gs_relax/pw.in" ]; then
    echo "=============================================================================="
    echo " [1/2] Launching Ground-State DFT Relaxation in qe_gs_relax..."
    echo "=============================================================================="
    cd "$WORKSPACE_DIR/qe_gs_relax"
    srun --cpu-bind=cores pw.x -in pw.in > pw.out 2>&1
    echo "[+] Ground-state relaxation finished at $(date)"
    cd "$WORKSPACE_DIR"
fi

# 2. Run Excited State Relaxation
if [ -f "$WORKSPACE_DIR/qe_es_relax/pw.in" ]; then
    echo "=============================================================================="
    echo " [2/2] Launching Excited-State Delta-SCF Relaxation in qe_es_relax..."
    echo "=============================================================================="
    cd "$WORKSPACE_DIR/qe_es_relax"
    srun --cpu-bind=cores pw.x -in pw.in > pw.out 2>&1
    echo "[+] Excited-state relaxation finished at $(date)"
    cd "$WORKSPACE_DIR"
fi

# 3. Post-relaxation geometry extraction
echo "=============================================================================="
echo " [Extraction] Extracting relaxed geometries to relaxed_gs.xyz and relaxed_es.xyz..."
echo "=============================================================================="
python3 -c "
import os, ase.io
from hydeph.qe_runner import parse_qe_relaxation_output
for sub_d, out_f in [('qe_gs_relax', 'relaxed_gs.xyz'), ('qe_es_relax', 'relaxed_es.xyz')]:
    p_out = os.path.join('$WORKSPACE_DIR', sub_d, 'pw.out')
    if os.path.isfile(p_out):
        try:
            at, e = parse_qe_relaxation_output(p_out)
            if at is not None:
                target_p = os.path.join('$WORKSPACE_DIR', out_f)
                ase.io.write(target_p, at, format='extxyz')
                print(f'[+] Successfully extracted {{target_p}} (energy = {{e:.4f}} eV)')
        except Exception as exc:
            print(f'[-] Extraction notice for {{out_f}}: {{exc}}')
" 2>/dev/null || true

# 4. Conjugate Ground-State Single-Point SCF at R_ES (for exact DFT emission forces)
if [ -f "$WORKSPACE_DIR/relaxed_es.xyz" ] && [ -f "$WORKSPACE_DIR/qe_gs_scf/pw.in" ]; then
    echo "=============================================================================="
    echo " [3/3] Launching Conjugate Ground-State SCF at R_ES in qe_gs_on_es..."
    echo "=============================================================================="
    mkdir -p "$WORKSPACE_DIR/qe_gs_on_es"
    python3 -c "
import os, re, ase.io
at_es = ase.io.read('$WORKSPACE_DIR/relaxed_es.xyz')
with open('$WORKSPACE_DIR/qe_gs_scf/pw.in', 'r', encoding='utf-8') as f:
    tpl = f.read()
lines_pos = ['ATOMIC_POSITIONS (angstrom)'] + [f'  {{s:4s}} {{x:14.8f}} {{y:14.8f}} {{z:14.8f}}' for s, (x, y, z) in zip(at_es.get_chemical_symbols(), at_es.positions)]
out_txt = re.sub(r'ATOMIC_POSITIONS.*?(?=K_POINTS|\\Z)', '\\n'.join(lines_pos) + '\\n\\n', tpl, flags=re.S)
with open('$WORKSPACE_DIR/qe_gs_on_es/pw.in', 'w', encoding='utf-8') as f:
    f.write(out_txt)
"
    if [ -n "$PSEUDO_DIR" ] && [ ! -e "$WORKSPACE_DIR/qe_gs_on_es/pseudo" ]; then
        ln -sfn "$PSEUDO_DIR" "$WORKSPACE_DIR/qe_gs_on_es/pseudo"
    fi
    cd "$WORKSPACE_DIR/qe_gs_on_es"
    srun --cpu-bind=cores pw.x -in pw.in > pw.out 2>&1
    echo "[+] Conjugate GS-on-ES SCF finished at $(date)"
    cd "$WORKSPACE_DIR"
fi

echo "=============================================================================="
echo " Relaxations completed at $(date)"
echo " To resume the pipeline, execute:"
echo "   python -m hydeph run config.yaml --from-stage 2"
echo "=============================================================================="
"""
            slurm_path = os.path.join(calc_base, "submit_relaxations_pure_qe.slurm")
            with open(slurm_path, "w", encoding="utf-8", newline="\n") as f:
                f.write(slurm_content)

            parent_dir = os.path.dirname(os.path.abspath(calc_base))
            if parent_dir and os.path.isdir(parent_dir) and parent_dir != os.path.abspath(calc_base):
                for fname in ("submit_relaxations_pure_qe.slurm",):
                    src_f = os.path.join(calc_base, fname)
                    if os.path.isfile(src_f):
                        try:
                            shutil.copy2(src_f, os.path.join(parent_dir, fname))
                        except Exception:
                            pass

        return {
            "workspace_dir": calc_base,
            "gs_relax_in": gs_pw_in,
            "gs_scf_in": gs_scf_in,
            "es_relax_in": es_pw_in if es_prepared else None,
            "slurm_script": slurm_path,
        }

    def prepare_all_dft_inputs(
        self,
        candidate_frames: List[Atoms],
        at_gs: Optional[Atoms] = None,
        calc_dir_base: Optional[str] = None,
        write_slurm_scripts: bool = True,
    ) -> Dict[str, Any]:
        """Prepares Quantum ESPRESSO pw.in input files and standalone SLURM batch scripts.

        Enables fully decoupled HPC execution:
        - Login node runs pre-processing & generates all pw.in files (no GPU needed).
        - Compute node runs pure pw.x via SLURM (no Python or Conda needed on compute nodes).
        - Login node resumes post-processing, Delta-NEP training, and lineshapes.

        Parameters
        ----------
        candidate_frames : List[Atoms]
            Candidate geometries to be labeled.
        at_gs : Atoms, optional
            Reference relaxed ground-state structure.
        calc_dir_base : str, optional
            Base workspace directory (defaults to config.workspace_dir).
        write_slurm_scripts : bool
            Whether to generate standalone SLURM runner scripts (submit_dft_pure_qe.slurm).

        Returns
        -------
        Dict[str, Any]
            Dictionary containing input paths, counts, and SLURM script paths.
        """
        calc_base = calc_dir_base or self.config.workspace_dir
        os.makedirs(calc_base, exist_ok=True)
        dft_dir = os.path.join(calc_base, "dft_calculations")
        os.makedirs(dft_dir, exist_ok=True)

        logger.info(f"[-] Generating Quantum ESPRESSO input decks for {len(candidate_frames)} candidate frames in {dft_dir}...")

        prepared_gs = []
        for idx, fr in enumerate(candidate_frames):
            cand_idx = fr.info.get("candidate_index", idx)
            frame_id = fr.info.get("frame_id") or fr.info.get("label") or f"frame_{cand_idx:02d}"
            frame_dir = os.path.join(dft_dir, f"frame_{cand_idx:03d}_{frame_id}")
            os.makedirs(os.path.join(frame_dir, "tmp"), exist_ok=True)
            pw_in = os.path.join(frame_dir, "pw.in")
            write_qe_input(fr, pw_in, self.config)
            prepared_gs.append(frame_dir)

        # Handle optical excited state frames if paired Delta-NEP is requested
        prepared_es = []
        if getattr(self.config, "paired_delta_nep", True) and getattr(self.config, "excited_state_method", "delta_scf") == "delta_scf":
            dft_es_dir = os.path.join(calc_base, "dft_calculations_es")
            os.makedirs(dft_es_dir, exist_ok=True)

            es_cand_mode = getattr(self.config, "dft_es_candidates", "all").lower()
            if es_cand_mode == "fc_path":
                es_targets = [f for f in candidate_frames if f.info.get("sample_type") == "franck_condon"]
            else:
                es_targets = list(candidate_frames)

            ref_structure = at_gs if at_gs is not None else candidate_frames[0]
            unique_syms = sorted(list(set(ref_structure.get_chemical_symbols())))
            z_vals = {}
            for s in unique_syms:
                upf = self.config.pseudopotentials.get(s, "")
                if upf:
                    p = resolve_pseudo_path(upf, self.config.pseudo_dir)
                    z_vals[s] = inspect_upf_metadata(p).get("z_valence")

            if all(v is not None for v in z_vals.values()):
                nelec_total = int(round(sum(z_vals[s] for s in ref_structure.get_chemical_symbols()) - float(self.config.tot_charge)))
            else:
                nelec_total = len(ref_structure) * 4

            tot_mag = int(self.config.spin_multiplicity - 1) if self.config.spin_polarized else 0
            if (nelec_total % 2) != (tot_mag % 2):
                if nelec_total % 2 != 0:
                    valid_mag = 1 if tot_mag <= 0 else (tot_mag if tot_mag % 2 != 0 else tot_mag + 1)
                else:
                    valid_mag = 0 if tot_mag <= 0 else (tot_mag if tot_mag % 2 == 0 else tot_mag + 1)
                tot_mag = valid_mag

            n_occ_max = (nelec_total + abs(tot_mag)) // 2
            spin_chan = getattr(self.config, "delta_scf_spin_channel", None)
            homo_b = getattr(self.config, "delta_scf_homo_band", None)
            lumo_b = getattr(self.config, "delta_scf_lumo_band", None)
            homo_deg = getattr(self.config, "delta_scf_homo_degeneracy", None)
            lumo_deg = getattr(self.config, "delta_scf_lumo_degeneracy", None)

            # Auto-detect from converged ground state pw.out if available
            gs_out = os.path.join(calc_base, "qe_gs_relax", "pw.out")
            if not os.path.isfile(gs_out):
                cand_outs = glob.glob(os.path.join(dft_dir, "frame_*", "pw.out"))
                if cand_outs:
                    gs_out = sorted(cand_outs)[0]
            gap_info = parse_qe_band_gap(gs_out, target_spin_channel=spin_chan) if os.path.isfile(gs_out) else None
            if gap_info is not None:
                spin_chan = spin_chan or gap_info.get("spin_channel") or 2
                homo_b = homo_b or gap_info.get("homo_band") or n_occ_max
                lumo_b = lumo_b or gap_info.get("lumo_band") or (n_occ_max + 1)
                homo_deg = max(1, int(homo_deg or gap_info.get("homo_degeneracy", 1) or 1))
                lumo_deg = max(1, int(lumo_deg or gap_info.get("lumo_degeneracy", 1) or 1))
                logger.info(
                    f"    [i] Auto-detected lowest optical transition from GS DFT: "
                    f"spin channel {spin_chan}, bands {homo_b} (deg={homo_deg}) -> {lumo_b} (deg={lumo_deg}) "
                    f"[gap = {gap_info.get('gap_eV', 0.0):.3f} eV]"
                )
            else:
                spin_chan = spin_chan or 2
                homo_b = homo_b or n_occ_max
                lumo_b = lumo_b or (n_occ_max + 1)
                homo_deg = max(1, int(homo_deg or 1))
                lumo_deg = max(1, int(lumo_deg or 1))
            nbnd = max(lumo_b + lumo_deg - 1, homo_b + 1, n_occ_max) + 8

            constrained_occupations = build_delta_scf_occupations(
                nelec=nelec_total,
                tot_magnetization=tot_mag,
                nbnd=nbnd,
                homo_band=homo_b,
                lumo_band=lumo_b,
                spin_channel=spin_chan,
                lumo_degeneracy=lumo_deg,
                homo_degeneracy=homo_deg,
            )

            for i, fr in enumerate(es_targets):
                cand_idx = fr.info.get("candidate_index", i)
                frame_id = fr.info.get("frame_id") or fr.info.get("label") or f"frame_{cand_idx:02d}"
                frame_dir_es = os.path.join(dft_es_dir, f"frame_es_{cand_idx:03d}_{frame_id}")
                os.makedirs(os.path.join(frame_dir_es, "tmp"), exist_ok=True)
                pw_in = os.path.join(frame_dir_es, "pw.in")
                write_qe_input(
                    fr,
                    pw_in,
                    self.config,
                    occupations_mode="from_input",
                    constrained_occupations=constrained_occupations,
                )
                prepared_es.append(frame_dir_es)

        logger.info(f"[+] Prepared {len(prepared_gs)} GS pw.in files in {dft_dir}")
        if prepared_es:
            logger.info(f"[+] Prepared {len(prepared_es)} ES Delta-SCF pw.in files in {dft_es_dir}")

        slurm_script_path = None
        array_script_path = None
        list_file_path = None

        if write_slurm_scripts:
            mods = self._get_slurm_modules()
            if not mods:
                logger.warning(
                    "[!] Warning: No cluster environment modules specified ('modules' or 'slurm_modules' is empty in config). "
                    "Cluster jobs may fail to find Quantum ESPRESSO binaries unless loaded in the shell environment."
                )
            # Standalone multi-GPU Slurm batch script
            acct = (
                getattr(self.config, "slurm_account", None)
                or getattr(self.config, "hpc_account", None)
                or os.environ.get("CINECA_ACCOUNT", os.environ.get("SLURM_ACCOUNT", "cin_staff"))
            )
            part = getattr(self.config, "slurm_partition", None) or "boost_usr_prod"
            qos = getattr(self.config, "slurm_qos", None)
            qos_line = f"#SBATCH --qos={qos}\n" if qos else ""
            nodes = getattr(self.config, "slurm_nodes", 1) or 1
            gpus = getattr(self.config, "slurm_gpus_per_node", 4) or 4
            cpus_per_task = getattr(self.config, "slurm_cpus_per_task", 8) or 8
            module_block = format_slurm_module_lines(mods)

            slurm_content = f"""#!/bin/bash
# ==============================================================================
# Standalone Quantum ESPRESSO DFT Runner
# ==============================================================================
# Executes candidate frame DFT calculations across {gpus}x GPUs.
# Runs standalone pw.x across compute nodes without requiring Python or Conda.
# ==============================================================================
#SBATCH --job-name=hydeph_pw_dft
#SBATCH --account={acct}
#SBATCH --partition={part}
{qos_line}#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node={gpus}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --gres=gpu:{gpus}
#SBATCH --time=04:00:00
#SBATCH --output=hydeph_dft_%j.out
#SBATCH --error=hydeph_dft_%j.err

set -e

WORK_DIR="${{SLURM_SUBMIT_DIR:-$PWD}}"
cd "$WORK_DIR"

# Locate workspace directory where dft_calculations lives
WORKSPACE_DIR=""
for cand in "$WORK_DIR" "$WORK_DIR/output" "$WORK_DIR/hydeph_run" "$WORK_DIR/../output" "$WORK_DIR/.."; do
    if [ -d "$cand/dft_calculations" ]; then
        WORKSPACE_DIR="$(cd "$cand" && pwd)"
        break
    fi
done
if [ -n "$WORKSPACE_DIR" ]; then
    cd "$WORKSPACE_DIR"
fi

echo "=============================================================================="
echo " Starting Pure Quantum ESPRESSO Calculations"
echo " Host: $(hostname) | Job ID: $SLURM_JOB_ID | Start: $(date)"
echo " Working Dir: $PWD"
echo "=============================================================================="

{module_block}

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Resolve pseudo directory and symlink into calculation workspace
PSEUDO_DIR=""
for cand in "$PWD/pseudo" "$PWD/../pseudo" "$PWD/../../pseudo" "$WORK_DIR/pseudo" "$WORK_DIR/../pseudo" "$WORK_DIR/../../pseudo"; do
    if [ -d "$cand" ]; then
        PSEUDO_DIR="$(cd "$cand" && pwd)"
        break
    fi
done
if [ -n "$PSEUDO_DIR" ] && [ ! -e "$PWD/pseudo" ]; then
    ln -sfn "$PSEUDO_DIR" "$PWD/pseudo"
fi

PW_CMD="pw.x"

# Collect all frame directories containing pw.in
DIRS=()
if [ -d "dft_calculations" ]; then
    while IFS= read -r d; do
        DIRS+=("$d")
    done < <(find dft_calculations -mindepth 1 -maxdepth 1 -type d | sort)
fi
if [ -d "dft_calculations_es" ]; then
    while IFS= read -r d; do
        DIRS+=("$d")
    done < <(find dft_calculations_es -mindepth 1 -maxdepth 1 -type d | sort)
fi

TOTAL=${{#DIRS[@]}}
echo "[+] Total candidate frame directories: $TOTAL"

MAX_CONCURRENT=4
RUNNING=0

for i in "${{!DIRS[@]}}"; do
    DIR="${{DIRS[$i]}}"
    IN_FILE="$DIR/pw.in"
    OUT_FILE="$DIR/pw.out"

    if [ ! -f "$IN_FILE" ]; then
        continue
    fi

    # Skip already converged calculations
    if [ -f "$OUT_FILE" ] && grep -q "JOB DONE." "$OUT_FILE" 2>/dev/null && ! grep -q "convergence NOT achieved" "$OUT_FILE" 2>/dev/null; then
        echo "[SKIP] Frame $((i+1))/$TOTAL ($DIR) is already converged."
        continue
    fi

    # Ensure pseudo directory is accessible from within frame directory
    if [ -n "$PSEUDO_DIR" ] && [ ! -e "$DIR/pseudo" ]; then
        ln -sfn "$PSEUDO_DIR" "$DIR/pseudo"
    fi

    GPU_ID=$(( i % MAX_CONCURRENT ))
    echo "[RUN]  Frame $((i+1))/$TOTAL ($DIR) on GPU $GPU_ID..."

    (
        export CUDA_VISIBLE_DEVICES=$GPU_ID
        cd "$DIR"
        srun --overlap --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK --cpu-bind=cores $PW_CMD -in pw.in > pw.out 2>&1
    ) &

    RUNNING=$((RUNNING + 1))
    if [ $RUNNING -ge $MAX_CONCURRENT ]; then
        wait -n 2>/dev/null || wait
        RUNNING=$((RUNNING - 1))
    fi
done

wait
echo "=============================================================================="
echo " All DFT calculations finished at $(date)"
echo " Resume post-processing on login node with:"
echo "     python -m hydeph run config.yaml --from-stage 4"
echo "=============================================================================="
"""
            slurm_script_path = os.path.join(calc_base, "submit_dft_pure_qe.slurm")
            with open(slurm_script_path, "w", encoding="utf-8", newline="\n") as f:
                f.write(slurm_content)

            # Frame list for SLURM Job Array (enforce forward slashes for Linux compatibility)
            all_dirs_rel = [os.path.relpath(d, calc_base).replace("\\", "/") for d in (prepared_gs + prepared_es)]
            list_file_path = os.path.join(calc_base, "dft_frames_list.txt")
            with open(list_file_path, "w", encoding="utf-8", newline="\n") as f:
                for d in all_dirs_rel:
                    f.write(f"{d}\n")

            total_count = len(prepared_gs) + len(prepared_es)
            array_max_idx = max(0, total_count - 1)

            array_content = f"""#!/bin/bash
# ==============================================================================
# Pure Quantum ESPRESSO SLURM Job Array Script
# ==============================================================================
# Usage (total {total_count} candidate frames):
#   sbatch --array=0-{array_max_idx}%16 submit_dft_array_pure_qe.slurm
# Or dynamic:
#   sbatch --array=0-$(($(wc -l < dft_frames_list.txt) - 1))%16 submit_dft_array_pure_qe.slurm
# ==============================================================================
#SBATCH --job-name=hydeph_pw_array
#SBATCH --account={acct}
#SBATCH --partition={part}
{qos_line}#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --gres=gpu:1
#SBATCH --array=0-{array_max_idx}%16
#SBATCH --time=01:00:00
#SBATCH --output=hydeph_array_%A_%a.out
#SBATCH --error=hydeph_array_%A_%a.err

set -e

WORK_DIR="${{SLURM_SUBMIT_DIR:-$PWD}}"
cd "$WORK_DIR"

echo "=============================================================================="
echo " [HPC Array] Task $SLURM_ARRAY_TASK_ID on $(hostname) at $(date)"
echo " Job ID: $SLURM_ARRAY_JOB_ID | Task ID: $SLURM_ARRAY_TASK_ID"
echo " Submit Dir: $WORK_DIR"
echo "=============================================================================="

{module_block}

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Locate dft_frames_list.txt across candidate paths relative to submit directory
LIST_FILE=""
for cand in "$WORK_DIR/dft_frames_list.txt" "$WORK_DIR/output/dft_frames_list.txt" "$WORK_DIR/hydeph_run/dft_frames_list.txt" "$WORK_DIR/../dft_frames_list.txt" "$WORK_DIR/../output/dft_frames_list.txt"; do
    if [ -f "$cand" ]; then
        LIST_FILE="$(cd "$(dirname "$cand")" && pwd)/$(basename "$cand")"
        break
    fi
done

if [ -z "$LIST_FILE" ]; then
    echo "[!] Error: 'dft_frames_list.txt' not found relative to submit dir ($WORK_DIR)!"
    echo "    Make sure you have executed Stage 2 first on the login node:"
    echo "        hydeph config.yaml --to-stage 2"
    exit 1
fi

WORKSPACE_DIR="$(dirname "$LIST_FILE")"
cd "$WORKSPACE_DIR"

echo "[+] Using frame list: $LIST_FILE"
echo "[+] Workspace directory: $WORKSPACE_DIR"

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    echo "[!] Error: SLURM_ARRAY_TASK_ID is unset. Ensure this script is submitted as a SLURM job array (e.g. sbatch submit_dft_array_pure_qe.slurm)."
    exit 1
fi

RAW_DIR=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$LIST_FILE")
# Normalize slashes in case of Windows formatting
RAW_DIR=$(echo "$RAW_DIR" | tr '\\\\' '/')

if [ -z "$RAW_DIR" ]; then
    echo "[!] Error: No entry found at line $((SLURM_ARRAY_TASK_ID + 1)) of $LIST_FILE"
    exit 1
fi

# Locate the target frame directory containing pw.in
DIR=""
for cand in "$WORKSPACE_DIR/$RAW_DIR" "$WORKSPACE_DIR/output/$RAW_DIR" "$WORKSPACE_DIR/hydeph_run/$RAW_DIR" "$WORK_DIR/$RAW_DIR" "$WORK_DIR/output/$RAW_DIR" "$WORK_DIR/hydeph_run/$RAW_DIR" "$RAW_DIR"; do
    if [ -d "$cand" ] && [ -f "$cand/pw.in" ]; then
        DIR="$(cd "$cand" && pwd)"
        break
    fi
done

if [ -z "$DIR" ]; then
    echo "[!] Error: Target directory '$RAW_DIR' with pw.in not found!"
    echo "    Checked relative to $WORKSPACE_DIR and $WORK_DIR"
    exit 1
fi

# Skip already converged calculations
if [ -f "$DIR/pw.out" ] && grep -q "JOB DONE." "$DIR/pw.out" 2>/dev/null && ! grep -q "convergence NOT achieved" "$DIR/pw.out" 2>/dev/null; then
    echo "[SKIP] Target directory '$RAW_DIR' is already converged. Skipping!"
    exit 0
fi

# Resolve pseudo directory and ensure it is accessible inside the target frame directory
PSEUDO_DIR=""
for cand in "$WORKSPACE_DIR/pseudo" "$WORKSPACE_DIR/../pseudo" "$WORKSPACE_DIR/../../pseudo" "$WORK_DIR/pseudo" "$WORK_DIR/../pseudo" "$WORK_DIR/../../pseudo"; do
    if [ -d "$cand" ]; then
        PSEUDO_DIR="$(cd "$cand" && pwd)"
        break
    fi
done
if [ -n "$PSEUDO_DIR" ] && [ ! -e "$DIR/pseudo" ]; then
    ln -sfn "$PSEUDO_DIR" "$DIR/pseudo"
fi

echo "[+] Launching QE in $DIR..."
echo "[+] Visible GPU: $CUDA_VISIBLE_DEVICES | CPUs per task: $SLURM_CPUS_PER_TASK"
cd "$DIR"

# Launch using srun with core binding (required for HPC-X / OpenMPI under Slurm)
srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK --cpu-bind=cores pw.x -in pw.in > pw.out 2>&1

echo "[+] Array Task $SLURM_ARRAY_TASK_ID finished successfully at $(date)"
"""
            array_script_path = os.path.join(calc_base, "submit_dft_array_pure_qe.slurm")
            with open(array_script_path, "w", encoding="utf-8", newline="\n") as f:
                f.write(array_content)

            # Also mirror SLURM scripts and dft_frames_list.txt to parent directory for convenience
            parent_dir = os.path.dirname(os.path.abspath(calc_base))
            if parent_dir and os.path.isdir(parent_dir) and parent_dir != os.path.abspath(calc_base):
                for fname in ("submit_dft_pure_qe.slurm", "submit_dft_array_pure_qe.slurm", "dft_frames_list.txt"):
                    src_f = os.path.join(calc_base, fname)
                    if os.path.isfile(src_f):
                        try:
                            shutil.copy2(src_f, os.path.join(parent_dir, fname))
                        except Exception:
                            pass

            logger.info(f"[+] Written pure QE batch script: {slurm_script_path}")
            logger.info(f"[+] Written pure QE array script: {array_script_path}")
            logger.info(f"    Cluster array command: sbatch --array=0-{array_max_idx}%16 {os.path.basename(array_script_path)}")

        return {
            "n_gs_frames": len(prepared_gs),
            "n_es_frames": len(prepared_es),
            "gs_dirs": prepared_gs,
            "es_dirs": prepared_es,
            "slurm_script": slurm_script_path,
            "array_script": array_script_path,
            "list_file": list_file_path,
        }

    def extract_and_attach_adiabatic_energetics(
        self,
        labeled_gs: List[Atoms],
        labeled_es: List[Atoms],
        at_gs: Optional[Atoms] = None,
        at_es: Optional[Atoms] = None,
    ) -> Dict[str, float]:
        """Computes exact adiabatic DFT energetics (ZPL, E_abs, E_em, Stokes shift, relaxation energies)
        from paired Franck-Condon candidate frames and attaches them to at_gs and at_es.

        Parameters
        ----------
        labeled_gs : List[Atoms]
            Converged ground-state DFT frames.
        labeled_es : List[Atoms]
            Converged excited-state Delta-SCF DFT frames.
        at_gs : Atoms, optional
            Reference ground state structure to update.
        at_es : Atoms, optional
            Reference excited state structure to update.

        Returns
        -------
        Dict[str, float]
            Dictionary of exact adiabatic energetics in eV.
        """
        return extract_and_attach_adiabatic_energetics(labeled_gs, labeled_es, at_gs=at_gs, at_es=at_es)


    def ingest_completed_calculations(
        self,
        candidate_frames: Optional[List[Atoms]] = None,
        at_gs: Optional[Atoms] = None,
        at_es: Optional[Atoms] = None,
        calc_dir_base: Optional[str] = None,
    ) -> Tuple[List[Atoms], List[Atoms]]:
        """Ingests completed Quantum ESPRESSO calculations from disk without executing `pw.x`.

        Parses converged energies, forces, and electronic band gaps from `dft_calculations/`
        and `dft_calculations_es/`, writes `labeled_dft_frames.xyz` and `labeled_frames_es.xyz`,
        computes exact adiabatic energetics (ZPL and Stokes shift), and updates `at_gs` / `at_es`.

        Parameters
        ----------
        candidate_frames : List[Atoms], optional
            Original candidate geometries. If None or empty, geometries are discovered
            directly from pw.in input files on disk.
        at_gs : Atoms, optional
            Reference ground state structure to update with DFT ground state properties.
        at_es : Atoms, optional
            Reference excited state structure to update with DFT excited state properties.
        calc_dir_base : str, optional
            Workspace root directory (defaults to config.workspace_dir).

        Returns
        -------
        Tuple[List[Atoms], List[Atoms]]
            `(labeled_frames_gs, labeled_frames_es)`
        """
        calc_base = calc_dir_base or self.config.workspace_dir
        dft_dir = os.path.join(calc_base, "dft_calculations")
        dft_es_dir = os.path.join(calc_base, "dft_calculations_es")
        out_xyz = os.path.join(calc_base, "labeled_dft_frames.xyz")
        out_es_xyz = os.path.join(calc_base, "labeled_frames_es.xyz")

        logger.info(f"[-] Ingesting completed DFT calculations from {dft_dir}...")

        # If candidate_frames is not provided, auto-discover from dft_dir
        eff_cand = list(candidate_frames) if candidate_frames else []
        if not eff_cand and os.path.isdir(dft_dir):
            subdirs = sorted([d for d in os.listdir(dft_dir) if os.path.isdir(os.path.join(dft_dir, d)) and d.startswith("frame_")])
            for idx, sd in enumerate(subdirs):
                sdir = os.path.join(dft_dir, sd)
                pw_in = os.path.join(sdir, "pw.in")
                if os.path.isfile(pw_in):
                    try:
                        at_read = ase.io.read(pw_in, format="espresso-in")
                        parts = sd.split("_", 2)
                        try:
                            cand_idx = int(parts[1]) if len(parts) >= 2 else idx
                        except ValueError:
                            cand_idx = idx
                        frame_id = parts[2] if len(parts) >= 3 else sd
                        stype = "franck_condon" if "fc" in frame_id else ("rattle" if "rattle" in frame_id else "md")
                        at_read.info["candidate_index"] = cand_idx
                        at_read.info["frame_id"] = frame_id
                        at_read.info["sample_type"] = stype
                        if "fc" in frame_id:
                            try:
                                fc_num = int(frame_id.split("_")[-1])
                                at_read.info["lambda"] = float(fc_num) / max(1.0, float(getattr(self.config, "n_fc_frames", 7) - 1))
                            except Exception:
                                pass
                        eff_cand.append(at_read)
                    except Exception as e:
                        logger.warning(f"[-] Could not parse structure from {pw_in}: {e}")

        # 1. Ingest Ground State frames
        labeled_gs: List[Atoms] = []
        missing_gs: List[str] = []

        for idx, fr in enumerate(eff_cand):
            cand_idx = fr.info.get("candidate_index", idx)
            frame_id = fr.info.get("frame_id") or fr.info.get("label") or f"frame_{cand_idx:02d}"
            stype = fr.info.get("sample_type", "unknown")

            cdir, is_done = find_frame_calc_dir(dft_dir, cand_idx, frame_id, fr, config=self.config)
            pw_out = os.path.join(cdir, "pw.out")

            if is_done and os.path.isfile(pw_out):
                e_ev, f_ev_a = parse_qe_output(pw_out)
                if e_ev is not None and f_ev_a is not None and len(f_ev_a) == len(fr):
                    at = fr.copy()
                    at.calc = None
                    at.info["energy"] = e_ev
                    at.arrays["forces"] = f_ev_a
                    at.info["qe_status"] = "converged"
                    at.info["sample_type"] = stype
                    if "lambda" in fr.info:
                        at.info["lambda"] = fr.info["lambda"]
                    _attach_gap_info(at, parse_qe_band_gap(pw_out))
                    labeled_gs.append(at)
                    continue

            missing_gs.append(f"{frame_id} ({os.path.basename(cdir)})")

        if missing_gs and eff_cand:
            logger.warning(
                f"[!] Notice: {len(missing_gs)}/{len(eff_cand)} GS frames are not yet converged or missing."
            )
            if len(missing_gs) <= 5:
                logger.warning(f"    Missing: {', '.join(missing_gs)}")

        if not labeled_gs:
            raise FileNotFoundError(
                f"No converged Quantum ESPRESSO calculations found in '{dft_dir}'. "
                "Please run the compute job (e.g. sbatch submit_dft_pure_qe.slurm or submit_dft_array_pure_qe.slurm) before resuming Stage 4."
            )

        # Write labeled_dft_frames.xyz
        for f in labeled_gs:
            f.calc = None
        tmp_xyz = out_xyz + ".tmp"
        ase.io.write(tmp_xyz, labeled_gs, format="extxyz")
        os.replace(tmp_xyz, out_xyz)
        logger.info(f"[+] Ingested {len(labeled_gs)} converged GS frames -> {out_xyz}")

        # Attach DFT properties to at_gs if present
        if at_gs is not None:
            gs_cand = next(
                (f for f in labeled_gs if f.info.get("sample_type") == "franck_condon" and abs(f.info.get("lambda", 1.0)) < 1e-4),
                labeled_gs[0] if labeled_gs else None,
            )
            if gs_cand is not None:
                if "energy" in gs_cand.info:
                    at_gs.info["energy"] = float(gs_cand.info["energy"])
                for k in ("dft_gap_eV", "dft_homo_eV", "dft_lumo_eV", "dft_fermi_eV"):
                    if k in gs_cand.info:
                        at_gs.info[k] = float(gs_cand.info[k])

        # 2. Ingest Excited State frames if dft_calculations_es exists
        labeled_es: List[Atoms] = []
        if os.path.isdir(dft_es_dir):
            logger.info(f"[-] Ingesting optical excited-state (Delta-SCF) calculations from {dft_es_dir}...")
            es_cand_mode = getattr(self.config, "dft_es_candidates", "all").lower()
            if es_cand_mode == "fc_path":
                es_targets = [f for f in eff_cand if f.info.get("sample_type") == "franck_condon"]
            else:
                es_targets = list(eff_cand)

            if not es_targets and os.path.isdir(dft_es_dir):
                # Auto-discover ES candidate frames from dft_es_dir
                subdirs_es = sorted([d for d in os.listdir(dft_es_dir) if os.path.isdir(os.path.join(dft_es_dir, d)) and d.startswith("frame_es_")])
                for idx, sd in enumerate(subdirs_es):
                    sdir = os.path.join(dft_es_dir, sd)
                    pw_in = os.path.join(sdir, "pw.in")
                    if os.path.isfile(pw_in):
                        try:
                            at_read = ase.io.read(pw_in, format="espresso-in")
                            parts = sd.split("_", 3)
                            try:
                                cand_idx = int(parts[2]) if len(parts) >= 3 else idx
                            except ValueError:
                                cand_idx = idx
                            frame_id = parts[3] if len(parts) >= 4 else sd
                            stype = "franck_condon" if "fc" in frame_id else ("rattle" if "rattle" in frame_id else "md")
                            at_read.info["candidate_index"] = cand_idx
                            at_read.info["frame_id"] = frame_id
                            at_read.info["sample_type"] = stype
                            if "fc" in frame_id:
                                try:
                                    fc_num = int(frame_id.split("_")[-1])
                                    at_read.info["lambda"] = float(fc_num) / max(1.0, float(getattr(self.config, "n_fc_frames", 7) - 1))
                                except Exception:
                                    pass
                            es_targets.append(at_read)
                        except Exception as e:
                            logger.warning(f"[-] Could not parse ES structure from {pw_in}: {e}")

            for i, fr in enumerate(es_targets):
                cand_idx = fr.info.get("candidate_index", i)
                frame_id = fr.info.get("frame_id") or fr.info.get("label") or f"frame_{cand_idx:02d}"
                stype = fr.info.get("sample_type", "franck_condon")

                cdir_es = os.path.join(dft_es_dir, f"frame_es_{cand_idx:03d}_{frame_id}")
                pw_out = os.path.join(cdir_es, "pw.out")
                if os.path.isfile(pw_out):
                    try:
                        with open(pw_out, "r", encoding="utf-8", errors="replace") as pf:
                            content = pf.read()
                        if "JOB DONE." in content and "convergence NOT achieved" not in content:
                            e_ev, f_ev_a = parse_qe_output(pw_out)
                            if e_ev is not None and f_ev_a is not None and len(f_ev_a) == len(fr):
                                at = fr.copy()
                                at.calc = None
                                at.info["energy"] = e_ev
                                at.arrays["forces"] = f_ev_a
                                at.info["electronic_state"] = "excited"
                                at.info["sample_type"] = stype
                                at.info["qe_status"] = "converged"
                                if "lambda" in fr.info:
                                    at.info["lambda"] = fr.info["lambda"]
                                labeled_es.append(at)
                    except Exception:
                        pass

            if labeled_es:
                for f in labeled_es:
                    f.calc = None
                tmp_es_xyz = out_es_xyz + ".tmp"
                ase.io.write(tmp_es_xyz, labeled_es, format="extxyz")
                os.replace(tmp_es_xyz, out_es_xyz)
                logger.info(f"[+] Ingested {len(labeled_es)} converged ES frames -> {out_es_xyz}")

        # Compute exact adiabatic energetics and attach to at_gs and at_es
        extract_and_attach_adiabatic_energetics(labeled_gs, labeled_es, at_gs=at_gs, at_es=at_es)

        return labeled_gs, labeled_es

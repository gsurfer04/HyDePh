"""Quantum ESPRESSO `pw.x` runner with caching and type-balanced resumption."""

import os
import sys
import re
import time
import signal
import shutil
import threading
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import ase.io
from ase import Atoms
from ase.units import Ry, Bohr


def is_windows():
    """Returns True if running on Windows."""
    return sys.platform.startswith("win")


def write_qe_input(atoms, pw_in_path, config, calculation="scf", retry=False):
    """Writes a Quantum ESPRESSO `pw.x` input card."""
    cell = atoms.get_cell()
    pos = atoms.positions
    symbols = atoms.get_chemical_symbols()

    unique_symbols = sorted(list(set(symbols)))
    from ase.data import atomic_masses, atomic_numbers
    mass_dict = {"Li": 6.941, "Cl": 35.453, "C": 12.011, "Si": 28.085, "N": 14.007, "F": 18.9984}

    # Fortran namelist parser requires forward slashes
    clean_pseudo_dir = config.pseudo_dir.replace("\\", "/") if getattr(config, "pseudo_dir", None) else "./pseudo"
    mixing_beta = 0.2 if retry else 0.4
    retry_lines = "\n  mixing_mode = 'local-TF',\n  electron_maxstep = 150," if retry else ""

    header = f"""&CONTROL
  calculation = '{calculation}',
  restart_mode = 'from_scratch',
  pseudo_dir = '{clean_pseudo_dir}',
  outdir = './tmp',
  prefix = 'calc',
  tprnfor = .true.,
  tstress = .false.,
  disk_io = 'none'
/
&SYSTEM
  ibrav = 0,
  nat = {len(atoms)},
  ntyp = {len(unique_symbols)},
  ecutwfc = {config.ecutwfc},
  ecutrho = {config.ecutrho},
  nspin = {2 if config.spin_polarized else 1},
  tot_magnetization = {config.spin_multiplicity - 1 if config.spin_polarized else 0},
  tot_charge = {config.tot_charge},
  occupations = 'smearing',
  smearing = 'gaussian',
  degauss = 0.005
/
&ELECTRONS
  conv_thr = 1.0d-5,
  mixing_beta = {mixing_beta}{retry_lines}
/
ATOMIC_SPECIES
"""
    for s in unique_symbols:
        upf_name = os.path.basename(config.pseudopotentials[s])
        mass = mass_dict.get(s, float(atomic_masses[atomic_numbers.get(s, 0)]))
        header += f"  {s:<3} {mass:8.4f}  {upf_name}\n"

    header += "\nCELL_PARAMETERS (angstrom)\n"
    for vec in cell:
        header += f"  {vec[0]:14.8f} {vec[1]:14.8f} {vec[2]:14.8f}\n"

    header += "\nATOMIC_POSITIONS (angstrom)\n"
    for s, p in zip(symbols, pos):
        header += f"  {s:<3} {p[0]:14.8f} {p[1]:14.8f} {p[2]:14.8f}\n"

    header += f"\nK_POINTS (automatic)\n  {config.kpoints[0]} {config.kpoints[1]} {config.kpoints[2]} 0 0 0\n"

    with open(pw_in_path, "w") as f:
        f.write(header)


def parse_qe_output(pw_out_path):
    """Parses total energy (eV) and forces (eV/A) from `pw.x` output, returning `(None, None)` if unconverged."""
    if not os.path.isfile(pw_out_path):
        return None, None

    energy_ev = None
    forces_list = []

    with open(pw_out_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    for line in lines:
        if "convergence NOT achieved" in line:
            return None, None

    for i, line in enumerate(lines):
        if "!    total energy" in line:
            parts = line.split("=")
            if len(parts) > 1:
                e_ry = float(parts[1].replace("Ry", "").strip())
                energy_ev = e_ry * Ry

        if "Forces acting on atoms (cartesian axes, Ry/au):" in line:
            forces_list = []
            for j in range(i + 1, len(lines)):
                fline = lines[j].strip()
                if "total force" in fline:
                    break
                if not fline:
                    continue
                m = re.search(r"force =\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)", fline)
                if m:
                    fx = float(m.group(1)) * (Ry / Bohr)
                    fy = float(m.group(2)) * (Ry / Bohr)
                    fz = float(m.group(3)) * (Ry / Bohr)
                    forces_list.append([fx, fy, fz])

    forces_ev_a = np.array(forces_list) if forces_list else None
    return energy_ev, forces_ev_a


def resolve_pw_executable(pw_exe=None):
    """Resolves the Quantum ESPRESSO `pw` executable path across Linux and Windows."""
    if pw_exe:
        if os.path.isfile(pw_exe):
            return os.path.abspath(pw_exe)
        found = shutil.which(pw_exe)
        if found:
            return found
        return pw_exe

    for env_var in ["PW_EXE", "QE_PW_EXE"]:
        env_exe = os.environ.get(env_var)
        if env_exe:
            if os.path.isfile(env_exe):
                return os.path.abspath(env_exe)
            found = shutil.which(env_exe)
            if found:
                return found
            return env_exe

    bin_dir = os.environ.get("QE_BIN_DIR") or os.environ.get("QE_PATH")
    if bin_dir and os.path.isdir(bin_dir):
        for candidate in (["pw.exe", "pw.x.exe", "pw.x"] if is_windows() else ["pw.x", "pw"]):
            cand_path = os.path.join(bin_dir, candidate)
            if os.path.isfile(cand_path):
                return os.path.abspath(cand_path)

    if is_windows():
        for name in ["pw.exe", "pw.x.exe", "pw.x"]:
            found = shutil.which(name)
            if found:
                return found

        raise ValueError(
            "Quantum ESPRESSO 'pw.exe' not found. On Windows, Quantum ESPRESSO is not installed "
            "as a standard system program and its location is not assumed. Please supply its path via "
            "ToolkitConfig(pw_exe=r'C:\\path\\to\\pw.exe') or by setting the PW_EXE environment variable. "
            "Windows binaries can be downloaded from: https://github.com/QMatSuite/quantum-espresso-windows-exe"
        )
    else:
        found = shutil.which("pw.x") or shutil.which("pw")
        if found:
            return found
        for p in ["/usr/local/bin/pw.x", "/usr/bin/pw.x"]:
            if os.path.isfile(p):
                return p
        return "/usr/local/bin/pw.x"


def resolve_mpi_runner(mpi_command=None, n_procs=2):
    """Resolves the MPI launcher and process-count flag, returning `(None, None)` for `n_procs <= 1`."""
    if n_procs <= 1:
        return None, None

    if mpi_command:
        cmd_lower = os.path.basename(mpi_command).lower()
        flag = "-n" if ("mpiexec" in cmd_lower and is_windows()) else ("-np" if "mpirun" in cmd_lower else "-n")
        return mpi_command, flag

    for env_var in ["HYDEPH_MPI_CMD", "MPI_CMD"]:
        env_mpi = os.environ.get(env_var)
        if env_mpi:
            flag = "-n" if is_windows() else "-np"
            return env_mpi, flag

    if is_windows():
        msmpi_bin = os.environ.get("MSMPI_BIN")
        if msmpi_bin:
            msmpi_exe = os.path.join(msmpi_bin, "mpiexec.exe")
            if os.path.isfile(msmpi_exe):
                return os.path.abspath(msmpi_exe), "-n"

        for name in ["mpiexec.exe", "mpiexec"]:
            found = shutil.which(name)
            if found:
                return found, "-n"

        for p in [
            r"C:\Program Files\Microsoft MPI\Bin\mpiexec.exe",
            r"C:\Program Files (x86)\Microsoft MPI\Bin\mpiexec.exe",
        ]:
            if os.path.isfile(p):
                return os.path.abspath(p), "-n"

        conda_prefix = os.environ.get("CONDA_PREFIX")
        if conda_prefix:
            conda_mpi = os.path.join(conda_prefix, "Library", "bin", "mpiexec.exe")
            if os.path.isfile(conda_mpi):
                return os.path.abspath(conda_mpi), "-n"

        return "mpiexec", "-n"
    else:
        found = shutil.which("mpirun")
        if found:
            return found, "-np"
        found = shutil.which("mpiexec")
        if found:
            return found, "-np"
        return "/usr/bin/mpirun", "-np"


def build_qe_command(pw_exe, pw_in="pw.in", n_procs=2, mpi_command=None, mpi_args=None):
    """Builds the command argument list for running Quantum ESPRESSO."""
    cmd_args = []
    if n_procs > 1:
        mpi_runner, n_flag = resolve_mpi_runner(mpi_command, n_procs=n_procs)
        if mpi_runner:
            cmd_args.append(mpi_runner)
            cmd_args.extend([n_flag, str(n_procs)])
            if mpi_args:
                cmd_args.extend(mpi_args)

    cmd_args.append(pw_exe)
    cmd_args.extend(["-in", pw_in])
    return cmd_args


def check_pw_in_matches_atoms(pw_in, atoms, tol=1e-3):
    """Verifies that atomic positions in existing `pw.in` match `atoms` within `tol`."""
    if not os.path.isfile(pw_in):
        return True
    try:
        with open(pw_in, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        in_pos = False
        coords = []
        for line in lines:
            line_str = line.strip()
            if "ATOMIC_POSITIONS" in line_str.upper():
                in_pos = True
                continue
            if in_pos:
                if line_str.startswith("/") or line_str.startswith("&") or "K_POINTS" in line_str.upper():
                    break
                parts = line_str.split()
                if len(parts) >= 4:
                    coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
        if len(coords) != len(atoms):
            return False
        diff = np.abs(np.array(coords) - atoms.positions)
        return bool(np.max(diff) < tol)
    except Exception:
        return True


def find_frame_calc_dir(dft_dir, cand_idx, frame_id, atoms):
    """Finds the calculation directory for a frame and returns `(calc_dir, is_converged)`."""
    canonical_dir = os.path.join(dft_dir, f"frame_{cand_idx:03d}_{frame_id}")
    candidates = [
        canonical_dir,
        os.path.join(dft_dir, frame_id),
        os.path.join(dft_dir, f"frame_{cand_idx:03d}"),
    ]
    for cdir in candidates:
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
                        and check_pw_in_matches_atoms(pw_in, atoms)
                    ):
                        e_ev, f_ev_a = parse_qe_output(pw_out)
                        if e_ev is not None and f_ev_a is not None and len(f_ev_a) == len(atoms):
                            return cdir, True
                except Exception:
                    pass
    return canonical_dir, False


class QERunner:
    def __init__(self, config, n_procs=2, pw_exe=None, mpi_command=None, mpi_args=None, n_workers=None, timeout_s=None):
        self.config = config
        self.n_procs = n_procs

        chosen_pw = pw_exe or getattr(config, "pw_exe", None)
        self.pw_exe = resolve_pw_executable(chosen_pw)

        self.mpi_command = mpi_command or getattr(config, "mpi_command", None)
        self.mpi_args = mpi_args or getattr(config, "mpi_args", None)
        self.n_workers = int(n_workers if n_workers is not None else getattr(config, "n_workers_qe", 1))
        self.timeout_s = timeout_s if timeout_s is not None else getattr(config, "qe_timeout_s", None)
        self.last_failed_entries = []

    def label_single_frame(self, atoms, calc_dir, retry=False, timeout_s=None):
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
                        print(f"      [Cache hit] Reusing converged calculation in {calc_dir}", flush=True)
                        at_labeled = atoms.copy()
                        at_labeled.info["energy"] = e_ev
                        at_labeled.arrays["forces"] = f_ev_a
                        return at_labeled
                else:
                    print(f"      [Cache mismatch] Atomic positions in {pw_in} differ from frame. Recomputing...", flush=True)
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
                print(f"      [Resuming] Incomplete calculation in {calc_dir}. Cleaned scratch tmp/ and restarting...", flush=True)

        write_qe_input(atoms, pw_in, self.config, retry=retry)
        os.makedirs(os.path.join(calc_dir, "tmp"), exist_ok=True)

        cmd_args = build_qe_command(
            self.pw_exe,
            pw_in="pw.in",
            n_procs=self.n_procs,
            mpi_command=self.mpi_command,
            mpi_args=self.mpi_args,
        )

        # Prepend pw_exe directory to PATH so colocated MKL/MS-MPI DLLs resolve on Windows
        env = os.environ.copy()
        if os.path.isabs(self.pw_exe):
            pw_dir = os.path.dirname(self.pw_exe)
            if pw_dir:
                env["PATH"] = pw_dir + os.pathsep + env.get("PATH", "")

        try:
            with open(pw_out, "wb") as f_out:
                proc = subprocess.Popen(
                    cmd_args,
                    stdout=f_out,
                    stderr=subprocess.STDOUT,
                    cwd=calc_dir,
                    env=env,
                )

                last_iter = 0
                t_start = time.time()
                t_last_log = t_start
                while proc.poll() is None:
                    now = time.time()
                    if eff_timeout is not None and (now - t_start) > float(eff_timeout):
                        try:
                            proc.kill()
                            proc.wait()
                        except Exception:
                            pass
                        raise RuntimeError(f"QE calculation timed out after {float(eff_timeout):.1f}s in {calc_dir}")
                    time.sleep(0.05)
                    now = time.time()
                    if (now - t_last_log) >= 2.0 and os.path.isfile(pw_out):
                        t_last_log = now
                        try:
                            with open(pw_out, "r", encoding="utf-8", errors="replace") as f:
                                lines = f.readlines()
                            for line in reversed(lines[-30:]):
                                if "iteration #" in line:
                                    m = re.search(r"iteration #\s+(\d+)", line)
                                    if m:
                                        cur_iter = int(m.group(1))
                                        if cur_iter != last_iter:
                                            last_iter = cur_iter
                                            print(f"      [pw] SCF iteration {cur_iter}...", flush=True)
                                    break
                                elif "Forces acting on atoms" in line:
                                    print("      [pw] SCF converged! Calculating atomic forces...", flush=True)
                                    break
                        except Exception:
                            pass

                proc.wait()
        except OSError as exc:
            raise RuntimeError(f"QE calculation failed in {calc_dir}: {exc}") from exc

        e_ev, f_ev_a = parse_qe_output(pw_out)
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
                )
            raise RuntimeError(f"QE calculation failed in {calc_dir}.\n{tail_msg}{win_msg}")

        at_labeled = atoms.copy()
        at_labeled.info["energy"] = e_ev
        at_labeled.arrays["forces"] = f_ev_a
        return at_labeled

    def _obtain_replacement_entry(
        self, stype, unused_by_type, frames, dft_dir, sampler, at_gs, type_total_counts, next_cand_idx
    ):
        """Returns the next unused candidate entry of `stype`, or generates a new frame if none remain."""
        pool = unused_by_type.get(stype, [])
        if pool:
            return pool.pop(0), next_cand_idx

        if sampler is None:
            from .frame_sampler import FrameSampler
            sampler = FrameSampler(self.config)

        base_atoms = at_gs if at_gs is not None else frames[0]
        type_idx = type_total_counts.get(stype, 0)
        type_total_counts[stype] = type_idx + 1
        cand_idx = next_cand_idx
        next_cand_idx += 1

        new_fr = sampler.generate_replacement_frame(
            stype, base_atoms, type_index=type_idx, candidate_index=cand_idx
        )
        frames.append(new_fr)
        cand_xyz = os.path.join(self.config.workspace_dir, "candidate_frames.xyz")
        if os.path.isfile(cand_xyz):
            try:
                ase.io.write(cand_xyz, frames, format="extxyz")
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

    def label_all_frames(
        self,
        frames,
        max_frames=None,
        max_new_frames=None,
        replace_failed_frames=None,
        sampler=None,
        at_gs=None,
        n_workers=None,
    ):
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
            cdir, is_done = find_frame_calc_dir(dft_dir, cand_idx, frame_id, fr)
            entry = {
                "index": cand_idx,
                "frame_id": frame_id,
                "sample_type": stype,
                "atoms": fr,
                "calc_dir": cdir,
            }
            if is_done:
                completed_entries.append(entry)
            else:
                pending_entries.append(entry)

        next_cand_idx = max_cand_idx + 1

        print(f"\n[-] High-Fidelity DFT Labeling (Quantum ESPRESSO, np={self.n_procs}, workers={eff_workers})...")
        print(f"    Total candidate pool: {len(frames)} frames")

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
            print(f"    [i] Already converged: {len(completed_entries)} frames ({comp_str})")
        if pending_entries:
            pend_str = ", ".join(f"{k}: {v}" for k, v in sorted(type_pending.items()))
            print(f"    [i] Pending / uncalculated: {len(pending_entries)} frames ({pend_str})")
        else:
            print(f"    [i] All {len(frames)} candidate frames are already converged!")

        resume_mode = getattr(self.config, "dft_resume_mode", "target_total")
        dft_target = max_frames if max_frames is not None else getattr(self.config, "max_dft_frames", None)
        new_limit = max_new_frames if max_new_frames is not None else getattr(self.config, "max_new_dft_frames", None)
        do_retry = getattr(self.config, "retry_failed_dft", True)
        do_replace = (
            replace_failed_frames
            if replace_failed_frames is not None
            else getattr(self.config, "replace_failed_frames", False)
        )

        to_calculate = []
        if not pending_entries:
            to_calculate = []
        elif resume_mode == "max_new" or (dft_target is None and new_limit is not None):
            budget = new_limit if new_limit is not None else len(pending_entries)
            sorted_pending = sorted(
                pending_entries,
                key=lambda e: (type_completed.get(e["sample_type"], 0), e["index"])
            )
            to_calculate = sorted_pending[:budget]
        elif dft_target is not None:
            n_done = len(completed_entries)
            if n_done >= dft_target:
                print(f"    [i] Target max_dft_frames={dft_target} already satisfied by {n_done} converged frames.")
                print("        Reusing all converged frames without duplicate DFT execution.")
                to_calculate = []
            else:
                needed = dft_target - n_done
                if new_limit is not None:
                    needed = min(needed, new_limit)
                sorted_pending = sorted(
                    pending_entries,
                    key=lambda e: (type_completed.get(e["sample_type"], 0), e["index"])
                )
                to_calculate = sorted_pending[:needed]
                calc_str = ", ".join(f"{e['frame_id']} ({e['sample_type']})" for e in to_calculate)
                print(f"    [i] Need {needed} additional frames to reach target {dft_target}. Selected: {calc_str}")
        else:
            to_calculate = list(pending_entries)

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
            print(f"\n    ---> [{step_num}/{len(to_calculate)}] Calculating {label} ({stype}) in {os.path.basename(cdir)}...")
            try:
                labeled_fr = self.label_single_frame(fr, cdir, retry=False)
                labeled_fr.info["qe_status"] = "converged"
                dt = time.time() - t0
                with state_lock:
                    labeled_dict[idx] = labeled_fr
                    type_completed[stype] = type_completed.get(stype, 0) + 1
                    current_frames = [labeled_dict[k] for k in sorted(labeled_dict.keys())]
                    ase.io.write(out_xyz, current_frames, format="extxyz")
                print(f"    [+] {label} labeled | E = {labeled_fr.info['energy']:.4f} eV | Max F = {np.abs(labeled_fr.arrays['forces']).max():.4f} eV/A ({dt:.1f} s)")
            except RuntimeError as err:
                with state_lock:
                    if do_retry:
                        fr.info["qe_status"] = "retry_pending"
                        entry["status"] = "retry_pending"
                        entry["attempts"] = 1
                        entry["last_error"] = str(err)
                        retry_queue.append(entry)
                        print(f"    [!] {label} ({stype}) failed on attempt 1. Tagged for retry after remaining frames.")
                    else:
                        fr.info["qe_status"] = "failed"
                        entry["status"] = "failed"
                        entry["attempts"] = 1
                        entry["last_error"] = str(err)
                        failed_entries.append(entry)
                        print(f"    [!] {label} ({stype}) failed and retry is disabled. Ignoring frame.")

        if to_calculate:
            print(f"\n[-] Executing {len(to_calculate)} DFT calculation(s)...")
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
            print(f"\n[-] Retrying {len(retry_queue)} tagged DFT calculation(s)...")
            for retry_num, entry in enumerate(retry_queue, 1):
                idx = entry["index"]
                fr = entry["atoms"]
                cdir = entry["calc_dir"]
                label = entry["frame_id"]
                stype = entry["sample_type"]

                t0 = time.time()
                print(f"\n    ---> [Retry {retry_num}/{len(retry_queue)}] Re-running {label} ({stype}) in {os.path.basename(cdir)}...")
                try:
                    labeled_fr = self.label_single_frame(fr, cdir, retry=True)
                    labeled_fr.info["qe_status"] = "converged_on_retry"
                    entry["status"] = "converged_on_retry"
                    entry["attempts"] = 2
                    labeled_dict[idx] = labeled_fr
                    dt = time.time() - t0
                    print(f"    [+] {label} converged on retry | E = {labeled_fr.info['energy']:.4f} eV | Max F = {np.abs(labeled_fr.arrays['forces']).max():.4f} eV/A ({dt:.1f} s)")
                    type_completed[stype] = type_completed.get(stype, 0) + 1
                    current_frames = [labeled_dict[k] for k in sorted(labeled_dict.keys())]
                    ase.io.write(out_xyz, current_frames, format="extxyz")
                except RuntimeError as err:
                    fr.info["qe_status"] = "failed"
                    entry["status"] = "failed"
                    entry["attempts"] = 2
                    entry["last_error"] = str(err)
                    failed_entries.append(entry)
                    print(f"    [!] {label} ({stype}) failed a second time.")

                    if do_replace and stype in ("core_rattle", "foundation_md_fps"):
                        repl_entry, next_cand_idx = self._obtain_replacement_entry(
                            stype, unused_by_type, frames, dft_dir, sampler, at_gs, type_total_counts, next_cand_idx
                        )
                        r_idx = repl_entry["index"]
                        r_fr = repl_entry["atoms"]
                        r_cdir = repl_entry["calc_dir"]
                        r_label = repl_entry["frame_id"]
                        print(f"    [->] Taking replacement frame {r_label} ({stype}) in {os.path.basename(r_cdir)} to fulfill quota...")
                        try:
                            try:
                                r_labeled = self.label_single_frame(r_fr, r_cdir, retry=False)
                            except RuntimeError:
                                print(f"    [!] Replacement {r_label} ({stype}) failed on attempt 1; retrying...")
                                r_labeled = self.label_single_frame(r_fr, r_cdir, retry=True)
                            r_labeled.info["qe_status"] = "converged"
                            labeled_dict[r_idx] = r_labeled
                            type_completed[stype] = type_completed.get(stype, 0) + 1
                            current_frames = [labeled_dict[k] for k in sorted(labeled_dict.keys())]
                            ase.io.write(out_xyz, current_frames, format="extxyz")
                            print(f"    [+] Replacement {r_label} ({stype}) converged | E = {r_labeled.info['energy']:.4f} eV")
                        except RuntimeError as repl_err:
                            r_fr.info["qe_status"] = "failed"
                            repl_entry["status"] = "failed"
                            repl_entry["last_error"] = str(repl_err)
                            failed_entries.append(repl_entry)
                            print(f"    [!] Replacement {r_label} ({stype}) also failed. Ignoring slot and continuing.")
                    else:
                        print(f"    [i] Ignoring failed calculation {label} ({stype}) and continuing with remaining frames.")

        self.last_failed_entries = failed_entries
        final_labeled = [labeled_dict[k] for k in sorted(labeled_dict.keys())]
        if final_labeled:
            ase.io.write(out_xyz, final_labeled, format="extxyz")

        if failed_entries:
            fail_str = ", ".join(f"{e['frame_id']} ({e['sample_type']})" for e in failed_entries)
            print(f"\n[!] Ignored {len(failed_entries)} failed DFT calculation(s): {fail_str}")

        if not final_labeled and (to_calculate or completed_entries):
            raise RuntimeError("All QE calculations failed; no converged frames available for training.")

        final_types = {}
        for fr in final_labeled:
            st = fr.info.get("sample_type", "unknown")
            final_types[st] = final_types.get(st, 0) + 1
        summary_str = ", ".join(f"{k}: {v}" for k, v in sorted(final_types.items()))
        print(f"\n[+] DFT Labeling Complete: {len(final_labeled)} total frames ({summary_str}) -> {out_xyz}")
        return final_labeled


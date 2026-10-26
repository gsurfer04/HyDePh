"""Quantum ESPRESSO `pw.x` runner with single-pass parsing, caching, and type-balanced resumption."""

from __future__ import annotations

import os
import sys
import re
import time
import shutil
import threading
import subprocess
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
import ase.io
from ase import Atoms
from ase.units import Ry, Bohr
from ase.data import atomic_masses, atomic_numbers

from .config import ToolkitConfig

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

logger = logging.getLogger(__name__)

RY_TO_EV = float(Ry)
RY_BOHR_TO_EV_A = float(Ry / Bohr)


def is_windows() -> bool:
    """Returns True if running on Windows."""
    return sys.platform.startswith("win")


def inspect_upf_metadata(upf_path: str) -> Dict[str, Any]:
    """Reads lightweight header metadata (`is_norm_conserving`, `z_valence`) from a UPF file."""
    info: Dict[str, Any] = {"is_norm_conserving": None, "z_valence": None}
    if not upf_path or not os.path.isfile(upf_path):
        return info
    try:
        header_lines = []
        with open(upf_path, "r", encoding="utf-8", errors="replace") as f:
            for _ in range(200):
                line = f.readline()
                if not line:
                    break
                header_lines.append(line)
                if "<PP_MESH" in line or "<PP_R " in line or "</PP_HEADER>" in line:
                    break
        text = "".join(header_lines)

        # Parse z_valence (UPF v2 attribute or UPF v1 line)
        m_z2 = re.search(r'z_valence\s*=\s*"\s*([-\d.Ee+]+)\s*"', text, re.IGNORECASE)
        if m_z2:
            info["z_valence"] = float(m_z2.group(1))
        else:
            m_z1 = re.search(r"([-\d.Ee+]+)\s+Z\s+valence", text, re.IGNORECASE)
            if m_z1:
                info["z_valence"] = float(m_z1.group(1))

        # Parse pseudopotential type (Norm-Conserving vs Ultrasoft/PAW)
        m_us = re.search(r'is_ultrasoft\s*=\s*"\s*([TFtf])\s*"', text)
        m_paw = re.search(r'is_paw\s*=\s*"\s*([TFtf])\s*"', text)
        m_type = re.search(r'pseudo_type\s*=\s*"\s*([A-Za-z0-9_-]+)\s*"', text)
        if m_us and m_us.group(1).upper() == "T":
            info["is_norm_conserving"] = False
        elif m_paw and m_paw.group(1).upper() == "T":
            info["is_norm_conserving"] = False
        elif m_type:
            ptype = m_type.group(1).upper()
            if ptype in ("NC", "SL", "NCPP"):
                info["is_norm_conserving"] = True
            elif ptype in ("US", "USPP", "PAW"):
                info["is_norm_conserving"] = False
        elif m_us and m_us.group(1).upper() == "F":
            info["is_norm_conserving"] = True
    except Exception:
        pass
    return info


_PARITY_WARNED_KEYS = set()


def write_qe_input(
    atoms: Atoms,
    pw_in_path: str,
    config: ToolkitConfig,
    calculation: str = "scf",
    retry: bool = False,
    tot_magnetization: Optional[int] = None,
    nspin: Optional[int] = None,
):
    """Writes a Quantum ESPRESSO `pw.x` input card.
    
    Uses universal ASE atomic masses, sets `K_POINTS gamma` for (1, 1, 1),
    auto-detects Norm-Conserving UPF cutoffs (`ecutrho = 4 * ecutwfc`),
    validates spin-multiplicity parity against total valence electrons,
    disables symmetry reduction via `nosym = .true.` for distorted defect supercells,
    supports `calculation = 'relax'` with `&IONS` BFGS dynamics,
    and trims excess empty conduction bands (`nbnd`) and Davidson subspace size.
    """
    cell = atoms.get_cell()
    pos = atoms.positions
    symbols = atoms.get_chemical_symbols()
    unique_symbols = sorted(list(set(symbols)))

    # Fortran namelist parser requires forward slashes
    clean_pseudo_dir = config.pseudo_dir.replace("\\", "/") if config.pseudo_dir else "./pseudo"

    # Inspect UPF files for Norm-Conserving status and valence electron counts
    meta_by_sym: Dict[str, Dict[str, Any]] = {}
    for s in unique_symbols:
        upf_rel = config.pseudopotentials.get(s, "")
        full_upf = upf_rel if os.path.isabs(upf_rel) else os.path.join(config.pseudo_dir or ".", upf_rel)
        meta_by_sym[s] = inspect_upf_metadata(full_upf)

    all_nc = bool(unique_symbols) and all(meta_by_sym[s]["is_norm_conserving"] is True for s in unique_symbols)
    all_z = bool(unique_symbols) and all(meta_by_sym[s]["z_valence"] is not None for s in unique_symbols)

    eff_ecutrho = float(config.ecutrho)
    if all_nc and config.ecutrho == 320.0 and (4.0 * config.ecutwfc) < config.ecutrho:
        eff_ecutrho = 4.0 * float(config.ecutwfc)

    if tot_magnetization is not None:
        tot_mag = int(tot_magnetization)
    else:
        tot_mag = int(config.spin_multiplicity - 1) if config.spin_polarized else 0
    eff_nspin = nspin if nspin is not None else (2 if (config.spin_polarized or tot_mag > 0) else 1)
    eff_nbnd: Optional[int] = int(config.nbnd) if getattr(config, "nbnd", None) is not None else None

    if all_z:
        total_nelec = sum(float(meta_by_sym[s]["z_valence"]) for s in symbols) - float(config.tot_charge)
        nelec_int = int(round(total_nelec))
        if eff_nspin == 2 and abs(total_nelec - nelec_int) < 1e-4:
            if (nelec_int - tot_mag) % 2 != 0:
                warn_key = (nelec_int, tot_mag, getattr(config, "auto_spin_parity", True))
                if getattr(config, "auto_spin_parity", True):
                    new_mag = tot_mag + 1 if tot_mag >= 1 else 0
                    if warn_key not in _PARITY_WARNED_KEYS:
                        _PARITY_WARNED_KEYS.add(warn_key)
                        logger.warning(
                            f"    [Spin Parity] Total valence electrons ({nelec_int}) and tot_magnetization ({tot_mag}) "
                            f"have opposite parity, which forces fractional spin occupations "
                            f"({0.5*(nelec_int+tot_mag):.1f} up, {0.5*(nelec_int-tot_mag):.1f} down). "
                            f"Auto-adjusting tot_magnetization -> {new_mag} (spin_multiplicity={new_mag+1})."
                        )
                    tot_mag = new_mag
                else:
                    if warn_key not in _PARITY_WARNED_KEYS:
                        _PARITY_WARNED_KEYS.add(warn_key)
                        logger.warning(
                            f"    [Spin Parity] WARNING: Total valence electrons ({nelec_int}) and tot_magnetization ({tot_mag}) "
                            f"have opposite parity! QE will use fractional spin occupations."
                        )

        if eff_nbnd is None and abs(total_nelec - nelec_int) < 1e-4 and nelec_int > 0:
            n_occ_max = (nelec_int + abs(tot_mag)) // 2 if eff_nspin == 2 else (nelec_int + 1) // 2
            buffer_bands = max(6, min(16, int(np.ceil(0.05 * n_occ_max))))
            eff_nbnd = n_occ_max + buffer_bands

    conv_thr = config.conv_thr
    diagonalization = config.diagonalization
    mixing_ndim = config.mixing_ndim
    base_mixing_beta = config.mixing_beta
    mixing_beta = 0.2 if retry else base_mixing_beta
    david_ndim = getattr(config, "diago_david_ndim", 2)
    david_line = f"\n  diago_david_ndim = {int(david_ndim)}," if (diagonalization.startswith("david") and david_ndim) else ""
    nbnd_line = f"\n  nbnd = {eff_nbnd}," if eff_nbnd is not None else ""
    retry_lines = "\n  mixing_mode = 'local-TF',\n  electron_maxstep = 150," if retry else ""
    forc_conv_line = f"\n  forc_conv_thr = {float(getattr(config, 'relax_forc_conv_thr', 1.0e-3))}," if calculation == "relax" else ""
    nstep_line = f"\n  nstep = {int(getattr(config, 'relax_nstep', 100))}," if calculation == "relax" else ""

    header = f"""&CONTROL
  calculation = '{calculation}',
  restart_mode = 'from_scratch',
  pseudo_dir = '{clean_pseudo_dir}',
  outdir = './tmp',
  prefix = 'calc',
  tprnfor = .true.,
  tstress = .false.,{forc_conv_line}{nstep_line}
  disk_io = 'none'
/
&SYSTEM
  ibrav = 0,
  nat = {len(atoms)},
  ntyp = {len(unique_symbols)},{nbnd_line}
  ecutwfc = {config.ecutwfc},
  ecutrho = {eff_ecutrho},
  nosym = .true.,
  nspin = {eff_nspin},
  tot_magnetization = {tot_mag},
  tot_charge = {config.tot_charge},
  occupations = 'smearing',
  smearing = 'gaussian',
  degauss = 0.005
/
&ELECTRONS
  conv_thr = {conv_thr},
  mixing_beta = {mixing_beta},
  mixing_ndim = {mixing_ndim},
  diagonalization = '{diagonalization}',{david_line}{retry_lines}
/
"""
    if calculation == "relax":
        header += """&IONS
  ion_dynamics = 'bfgs'
/
"""

    header += """ATOMIC_SPECIES
"""
    for s in unique_symbols:
        upf_name = os.path.basename(config.pseudopotentials[s])
        mass = float(atomic_masses[atomic_numbers[s]])
        header += f"  {s:<3} {mass:8.4f}  {upf_name}\n"

    header += "\nCELL_PARAMETERS (angstrom)\n"
    for vec in cell:
        header += f"  {vec[0]:14.8f} {vec[1]:14.8f} {vec[2]:14.8f}\n"

    header += "\nATOMIC_POSITIONS (angstrom)\n"
    for s, p in zip(symbols, pos):
        header += f"  {s:<3} {p[0]:14.8f} {p[1]:14.8f} {p[2]:14.8f}\n"

    if tuple(config.kpoints) == (1, 1, 1):
        header += "\nK_POINTS gamma\n"
    else:
        k = config.kpoints
        header += f"\nK_POINTS (automatic)\n  {k[0]} {k[1]} {k[2]} 0 0 0\n"

    with open(pw_in_path, "w") as f:
        f.write(header)


def parse_qe_output(pw_out_path: str) -> Tuple[Optional[float], Optional[np.ndarray]]:
    """Parses total energy (eV) and forces (eV/A) from `pw.x` output in a single pass.
    
    Returns `(None, None)` if calculation is unconverged or file does not exist.
    """
    if not os.path.isfile(pw_out_path):
        return None, None

    energy_ev: Optional[float] = None
    forces_list: List[List[float]] = []
    in_forces = False
    not_converged = False

    with open(pw_out_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if "convergence NOT achieved" in line:
                not_converged = True
                break
            if "!    total energy" in line:
                parts = line.split("=")
                if len(parts) > 1:
                    try:
                        e_ry = float(parts[1].replace("Ry", "").strip())
                        energy_ev = e_ry * RY_TO_EV
                    except ValueError:
                        pass
            elif "Forces acting on atoms (cartesian axes, Ry/au):" in line:
                in_forces = True
                forces_list = []
            elif in_forces:
                fline = line.strip()
                if "total force" in fline:
                    in_forces = False
                elif fline:
                    m = re.search(r"force =\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)", fline)
                    if m:
                        fx = float(m.group(1)) * RY_BOHR_TO_EV_A
                        fy = float(m.group(2)) * RY_BOHR_TO_EV_A
                        fz = float(m.group(3)) * RY_BOHR_TO_EV_A
                        forces_list.append([fx, fy, fz])

    if not_converged:
        return None, None

    f_arr = np.array(forces_list, dtype=float) if forces_list else None
    return energy_ev, f_arr


def parse_qe_relaxation_output(pw_out_path: str, template_atoms: Atoms) -> Tuple[Optional[Atoms], Optional[float]]:
    """Parses the final relaxed geometry and total energy from a Quantum ESPRESSO `pw.x` relaxation output.
    
    Parameters
    ----------
    pw_out_path : str
        Path to `pw.out`.
    template_atoms : Atoms
        Reference Atoms object containing cell, symbols, and pbc.
        
    Returns
    -------
    Tuple[Optional[Atoms], Optional[float]]
        `(relaxed_atoms, energy_eV)`, or `(None, None)` if relaxation failed or output is missing.
    """
    if not os.path.isfile(pw_out_path):
        return None, None

    final_positions: List[List[float]] = []
    energy_ev: Optional[float] = None
    in_final_coords = False
    in_atomic_positions = False

    try:
        with open(pw_out_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except Exception:
        return None, None

    # Pass 1: Check for "Begin final coordinates" block (standard upon BFGS completion)
    for line in lines:
        if "Final energy" in line or "!    total energy" in line:
            parts = line.split("=")
            if len(parts) > 1:
                try:
                    e_ry = float(parts[1].replace("Ry", "").strip())
                    energy_ev = e_ry * RY_TO_EV
                except ValueError:
                    pass
        if "Begin final coordinates" in line:
            in_final_coords = True
            final_positions = []
            continue
        if in_final_coords:
            if "End final coordinates" in line:
                in_final_coords = False
                break
            if "ATOMIC_POSITIONS" in line:
                in_atomic_positions = True
                continue
            if in_atomic_positions:
                parts = line.strip().split()
                if len(parts) >= 4:
                    try:
                        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                        final_positions.append([x, y, z])
                    except ValueError:
                        pass

    # Pass 2 fallback: If "Begin final coordinates" was omitted, retrieve the final ATOMIC_POSITIONS block
    if len(final_positions) != len(template_atoms):
        blocks: List[List[List[float]]] = []
        cur_block: List[List[float]] = []
        capturing = False
        for line in lines:
            if "ATOMIC_POSITIONS" in line:
                if cur_block:
                    blocks.append(cur_block)
                cur_block = []
                capturing = True
                continue
            if capturing:
                if line.startswith("/") or line.startswith("&") or "End final" in line or "Writing" in line:
                    capturing = False
                    if cur_block:
                        blocks.append(cur_block)
                    cur_block = []
                    continue
                parts = line.strip().split()
                if len(parts) >= 4:
                    try:
                        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                        cur_block.append([x, y, z])
                    except ValueError:
                        pass
        if cur_block:
            blocks.append(cur_block)
        for b in reversed(blocks):
            if len(b) == len(template_atoms):
                final_positions = b
                break

    if len(final_positions) != len(template_atoms):
        return None, None

    at_relaxed = template_atoms.copy()
    at_relaxed.positions = np.array(final_positions, dtype=float)
    if energy_ev is not None:
        at_relaxed.info["energy"] = energy_ev

    return at_relaxed, energy_ev


def parse_qe_band_gap(pw_out_path: str) -> Optional[Dict[str, float]]:
    """Parses Kohn-Sham defect eigenvalue gap (eV), HOMO, LUMO, and Fermi level from `pw.x` output.
    
    Extracts the fundamental optical transition gap across spin channels without additional DFT cost.
    """
    if not os.path.isfile(pw_out_path):
        return None
    try:
        with open(pw_out_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    except Exception:
        return None

    # 1. Direct explicit HOMO-LUMO string if reported by QE
    homo_lumo = re.findall(
        r"highest occupied,\s+lowest unoccupied level\s*\(ev\):\s*([-\d\.]+)\s+([-\d\.]+)",
        text,
        re.IGNORECASE,
    )
    if homo_lumo:
        h, l = float(homo_lumo[-1][0]), float(homo_lumo[-1][1])
        if l > h:
            return {"gap_eV": l - h, "homo_eV": h, "lumo_eV": l, "fermi_eV": (h + l) / 2.0}

    # 2. Spin-polarized bands (nspin = 2)
    spin_up = re.search(r"------ SPIN UP ------------.*?bands \(ev\):\s*(.*?)\n\s*------ SPIN DOWN", text, re.DOTALL)
    spin_dw = re.search(r"------ SPIN DOWN ----------.*?bands \(ev\):\s*(.*?)\n\s*the spin", text, re.DOTALL)
    fermi_spin = re.findall(r"the spin up/dw Fermi energies are\s+([-\d\.]+)\s+([-\d\.]+)\s+ev", text, re.IGNORECASE)

    if fermi_spin and spin_up and spin_dw:
        ef_up, ef_dw = float(fermi_spin[-1][0]), float(fermi_spin[-1][1])
        u_bands = [float(x) for x in spin_up.group(1).split()]
        d_bands = [float(x) for x in spin_dw.group(1).split()]

        gaps = []
        for bands, ef, spin_name in [(u_bands, ef_up, "up"), (d_bands, ef_dw, "down")]:
            occ = [b for b in bands if b <= ef + 1e-3]
            unocc = [b for b in bands if b > ef + 1e-3]
            if occ and unocc:
                gap = unocc[0] - occ[-1]
                if gap > 0.05:
                    gaps.append({"gap_eV": gap, "homo_eV": occ[-1], "lumo_eV": unocc[0], "fermi_eV": ef, "spin": spin_name})

        if gaps:
            gaps.sort(key=lambda x: x["gap_eV"])
            return gaps[0]

    # 3. Non-spin-polarized bands (nspin = 1)
    single_bands = re.search(
        r"(?:End of self-consistent calculation|End of band structure calculation).*?bands \(ev\):\s*(.*?)\n\s*(?:the Fermi|highest)",
        text,
        re.DOTALL,
    )
    single_fermi = re.findall(r"the Fermi energy is\s+([-\d\.]+)\s+ev", text, re.IGNORECASE)
    if single_bands and single_fermi:
        ef = float(single_fermi[-1])
        bands = [float(x) for x in single_bands.group(1).split()]
        occ = [b for b in bands if b <= ef + 1e-3]
        unocc = [b for b in bands if b > ef + 1e-3]
        if occ and unocc and (unocc[0] - occ[-1]) > 0.05:
            return {"gap_eV": unocc[0] - occ[-1], "homo_eV": occ[-1], "lumo_eV": unocc[0], "fermi_eV": ef}

    return None

    forces_ev_a = np.array(forces_list) if forces_list else None
    return energy_ev, forces_ev_a


def resolve_pw_executable(pw_exe: Optional[str] = None) -> str:
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
        for name in ["pw.x", "pw"]:
            found = shutil.which(name)
            if found:
                return found
        return "/usr/bin/pw.x"


def resolve_mpi_runner(mpi_command: Optional[str] = None, n_procs: int = 2) -> Tuple[Optional[str], Optional[str]]:
    """Resolves the MPI launcher and process-count flag, returning `(None, None)` for `n_procs <= 1`."""
    if n_procs <= 1:
        return None, None

    if mpi_command:
        cmd_lower = os.path.basename(mpi_command).lower()
        flag = "-n" if ("mpiexec" in cmd_lower and is_windows()) else ("-np" if "mpirun" in cmd_lower else "-n")
        return mpi_command, flag

    for env_var in ["HYDEPH_MPI_CMD", "MPI_CMD", "MPIEXEC", "MPIRUN"]:
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


def build_qe_command(
    pw_exe: str,
    pw_in: str = "pw.in",
    n_procs: int = 2,
    mpi_command: Optional[str] = None,
    mpi_args: Optional[List[str]] = None,
    pw_args: Optional[List[str]] = None,
) -> List[str]:
    """Builds the command argument list for running Quantum ESPRESSO."""
    cmd_args = []
    if n_procs > 1:
        mpi_runner, n_flag = resolve_mpi_runner(mpi_command, n_procs=n_procs)
        if mpi_runner:
            cmd_args.append(mpi_runner)
            cmd_args.extend([n_flag, str(n_procs)])
            if mpi_args:
                cmd_args.extend(mpi_args)

    if pw_exe.endswith(".py"):
        cmd_args.extend([sys.executable, pw_exe])
    else:
        cmd_args.append(pw_exe)

    cmd_args.extend(["-in", pw_in])
    if pw_args:
        cmd_args.extend(pw_args)
    return cmd_args


def check_pw_in_matches_atoms(pw_in: str, atoms: Atoms, tol: float = 1e-3) -> bool:
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


def find_frame_calc_dir(dft_dir: str, cand_idx: int, frame_id: str, atoms: Atoms) -> Tuple[str, bool]:
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
        self.pw_exe = resolve_pw_executable(pw_exe or config.pw_exe)
        self.mpi_command = mpi_command or config.mpi_command
        self.mpi_args = mpi_args or config.mpi_args
        self.n_workers = int(n_workers if n_workers is not None else config.n_workers_qe)
        self.timeout_s = timeout_s if timeout_s is not None else config.qe_timeout_s
        self.last_failed_entries: List[Dict[str, Any]] = []

    def _resolve_omp_threads(self, n_workers_override: Optional[int] = None) -> int:
        """Determines OpenMP threads per MPI rank so all CPU cores are utilized without oversubscription."""
        if getattr(self.config, "omp_num_threads", None) is not None:
            return max(1, int(self.config.omp_num_threads))
        total_cores = os.cpu_count() or 1
        eff_workers = max(1, int(n_workers_override if n_workers_override is not None else self.n_workers))
        active_ranks = max(1, self.n_procs * eff_workers)
        return max(1, total_cores // active_ranks)

    def label_single_frame(
        self, atoms: Atoms, calc_dir: str, retry: bool = False, timeout_s: Optional[float] = None
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
                        at_labeled.info["energy"] = e_ev
                        at_labeled.arrays["forces"] = f_ev_a
                        gap_info = parse_qe_band_gap(pw_out)
                        if gap_info:
                            for k, v in gap_info.items():
                                at_labeled.info[f"dft_{k}"] = v
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

        write_qe_input(atoms, pw_in, self.config, retry=retry)
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
                sleep_interval = min(0.5, max(0.05, float(eff_timeout) / 5.0 if eff_timeout else 0.5))

                while proc.poll() is None:
                    now = time.time()
                    if eff_timeout is not None and (now - t_start) > float(eff_timeout):
                        try:
                            proc.kill()
                            proc.wait()
                        except Exception:
                            pass
                        raise RuntimeError(f"QE calculation timed out after {float(eff_timeout):.1f}s in {calc_dir}")
                    time.sleep(sleep_interval)
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
                                            logger.info(f"      [pw] SCF iteration {cur_iter}...")
                                    break
                                elif "Forces acting on atoms" in line:
                                    logger.info("      [pw] SCF converged! Calculating atomic forces...")
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
                    "\n  * Tip: For massive speedups on NVIDIA GPUs, running HyDePh inside WSL2 with GPU-accelerated Quantum ESPRESSO is strongly recommended."
                )
            raise RuntimeError(f"QE calculation failed in {calc_dir}.\n{tail_msg}{win_msg}")

        at_labeled = atoms.copy()
        at_labeled.info["energy"] = e_ev
        at_labeled.arrays["forces"] = f_ev_a
        gap_info = parse_qe_band_gap(pw_out)
        if gap_info:
            for k, v in gap_info.items():
                at_labeled.info[f"dft_{k}"] = v
        return at_labeled

    def run_relaxation(
        self,
        atoms: Atoms,
        calc_dir: str,
        tot_magnetization: Optional[int] = None,
        nspin: Optional[int] = None,
        timeout_s: Optional[float] = None,
        force_rerun: bool = False,
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
            at_rel, e_ev = parse_qe_relaxation_output(pw_out, atoms)
            if at_rel is not None:
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

        write_qe_input(
            atoms,
            pw_in,
            self.config,
            calculation="relax",
            tot_magnetization=tot_magnetization,
            nspin=nspin,
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

        logger.info(f"      [pw relax] Starting QE BFGS relaxation in {calc_dir} (tot_magnetization={tot_magnetization})...")
        t_start = time.time()
        last_step = 0
        last_iter = 0

        try:
            with open(pw_out, "wb") as f_out:
                proc = subprocess.Popen(
                    cmd_args,
                    stdout=f_out,
                    stderr=subprocess.STDOUT,
                    cwd=calc_dir,
                    env=env,
                )

                sleep_interval = min(0.5, max(0.05, float(eff_timeout) / 5.0 if eff_timeout else 0.5))
                while proc.poll() is None:
                    now = time.time()
                    if eff_timeout is not None and (now - t_start) > float(eff_timeout):
                        try:
                            proc.kill()
                            proc.wait()
                        except Exception:
                            pass
                        at_partial, e_part = parse_qe_relaxation_output(pw_out, atoms)
                        if at_partial is not None:
                            logger.warning(
                                f"      [pw relax] Relaxation reached timeout ({float(eff_timeout):.1f}s) but successfully "
                                f"recovered latest BFGS geometry ({at_partial.info.get('energy', 0.0):.4f} eV)."
                            )
                            return at_partial
                        raise RuntimeError(f"QE relaxation timed out after {float(eff_timeout):.1f}s in {calc_dir}")
                    time.sleep(sleep_interval)
                    if os.path.isfile(pw_out):
                        try:
                            with open(pw_out, "r", encoding="utf-8", errors="replace") as f:
                                lines = f.readlines()
                            for line in reversed(lines[-30:]):
                                if "number of scf cycles" in line or "iteration #" in line:
                                    m = re.search(r"iteration #\s+(\d+)", line)
                                    if m and int(m.group(1)) != last_iter:
                                        last_iter = int(m.group(1))
                                elif "step" in line and ("search" in line or "new bfgs" in line):
                                    m = re.search(r"step\s+(\d+)", line)
                                    if m and int(m.group(1)) != last_step:
                                        last_step = int(m.group(1))
                                        logger.info(f"      [pw relax] BFGS Step {last_step}...")
                        except Exception:
                            pass
                proc.wait()
        except OSError as exc:
            raise RuntimeError(f"QE relaxation failed in {calc_dir}: {exc}") from exc

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
        return at_rel

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
        frames: List[Atoms],
        max_frames: Optional[int] = None,
        max_new_frames: Optional[int] = None,
        replace_failed_frames: Optional[bool] = None,
        sampler=None,
        at_gs: Optional[Atoms] = None,
        n_workers: Optional[int] = None,
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
                logger.info(f"    [i] Target max_dft_frames={dft_target} already satisfied by {n_done} converged frames.")
                logger.info("        Reusing all converged frames without duplicate DFT execution.")
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
                logger.info(f"    [i] Need {needed} additional frames to reach target {dft_target}. Selected: {calc_str}")
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
            logger.info(f"\n    ---> [{step_num}/{len(to_calculate)}] Calculating {label} ({stype}) in {os.path.basename(cdir)}...")
            try:
                labeled_fr = self.label_single_frame(fr, cdir, retry=False)
                labeled_fr.info["qe_status"] = "converged"
                dt = time.time() - t0
                with state_lock:
                    labeled_dict[idx] = labeled_fr
                    type_completed[stype] = type_completed.get(stype, 0) + 1
                    current_frames = [labeled_dict[k] for k in sorted(labeled_dict.keys())]
                    ase.io.write(out_xyz, current_frames, format="extxyz")
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
                    ase.io.write(out_xyz, current_frames, format="extxyz")
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
                            ase.io.write(out_xyz, current_frames, format="extxyz")
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
            ase.io.write(out_xyz, final_labeled, format="extxyz")

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

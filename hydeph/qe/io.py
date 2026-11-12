"""Quantum ESPRESSO input deck generation, executable discovery, and file validation."""

from __future__ import annotations

import os
import sys
import re
import shutil
import threading
import logging
from typing import Optional, List, Tuple, Dict, Any, Union
import numpy as np
from ase import Atoms
from ase.units import Ry, Bohr
from ase.data import atomic_masses, atomic_numbers

from ..config import (
    ToolkitConfig,
    discover_pseudopotential,
    inspect_upf_metadata,
    HARD_ELEMENT_CUTOFFS,
    resolve_pseudo_path,
)

__all__ = [
    "is_windows",
    "resolve_pw_executable",
    "resolve_mpi_runner",
    "build_qe_command",
    "build_delta_scf_occupations",
    "write_qe_input",
    "check_pw_in_matches_atoms",
    "discover_pseudopotential",
    "inspect_upf_metadata",
    "HARD_ELEMENT_CUTOFFS",
    "resolve_pseudo_path",
    "RY_TO_EV",
    "RY_BOHR_TO_EV_A",
    "BOHR_TO_ANGSTROM",
]

logger = logging.getLogger(__name__)

RY_TO_EV = float(Ry)
RY_BOHR_TO_EV_A = float(Ry / Bohr)
BOHR_TO_ANGSTROM = float(Bohr)


def is_windows() -> bool:
    """Returns True if running on Windows."""
    mod = sys.modules.get("hydeph.qe_runner")
    if mod is not None and hasattr(mod, "is_windows") and mod.is_windows is not is_windows:
        try:
            return bool(mod.is_windows())
        except Exception:
            pass
    return sys.platform.startswith("win")


_PARITY_WARNED_KEYS = set()
_PARITY_LOCK = threading.Lock()
_UPF_RESOLVE_LOCK = threading.Lock()


def build_delta_scf_occupations(
    nelec: int,
    tot_magnetization: int,
    nbnd: int,
    homo_band: int,
    lumo_band: int,
    spin_channel: int = 1,
    lumo_degeneracy: int = 1,
    homo_degeneracy: int = 1,
) -> Tuple[List[float], List[float]]:
    """Builds constrained occupation arrays for Delta-SCF optical excitation.

    Promotes one electron from `homo_band` (or distributed across `homo_degeneracy` bands ending at `homo_band`)
    to `lumo_band` (or distributed across `lumo_degeneracy` bands starting at `lumo_band`) in `spin_channel` (1=up, 2=down),
    conserving total electron number and total magnetization (2S).
    If `lumo_degeneracy > 1` or `homo_degeneracy > 1`, fractional occupations preserve spatial symmetry
    and eliminate SCF orbital-rotation oscillations and charge sloshing.
    """
    n_up = (int(nelec) + int(tot_magnetization)) // 2
    n_dw = (int(nelec) - int(tot_magnetization)) // 2

    if n_up + n_dw != int(nelec):
        raise ValueError(
            f"Electron count {nelec} and magnetization {tot_magnetization} have incompatible parity: "
            f"n_up={n_up}, n_dw={n_dw}"
        )

    deg_lumo = max(1, int(lumo_degeneracy))
    deg_homo = max(1, int(homo_degeneracy))
    min_homo = homo_band - deg_homo + 1
    if min_homo < 1:
        raise ValueError(f"homo_band={homo_band} and homo_degeneracy={deg_homo} require band index < 1")

    max_b = max(n_up, n_dw, homo_band, lumo_band + deg_lumo - 1)
    if nbnd < max_b:
        raise ValueError(
            f"nbnd ({nbnd}) must be at least max(n_up={n_up}, n_dw={n_dw}, "
            f"homo={homo_band}, lumo_end={lumo_band + deg_lumo - 1})"
        )

    occ_up = [1.0] * n_up + [0.0] * (nbnd - n_up)
    occ_dw = [1.0] * n_dw + [0.0] * (nbnd - n_dw)

    frac_hole = 1.0 / deg_homo
    frac_occ = 1.0 / deg_lumo

    if spin_channel == 1:
        for d_idx in range(deg_homo):
            b_idx = (homo_band - 1) - d_idx
            if occ_up[b_idx] < frac_hole - 1e-4:
                logger.warning(f"[!] Warning: HOMO band {b_idx + 1} in spin up has low occupation: {occ_up[b_idx]}")
            occ_up[b_idx] -= frac_hole
        for d_idx in range(deg_lumo):
            b_idx = (lumo_band - 1) + d_idx
            occ_up[b_idx] += frac_occ
    elif spin_channel == 2:
        for d_idx in range(deg_homo):
            b_idx = (homo_band - 1) - d_idx
            if occ_dw[b_idx] < frac_hole - 1e-4:
                logger.warning(f"[!] Warning: HOMO band {b_idx + 1} in spin down has low occupation: {occ_dw[b_idx]}")
            occ_dw[b_idx] -= frac_hole
        for d_idx in range(deg_lumo):
            b_idx = (lumo_band - 1) + d_idx
            occ_dw[b_idx] += frac_occ
    else:
        raise ValueError(f"Invalid spin_channel: {spin_channel}. Must be 1 (up) or 2 (down).")

    return occ_up, occ_dw


def write_qe_input(
    atoms: Atoms,
    pw_in_path: str,
    config: ToolkitConfig,
    calculation: str = "scf",
    retry: bool = False,
    tot_magnetization: Optional[int] = None,
    nspin: Optional[int] = None,
    occupations_mode: Optional[str] = None,
    constrained_occupations: Optional[Union[List[List[float]], Tuple[List[float], ...]]] = None,
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

    # Inspect UPF files for Norm-Conserving status and valence electron counts
    meta_by_sym: Dict[str, Dict[str, Any]] = {}
    for s in unique_symbols:
        upf_rel = config.pseudopotentials.get(s, "")
        if not upf_rel and config.pseudo_dir:
            disc = discover_pseudopotential(s, config.pseudo_dir)
            if disc:
                full_disc = resolve_pseudo_path(disc, config.pseudo_dir)
                with _UPF_RESOLVE_LOCK:
                    config.pseudopotentials[s] = full_disc
                upf_rel = full_disc
                logger.info(f"[+] Auto-discovered pseudopotential for '{s}' during QE input prep: {os.path.basename(disc)}")
        full_upf = resolve_pseudo_path(upf_rel, config.pseudo_dir)
        meta_by_sym[s] = inspect_upf_metadata(full_upf)

    # Resolve target pseudopotential directory on disk and compute exact relative path from pw_in_path
    pw_dir = os.path.dirname(os.path.abspath(pw_in_path))
    target_pseudo_dir = None
    for s in unique_symbols:
        upf_cand = config.pseudopotentials.get(s, "")
        if upf_cand:
            p = resolve_pseudo_path(upf_cand, config.pseudo_dir)
            if os.path.isfile(p):
                target_pseudo_dir = os.path.dirname(os.path.abspath(p))
                break

    if not target_pseudo_dir:
        raw_pseudo = getattr(config, "pseudo_dir", None) or "./pseudo"
        cand_p = os.path.abspath(raw_pseudo)
        if os.path.isdir(cand_p):
            target_pseudo_dir = cand_p
        else:
            ws_parent = os.path.dirname(os.path.abspath(getattr(config, "workspace_dir", "./output")))
            cand_ws_p = os.path.join(ws_parent, raw_pseudo.lstrip("./\\"))
            target_pseudo_dir = cand_ws_p if os.path.isdir(cand_ws_p) else cand_p

    try:
        rel_pseudo = os.path.relpath(target_pseudo_dir, pw_dir).replace("\\", "/")
        clean_pseudo_dir = rel_pseudo
    except (ValueError, Exception):
        clean_pseudo_dir = config.pseudo_dir.replace("\\", "/") if config.pseudo_dir else "./pseudo"

    try:
        local_pseudo_link = os.path.join(pw_dir, "pseudo")
        if not os.path.exists(local_pseudo_link) and target_pseudo_dir and os.path.isdir(target_pseudo_dir):
            rel_to_target = os.path.relpath(target_pseudo_dir, pw_dir)
            os.symlink(rel_to_target, local_pseudo_link)
    except Exception:
        pass

    all_nc = bool(unique_symbols) and all(meta_by_sym[s]["is_norm_conserving"] is True for s in unique_symbols)
    all_z = bool(unique_symbols) and all(meta_by_sym[s]["z_valence"] is not None for s in unique_symbols)

    if getattr(config, "ecutwfc", None) is None or getattr(config, "ecutrho", None) is None:
        config.auto_resolve_cutoffs(elements=unique_symbols)

    eff_ecutwfc = float(config.ecutwfc)
    eff_ecutrho = float(config.ecutrho)
    if all_nc:
        eff_ecutrho = 4.0 * eff_ecutwfc
    else:
        eff_ecutrho = max(eff_ecutrho, 4.0 * eff_ecutwfc)
        for m in meta_by_sym.values():
            if m.get("suggested_cutoff_rho") is not None:
                eff_ecutrho = max(eff_ecutrho, float(m["suggested_cutoff_rho"]))

    for s in unique_symbols:
        meta = meta_by_sym.get(s, {})
        rec_wfc = meta.get("suggested_cutoff_wfc")
        if rec_wfc is not None and eff_ecutwfc < float(rec_wfc):
            logger.warning(
                f"    [!] Warning: Configured ecutwfc={eff_ecutwfc} Ry is lower than "
                f"UPF suggested cutoff {rec_wfc} Ry for element '{s}'."
            )
        elif s in HARD_ELEMENT_CUTOFFS and eff_ecutwfc < 60.0:
            logger.warning(
                f"    [!] Warning: ecutwfc={eff_ecutwfc} Ry may be too low for hard element '{s}' "
                f"(recommended >= 60-80 Ry for converged forces/Hessians)."
            )

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
                if nelec_int % 2 != 0:
                    valid_mag = 1 if tot_mag <= 0 else (tot_mag if tot_mag % 2 != 0 else tot_mag + 1)
                else:
                    valid_mag = 0 if tot_mag <= 0 else (tot_mag if tot_mag % 2 == 0 else tot_mag + 1)

                auto_parity = getattr(config, "auto_spin_parity", False)
                if not auto_parity:
                    raise ValueError(
                        f"Spin parity mismatch: Total valence electrons ({nelec_int}) and tot_magnetization ({tot_mag}) "
                        f"have opposite parity, forcing fractional spin occupations "
                        f"({0.5*(nelec_int+tot_mag):.1f} up, {0.5*(nelec_int-tot_mag):.1f} down). "
                        f"Set tot_magnetization to an {'odd' if nelec_int % 2 != 0 else 'even'} integer "
                        f"(e.g. tot_magnetization={valid_mag}, spin_multiplicity={valid_mag+1}) "
                        f"or set auto_spin_parity=True to allow automatic parity adjustment."
                    )
                else:
                    warn_key = (nelec_int, tot_mag, auto_parity)
                    with _PARITY_LOCK:
                        if warn_key not in _PARITY_WARNED_KEYS:
                            _PARITY_WARNED_KEYS.add(warn_key)
                            logger.warning(
                                f"    [Spin Parity] Total valence electrons ({nelec_int}) and tot_magnetization ({tot_mag}) "
                                f"have opposite parity, which forces fractional spin occupations "
                                f"({0.5*(nelec_int+tot_mag):.1f} up, {0.5*(nelec_int-tot_mag):.1f} down). "
                                f"Auto-adjusting tot_magnetization -> {valid_mag} (spin_multiplicity={valid_mag+1})."
                            )
                    tot_mag = valid_mag
                    try:
                        config.tot_magnetization = valid_mag
                        config.spin_multiplicity = valid_mag + 1
                    except Exception:
                        pass

        if eff_nbnd is None and abs(total_nelec - nelec_int) < 1e-4 and nelec_int > 0:
            n_occ_max = (nelec_int + abs(tot_mag)) // 2 if eff_nspin == 2 else (nelec_int + 1) // 2
            buffer_bands = max(6, min(16, int(np.ceil(0.05 * n_occ_max))))
            eff_nbnd = n_occ_max + buffer_bands

    if constrained_occupations is not None:
        eff_nbnd = len(constrained_occupations[0])

    occ_mode = occupations_mode or getattr(config, "qe_occupations", "fixed")
    if constrained_occupations is not None:
        occ_mode = "from_input"

    if occ_mode == "from_input":
        occ_lines = "  occupations = 'from_input'"
    elif occ_mode == "fixed":
        occ_lines = "  occupations = 'fixed'"
    elif occ_mode == "smearing":
        smearing = getattr(config, "smearing", "gaussian")
        degauss = getattr(config, "degauss", 0.005)
        occ_lines = f"  occupations = 'smearing',\n  smearing = '{smearing}',\n  degauss = {degauss}"
    else:
        raise ValueError(f"Unsupported occupations_mode: '{occ_mode}'. Must be 'fixed', 'smearing', or 'from_input'.")

    if getattr(config, "conv_thr", None) is not None:
        conv_thr = str(config.conv_thr)
    else:
        nat = len(atoms)
        target_conv = max(1.0e-11, nat * 1.0e-10)
        conv_thr = f"{target_conv:.1e}".replace("e", "d")

    diagonalization = config.diagonalization
    mixing_ndim = config.mixing_ndim
    base_mixing_beta = config.mixing_beta
    mixing_beta = 0.2 if retry else base_mixing_beta
    if occupations_mode == "from_input":
        mixing_beta = min(mixing_beta, 0.20 if len(atoms) < 100 else 0.15)
    elif len(atoms) > 100 and mixing_beta > 0.25:
        mixing_beta = 0.20
    david_ndim = getattr(config, "diago_david_ndim", 2)
    david_line = f"\n  diago_david_ndim = {int(david_ndim)}," if (diagonalization.startswith("david") and david_ndim) else ""
    nbnd_line = f"\n  nbnd = {eff_nbnd}," if eff_nbnd is not None else ""
    mix_mode = getattr(config, "mixing_mode", None)
    mix_mode_line = f"\n  mixing_mode = '{mix_mode}'," if mix_mode else ""
    if retry:
        retry_lines = f"{mix_mode_line}\n  electron_maxstep = 150,"
    elif mix_mode_line:
        retry_lines = mix_mode_line
    else:
        retry_lines = ""
    forc_conv_line = f"\n  forc_conv_thr = {float(getattr(config, 'relax_forc_conv_thr', 1.0e-3))}," if calculation in ("relax", "vc-relax") else ""
    nstep_line = f"\n  nstep = {int(getattr(config, 'relax_nstep', 100))}," if calculation in ("relax", "vc-relax") else ""

    hybrid_lines = ""
    if getattr(config, "input_dft", None):
        hybrid_lines += f"\n  input_dft = '{config.input_dft}',"
    if getattr(config, "exx_fraction", None) is not None:
        hybrid_lines += f"\n  exx_fraction = {float(config.exx_fraction)},"
    if getattr(config, "nqx", None) is not None:
        hybrid_lines += f"\n  nqx1 = {int(config.nqx[0])},\n  nqx2 = {int(config.nqx[1])},\n  nqx3 = {int(config.nqx[2])},"
    if getattr(config, "ecutfock", None) is not None:
        hybrid_lines += f"\n  ecutfock = {float(config.ecutfock)},"

    tstress_val = ".true." if calculation == "vc-relax" else ".false."

    header = f"""&CONTROL
  calculation = '{calculation}',
  restart_mode = 'from_scratch',
  pseudo_dir = '{clean_pseudo_dir}',
  outdir = './tmp',
  prefix = 'calc',
  tprnfor = .true.,
  tstress = {tstress_val},{forc_conv_line}{nstep_line}
  disk_io = 'none'
/
&SYSTEM
  ibrav = 0,
  nat = {len(atoms)},
  ntyp = {len(unique_symbols)},{nbnd_line}
  ecutwfc = {eff_ecutwfc},
  ecutrho = {eff_ecutrho},
  nosym = .true.,
  nspin = {eff_nspin},
  tot_magnetization = {tot_mag},
  tot_charge = {config.tot_charge},{hybrid_lines}
{occ_lines}
/
&ELECTRONS
  conv_thr = {conv_thr},
  mixing_beta = {mixing_beta},
  mixing_ndim = {mixing_ndim},
  diagonalization = '{diagonalization}',{david_line}{retry_lines}
/
"""
    if calculation in ("relax", "vc-relax"):
        header += """&IONS
  ion_dynamics = 'bfgs'
/
"""
    if calculation == "vc-relax":
        cell_dyn = getattr(config, "cell_dynamics", "bfgs")
        cell_dofree = getattr(config, "cell_dofree", "all")
        press = float(getattr(config, "press_kbar", getattr(config, "cell_pressure_kbar", 0.0)))
        cell_factor = float(getattr(config, "cell_factor", 2.0))
        header += f"""&CELL
  cell_dynamics = '{cell_dyn}',
  press = {press:.4f},
  cell_dofree = '{cell_dofree}',
  cell_factor = {cell_factor:.2f},
/
"""

    header += """ATOMIC_SPECIES
"""
    for s in unique_symbols:
        if s not in config.pseudopotentials or not config.pseudopotentials[s]:
            disc = discover_pseudopotential(s, config.pseudo_dir) if config.pseudo_dir else None
            if disc:
                full_disc = resolve_pseudo_path(disc, config.pseudo_dir)
                with _UPF_RESOLVE_LOCK:
                    config.pseudopotentials[s] = full_disc
            else:
                raise KeyError(
                    f"Pseudopotential for element '{s}' was not specified in config.pseudopotentials "
                    f"and could not be auto-discovered in pseudo_dir='{config.pseudo_dir}'. "
                    f"Please specify pseudopotentials={{'{s}': '<filename.upf>'}} in ToolkitConfig or place a matching UPF in '{config.pseudo_dir}'."
                )
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

    if occ_mode == "from_input" and constrained_occupations is not None:
        header += "\nOCCUPATIONS\n"
        for spin_occs in constrained_occupations:
            formatted = [f"{float(x):.4f}" if abs(float(x) - round(float(x))) > 1e-4 else f"{float(x):.1f}" for x in spin_occs]
            for chunk_start in range(0, len(formatted), 10):
                chunk = formatted[chunk_start:chunk_start + 10]
                header += "  " + "  ".join(chunk) + "\n"

    with open(pw_in_path, "w") as f:
        f.write(header)


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
        # Common user-compiled or local paths
        user_home = os.path.expanduser("~")
        for local_cand in [
            os.path.join(user_home, "qe-7.5", "bin", "pw.x"),
            os.path.join(user_home, "qe-7.5", "PW", "src", "pw.x"),
            "/usr/local/bin/pw.x",
            "/usr/bin/pw.x",
        ]:
            if os.path.isfile(local_cand):
                return local_cand
        return "pw.x"


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


def check_pw_in_matches_atoms(
    pw_in: str,
    atoms: Atoms,
    config: Optional[ToolkitConfig] = None,
    tol: float = 1e-3,
) -> bool:
    """Verifies that atomic positions, cell, species, and configuration parameters in existing `pw.in` match `atoms` within `tol`.
    
    Fail-closed: returns False if `pw.in` is missing, unreadable, or fails to parse.
    """
    if not pw_in or not os.path.isfile(pw_in):
        return False
    try:
        with open(pw_in, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
            lines = text.splitlines()

        in_pos = False
        coords = []
        species = []
        in_cell = False
        cell_vecs = []
        pos_unit = "angstrom"

        for line in lines:
            line_str = line.strip()
            if "CELL_PARAMETERS" in line_str.upper():
                in_cell = True
                in_pos = False
                continue
            if in_cell:
                if line_str.startswith("/") or line_str.startswith("&") or "ATOMIC" in line_str.upper():
                    in_cell = False
                else:
                    parts = line_str.split()
                    if len(parts) >= 3:
                        try:
                            cell_vecs.append([float(parts[0]), float(parts[1]), float(parts[2])])
                        except ValueError:
                            pass
                        if len(cell_vecs) == 3:
                            in_cell = False
                            continue

            if "ATOMIC_POSITIONS" in line_str.upper():
                in_pos = True
                in_cell = False
                p_parts = line_str.split()
                if len(p_parts) > 1:
                    pos_unit = p_parts[1].lower().strip("()")
                else:
                    pos_unit = "alat"
                continue
            if in_pos:
                if line_str.startswith("/") or line_str.startswith("&") or "K_POINTS" in line_str.upper():
                    break
                parts = line_str.split()
                if len(parts) >= 4:
                    species.append(parts[0])
                    coords.append([float(parts[1]), float(parts[2]), float(parts[3])])

        if len(coords) != len(atoms):
            return False

        # Match species symbols
        if species != list(atoms.get_chemical_symbols()):
            return False

        # Convert coords to Cartesian Angstroms based on pos_unit
        coords_arr = np.array(coords, dtype=float)
        cell_mat = atoms.cell.array if (atoms.cell is not None and atoms.pbc.any()) else (np.array(cell_vecs) if len(cell_vecs) == 3 else np.eye(3))
        if pos_unit == "crystal":
            coords_cart = np.dot(coords_arr, cell_mat)
        elif pos_unit == "bohr":
            coords_cart = coords_arr * Bohr
        else:
            coords_cart = coords_arr

        # Match positions
        if atoms.cell is not None and atoms.pbc.any():
            from ase.geometry import find_mic
            _, d_mic = find_mic(coords_cart - atoms.positions, cell_mat, pbc=atoms.pbc)
            if np.max(d_mic) >= tol:
                return False
        else:
            diff = np.abs(coords_cart - atoms.positions)
            if np.max(diff) >= tol:
                return False

        # Match cell vectors if parsed and atoms has a cell
        if len(cell_vecs) == 3 and atoms.cell is not None and atoms.pbc.any():
            cell_diff = np.abs(np.array(cell_vecs) - atoms.cell.array)
            if np.max(cell_diff) >= tol:
                return False

        # If config is provided, verify matching cutoffs, charge, spin, kpoints, functional, pseudos, occupations, nbnd, conv_thr
        if config is not None:
            # 1. ecutwfc
            if getattr(config, "ecutwfc", None) is not None:
                m_wfc = re.search(r"ecutwfc\s*=\s*([-\d.EeDd+]+)", text, re.IGNORECASE)
                if not m_wfc or abs(float(m_wfc.group(1).replace("d", "e").replace("D", "e")) - float(config.ecutwfc)) > 1e-3:
                    return False

            # 2. ecutrho
            if getattr(config, "ecutrho", None) is not None:
                m_rho = re.search(r"ecutrho\s*=\s*([-\d.EeDd+]+)", text, re.IGNORECASE)
                if not m_rho or abs(float(m_rho.group(1).replace("d", "e").replace("D", "e")) - float(config.ecutrho)) > 1e-3:
                    return False

            # 3. tot_charge
            if getattr(config, "tot_charge", None) is not None:
                m_chg = re.search(r"tot_charge\s*=\s*([-\d.EeDd+]+)", text, re.IGNORECASE)
                if not m_chg or abs(float(m_chg.group(1).replace("d", "e").replace("D", "e")) - float(config.tot_charge)) > 1e-3:
                    return False

            # 4. tot_magnetization
            if getattr(config, "tot_magnetization", None) is not None:
                m_mag = re.search(r"tot_magnetization\s*=\s*([-\d]+)", text, re.IGNORECASE)
                if not m_mag or int(m_mag.group(1)) != int(config.tot_magnetization):
                    return False

            # 5. input_dft (exchange-correlation functional)
            input_dft_cfg = getattr(config, "input_dft", None)
            m_dft = re.search(r"input_dft\s*=\s*['\"]([^'\"]+)['\"]", text, re.IGNORECASE)
            if input_dft_cfg:
                if not m_dft or m_dft.group(1).strip().lower() != str(input_dft_cfg).strip().lower():
                    return False
            elif m_dft:
                return False

            # 6. conv_thr
            if getattr(config, "conv_thr", None) is not None:
                m_thr = re.search(r"conv_thr\s*=\s*([-\d.EeDd+]+)", text, re.IGNORECASE)
                if m_thr:
                    val_thr = float(m_thr.group(1).replace("d", "e").replace("D", "e"))
                    cfg_thr = float(config.conv_thr)
                    if abs(val_thr - cfg_thr) / max(1e-15, cfg_thr) > 0.05:
                        return False

            # 7. nbnd
            nbnd_cfg = getattr(config, "nbnd", None)
            if nbnd_cfg is not None:
                m_nbnd = re.search(r"nbnd\s*=\s*([-\d]+)", text, re.IGNORECASE)
                if not m_nbnd or int(m_nbnd.group(1)) != int(nbnd_cfg):
                    return False

            # 8. Occupations mode
            m_occ = re.search(r"occupations\s*=\s*['\"]([^'\"]+)['\"]", text, re.IGNORECASE)
            occ_mode_cfg = getattr(config, "occupations_mode", None)
            if occ_mode_cfg:
                exp_occ = str(occ_mode_cfg).strip().lower()
                if not m_occ or m_occ.group(1).strip().lower() != exp_occ:
                    return False

            # 9. K_POINTS
            kpts_cfg = getattr(config, "kpoints", (1, 1, 1))
            is_gamma_cfg = tuple(kpts_cfg) == (1, 1, 1) or str(kpts_cfg).lower() == "gamma"
            is_gamma_in = bool(re.search(r"K_POINTS\s+gamma", text, re.IGNORECASE))
            if is_gamma_cfg != is_gamma_in:
                return False
            if not is_gamma_cfg:
                m_kpts = re.search(r"K_POINTS\s+automatic\s*\n\s*(\d+)\s+(\d+)\s+(\d+)", text, re.IGNORECASE)
                if not m_kpts or (int(m_kpts.group(1)), int(m_kpts.group(2)), int(m_kpts.group(3))) != tuple(kpts_cfg):
                    return False

            # 10. Pseudopotential filenames in ATOMIC_SPECIES
            pseudos = getattr(config, "pseudopotentials", {}) or {}
            if pseudos:
                for sym, upf_spec in pseudos.items():
                    exp_upf_base = os.path.basename(str(upf_spec)).lower()
                    m_species = re.search(rf"^\s*{re.escape(sym)}\s+[\d.]+\s+([^\s]+)", text, re.IGNORECASE | re.MULTILINE)
                    if m_species:
                        actual_upf_base = os.path.basename(m_species.group(1).strip()).lower()
                        if actual_upf_base != exp_upf_base:
                            return False
                    else:
                        return False

        return True
    except Exception:
        return False

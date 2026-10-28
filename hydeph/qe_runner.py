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
from typing import Optional, List, Tuple, Dict, Any, Union
import numpy as np
import ase.io
from ase import Atoms
from ase.units import Ry, Bohr
from ase.data import atomic_masses, atomic_numbers

from .config import ToolkitConfig, discover_pseudopotential

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
    """Reads lightweight header metadata (`is_norm_conserving`, `z_valence`, suggested cutoffs) from a UPF file."""
    info: Dict[str, Any] = {
        "is_norm_conserving": None,
        "z_valence": None,
        "suggested_cutoff_wfc": None,
        "suggested_cutoff_rho": None,
    }
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

        # Parse suggested cutoffs (UPF v2 attributes or comments)
        m_wfc = re.search(r'(?:suggested_cutoff_wfc|wfc_cutoff)\s*=\s*"\s*([-\d.Ee+]+)\s*"', text, re.IGNORECASE)
        if not m_wfc:
            m_wfc = re.search(r'(?:suggested|recommended)?\s*cutoff(?:\s*for)?\s*wavefunctions?\s*[:=]\s*([-\d.Ee+]+)', text, re.IGNORECASE)
        if m_wfc:
            try:
                info["suggested_cutoff_wfc"] = float(m_wfc.group(1))
            except ValueError:
                pass

        m_rho = re.search(r'(?:suggested_cutoff_rho|rho_cutoff)\s*=\s*"\s*([-\d.Ee+]+)\s*"', text, re.IGNORECASE)
        if not m_rho:
            m_rho = re.search(r'(?:suggested|recommended)?\s*cutoff(?:\s*for)?\s*(?:charge\s*density|density|rho)\s*[:=]\s*([-\d.Ee+]+)', text, re.IGNORECASE)
        if m_rho:
            try:
                info["suggested_cutoff_rho"] = float(m_rho.group(1))
            except ValueError:
                pass

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
_PARITY_LOCK = threading.Lock()
_UPF_RESOLVE_LOCK = threading.Lock()


def run_monitored_qe_subprocess(
    cmd_args: List[str],
    calc_dir: str,
    pw_out: str,
    env: Dict[str, str],
    timeout: Optional[float] = None,
    mode: str = "scf",
    allow_fallbacks: bool = False,
    atoms_template: Optional[Atoms] = None,
) -> Optional[Atoms]:
    """Executes a Quantum ESPRESSO subprocess, continuously monitoring progress and enforcing timeouts.

    Parameters
    ----------
    cmd_args : List[str]
        Command line arguments for the subprocess.
    calc_dir : str
        Working directory for the calculation.
    pw_out : str
        Path to output file (pw.out).
    env : Dict[str, str]
        Environment variables for subprocess execution.
    timeout : float, optional
        Maximum execution walltime in seconds.
    mode : str
        "scf" (logs SCF iterations) or "relax" (logs BFGS steps).
    allow_fallbacks : bool
        If True and mode == "relax", attempts to recover latest partial BFGS geometry upon timeout.
    atoms_template : Atoms, optional
        Reference Atoms object for parsing partial relaxation coordinates.

    Returns
    -------
    Optional[Atoms]
        Partial relaxed geometry if timed out under allow_fallbacks=True, otherwise None.

    Raises
    ------
    RuntimeError
        If the process execution fails or times out.
    """
    t_start = time.time()
    t_last_log = t_start
    last_iter = 0
    last_step = 0
    sleep_interval = min(0.5, max(0.05, float(timeout) / 5.0 if timeout else 0.5))
    calc_type = "relaxation" if mode == "relax" else "calculation"

    try:
        with open(pw_out, "wb") as f_out:
            proc = subprocess.Popen(
                cmd_args,
                stdout=f_out,
                stderr=subprocess.STDOUT,
                cwd=calc_dir,
                env=env,
            )

            while proc.poll() is None:
                now = time.time()
                if timeout is not None and (now - t_start) > float(timeout):
                    try:
                        proc.kill()
                        proc.wait()
                    except Exception:
                        pass
                    if mode == "relax" and allow_fallbacks and atoms_template is not None:
                        at_part, e_part = parse_qe_relaxation_output(pw_out, atoms_template, require_converged=False)
                        if at_part is not None:
                            logger.warning(
                                f"      [pw relax] Relaxation reached timeout ({float(timeout):.1f}s) but recovered latest BFGS geometry "
                                f"({at_part.info.get('energy', 0.0):.4f} eV) under allow_fallbacks=True."
                            )
                            return at_part
                    raise RuntimeError(f"QE {calc_type} timed out after {float(timeout):.1f}s in {calc_dir}")

                time.sleep(sleep_interval)
                now = time.time()
                if (now - t_last_log) >= 2.0 and os.path.isfile(pw_out):
                    t_last_log = now
                    try:
                        with open(pw_out, "r", encoding="utf-8", errors="replace") as f:
                            lines = f.readlines()
                        for line in reversed(lines[-30:]):
                            if mode == "scf":
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
                            elif mode == "relax":
                                if "number of scf cycles" in line or "iteration #" in line:
                                    m = re.search(r"iteration #\s+(\d+)", line)
                                    if m and int(m.group(1)) != last_iter:
                                        last_iter = int(m.group(1))
                                elif "step" in line and ("search" in line or "new bfgs" in line):
                                    m = re.search(r"step\s+(\d+)", line)
                                    if m and int(m.group(1)) != last_step:
                                        last_step = int(m.group(1))
                                        logger.info(f"      [pw relax] BFGS Step {last_step}...")
                                    break
                    except Exception:
                        pass

            proc.wait()
    except OSError as exc:
        raise RuntimeError(f"QE {calc_type} failed in {calc_dir}: {exc}") from exc

    return None


def build_delta_scf_occupations(
    nelec: int,
    tot_magnetization: int,
    nbnd: int,
    homo_band: int,
    lumo_band: int,
    spin_channel: int = 1,
) -> Tuple[List[float], List[float]]:
    """Builds constrained occupation arrays for Delta-SCF optical excitation.

    Promotes one electron from `homo_band` to `lumo_band` in `spin_channel` (1=up, 2=down),
    conserving total electron number and total magnetization (2S).
    """
    n_up = (int(nelec) + int(tot_magnetization)) // 2
    n_dw = (int(nelec) - int(tot_magnetization)) // 2

    if n_up + n_dw != int(nelec):
        raise ValueError(
            f"Electron count {nelec} and magnetization {tot_magnetization} have incompatible parity: "
            f"n_up={n_up}, n_dw={n_dw}"
        )

    if nbnd < max(n_up, n_dw, homo_band, lumo_band):
        raise ValueError(
            f"nbnd ({nbnd}) must be at least max(n_up={n_up}, n_dw={n_dw}, "
            f"homo={homo_band}, lumo={lumo_band})"
        )

    occ_up = [1.0] * n_up + [0.0] * (nbnd - n_up)
    occ_dw = [1.0] * n_dw + [0.0] * (nbnd - n_dw)

    if spin_channel == 1:
        if occ_up[homo_band - 1] < 0.5:
            logger.warning(f"[!] Warning: HOMO band {homo_band} in spin up has low occupation: {occ_up[homo_band - 1]}")
        occ_up[homo_band - 1] = 0.0
        occ_up[lumo_band - 1] = 1.0
    elif spin_channel == 2:
        if occ_dw[homo_band - 1] < 0.5:
            logger.warning(f"[!] Warning: HOMO band {homo_band} in spin down has low occupation: {occ_dw[homo_band - 1]}")
        occ_dw[homo_band - 1] = 0.0
        occ_dw[lumo_band - 1] = 1.0
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

    # Fortran namelist parser requires forward slashes
    clean_pseudo_dir = config.pseudo_dir.replace("\\", "/") if config.pseudo_dir else "./pseudo"

    # Inspect UPF files for Norm-Conserving status and valence electron counts
    meta_by_sym: Dict[str, Dict[str, Any]] = {}
    for s in unique_symbols:
        upf_rel = config.pseudopotentials.get(s, "")
        if not upf_rel and config.pseudo_dir:
            disc = discover_pseudopotential(s, config.pseudo_dir)
            if disc:
                full_disc = disc if os.path.isabs(disc) else os.path.join(config.pseudo_dir, disc)
                with _UPF_RESOLVE_LOCK:
                    config.pseudopotentials[s] = full_disc
                upf_rel = full_disc
                logger.info(f"[+] Auto-discovered pseudopotential for '{s}' during QE input prep: {os.path.basename(disc)}")
        full_upf = upf_rel if os.path.isabs(upf_rel) else os.path.join(config.pseudo_dir or ".", upf_rel)
        meta_by_sym[s] = inspect_upf_metadata(full_upf)

    all_nc = bool(unique_symbols) and all(meta_by_sym[s]["is_norm_conserving"] is True for s in unique_symbols)
    all_z = bool(unique_symbols) and all(meta_by_sym[s]["z_valence"] is not None for s in unique_symbols)

    eff_ecutrho = float(config.ecutrho)
    if all_nc:
        eff_ecutrho = 4.0 * float(config.ecutwfc)
    else:
        eff_ecutrho = max(eff_ecutrho, 4.0 * float(config.ecutwfc))
        for m in meta_by_sym.values():
            if m.get("suggested_cutoff_rho") is not None:
                eff_ecutrho = max(eff_ecutrho, float(m["suggested_cutoff_rho"]))

    HARD_ELEMENTS = {"C", "N", "O", "B", "F"}
    for s in unique_symbols:
        meta = meta_by_sym.get(s, {})
        rec_wfc = meta.get("suggested_cutoff_wfc")
        if rec_wfc is not None and float(config.ecutwfc) < float(rec_wfc):
            logger.warning(
                f"    [!] Warning: Configured ecutwfc={config.ecutwfc} Ry is lower than "
                f"UPF suggested cutoff {rec_wfc} Ry for element '{s}'."
            )
        elif s in HARD_ELEMENTS and float(config.ecutwfc) < 60.0:
            logger.warning(
                f"    [!] Warning: ecutwfc={config.ecutwfc} Ry may be too low for hard element '{s}' "
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
    david_ndim = getattr(config, "diago_david_ndim", 2)
    david_line = f"\n  diago_david_ndim = {int(david_ndim)}," if (diagonalization.startswith("david") and david_ndim) else ""
    nbnd_line = f"\n  nbnd = {eff_nbnd}," if eff_nbnd is not None else ""
    retry_lines = "\n  mixing_mode = 'local-TF',\n  electron_maxstep = 150," if retry else ""
    forc_conv_line = f"\n  forc_conv_thr = {float(getattr(config, 'relax_forc_conv_thr', 1.0e-3))}," if calculation == "relax" else ""
    nstep_line = f"\n  nstep = {int(getattr(config, 'relax_nstep', 100))}," if calculation == "relax" else ""

    hybrid_lines = ""
    if getattr(config, "input_dft", None):
        hybrid_lines += f"\n  input_dft = '{config.input_dft}',"
    if getattr(config, "exx_fraction", None) is not None:
        hybrid_lines += f"\n  exx_fraction = {float(config.exx_fraction)},"
    if getattr(config, "nqx", None) is not None:
        hybrid_lines += f"\n  nqx1 = {int(config.nqx[0])},\n  nqx2 = {int(config.nqx[1])},\n  nqx3 = {int(config.nqx[2])},"
    if getattr(config, "ecutfock", None) is not None:
        hybrid_lines += f"\n  ecutfock = {float(config.ecutfock)},"

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
    if calculation == "relax":
        header += """&IONS
  ion_dynamics = 'bfgs'
/
"""

    header += """ATOMIC_SPECIES
"""
    for s in unique_symbols:
        if s not in config.pseudopotentials or not config.pseudopotentials[s]:
            disc = discover_pseudopotential(s, config.pseudo_dir) if config.pseudo_dir else None
            if disc:
                full_disc = disc if os.path.isabs(disc) else os.path.join(config.pseudo_dir, disc)
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


def parse_qe_relaxation_output(
    pw_out_path: str,
    template_atoms: Atoms,
    require_converged: bool = True,
) -> Tuple[Optional[Atoms], Optional[float]]:
    """Parses the final relaxed geometry and total energy from a Quantum ESPRESSO `pw.x` relaxation output.
    
    Parameters
    ----------
    pw_out_path : str
        Path to `pw.out`.
    template_atoms : Atoms
        Reference Atoms object containing cell, symbols, and pbc.
    require_converged : bool
        If True (default), strictly verifies BFGS convergence marker ('Begin final coordinates' or
        'bfgs converged in') and rejects timed-out, nstep-exhausted, or incomplete relaxations.
        
    Returns
    -------
    Tuple[Optional[Atoms], Optional[float]]
        `(relaxed_atoms, energy_eV)`, or `(None, None)` if relaxation failed or output is missing/unconverged.
    """
    if not os.path.isfile(pw_out_path):
        return None, None

    try:
        with open(pw_out_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except Exception:
        return None, None

    text = "".join(lines)
    text_lower = text.lower()

    if require_converged:
        has_final_coords = "begin final coordinates" in text_lower
        has_bfgs_converged = "bfgs converged in" in text_lower
        if not (has_final_coords or has_bfgs_converged):
            return None, None
        if "the maximum number of steps has been reached." in text_lower:
            return None, None
        if "convergence not achieved" in text_lower and not has_final_coords:
            return None, None

    final_positions: List[List[float]] = []
    energy_ev: Optional[float] = None
    in_final_coords = False

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
                continue
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


def parse_qe_band_gap(pw_out_path: str) -> Optional[Dict[str, Any]]:
    """Parses Kohn-Sham defect eigenvalue gap (eV), HOMO, LUMO, and Fermi level from `pw.x` output.
    
    Extracts the fundamental optical transition gap across spin channels without additional DFT cost.
    Supports spin-polarized and non-spin-polarized outputs, with smearing or fixed occupations.
    """
    if not os.path.isfile(pw_out_path):
        return None
    try:
        with open(pw_out_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    except Exception:
        return None

    # Read number of electrons and magnetization if reported
    nelec_m = re.findall(
        r"number of electrons\s*=\s*([\d\.]+)\s*(?:\(up:\s*([\d\.]+),\s*down:\s*([\d\.]+)\))?",
        text,
        re.IGNORECASE,
    )
    nelec, n_up_elec, n_dw_elec = None, None, None
    if nelec_m:
        nelec = float(nelec_m[-1][0])
        if nelec_m[-1][1] and nelec_m[-1][2]:
            n_up_elec = int(round(float(nelec_m[-1][1])))
            n_dw_elec = int(round(float(nelec_m[-1][2])))

    mag_m = re.findall(r"total magnetization\s*=\s*([-\d\.]+)", text, re.IGNORECASE)
    tot_mag = float(mag_m[-1]) if mag_m else 0.0

    if nelec is not None and (n_up_elec is None or n_dw_elec is None):
        n_up_elec = int(round((nelec + tot_mag) / 2.0))
        n_dw_elec = int(round((nelec - tot_mag) / 2.0))

    # 1. Spin-polarized bands (nspin = 2)
    sp_up_matches = re.findall(
        r"------ SPIN UP ------------.*?bands \(ev\):\s*(.*?)\n\s*------ SPIN DOWN",
        text,
        re.DOTALL,
    )
    sp_dw_matches = re.findall(
        r"------ SPIN DOWN ----------.*?bands \(ev\):\s*(.*?)\n\s*(?:highest occupied|the spin|!|total energy|Forces|End of|convergence)",
        text,
        re.DOTALL,
    )
    fermi_spin = re.findall(
        r"the spin up/dw Fermi energies are\s+([-\d\.]+)\s+([-\d\.]+)\s+ev",
        text,
        re.IGNORECASE,
    )

    if sp_up_matches and sp_dw_matches:
        u_bands = [float(x) for x in sp_up_matches[-1].split()]
        d_bands = [float(x) for x in sp_dw_matches[-1].split()]

        ef_up, ef_dw = None, None
        if fermi_spin:
            ef_up, ef_dw = float(fermi_spin[-1][0]), float(fermi_spin[-1][1])

        # Spin UP (Channel 1)
        if ef_up is not None:
            occ_u = [b for b in u_bands if b <= ef_up + 1e-3]
            unocc_u = [b for b in u_bands if b > ef_up + 1e-3]
            gap_u = (unocc_u[0] - occ_u[-1]) if (occ_u and unocc_u) else None
            homo_b_u = len(occ_u)
            lumo_b_u = len(occ_u) + 1
            homo_u = occ_u[-1] if occ_u else None
            lumo_u = unocc_u[0] if unocc_u else None
        elif n_up_elec is not None and 0 < n_up_elec < len(u_bands):
            homo_b_u = n_up_elec
            lumo_b_u = n_up_elec + 1
            homo_u = u_bands[homo_b_u - 1]
            lumo_u = u_bands[lumo_b_u - 1]
            gap_u = lumo_u - homo_u
            ef_up = (homo_u + lumo_u) / 2.0
        else:
            gap_u, homo_b_u, lumo_b_u, homo_u, lumo_u = None, None, None, None, None

        # Spin DOWN (Channel 2)
        if ef_dw is not None:
            occ_d = [b for b in d_bands if b <= ef_dw + 1e-3]
            unocc_d = [b for b in d_bands if b > ef_dw + 1e-3]
            gap_d = (unocc_d[0] - occ_d[-1]) if (occ_d and unocc_d) else None
            homo_b_d = len(occ_d)
            lumo_b_d = len(occ_d) + 1
            homo_d = occ_d[-1] if occ_d else None
            lumo_d = unocc_d[0] if unocc_d else None
        elif n_dw_elec is not None and 0 < n_dw_elec < len(d_bands):
            homo_b_d = n_dw_elec
            lumo_b_d = n_dw_elec + 1
            homo_d = d_bands[homo_b_d - 1]
            lumo_d = d_bands[lumo_b_d - 1]
            gap_d = lumo_d - homo_d
            ef_dw = (homo_d + lumo_d) / 2.0
        else:
            gap_d, homo_b_d, lumo_b_d, homo_d, lumo_d = None, None, None, None, None

        # Prioritize the minority or in-gap defect channel (smaller positive gap)
        chosen = None
        if gap_d is not None and gap_u is not None:
            if 0.05 < gap_d < gap_u:
                chosen = ("down", 2, gap_d, homo_d, lumo_d, homo_b_d, lumo_b_d, ef_dw)
            else:
                chosen = ("up", 1, gap_u, homo_u, lumo_u, homo_b_u, lumo_b_u, ef_up)
        elif gap_d is not None:
            chosen = ("down", 2, gap_d, homo_d, lumo_d, homo_b_d, lumo_b_d, ef_dw)
        elif gap_u is not None:
            chosen = ("up", 1, gap_u, homo_u, lumo_u, homo_b_u, lumo_b_u, ef_up)

        if chosen:
            spin_name, spin_chan, gap_v, homo_v, lumo_v, h_b, l_b, ef_v = chosen
            return {
                "gap_eV": gap_v,
                "homo_eV": homo_v,
                "lumo_eV": lumo_v,
                "fermi_eV": ef_v,
                "spin": spin_name,
                "spin_channel": spin_chan,
                "homo_band": h_b,
                "lumo_band": l_b,
                "nbnd": max(len(u_bands), len(d_bands)),
                "gap_up_eV": gap_u,
                "gap_down_eV": gap_d,
                "nelec": nelec,
                "n_up": n_up_elec,
                "n_dw": n_dw_elec,
                "tot_mag": tot_mag,
            }

    # 2. Non-spin-polarized bands (nspin = 1)
    single_bands = re.findall(
        r"(?:End of self-consistent calculation|End of band structure calculation).*?bands \(ev\):\s*(.*?)\n\s*(?:the Fermi|highest|!|Forces)",
        text,
        re.DOTALL,
    )
    single_fermi = re.findall(r"the Fermi energy is\s+([-\d\.]+)\s+ev", text, re.IGNORECASE)
    if single_bands:
        bands = [float(x) for x in single_bands[-1].split()]
        if single_fermi:
            ef = float(single_fermi[-1])
            occ = [b for b in bands if b <= ef + 1e-3]
            unocc = [b for b in bands if b > ef + 1e-3]
            if occ and unocc and (unocc[0] - occ[-1]) > 0.05:
                return {
                    "gap_eV": unocc[0] - occ[-1],
                    "homo_eV": occ[-1],
                    "lumo_eV": unocc[0],
                    "fermi_eV": ef,
                    "spin": "down",
                    "spin_channel": 2,
                    "homo_band": len(occ),
                    "lumo_band": len(occ) + 1,
                    "nbnd": len(bands),
                    "nelec": nelec,
                }
        elif nelec is not None:
            n_occ = int(round(nelec / 2.0))
            if 0 < n_occ < len(bands):
                homo_val = bands[n_occ - 1]
                lumo_val = bands[n_occ]
                return {
                    "gap_eV": lumo_val - homo_val,
                    "homo_eV": homo_val,
                    "lumo_eV": lumo_val,
                    "fermi_eV": (homo_val + lumo_val) / 2.0,
                    "spin": "down",
                    "spin_channel": 2,
                    "homo_band": n_occ,
                    "lumo_band": n_occ + 1,
                    "nbnd": len(bands),
                    "nelec": nelec,
                }

    # 3. Direct explicit HOMO-LUMO string fallback if reported by QE
    homo_lumo = re.findall(
        r"highest occupied,\s+lowest unoccupied level\s*\(ev\):\s*([-\d\.]+)\s+([-\d\.]+)",
        text,
        re.IGNORECASE,
    )
    if homo_lumo:
        homo_val, lumo_val = float(homo_lumo[-1][0]), float(homo_lumo[-1][1])
        if lumo_val > homo_val:
            n_occ = int(round(nelec / 2.0)) if nelec else 1
            return {
                "gap_eV": lumo_val - homo_val,
                "homo_eV": homo_val,
                "lumo_eV": lumo_val,
                "fermi_eV": (homo_val + lumo_val) / 2.0,
                "spin": "down",
                "spin_channel": 2,
                "homo_band": n_occ,
                "lumo_band": n_occ + 1,
                "nbnd": n_occ + 8,
                "nelec": nelec,
            }

    return None


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

        # Match positions
        diff = np.abs(np.array(coords) - atoms.positions)
        if np.max(diff) >= tol:
            return False

        # Match cell vectors if parsed and atoms has a cell
        if len(cell_vecs) == 3 and atoms.cell is not None and atoms.pbc.any():
            cell_diff = np.abs(np.array(cell_vecs) - atoms.cell.array)
            if np.max(cell_diff) >= tol:
                return False

        # If config is provided, verify matching cutoffs, charge, and spin
        if config is not None:
            m_wfc = re.search(r"ecutwfc\s*=\s*([-\d.Ee+]+)", text)
            if m_wfc and abs(float(m_wfc.group(1)) - float(config.ecutwfc)) > 1e-3:
                return False

            m_chg = re.search(r"tot_charge\s*=\s*([-\d.Ee+]+)", text)
            if m_chg and abs(float(m_chg.group(1)) - float(config.tot_charge)) > 1e-3:
                return False

            m_mag = re.search(r"tot_magnetization\s*=\s*([-\d]+)", text)
            if m_mag and config.tot_magnetization is not None:
                if int(m_mag.group(1)) != int(config.tot_magnetization):
                    return False

        return True
    except Exception:
        return False


def find_frame_calc_dir(
    dft_dir: str,
    cand_idx: int,
    frame_id: str,
    atoms: Atoms,
    config: Optional[ToolkitConfig] = None,
) -> Tuple[str, bool]:
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
                        and check_pw_in_matches_atoms(pw_in, atoms, config=config)
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
        self,
        atoms: Atoms,
        calc_dir: str,
        retry: bool = False,
        timeout_s: Optional[float] = None,
        occupations_mode: Optional[str] = None,
        constrained_occupations: Optional[Union[List[List[float]], Tuple[List[float], ...]]] = None,
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
        gap_info = parse_qe_band_gap(pw_out)
        if gap_info:
            for k, v in gap_info.items():
                at_labeled.info[f"dft_{k}"] = v

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
            if at_rel is not None and check_pw_in_matches_atoms(pw_in, atoms, config=self.config):
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

        logger.info(f"      [pw relax] Starting QE BFGS relaxation in {calc_dir} (tot_magnetization={tot_magnetization})...")
        t_start = time.time()

        at_fallback = run_monitored_qe_subprocess(
            cmd_args=cmd_args,
            calc_dir=calc_dir,
            pw_out=pw_out,
            env=env,
            timeout=eff_timeout,
            mode="relax",
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
            gs_out = os.path.join(calc_base, "qe_gs_relax", "pw.out")
            gap_info = parse_qe_band_gap(gs_out) if os.path.isfile(gs_out) else None

            unique_syms = sorted(list(set(at_gs.get_chemical_symbols())))
            z_vals = {}
            for s in unique_syms:
                upf = self.config.pseudopotentials.get(s, "")
                if upf:
                    p = upf if os.path.isabs(upf) else os.path.join(self.config.pseudo_dir or ".", upf)
                    z_vals[s] = inspect_upf_metadata(p).get("z_valence")

            if all(v is not None for v in z_vals.values()):
                nelec_total = int(round(sum(z_vals[s] for s in at_gs.get_chemical_symbols()) - float(self.config.tot_charge)))
            else:
                nelec_total = len(at_gs) * 4

            if gap_info is not None and gap_info.get("nelec") is not None:
                nelec_total = int(round(gap_info["nelec"]))

            tot_mag = int(self.config.spin_multiplicity - 1) if self.config.spin_polarized else 0
            if (nelec_total % 2) != (tot_mag % 2):
                if nelec_total % 2 != 0:
                    valid_mag = 1 if tot_mag <= 0 else (tot_mag if tot_mag % 2 != 0 else tot_mag + 1)
                else:
                    valid_mag = 0 if tot_mag <= 0 else (tot_mag if tot_mag % 2 == 0 else tot_mag + 1)
                if not getattr(self.config, "auto_spin_parity", False):
                    raise ValueError(
                        f"Spin parity mismatch in Delta-SCF: Total valence electrons ({nelec_total}) and "
                        f"tot_magnetization ({tot_mag}) have opposite parity ({0.5*(nelec_total+tot_mag):.1f} up, {0.5*(nelec_total-tot_mag):.1f} down). "
                        f"Set tot_magnetization={valid_mag} (spin_multiplicity={valid_mag+1}) or set auto_spin_parity=True."
                    )
                else:
                    tot_mag = valid_mag

            n_occ_max = (nelec_total + abs(tot_mag)) // 2
            spin_chan = getattr(self.config, "delta_scf_spin_channel", None)
            homo_b = getattr(self.config, "delta_scf_homo_band", None)
            lumo_b = getattr(self.config, "delta_scf_lumo_band", None)

            if gap_info is not None:
                spin_chan = spin_chan or gap_info.get("spin_channel") or 2
                homo_b = homo_b or gap_info.get("homo_band") or n_occ_max
                lumo_b = lumo_b or gap_info.get("lumo_band") or (n_occ_max + 1)
                nbnd = gap_info.get("nbnd") or max(homo_b + 1, lumo_b, n_occ_max) + 8
            else:
                spin_chan = spin_chan or 2
                homo_b = homo_b or n_occ_max
                lumo_b = lumo_b or (n_occ_max + 1)
                nbnd = max(lumo_b, n_occ_max) + 8

            nbnd = max(nbnd, homo_b + 1, lumo_b, n_occ_max)

            constrained_occupations = build_delta_scf_occupations(
                nelec=nelec_total,
                tot_magnetization=tot_mag,
                nbnd=nbnd,
                homo_band=homo_b,
                lumo_band=lumo_b,
                spin_channel=spin_chan,
            )

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

        labeled_dict = {}
        for e in completed:
            at = self.label_single_frame(
                e["atoms"],
                e["calc_dir"],
                occupations_mode="from_input",
                constrained_occupations=constrained_occupations,
            )
            at.info["electronic_state"] = "excited"
            at.info["sample_type"] = e["sample_type"]
            at.info["qe_status"] = "converged"
            if "lambda" in e["atoms"].info:
                at.info["lambda"] = e["atoms"].info["lambda"]
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
            logger.info(f"\n    ---> [ES {step_num}/{len(pending)}] Calculating {label} ({stype}) in {os.path.basename(cdir)}...")
            try:
                at_labeled = self.label_single_frame(
                    fr,
                    cdir,
                    retry=False,
                    occupations_mode="from_input",
                    constrained_occupations=constrained_occupations,
                )
                at_labeled.info["electronic_state"] = "excited"
                at_labeled.info["sample_type"] = stype
                at_labeled.info["qe_status"] = "converged"
                if "lambda" in fr.info:
                    at_labeled.info["lambda"] = fr.info["lambda"]
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
                logger.info(f"    ---> [Retry {retry_num}/{len(retry_queue)}] {label} ({stype}) with mixing_mode='local-TF'...")
                try:
                    at_labeled = self.label_single_frame(
                        fr,
                        cdir,
                        retry=True,
                        occupations_mode="from_input",
                        constrained_occupations=constrained_occupations,
                    )
                    at_labeled.info["electronic_state"] = "excited"
                    at_labeled.info["sample_type"] = stype
                    at_labeled.info["qe_status"] = "converged"
                    if "lambda" in fr.info:
                        at_labeled.info["lambda"] = fr.info["lambda"]
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
            logger.info(f"\n    ---> [{step_num}/{len(to_calculate)}] Calculating {label} ({stype}) in {os.path.basename(cdir)}...")
            try:
                labeled_fr = self.label_single_frame(fr, cdir, retry=False)
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

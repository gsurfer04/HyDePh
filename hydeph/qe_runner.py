"""Quantum ESPRESSO `pw.x` runner with single-pass parsing, caching, and type-balanced resumption."""

from __future__ import annotations

import os
import sys
import re
import glob
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

from .config import ToolkitConfig, discover_pseudopotential, inspect_upf_metadata, HARD_ELEMENT_CUTOFFS, resolve_pseudo_path

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

RY_TO_EV = float(Ry)
RY_BOHR_TO_EV_A = float(Ry / Bohr)


def is_windows() -> bool:
    """Returns True if running on Windows."""
    return sys.platform.startswith("win")


def format_slurm_module_lines(modules: Optional[List[str]] = None, purge: bool = True) -> str:
    """Formats deterministic bash module purge and load lines in exact specified order without heuristic loops.

    Preserves exact user-defined library loading order (e.g. profile/chem-phys before quantum-espresso).
    """
    lines = []
    if purge:
        lines.append("module purge")
    if modules:
        for m in modules:
            m_clean = str(m).strip()
            if m_clean:
                lines.append(f"module load {m_clean}")
    else:
        lines.append("# Note: No cluster environment modules specified in configuration.")
    return "\n".join(lines)


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

    # Fortran namelist parser requires forward slashes
    clean_pseudo_dir = config.pseudo_dir.replace("\\", "/") if config.pseudo_dir else "./pseudo"

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
    template_atoms: Optional[Atoms] = None,
    require_converged: bool = True,
) -> Tuple[Optional[Atoms], Optional[float]]:
    """Parses the final relaxed geometry and total energy from a Quantum ESPRESSO `pw.x` relaxation output.
    
    Parameters
    ----------
    pw_out_path : str
        Path to `pw.out`.
    template_atoms : Atoms, optional
        Reference Atoms object containing cell, symbols, and pbc. If omitted, attempts to infer
        from the adjacent `pw.in` input deck.
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

    if template_atoms is None:
        calc_dir = os.path.dirname(pw_out_path)
        pw_in = os.path.join(calc_dir, "pw.in")
        if os.path.isfile(pw_in):
            try:
                template_atoms = ase.io.read(pw_in, format="espresso-in")
            except Exception:
                template_atoms = None

    if template_atoms is None:
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


def parse_qe_band_gap(pw_out_path: str, target_spin_channel: Optional[int] = None) -> Optional[Dict[str, Any]]:
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

        # Compute degeneracies for both spin channels (exact point-group multiplets < 5 meV)
        deg_lumo_u, deg_homo_u = 1, 1
        if u_bands and lumo_b_u is not None and lumo_b_u <= len(u_bands):
            le = u_bands[lumo_b_u - 1]
            while (lumo_b_u - 1 + deg_lumo_u) < len(u_bands) and abs(u_bands[lumo_b_u - 1 + deg_lumo_u] - le) < 0.005:
                deg_lumo_u += 1
        if u_bands and homo_b_u is not None and 1 <= homo_b_u <= len(u_bands):
            he = u_bands[homo_b_u - 1]
            while (homo_b_u - deg_homo_u) >= 1 and abs(u_bands[homo_b_u - deg_homo_u - 1] - he) < 0.005:
                deg_homo_u += 1

        deg_lumo_d, deg_homo_d = 1, 1
        if d_bands and lumo_b_d is not None and lumo_b_d <= len(d_bands):
            le = d_bands[lumo_b_d - 1]
            while (lumo_b_d - 1 + deg_lumo_d) < len(d_bands) and abs(d_bands[lumo_b_d - 1 + deg_lumo_d] - le) < 0.005:
                deg_lumo_d += 1
        if d_bands and homo_b_d is not None and 1 <= homo_b_d <= len(d_bands):
            he = d_bands[homo_b_d - 1]
            while (homo_b_d - deg_homo_d) >= 1 and abs(d_bands[homo_b_d - deg_homo_d - 1] - he) < 0.005:
                deg_homo_d += 1

        # Prioritize the minority or in-gap defect channel (smaller positive gap) or target_spin_channel
        chosen = None
        if target_spin_channel == 1 and gap_u is not None:
            chosen = ("up", 1, gap_u, homo_u, lumo_u, homo_b_u, lumo_b_u, ef_up)
        elif target_spin_channel == 2 and gap_d is not None:
            chosen = ("down", 2, gap_d, homo_d, lumo_d, homo_b_d, lumo_b_d, ef_dw)
        elif gap_d is not None and gap_u is not None:
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
            chosen_deg_lumo = deg_lumo_d if spin_chan == 2 else deg_lumo_u
            chosen_deg_homo = deg_homo_d if spin_chan == 2 else deg_homo_u
            return {
                "gap_eV": gap_v,
                "homo_eV": homo_v,
                "lumo_eV": lumo_v,
                "fermi_eV": ef_v,
                "spin": spin_name,
                "spin_channel": spin_chan,
                "homo_band": h_b,
                "homo_degeneracy": chosen_deg_homo,
                "lumo_band": l_b,
                "lumo_degeneracy": chosen_deg_lumo,
                "homo_deg_up": deg_homo_u,
                "lumo_deg_up": deg_lumo_u,
                "homo_deg_down": deg_homo_d,
                "lumo_deg_down": deg_lumo_d,
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
                    "homo_degeneracy": 1,
                    "lumo_band": len(occ) + 1,
                    "lumo_degeneracy": 1,
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
                    "homo_degeneracy": 1,
                    "lumo_band": n_occ + 1,
                    "lumo_degeneracy": 1,
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


def extract_and_attach_adiabatic_energetics(
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
    if not labeled_gs or not labeled_es:
        return {}

    gs_0 = next((f for f in labeled_gs if f.info.get("sample_type") == "franck_condon" and abs(f.info.get("lambda", 1.0)) < 1e-4), None)
    gs_1 = next((f for f in labeled_gs if f.info.get("sample_type") == "franck_condon" and abs(f.info.get("lambda", 0.0) - 1.0) < 1e-4), None)
    es_0 = next((f for f in labeled_es if f.info.get("sample_type") == "franck_condon" and abs(f.info.get("lambda", 1.0)) < 1e-4), None)
    es_1 = next((f for f in labeled_es if f.info.get("sample_type") == "franck_condon" and abs(f.info.get("lambda", 0.0) - 1.0) < 1e-4), None)

    if gs_0 is None and labeled_gs:
        gs_0 = next((f for f in labeled_gs if f.info.get("frame_id") in ("fc_00", "frame_00")), labeled_gs[0])
    if es_1 is None and labeled_es:
        fc_es = [f for f in labeled_es if f.info.get("sample_type") == "franck_condon"]
        if fc_es:
            es_1 = fc_es[-1]
        elif labeled_es:
            es_1 = labeled_es[-1]

    results: Dict[str, float] = {}
    if gs_0 is not None and es_1 is not None and "energy" in gs_0.info and "energy" in es_1.info:
        e_gs_0 = float(gs_0.info["energy"])
        e_es_1 = float(es_1.info["energy"])
        e_zpl_exact = float(e_es_1 - e_gs_0)

        results["e_gs_0"] = e_gs_0
        results["e_es_1"] = e_es_1
        if at_gs is not None:
            at_gs.info["energy"] = e_gs_0
            if e_zpl_exact > 0:
                at_gs.info["e_zpl_exact_eV"] = e_zpl_exact
                at_gs.info["zpl_source"] = "delta_scf_adiabatic"
        if at_es is not None:
            at_es.info["energy"] = e_es_1
        if e_zpl_exact > 0:
            results["e_zpl_exact_eV"] = e_zpl_exact

        logger.info("\n" + "-" * 60)
        logger.info("[+] EXACT ADIABATIC DFT ENERGETICS (Delta-SCF):")
        logger.info(f"    • Ground State Minimum E_GS(lambda=0): {e_gs_0:.4f} eV")
        logger.info(f"    • Excited State Minimum E_ES(lambda=1): {e_es_1:.4f} eV")
        logger.info(f"    • Exact Adiabatic ZPL E_ZPL: {e_zpl_exact:.4f} eV")

        if gs_1 is not None and es_0 is not None and "energy" in gs_1.info and "energy" in es_0.info:
            e_gs_1 = float(gs_1.info["energy"])
            e_es_0 = float(es_0.info["energy"])
            e_abs_exact = float(e_es_0 - e_gs_0)
            e_em_exact = float(e_es_1 - e_gs_1)
            erel_gs = float(e_gs_1 - e_gs_0)
            erel_es = float(e_es_0 - e_es_1)
            stokes_exact = float(e_abs_exact - e_em_exact)

            results["e_abs_exact_eV"] = e_abs_exact
            results["e_em_exact_eV"] = e_em_exact
            results["erel_gs_eV"] = erel_gs
            results["erel_es_eV"] = erel_es
            results["stokes_shift_exact_eV"] = stokes_exact

            logger.info(f"    • Vertical Absorption E_abs: {e_abs_exact:.4f} eV")
            logger.info(f"    • Vertical Emission E_em: {e_em_exact:.4f} eV")
            logger.info(f"    • GS Relaxation Energy Delta E_rel,GS: {erel_gs*1000:.1f} meV ({erel_gs:.4f} eV)")
            logger.info(f"    • ES Relaxation Energy Delta E_rel,ES: {erel_es*1000:.1f} meV ({erel_es:.4f} eV)")
            logger.info(f"    • Exact Stokes Shift Delta E_Stokes: {stokes_exact*1000:.1f} meV ({stokes_exact:.4f} eV)")
        logger.info("-" * 60 + "\n")

    return results


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
            gs_candidates = [
                os.path.join(calc_base, "qe_gs_relax", "pw.out"),
                os.path.join(calc_base, "dft_calculations", "frame_000_fc_00", "pw.out"),
            ]
            gs_out = next((p for p in gs_candidates if os.path.isfile(p)), None)
            target_sc = getattr(self.config, "delta_scf_spin_channel", None)
            gap_info = parse_qe_band_gap(gs_out, target_spin_channel=target_sc) if gs_out else None

            unique_syms = sorted(list(set(at_gs.get_chemical_symbols())))
            z_vals = {}
            for s in unique_syms:
                upf = self.config.pseudopotentials.get(s, "")
                if upf:
                    p = resolve_pseudo_path(upf, self.config.pseudo_dir)
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

            homo_deg = getattr(self.config, "delta_scf_homo_degeneracy", None)
            if homo_deg is None and gap_info is not None:
                homo_deg = gap_info.get("homo_degeneracy", 1)
            homo_deg = max(1, int(homo_deg or 1))

            lumo_deg = getattr(self.config, "delta_scf_lumo_degeneracy", None)
            if lumo_deg is None and gap_info is not None:
                lumo_deg = gap_info.get("lumo_degeneracy", 1)
            lumo_deg = max(1, int(lumo_deg or 1))

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
                logger.info(f"    ---> [Retry {retry_num}/{len(retry_queue)}] {label} ({stype}) with conservative mixing (beta=0.2, electron_maxstep=150)...")
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

        write_qe_input(
            defect_sc,
            gs_pw_in,
            self.config,
            calculation="relax",
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
            homo_b = getattr(self.config, "delta_scf_homo_band", None)
            homo_deg = getattr(self.config, "delta_scf_homo_degeneracy", None)
            lumo_b = getattr(self.config, "delta_scf_lumo_band", None)
            spin_chan = getattr(self.config, "delta_scf_spin_channel", None)
            lumo_deg = getattr(self.config, "delta_scf_lumo_degeneracy", None)

            # Auto-detect from converged ground state pw.out if available
            gs_out = os.path.join(gs_dir, "pw.out")
            gap_info = parse_qe_band_gap(gs_out, target_spin_channel=spin_chan) if os.path.isfile(gs_out) else None
            if gap_info is not None:
                spin_chan = spin_chan or gap_info.get("spin_channel") or 2
                homo_b = homo_b or gap_info.get("homo_band")
                lumo_b = lumo_b or gap_info.get("lumo_band")
                homo_deg = max(1, int(homo_deg or gap_info.get("homo_degeneracy", 1) or 1))
                lumo_deg = max(1, int(lumo_deg or gap_info.get("lumo_degeneracy", 1) or 1))
                logger.info(
                    f"    [i] Auto-detected lowest optical transition from GS DFT: "
                    f"spin channel {spin_chan}, bands {homo_b} (deg={homo_deg}) -> {lumo_b} (deg={lumo_deg}) "
                    f"[gap = {gap_info.get('gap_eV', 0.0):.3f} eV]"
                )

            # Count valence electrons
            unique_syms = sorted(list(set(defect_sc.get_chemical_symbols())))
            z_vals = {}
            for s in unique_syms:
                upf = self.config.pseudopotentials.get(s, "") if self.config.pseudopotentials else ""
                if upf:
                    p = resolve_pseudo_path(upf, self.config.pseudo_dir)
                    if os.path.isfile(p):
                        z_vals[s] = inspect_upf_metadata(p).get("z_valence")
            _DEFAULT_VALENCE = {
                "H": 1, "Li": 1, "Be": 2, "B": 3, "C": 4, "N": 5, "O": 6, "F": 7,
                "Na": 1, "Mg": 2, "Al": 3, "Si": 4, "P": 5, "S": 6, "Cl": 7,
                "K": 1, "Ca": 2, "Ga": 3, "Ge": 4, "As": 5, "Se": 6, "Br": 7,
                "Rb": 1, "Sr": 2, "In": 3, "Sn": 4, "Sb": 5, "Te": 6, "I": 7,
                "Cs": 1, "Ba": 2,
            }
            nelec_total = int(round(
                sum(z_vals.get(s) or _DEFAULT_VALENCE.get(s, 4) for s in defect_sc.get_chemical_symbols())
                - float(self.config.tot_charge)
            ))
            if gap_info is not None and gap_info.get("nelec") is not None:
                nelec_total = int(round(gap_info["nelec"]))

            tot_mag = int(self.config.spin_multiplicity - 1) if self.config.spin_polarized else 0
            if (nelec_total % 2) != (tot_mag % 2):
                tot_mag = tot_mag if tot_mag % 2 == nelec_total % 2 else (tot_mag + 1 if tot_mag >= 0 else 0)

            n_occ_max = (nelec_total + abs(tot_mag)) // 2
            homo_b = homo_b or n_occ_max
            lumo_b = lumo_b or (n_occ_max + 1)
            homo_deg = max(1, int(homo_deg or 1))
            lumo_deg = max(1, int(lumo_deg or 1))
            spin_chan = spin_chan or 2

            nbnd = max(homo_b + 1, lumo_b + lumo_deg - 1, n_occ_max) + 8
            occ_up, occ_dw = build_delta_scf_occupations(
                nelec=nelec_total,
                tot_magnetization=tot_mag,
                nbnd=nbnd,
                homo_band=homo_b,
                lumo_band=lumo_b,
                spin_channel=spin_chan,
                lumo_degeneracy=lumo_deg,
                homo_degeneracy=homo_deg,
            )
            start_es = init_es if init_es is not None else defect_sc
            write_qe_input(
                start_es,
                es_pw_in,
                self.config,
                calculation="relax",
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
            qos = getattr(self.config, "slurm_qos", None)
            qos_line = f"#SBATCH --qos={qos}\n" if qos else ""
            nodes = getattr(self.config, "slurm_nodes", 1) or 1
            gpus = getattr(self.config, "slurm_gpus_per_node", 4) or 4
            cpus_per_task = getattr(self.config, "slurm_cpus_per_task", 8) or 8
            module_block = format_slurm_module_lines(mods)

            slurm_content = f"""#!/bin/bash
# ==============================================================================
#  HyDePh Decoupled HPC Relaxation Launcher (Pure Quantum ESPRESSO)
#  Ground-State and Excited-State (Delta-SCF) Geometry Optimizations
#
#  Usage:
#    sbatch submit_relaxations_pure_qe.slurm
# ==============================================================================
#SBATCH --job-name=hydeph_relax
#SBATCH --account={acct}
#SBATCH --partition={part}
{qos_line}#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node={gpus}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --gres=gpu:{gpus}
#SBATCH --time=04:00:00
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
for cand in "$WORK_DIR" "$WORK_DIR/hydeph_run" "$WORK_DIR/../hydeph_run"; do
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
for cand in "$WORKSPACE_DIR/pseudo" "$WORKSPACE_DIR/../pseudo" "$WORK_DIR/pseudo" "$WORK_DIR/../pseudo"; do
    if [ -d "$cand" ]; then
        PSEUDO_DIR="$(cd "$cand" && pwd)"
        break
    fi
done
if [ -n "$PSEUDO_DIR" ]; then
    [ -d "$WORKSPACE_DIR/qe_gs_relax" ] && [ ! -e "$WORKSPACE_DIR/qe_gs_relax/pseudo" ] && ln -sfn "$PSEUDO_DIR" "$WORKSPACE_DIR/qe_gs_relax/pseudo"
    [ -d "$WORKSPACE_DIR/qe_es_relax" ] && [ ! -e "$WORKSPACE_DIR/qe_es_relax/pseudo" ] && ln -sfn "$PSEUDO_DIR" "$WORKSPACE_DIR/qe_es_relax/pseudo"
    [ -d "$WORKSPACE_DIR/qe_gs_scf" ] && [ ! -e "$WORKSPACE_DIR/qe_gs_scf/pseudo" ] && ln -sfn "$PSEUDO_DIR" "$WORKSPACE_DIR/qe_gs_scf/pseudo"
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
for cand in "$WORK_DIR" "$WORK_DIR/hydeph_run"; do
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
for cand in "$PWD/pseudo" "$PWD/../pseudo" "$WORK_DIR/pseudo" "$WORK_DIR/../pseudo"; do
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
for cand in "$WORK_DIR/dft_frames_list.txt" "$WORK_DIR/hydeph_run/dft_frames_list.txt" "$WORK_DIR/../dft_frames_list.txt"; do
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

RAW_DIR=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$LIST_FILE")
# Normalize slashes in case of Windows formatting
RAW_DIR=$(echo "$RAW_DIR" | tr '\\\\' '/')

if [ -z "$RAW_DIR" ]; then
    echo "[!] Error: No entry found at line $((SLURM_ARRAY_TASK_ID + 1)) of $LIST_FILE"
    exit 1
fi

# Locate the target frame directory containing pw.in
DIR=""
for cand in "$WORKSPACE_DIR/$RAW_DIR" "$WORK_DIR/$RAW_DIR" "$RAW_DIR"; do
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
for cand in "$WORKSPACE_DIR/pseudo" "$WORKSPACE_DIR/../pseudo" "$WORK_DIR/pseudo" "$WORK_DIR/../pseudo"; do
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
                    gap_info = parse_qe_band_gap(pw_out)
                    if gap_info:
                        for k, v in gap_info.items():
                            at.info[f"dft_{k}"] = v
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

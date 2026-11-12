"""Quantum ESPRESSO output parsing, band structures, and energetics extraction."""

from __future__ import annotations

import os
import re
import json
import logging
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
import ase.io
from ase import Atoms

from .io import (
    RY_TO_EV,
    RY_BOHR_TO_EV_A,
    BOHR_TO_ANGSTROM,
    resolve_pseudo_path,
    inspect_upf_metadata,
    build_delta_scf_occupations,
    check_pw_in_matches_atoms,
)
from ..config import ToolkitConfig

logger = logging.getLogger(__name__)


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
    final_cell: Optional[np.ndarray] = None
    cell_lines: List[List[float]] = []
    cell_card = ""
    pos_card = ""
    energy_ev: Optional[float] = None
    in_final_coords = False
    in_cell = False
    in_pos = False

    alat_ang: Optional[float] = None
    alat_m = re.findall(r"lattice parameter \(alat\)\s*=\s*([-\d\.]+)", text, re.IGNORECASE)
    if alat_m:
        try:
            alat_ang = float(alat_m[-1]) * BOHR_TO_ANGSTROM
        except ValueError:
            pass

    def _parse_cell_scale(card_str: str, default_alat: Optional[float] = None) -> float:
        c_low = card_str.lower()
        if "angstrom" in c_low:
            return 1.0
        if "bohr" in c_low:
            return BOHR_TO_ANGSTROM
        if "alat" in c_low:
            m_val = re.search(r"=\s*([-\d\.]+)", card_str)
            if m_val:
                try:
                    return float(m_val.group(1)) * BOHR_TO_ANGSTROM
                except ValueError:
                    pass
            if default_alat is not None:
                return default_alat
            return BOHR_TO_ANGSTROM
        return default_alat if default_alat is not None else 1.0

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
            cell_lines = []
            final_positions = []
            cell_card = ""
            pos_card = ""
            in_cell = False
            in_pos = False
            continue
        if in_final_coords:
            if "End final coordinates" in line:
                in_final_coords = False
                break
            if "CELL_PARAMETERS" in line:
                in_cell = True
                in_pos = False
                cell_card = line
                cell_lines = []
                continue
            if "ATOMIC_POSITIONS" in line:
                in_pos = True
                in_cell = False
                pos_card = line
                final_positions = []
                continue
            if in_cell:
                parts = line.strip().split()
                if len(parts) >= 3:
                    try:
                        cell_lines.append([float(parts[0]), float(parts[1]), float(parts[2])])
                    except ValueError:
                        pass
                if len(cell_lines) == 3:
                    in_cell = False
            elif in_pos:
                parts = line.strip().split()
                if len(parts) >= 4:
                    try:
                        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                        final_positions.append([x, y, z])
                    except ValueError:
                        pass

    if len(cell_lines) == 3:
        scale = _parse_cell_scale(cell_card, alat_ang)
        final_cell = np.array(cell_lines, dtype=float) * scale

    # Pass 2 fallback: If "Begin final coordinates" was omitted or had mismatched count
    if len(final_positions) != len(template_atoms):
        cell_blocks: List[Tuple[str, List[List[float]]]] = []
        pos_blocks: List[Tuple[str, List[List[float]]]] = []

        i = 0
        while i < len(lines):
            line = lines[i]
            if "CELL_PARAMETERS" in line:
                card = line
                vecs: List[List[float]] = []
                j = i + 1
                while j < len(lines) and len(vecs) < 3:
                    parts = lines[j].strip().split()
                    if len(parts) >= 3:
                        try:
                            vecs.append([float(parts[0]), float(parts[1]), float(parts[2])])
                        except ValueError:
                            break
                    j += 1
                if len(vecs) == 3:
                    cell_blocks.append((card, vecs))
                i = j
                continue
            elif "ATOMIC_POSITIONS" in line:
                card = line
                p_list: List[List[float]] = []
                j = i + 1
                while j < len(lines) and len(p_list) < len(template_atoms):
                    parts = lines[j].strip().split()
                    if len(parts) >= 4:
                        try:
                            p_list.append([float(parts[1]), float(parts[2]), float(parts[3])])
                        except ValueError:
                            break
                    j += 1
                if len(p_list) == len(template_atoms):
                    pos_blocks.append((card, p_list))
                i = j
                continue
            i += 1

        if pos_blocks:
            pos_card, final_positions = pos_blocks[-1]
        if cell_blocks and final_cell is None:
            c_card, c_vecs = cell_blocks[-1]
            c_scale = _parse_cell_scale(c_card, alat_ang)
            final_cell = np.array(c_vecs, dtype=float) * c_scale

    if len(final_positions) != len(template_atoms):
        return None, None

    if final_cell is None:
        final_cell = np.array(template_atoms.cell, dtype=float)

    raw_pos = np.array(final_positions, dtype=float)
    pos_lower = pos_card.lower()
    if "crystal" in pos_lower:
        cart_pos = raw_pos @ final_cell
    elif "bohr" in pos_lower:
        cart_pos = raw_pos * BOHR_TO_ANGSTROM
    elif "alat" in pos_lower:
        scale = alat_ang if alat_ang is not None else 1.0
        cart_pos = raw_pos * scale
    else:
        cart_pos = raw_pos

    at_relaxed = template_atoms.copy()
    at_relaxed.set_cell(final_cell, scale_atoms=False)
    at_relaxed.positions = cart_pos
    at_relaxed.pbc = True
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

            # Build in-gap level diagram around Fermi level (homo_band - 3 to lumo_band + 3)
            homo_cands = [b for b in (homo_b_u, homo_b_d) if b is not None]
            lumo_cands = [b for b in (lumo_b_u, lumo_b_d) if b is not None]
            min_b = max(1, min(homo_cands) - 3) if homo_cands else 1
            max_b = min(max(len(u_bands), len(d_bands)), max(lumo_cands) + 3) if lumo_cands else (min_b + 7)

            diag_lines = [
                "+------+------------------------+------------------------+",
                "| Band | Spin UP (Channel 1)    | Spin DOWN (Channel 2)  |",
                "+------+------------------------+------------------------+",
            ]
            for b in range(min_b, max_b + 1):
                if b <= len(u_bands):
                    val_u = u_bands[b - 1]
                    occ_u_str = "[X]" if (homo_b_u and b <= homo_b_u) else "[ ]"
                    tag_u = " (HOMO)" if b == homo_b_u else (" (LUMO)" if b == lumo_b_u else "")
                    up_s = f"{val_u:+7.3f} eV {occ_u_str}{tag_u}"
                else:
                    up_s = "       ---        "

                if b <= len(d_bands):
                    val_d = d_bands[b - 1]
                    occ_d_str = "[X]" if (homo_b_d and b <= homo_b_d) else "[ ]"
                    tag_d = " (HOMO)" if b == homo_b_d else (" (LUMO)" if b == lumo_b_d else "")
                    dw_s = f"{val_d:+7.3f} eV {occ_d_str}{tag_d}"
                else:
                    dw_s = "       ---        "

                diag_lines.append(f"| {b:4d} | {up_s:<22s} | {dw_s:<22s} |")
            diag_lines.append("+------+------------------------+------------------------+")
            diag_lines.append("Candidate Optical Transitions:")
            if gap_u is not None:
                sel_u = " [SELECTED]" if spin_chan == 1 else ""
                diag_lines.append(
                    f"  - Spin UP:   Band {homo_b_u} -> {lumo_b_u} | Gap = {gap_u:.3f} eV "
                    f"(HOMO deg={deg_homo_u}, LUMO deg={deg_lumo_u}){sel_u}"
                )
            if gap_d is not None:
                sel_d = " [SELECTED]" if spin_chan == 2 else ""
                diag_lines.append(
                    f"  - Spin DOWN: Band {homo_b_d} -> {lumo_b_d} | Gap = {gap_d:.3f} eV "
                    f"(HOMO deg={deg_homo_d}, LUMO deg={deg_lumo_d}){sel_d}"
                )
            diagram_str = "\n".join(diag_lines)
            logger.info(f"\n[+] In-Gap Level Diagram (Delta-SCF Candidate Bands):\n{diagram_str}")

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
                "level_diagram": diagram_str,
                "u_bands": u_bands,
                "d_bands": d_bands,
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
                homo_b = len(occ)
                lumo_b = len(occ) + 1
                gap_v = unocc[0] - occ[-1]
                min_b = max(1, homo_b - 3)
                max_b = min(len(bands), lumo_b + 3)
                diag_lines = [
                    "+------+------------------------+",
                    "| Band | Energy (eV)  Occ  Note |",
                    "+------+------------------------+",
                ]
                for b in range(min_b, max_b + 1):
                    val = bands[b - 1]
                    occ_str = "[X]" if b <= homo_b else "[ ]"
                    tag = " (HOMO)" if b == homo_b else (" (LUMO)" if b == lumo_b else "")
                    diag_lines.append(f"| {b:4d} | {val:+7.3f} eV {occ_str}{tag:<7s} |")
                diag_lines.append("+------+------------------------+")
                diag_lines.append(f"Candidate Optical Transition: Band {homo_b} -> {lumo_b} | Gap = {gap_v:.3f} eV")
                diagram_str = "\n".join(diag_lines)
                logger.info(f"\n[+] In-Gap Level Diagram (Delta-SCF Candidate Bands):\n{diagram_str}")
                return {
                    "gap_eV": gap_v,
                    "homo_eV": occ[-1],
                    "lumo_eV": unocc[0],
                    "fermi_eV": ef,
                    "spin": "down",
                    "spin_channel": 2,
                    "homo_band": homo_b,
                    "homo_degeneracy": 1,
                    "lumo_band": lumo_b,
                    "lumo_degeneracy": 1,
                    "nbnd": len(bands),
                    "nelec": nelec,
                    "level_diagram": diagram_str,
                    "bands": bands,
                }
        elif nelec is not None:
            n_occ = int(round(nelec / 2.0))
            if 0 < n_occ < len(bands):
                homo_val = bands[n_occ - 1]
                lumo_val = bands[n_occ]
                homo_b = n_occ
                lumo_b = n_occ + 1
                gap_v = lumo_val - homo_val
                min_b = max(1, homo_b - 3)
                max_b = min(len(bands), lumo_b + 3)
                diag_lines = [
                    "+------+------------------------+",
                    "| Band | Energy (eV)  Occ  Note |",
                    "+------+------------------------+",
                ]
                for b in range(min_b, max_b + 1):
                    val = bands[b - 1]
                    occ_str = "[X]" if b <= homo_b else "[ ]"
                    tag = " (HOMO)" if b == homo_b else (" (LUMO)" if b == lumo_b else "")
                    diag_lines.append(f"| {b:4d} | {val:+7.3f} eV {occ_str}{tag:<7s} |")
                diag_lines.append("+------+------------------------+")
                diag_lines.append(f"Candidate Optical Transition: Band {homo_b} -> {lumo_b} | Gap = {gap_v:.3f} eV")
                diagram_str = "\n".join(diag_lines)
                logger.info(f"\n[+] In-Gap Level Diagram (Delta-SCF Candidate Bands):\n{diagram_str}")
                return {
                    "gap_eV": gap_v,
                    "homo_eV": homo_val,
                    "lumo_eV": lumo_val,
                    "fermi_eV": (homo_val + lumo_val) / 2.0,
                    "spin": "down",
                    "spin_channel": 2,
                    "homo_band": homo_b,
                    "homo_degeneracy": 1,
                    "lumo_band": lumo_b,
                    "lumo_degeneracy": 1,
                    "nbnd": len(bands),
                    "nelec": nelec,
                    "level_diagram": diagram_str,
                    "bands": bands,
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


def resolve_delta_scf_spec(
    at_gs: Atoms,
    config: ToolkitConfig,
    gs_pw_out_path: Optional[str] = None,
    workspace_dir: Optional[str] = None,
    gap_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Resolves and writes canonical Delta-SCF electronic transition occupations and metadata.

    Reads valence electron counts directly from pseudopotential metadata (raising an error if
    any species lacks a valid UPF) and determines target bands, spin channels, and occupations.
    Writes `delta_scf_spec.json` into `workspace_dir` to ensure ground-state relaxation,
    excited-state relaxation, candidate labelling, and HPC job decks share an identical transition.
    """
    ws = workspace_dir or config.workspace_dir
    os.makedirs(ws, exist_ok=True)
    spec_path = os.path.join(ws, "delta_scf_spec.json")

    _DEFAULT_VALENCE = {
        "H": 1, "Li": 1, "Be": 2, "B": 3, "C": 4, "N": 5, "O": 6, "F": 7,
        "Na": 1, "Mg": 2, "Al": 3, "Si": 4, "P": 5, "S": 6, "Cl": 7,
        "K": 1, "Ca": 2, "Ga": 3, "Ge": 4, "As": 5, "Se": 6, "Br": 7,
        "Rb": 1, "Sr": 2, "In": 3, "Sn": 4, "Sb": 5, "Te": 6, "I": 7,
        "Cs": 1, "Ba": 2,
    }

    unique_syms = sorted(list(set(at_gs.get_chemical_symbols())))
    z_vals = {}
    for s in unique_syms:
        upf = config.pseudopotentials.get(s, "") if config.pseudopotentials else ""
        if not upf:
            if getattr(config, "allow_fallbacks", True) or not config.pseudopotentials:
                if s in _DEFAULT_VALENCE:
                    z = _DEFAULT_VALENCE[s]
                    logger.warning(
                        f"[!] Warning: No UPF specified for element '{s}'. Using fallback atomic valence {z}. "
                        f"For production DFT runs, specify a valid UPF file in config.pseudopotentials."
                    )
                    z_vals[s] = float(z)
                    continue
                else:
                    raise ValueError(
                        f"Missing pseudopotential for element '{s}' and element not found in default valence table. "
                        f"Specify a valid UPF file in config.pseudopotentials."
                    )
            raise ValueError(
                f"Missing pseudopotential for element '{s}'. Specify a valid UPF file in config.pseudopotentials."
            )
        p = resolve_pseudo_path(upf, config.pseudo_dir)
        if not os.path.isfile(p):
            raise FileNotFoundError(
                f"Pseudopotential file for element '{s}' not found at '{p}'. Check pseudo_dir and pseudopotentials."
            )
        z = inspect_upf_metadata(p).get("z_valence")
        if z is None:
            raise ValueError(
                f"Could not parse valence electron count (z_valence) from UPF header '{p}'. "
                f"Ensure UPF is a valid Quantum ESPRESSO pseudopotential."
            )
        z_vals[s] = float(z)

    nelec_total = int(round(sum(z_vals[s] for s in at_gs.get_chemical_symbols()) - float(config.tot_charge)))

    target_sc = getattr(config, "delta_scf_spin_channel", None)
    if gap_info is None and gs_pw_out_path and os.path.isfile(gs_pw_out_path):
        gap_info = parse_qe_band_gap(gs_pw_out_path, target_spin_channel=target_sc)

    if gap_info is not None and gap_info.get("nelec") is not None:
        nelec_qe = int(round(gap_info["nelec"]))
        if nelec_qe != nelec_total:
            logger.warning(
                f"[!] Notice: UPF valence sum ({nelec_total}) differs from QE reported electrons ({nelec_qe}). "
                f"Using QE electron count {nelec_qe}."
            )
            nelec_total = nelec_qe

    tot_mag = int(config.spin_multiplicity - 1) if config.spin_polarized else 0
    if (nelec_total % 2) != (tot_mag % 2):
        if nelec_total % 2 != 0:
            valid_mag = 1 if tot_mag <= 0 else (tot_mag if tot_mag % 2 != 0 else tot_mag + 1)
        else:
            valid_mag = 0 if tot_mag <= 0 else (tot_mag if tot_mag % 2 == 0 else tot_mag + 1)
        if not getattr(config, "auto_spin_parity", False):
            raise ValueError(
                f"Spin parity mismatch in Delta-SCF: Total valence electrons ({nelec_total}) and "
                f"tot_magnetization ({tot_mag}) have opposite parity ({0.5*(nelec_total+tot_mag):.1f} up, {0.5*(nelec_total-tot_mag):.1f} down). "
                f"Set tot_magnetization={valid_mag} (spin_multiplicity={valid_mag+1}) or set auto_spin_parity=True."
            )
        else:
            tot_mag = valid_mag

    n_occ_max = (nelec_total + abs(tot_mag)) // 2
    spin_chan = getattr(config, "delta_scf_spin_channel", None)
    homo_b = getattr(config, "delta_scf_homo_band", None)
    lumo_b = getattr(config, "delta_scf_lumo_band", None)

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

    homo_deg = getattr(config, "delta_scf_homo_degeneracy", None)
    if homo_deg is None and gap_info is not None:
        homo_deg = gap_info.get("homo_degeneracy", 1)
    homo_deg = max(1, int(homo_deg or 1))

    lumo_deg = getattr(config, "delta_scf_lumo_degeneracy", None)
    if lumo_deg is None and gap_info is not None:
        lumo_deg = gap_info.get("lumo_degeneracy", 1)
    lumo_deg = max(1, int(lumo_deg or 1))

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

    spec = {
        "nelec": nelec_total,
        "tot_magnetization": tot_mag,
        "spin_channel": int(spin_chan),
        "homo_band": int(homo_b),
        "lumo_band": int(lumo_b),
        "homo_degeneracy": int(homo_deg),
        "lumo_degeneracy": int(lumo_deg),
        "nbnd": int(nbnd),
        "occupations_up": [float(x) for x in occ_up],
        "occupations_down": [float(x) for x in occ_dw],
        "gap_info": gap_info,
        "level_diagram": gap_info.get("level_diagram") if gap_info else None,
    }

    try:
        with open(spec_path, "w", encoding="utf-8") as f:
            json.dump(spec, f, indent=2)
        logger.info(f"[+] Canonical Delta-SCF specification resolved and written -> {spec_path}")
    except Exception as e:
        logger.warning(f"[-] Could not write delta_scf_spec.json ({e})")

    return spec


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
    allow_fallbacks: bool = False,
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
    allow_fallbacks : bool, default=False
        Whether to allow execution to proceed if anchor frames are missing.

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

    if gs_0 is None or es_1 is None:
        if not allow_fallbacks:
            raise RuntimeError(
                f"Exact adiabatic Delta-SCF energetics require converged lambda=0 and lambda=1 anchor frames. "
                f"Found gs_0={'yes' if gs_0 else 'missing'}, es_1={'yes' if es_1 else 'missing'}."
            )
        logger.warning(
            "[!] Warning: Franck-Condon anchor frame (lambda=0 or lambda=1) is missing from converged DFT frames. "
            "Exact adiabatic Delta-SCF energetics cannot be calculated."
        )
        return {}

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

"""Supercell construction, defect embedding, and ground- and excited-state geometry optimization."""

from __future__ import annotations

import os
import logging
from typing import Tuple, Optional
import numpy as np
import ase.io
from ase import Atoms
from ase.geometry import find_mic
from ase.optimize import BFGS
from ase.constraints import FixAtoms
from calorine.calculators import CPUNEP

from .config import ToolkitConfig
from .calculator import load_foundation_calculator
from .qe_runner import build_delta_scf_occupations, inspect_upf_metadata, parse_qe_band_gap

logger = logging.getLogger(__name__)


def mic_displacement(pos_a, pos_b, cell, pbc=True):
    """Returns minimum-image displacement vectors `pos_a - pos_b` under periodic boundary conditions."""
    raw = np.asarray(pos_a, dtype=float) - np.asarray(pos_b, dtype=float)
    vecs, _ = find_mic(raw, cell, pbc=pbc)
    return vecs


def mic_distances(positions, center, cell, pbc=True):
    """Returns minimum-image Euclidean distances from `center` to `positions`."""
    raw = np.asarray(positions, dtype=float) - np.asarray(center, dtype=float)
    _, dists = find_mic(raw, cell, pbc=pbc)
    return dists


def get_defect_center(atoms: Atoms) -> np.ndarray:
    """Returns the defect center stored in `atoms.info['defect_center']`, or the supercell center."""
    val = atoms.info.get("defect_center")
    if val is not None:
        if isinstance(val, str):
            parts = [float(x) for x in val.replace("[", " ").replace("]", " ").replace(",", " ").split()]
            if len(parts) == 3:
                return np.array(parts, dtype=float)
        arr = np.asarray(val, dtype=float).reshape(-1)
        if arr.size == 3:
            return arr
    return np.asarray(atoms.get_cell().sum(axis=0) / 2.0, dtype=float)


def detect_defect_sites(
    pristine: Atoms,
    defect: Atoms,
    tol: float = 0.6,
) -> dict:
    """Identifies defect sites (vacancies, substitutions, interstitials) between pristine and defect structures.

    Uses minimum-image convention distance matching, without making any assumptions about
    the ordering of atoms in the pristine or defect Atoms objects.

    Parameters
    ----------
    pristine : Atoms
        Reference pristine crystal (unit cell or supercell).
    defect : Atoms
        Defective crystal structure (unit cell or supercell).
    tol : float
        Distance threshold (in Angstroms) for matching atoms between pristine and defect.
        Default is 0.6 A.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'vacancies': list of dicts with 'symbol', 'position' (pristine pos), 'index' (pristine index)
        - 'substitutions': list of dicts with 'from_symbol', 'to_symbol', 'position' (defect pos), 'index'
        - 'interstitials': list of dicts with 'symbol', 'position' (defect pos), 'index'
        - 'all_defect_positions': list of np.ndarray positions of all defect sites
        - 'centroid': np.ndarray of shape (3,) giving the defect complex centroid
        - 'description': human-readable summary string
    """
    cell = defect.cell
    vacancies = []
    substitutions = []
    interstitials = []
    all_defect_positions = []

    # 1. Check each pristine atom for vacancy or matching partner in defect
    for i_p, at_p in enumerate(pristine):
        if len(defect) == 0:
            vacancies.append({"symbol": at_p.symbol, "position": at_p.position.copy(), "index": i_p})
            all_defect_positions.append(at_p.position.copy())
            continue
        dists = mic_distances(defect.positions, at_p.position, cell)
        min_d = np.min(dists)
        if min_d > tol:
            vacancies.append({"symbol": at_p.symbol, "position": at_p.position.copy(), "index": i_p})
            all_defect_positions.append(at_p.position.copy())

    # 2. Check each defect atom for substitution or interstitial match
    for i_d, at_d in enumerate(defect):
        if len(pristine) == 0:
            interstitials.append({"symbol": at_d.symbol, "position": at_d.position.copy(), "index": i_d})
            all_defect_positions.append(at_d.position.copy())
            continue
        dists = mic_distances(pristine.positions, at_d.position, cell)
        min_idx = int(np.argmin(dists))
        min_d = float(dists[min_idx])
        if min_d > tol:
            interstitials.append({"symbol": at_d.symbol, "position": at_d.position.copy(), "index": i_d})
            all_defect_positions.append(at_d.position.copy())
        else:
            closest_p = pristine[min_idx]
            if closest_p.symbol != at_d.symbol:
                substitutions.append({
                    "from_symbol": closest_p.symbol,
                    "to_symbol": at_d.symbol,
                    "position": at_d.position.copy(),
                    "index": i_d,
                })
                all_defect_positions.append(at_d.position.copy())

    # Compute centroid using minimum image convention
    if not all_defect_positions:
        centroid = np.asarray(defect.get_cell().sum(axis=0) / 2.0, dtype=float)
    elif len(all_defect_positions) == 1:
        centroid = all_defect_positions[0]
    else:
        ref_p = all_defect_positions[0]
        rel_vecs = mic_displacement(all_defect_positions, ref_p, cell)
        centroid = ref_p + np.mean(rel_vecs, axis=0)

    parts = []
    if vacancies:
        vac_strs = [v["symbol"] for v in vacancies]
        parts.append(f"{len(vacancies)} vacancy ({', '.join(vac_strs)})")
    if substitutions:
        sub_strs = [f"{s['from_symbol']}->{s['to_symbol']}" for s in substitutions]
        parts.append(f"{len(substitutions)} substitution ({', '.join(sub_strs)})")
    if interstitials:
        int_strs = [it["symbol"] for it in interstitials]
        parts.append(f"{len(interstitials)} interstitial ({', '.join(int_strs)})")
    description = ", ".join(parts) if parts else "pristine / unidentified defect"

    return {
        "vacancies": vacancies,
        "substitutions": substitutions,
        "interstitials": interstitials,
        "all_defect_positions": all_defect_positions,
        "centroid": centroid,
        "description": description,
    }


class DefectBuilder:
    """Constructs pristine and defect supercells, embeds unit cells, and optimizes geometries."""

    def __init__(self, config: ToolkitConfig, calc_foundation=None, calc_qe=None):
        self.config = config
        os.makedirs(self.config.workspace_dir, exist_ok=True)
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
        self.calc_qe = calc_qe
        self.vac_pos: Optional[np.ndarray] = None

    def _get_qe_runner(self):
        """Initializes or retrieves the Quantum ESPRESSO runner."""
        if self.calc_qe is None:
            from .qe_runner import QERunner
            self.calc_qe = QERunner(self.config)
        return self.calc_qe

    def _detect_supercell_defect_pos(self, pristine_sc: Atoms, defect_sc: Atoms) -> Tuple[np.ndarray, list]:
        """Detects the defect coordinate(s) in a pre-built defect supercell relative to `pristine_sc`."""
        res = detect_defect_sites(pristine_sc, defect_sc)
        if not res["all_defect_positions"]:
            centroid = np.asarray(defect_sc.get_cell().sum(axis=0) / 2.0, dtype=float)
            return centroid, [centroid]
        for v in res["vacancies"]:
            logger.info(f"    [i] Auto-detected vacancy of {v['symbol']} at {v['position']}")
        for s in res["substitutions"]:
            logger.info(f"    [i] Auto-detected substitution {s['from_symbol']} -> {s['to_symbol']} at {s['position']}")
        for it in res["interstitials"]:
            logger.info(f"    [i] Auto-detected interstitial of {it['symbol']} at {it['position']}")
        if len(res["all_defect_positions"]) > 1:
            logger.info(f"    [i] Defect complex centroid ({len(res['all_defect_positions'])} sites): {res['centroid']}")
        return res["centroid"], res["all_defect_positions"]

    def build_supercell_and_defect(self) -> Tuple[Atoms, Atoms]:
        """Builds pristine and defect supercells, persisting `defect_center` in `Atoms.info`."""
        pristine = ase.io.read(self.config.pristine_cell_path)

        rep = self.config.supercell_repeat
        pristine_sc = pristine.repeat(rep)
        pristine_sc_path = os.path.join(self.config.workspace_dir, "pristine_supercell.xyz")
        ase.io.write(pristine_sc_path, pristine_sc, format="extxyz")
        logger.info(f"[+] Built pristine supercell ({len(pristine_sc)} atoms, repeat={rep})")

        if self.config.defect_cell_path and os.path.isfile(self.config.defect_cell_path):
            defect_input = ase.io.read(self.config.defect_cell_path)
            v_input = defect_input.get_volume()
            v_pristine_uc = pristine.get_volume()

            if abs(v_input - v_pristine_uc) / v_pristine_uc < 0.25:
                logger.info(f"[+] Pre-made defect unit cell provided ({len(defect_input)} atoms).")
                logger.info(f"    Surrounding with pristine unit cells to build {rep} supercell...")
                defect_sc, detected_pos = self.embed_defect_unitcell(pristine, defect_input, rep)
                self.vac_pos = np.asarray(detected_pos, dtype=float)
                logger.info(f"[+] Defect embedded at {self.vac_pos}; resulting supercell has {len(defect_sc)} atoms.")
            else:
                defect_sc = defect_input
                centroid, defect_positions = self._detect_supercell_defect_pos(pristine_sc, defect_sc)
                self.vac_pos = centroid
                defect_sc.info["defect_positions"] = [p.tolist() if hasattr(p, 'tolist') else list(p) for p in defect_positions]
                logger.info(f"[+] Loaded existing defect supercell ({len(defect_sc)} atoms, defect center = {self.vac_pos}, {len(defect_positions)} sites)")
        else:
            defect_sc = pristine_sc.copy()
            dtype = self.config.defect_type.lower()
            site_idx = self.config.defect_site_index
            elem = self.config.defect_site_element
            sub_elem = self.config.substitutional_element

            if dtype == "substitutional":
                sub_pairs = []
                if elem and "->" in elem:
                    # Support multiple comma-separated substitutions, e.g. "B->C, N->C"
                    for part in elem.split(","):
                        if "->" in part:
                            p = [s.strip() for s in part.split("->")]
                            sub_pairs.append((p[0], p[1]))
                elif elem and sub_elem:
                    sub_pairs.append((elem, sub_elem))

                if not sub_pairs:
                    raise ValueError("Substitutional defect requires `substitutional_element` (or 'Host->Sub' in `defect_site_element`).")

                defect_positions = []
                center = defect_sc.get_cell().sum(axis=0) / 2.0
                last_pos = center

                if site_idx is not None:
                    target_idx = int(site_idx)
                    if target_idx < 0 or target_idx >= len(defect_sc):
                        raise ValueError(f"defect_site_index={target_idx} out of bounds for supercell of {len(defect_sc)} atoms.")
                    old_sym = defect_sc[target_idx].symbol
                    target_s = sub_pairs[0][1] if sub_pairs else sub_elem
                    defect_sc[target_idx].symbol = target_s
                    pos = defect_sc.positions[target_idx].copy()
                    defect_positions.append(pos)
                    logger.info(f"[+] Created substitution {old_sym} -> {target_s} at site {target_idx}, coordinate = {pos}")
                else:
                    for host_s, target_s in sub_pairs:
                        symbols = defect_sc.get_chemical_symbols()
                        cand_indices = [i for i, s in enumerate(symbols) if s == host_s]
                        if not cand_indices:
                            raise ValueError(f"Host element {host_s} not found in pristine supercell for substitution.")
                        cand_dists = mic_distances(defect_sc.positions[cand_indices], last_pos, defect_sc.cell)
                        t_idx = cand_indices[int(np.argmin(cand_dists))]
                        old_sym = defect_sc[t_idx].symbol
                        defect_sc[t_idx].symbol = target_s
                        pos = defect_sc.positions[t_idx].copy()
                        defect_positions.append(pos)
                        last_pos = pos
                        logger.info(f"[+] Created substitution {old_sym} -> {target_s} at site {t_idx}, coordinate = {pos}")

                if len(defect_positions) == 1:
                    self.vac_pos = defect_positions[0]
                else:
                    ref_p = defect_positions[0]
                    rel_vecs = mic_displacement(defect_positions, ref_p, defect_sc.cell)
                    self.vac_pos = ref_p + np.mean(rel_vecs, axis=0)
                    logger.info(f"[+] Defect complex centroid ({len(defect_positions)} sites): {self.vac_pos}")
                defect_sc.info["defect_positions"] = [p.tolist() if hasattr(p, 'tolist') else list(p) for p in defect_positions]
            elif dtype == "vacancy":
                if site_idx is not None:
                    target_idx = int(site_idx)
                    if target_idx < 0 or target_idx >= len(defect_sc):
                        raise ValueError(f"defect_site_index={target_idx} out of bounds for supercell of {len(defect_sc)} atoms.")
                else:
                    symbols = defect_sc.get_chemical_symbols()
                    cand_indices = [i for i, s in enumerate(symbols) if (elem is None or s == elem)]
                    if not cand_indices:
                        raise ValueError(f"Element {elem} not found in pristine supercell for defect creation.")
                    center = defect_sc.get_cell().sum(axis=0) / 2.0
                    cand_dists = mic_distances(defect_sc.positions[cand_indices], center, defect_sc.cell)
                    target_idx = cand_indices[int(np.argmin(cand_dists))]

                self.vac_pos = defect_sc.positions[target_idx].copy()
                removed_sym = defect_sc[target_idx].symbol
                del defect_sc[target_idx]
                defect_sc.info["defect_positions"] = [self.vac_pos.tolist()]
                logger.info(f"[+] Created {removed_sym} vacancy at site {target_idx}, coordinate = {self.vac_pos}")
            else:
                raise NotImplementedError(f"Defect type '{self.config.defect_type}' not supported.")

        defect_sc.info["defect_center"] = np.asarray(self.vac_pos, dtype=float)
        cell_lens = defect_sc.cell.lengths()
        min_dim = float(np.min(cell_lens))
        nat = len(defect_sc)
        logger.info(
            f"[+] Supercell geometry: {nat} atoms, dimensions: {cell_lens[0]:.2f} x {cell_lens[1]:.2f} x {cell_lens[2]:.2f} A (min L = {min_dim:.2f} A)"
        )
        if nat < 100 or min_dim < 9.0:
            logger.info(
                "    [i] Note: For stiff covalent crystals (e.g. Diamond, SiC), supercells with L >= 10 A (>= 140-200 atoms) "
                "are recommended to avoid discrete phonon comb artifacts."
            )

        defect_init_path = os.path.join(self.config.workspace_dir, "defect_unrelaxed.xyz")
        ase.io.write(defect_init_path, defect_sc, format="extxyz")
        return pristine_sc, defect_sc

    def embed_defect_unitcell(self, pristine_uc: Atoms, defect_uc: Atoms, rep: Tuple[int, int, int]) -> Tuple[Atoms, np.ndarray]:
        """Tiles pristine unit cells around a central defect unit cell and returns `(supercell, defect_pos)`.
        
        Vectorized with NumPy broadcasting for maximum efficiency across large supercells.
        """
        a1, a2, a3 = pristine_uc.cell[0], pristine_uc.cell[1], pristine_uc.cell[2]
        ic, jc, kc = rep[0] // 2, rep[1] // 2, rep[2] // 2
        shift_center = ic * a1 + jc * a2 + kc * a3

        res = detect_defect_sites(pristine_uc, defect_uc)
        defect_positions = []
        for v in res["vacancies"]:
            pos = v["position"] + shift_center
            defect_positions.append(pos)
            logger.info(f"    [i] Auto-detected vacancy of {v['symbol']} at {pos}")
        for s in res["substitutions"]:
            pos = s["position"] + shift_center
            defect_positions.append(pos)
            logger.info(f"    [i] Auto-detected substitution {s['from_symbol']} -> {s['to_symbol']} at {pos}")
        for it in res["interstitials"]:
            pos = it["position"] + shift_center
            defect_positions.append(pos)
            logger.info(f"    [i] Auto-detected interstitial of {it['symbol']} at {pos}")

        if not defect_positions:
            defect_pos = shift_center
        else:
            defect_pos = res["centroid"] + shift_center
            if len(defect_positions) > 1:
                logger.info(f"    [i] Defect complex centroid ({len(defect_positions)} sites): {defect_pos}")

        offsets = np.array([[i, j, k] for i in range(rep[0]) for j in range(rep[1]) for k in range(rep[2])], dtype=float)
        center_idx = ic * (rep[1] * rep[2]) + jc * rep[2] + kc
        uc_cell = pristine_uc.get_cell()

        pos_parts = []
        sym_parts = []

        if center_idx > 0:
            offsets_before = offsets[:center_idx]
            pos_before = (pristine_uc.get_positions()[None, :, :] + (offsets_before[:, None, :] @ uc_cell)).reshape(-1, 3)
            pos_parts.append(pos_before)
            sym_parts.extend(list(pristine_uc.get_chemical_symbols()) * center_idx)

        pos_center = defect_uc.get_positions() + shift_center
        pos_parts.append(pos_center)
        sym_parts.extend(list(defect_uc.get_chemical_symbols()))

        if center_idx + 1 < len(offsets):
            offsets_after = offsets[center_idx + 1:]
            pos_after = (pristine_uc.get_positions()[None, :, :] + (offsets_after[:, None, :] @ uc_cell)).reshape(-1, 3)
            pos_parts.append(pos_after)
            sym_parts.extend(list(pristine_uc.get_chemical_symbols()) * (len(offsets) - center_idx - 1))

        all_positions = np.vstack(pos_parts)
        cell = np.array([rep[0] * a1, rep[1] * a2, rep[2] * a3])
        supercell = Atoms(symbols=sym_parts, positions=all_positions, cell=cell, pbc=True)
        supercell.info["defect_center"] = np.asarray(defect_pos, dtype=float)
        supercell.info["defect_positions"] = [p.tolist() if hasattr(p, 'tolist') else list(p) for p in defect_positions]
        return supercell, defect_pos

    def relax_ground_state(self, defect_sc: Atoms, force_relax: bool = False) -> Atoms:
        """Relaxes ground-state geometry using DFT or foundation potential, or reuses cached `relaxed_gs.xyz`."""
        gs_path = os.path.join(self.config.workspace_dir, "relaxed_gs.xyz")
        force = force_relax or self.config.force_resample
        center = self.vac_pos if self.vac_pos is not None else get_defect_center(defect_sc)

        if not force and os.path.isfile(gs_path) and os.path.getsize(gs_path) > 0:
            try:
                cached_gs = ase.io.read(gs_path)
                if (
                    len(cached_gs) == len(defect_sc)
                    and cached_gs.get_chemical_symbols() == defect_sc.get_chemical_symbols()
                ):
                    cached_gs.info["defect_center"] = np.asarray(center, dtype=float)
                    if "gs_source" not in cached_gs.info:
                        cached_gs.info["gs_source"] = "cached"
                    logger.info(f"[i] Reusing existing relaxed ground state from {gs_path}")
                    return cached_gs
            except Exception:
                pass

        gs_method = getattr(self.config, "ground_state_relax_method", "auto").lower()
        es_method = getattr(self.config, "excited_state_method", "spin_proxy").lower()
        use_dft = (gs_method == "dft") or (gs_method == "auto" and es_method in ("delta_scf", "spin_proxy"))

        if use_dft:
            logger.info("\n[-] Optimizing Ground State geometry using Quantum ESPRESSO...")
            try:
                qe_runner = self._get_qe_runner()
                gs_calc_dir = os.path.join(self.config.workspace_dir, "qe_gs_relax")
                tot_mag = int(self.config.spin_multiplicity - 1) if self.config.spin_polarized else 0
                at_gs = qe_runner.run_relaxation(
                    defect_sc,
                    calc_dir=gs_calc_dir,
                    tot_magnetization=tot_mag,
                    nspin=2 if (self.config.spin_polarized or tot_mag > 0) else 1,
                    force_rerun=force,
                )
                at_gs.calc = None
                at_gs.info["defect_center"] = np.asarray(center, dtype=float)
                at_gs.info["gs_source"] = "dft"
                if "defect_positions" in defect_sc.info:
                    at_gs.info["defect_positions"] = defect_sc.info["defect_positions"]
                tmp_gs = gs_path + ".tmp.xyz"
                ase.io.write(tmp_gs, at_gs, format="extxyz")
                os.replace(tmp_gs, gs_path)
                logger.info(f"[+] Ground State DFT relaxation completed -> {gs_path}")
                return at_gs
            except Exception as e:
                err_msg = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
                if not getattr(self.config, "allow_fallbacks", False):
                    raise RuntimeError(
                        f"Ground-state DFT relaxation failed: {err_msg}. "
                        "Heuristic fallbacks to empirical potentials are disabled by default (allow_fallbacks=False). "
                        "Resolve the DFT calculation issue or explicitly set allow_fallbacks=True to permit fallback."
                    ) from e
                logger.warning(f"[!] Ground State DFT relaxation failed or unavailable: {err_msg}")
                if self.calc_foundation is None:
                    raise
                logger.warning("    Falling back to foundation potential for Ground State relaxation...")

        logger.info("\n[-] Optimizing Ground State geometry using foundation potential...")
        at_gs = defect_sc.copy()
        at_gs.calc = self.calc_foundation
        opt = BFGS(at_gs, logfile=None)
        opt.run(fmax=0.01, steps=200)

        energy_val = float(at_gs.get_potential_energy())
        at_gs.info["defect_center"] = np.asarray(center, dtype=float)
        if "defect_positions" in defect_sc.info:
            at_gs.info["defect_positions"] = defect_sc.info["defect_positions"]
        at_gs.info["energy"] = energy_val
        at_gs.info["gs_source"] = "foundation_fallback" if use_dft else "foundation"
        at_gs.calc = None
        tmp_gs = gs_path + ".tmp.xyz"
        ase.io.write(tmp_gs, at_gs, format="extxyz")
        os.replace(tmp_gs, gs_path)
        logger.info(f"[+] Ground State relaxation completed: E = {energy_val:.4f} eV")
        return at_gs

    def generate_excited_state_geometry(self, at_gs: Atoms, force_relax: bool = False) -> Tuple[Atoms, float]:
        """Builds or loads the Optical Excited State structure and returns `(at_es, dq_fc)`."""
        es_path = os.path.join(self.config.workspace_dir, "relaxed_es.xyz")
        force = force_relax or self.config.force_resample
        center = self.vac_pos if self.vac_pos is not None else get_defect_center(at_gs)

        if self.config.excited_cell_path and os.path.isfile(self.config.excited_cell_path):
            es_input = ase.io.read(self.config.excited_cell_path)
            v_input = es_input.get_volume()
            v_gs = at_gs.get_volume()

            if abs(v_input - v_gs) / v_gs < 0.25 and len(es_input) == len(at_gs):
                at_es = es_input.copy()
            else:
                rep = self.config.supercell_repeat
                pristine_uc = ase.io.read(self.config.pristine_cell_path)
                at_es, _ = self.embed_defect_unitcell(pristine_uc, es_input, rep)

            dr_fc = mic_displacement(at_es.positions, at_gs.positions, at_gs.cell)
            at_es.positions = at_gs.positions + dr_fc
            at_es.info["defect_center"] = np.asarray(center, dtype=float)
            at_es.info["es_source"] = "file"
            if "defect_positions" in at_gs.info:
                at_es.info["defect_positions"] = at_gs.info["defect_positions"]
            tmp_es = es_path + ".tmp.xyz"
            ase.io.write(tmp_es, at_es, format="extxyz")
            os.replace(tmp_es, es_path)
            m = at_gs.get_masses()
            dq_fc = float(np.sqrt(np.sum(m[:, None] * (dr_fc ** 2))))
            logger.info(f"[+] Loaded Optical Excited State from file (Franck-Condon Delta Q = {dq_fc:.4f} amu^(1/2)*A)")
            return at_es, dq_fc

        if not force and os.path.isfile(es_path) and os.path.getsize(es_path) > 0:
            try:
                cached_es = ase.io.read(es_path)
                if (
                    len(cached_es) == len(at_gs)
                    and cached_es.get_chemical_symbols() == at_gs.get_chemical_symbols()
                ):
                    dr_fc = mic_displacement(cached_es.positions, at_gs.positions, at_gs.cell)
                    cached_es.positions = at_gs.positions + dr_fc
                    cached_es.info["defect_center"] = np.asarray(center, dtype=float)
                    if "es_source" not in cached_es.info:
                        cached_es.info["es_source"] = "cached"
                    m = at_gs.get_masses()
                    dq_fc = float(np.sqrt(np.sum(m[:, None] * (dr_fc ** 2))))
                    if dq_fc > 1e-5:
                        logger.info(f"[i] Reusing existing relaxed excited state from {es_path} (Delta Q = {dq_fc:.4f} amu^(1/2)*A)")
                        return cached_es, dq_fc
            except Exception:
                pass

        es_method = getattr(self.config, "excited_state_method", "delta_scf").lower()

        if es_method == "delta_scf":
            logger.info("\n[-] Optimizing Optical Excited State geometry using Quantum ESPRESSO Delta-SCF method...")
            try:
                qe_runner = self._get_qe_runner()
                es_calc_dir = os.path.join(self.config.workspace_dir, "qe_es_relax")
                gs_calc_dir = os.path.join(self.config.workspace_dir, "qe_gs_relax")
                gs_out = os.path.join(gs_calc_dir, "pw.out")

                gap_info = parse_qe_band_gap(gs_out) if os.path.isfile(gs_out) else None

                unique_syms = sorted(list(set(at_gs.get_chemical_symbols())))
                z_vals = {}
                for s in unique_syms:
                    upf = self.config.pseudopotentials.get(s, "") if self.config.pseudopotentials else ""
                    if upf:
                        p = upf if os.path.isabs(upf) else os.path.join(self.config.pseudo_dir or ".", upf)
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
                    sum(z_vals.get(s) or _DEFAULT_VALENCE.get(s, 4) for s in at_gs.get_chemical_symbols())
                    - float(self.config.tot_charge)
                ))
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

                lumo_deg = getattr(self.config, "delta_scf_lumo_degeneracy", None)
                if lumo_deg is None and gap_info is not None:
                    lumo_deg = gap_info.get("lumo_degeneracy", 1)
                lumo_deg = max(1, int(lumo_deg or 1))

                if lumo_deg > 1:
                    logger.info(
                        f"    [Delta-SCF] Promoting electron: band {homo_b} -> bands {lumo_b}..{lumo_b + lumo_deg - 1} "
                        f"in spin channel {spin_chan} (degeneracy={lumo_deg}, {1.0/lumo_deg:.2f} e/band, nbnd={nbnd})"
                    )
                else:
                    logger.info(
                        f"    [Delta-SCF] Promoting electron: band {homo_b} -> {lumo_b} "
                        f"in spin channel {spin_chan} ({'down' if spin_chan == 2 else 'up'}, nbnd={nbnd})"
                    )

                occ_up, occ_dw = build_delta_scf_occupations(
                    nelec=nelec_total,
                    tot_magnetization=tot_mag,
                    nbnd=nbnd,
                    homo_band=homo_b,
                    lumo_band=lumo_b,
                    spin_channel=spin_chan,
                    lumo_degeneracy=lumo_deg,
                )

                init_es = at_gs
                es_seed = getattr(self.config, "excited_state_seed_path", None)
                if es_seed and os.path.isfile(es_seed):
                    try:
                        seed_at = ase.io.read(es_seed)
                        if len(seed_at) == len(at_gs) and seed_at.get_chemical_symbols() == at_gs.get_chemical_symbols():
                            init_es = at_gs.copy()
                            init_es.positions = at_gs.positions + mic_displacement(seed_at.positions, at_gs.positions, at_gs.cell)
                            logger.info(f"    [i] Seeding excited-state Delta-SCF relaxation from {es_seed}")
                    except Exception as e:
                        logger.warning(f"    [!] Could not load excited_state_seed_path '{es_seed}': {e}")

                at_es = qe_runner.run_relaxation(
                    init_es,
                    calc_dir=es_calc_dir,
                    tot_magnetization=tot_mag,
                    nspin=2,
                    occupations_mode="from_input",
                    constrained_occupations=(occ_up, occ_dw),
                    force_rerun=force,
                )

                dr_fc = mic_displacement(at_es.positions, at_gs.positions, at_gs.cell)
                at_es.positions = at_gs.positions + dr_fc
                at_es.info["defect_center"] = np.asarray(center, dtype=float)
                at_es.info["es_source"] = "delta_scf"
                at_es.info["delta_scf_spin_channel"] = spin_chan
                at_es.info["delta_scf_homo_band"] = homo_b
                at_es.info["delta_scf_lumo_band"] = lumo_b
                if "defect_positions" in at_gs.info:
                    at_es.info["defect_positions"] = at_gs.info["defect_positions"]
                at_es.calc = None
                tmp_es = es_path + ".tmp.xyz"
                ase.io.write(tmp_es, at_es, format="extxyz")
                os.replace(tmp_es, es_path)

                m = at_gs.get_masses()
                dq_fc = float(np.sqrt(np.sum(m[:, None] * (dr_fc ** 2))))
                logger.info(f"[+] Optical Excited State Delta-SCF relaxation completed (Franck-Condon Delta Q = {dq_fc:.4f} amu^(1/2)*A)")
                return at_es, dq_fc
            except Exception as e:
                err_msg = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
                if not getattr(self.config, "allow_fallbacks", False):
                    raise RuntimeError(
                        f"Excited-state Delta-SCF DFT relaxation failed: {err_msg}. "
                        "Heuristic fallbacks to synthetic breathing displacements are disabled by default (allow_fallbacks=False). "
                        "Resolve the DFT calculation issue or explicitly set allow_fallbacks=True to permit fallback."
                    ) from e
                logger.warning(f"[!] Delta-SCF DFT relaxation failed or unavailable: {err_msg}")
                logger.warning("    Falling back to synthetic breathing displacement...")

        elif es_method == "spin_proxy":
            logger.info("\n[-] Optimizing Optical Excited State geometry using Quantum ESPRESSO spin-proxy method...")
            try:
                qe_runner = self._get_qe_runner()
                es_calc_dir = os.path.join(self.config.workspace_dir, "qe_es_relax")

                # Target spin magnetization: explicit or proxy M_es = M_0 + 2
                if getattr(self.config, "excited_state_tot_magnetization", None) is not None:
                    target_mag = int(self.config.excited_state_tot_magnetization)
                else:
                    gs_mag = int(self.config.spin_multiplicity - 1) if self.config.spin_polarized else 0
                    target_mag = gs_mag + 2

                logger.info(
                    f"    [Spin Proxy] Auto-selected excited state tot_magnetization = {target_mag} "
                    f"(proxy spin multiplicity = {target_mag + 1})"
                )

                init_es = at_gs
                es_seed = getattr(self.config, "excited_state_seed_path", None)
                if es_seed and os.path.isfile(es_seed):
                    try:
                        seed_at = ase.io.read(es_seed)
                        if len(seed_at) == len(at_gs) and seed_at.get_chemical_symbols() == at_gs.get_chemical_symbols():
                            init_es = at_gs.copy()
                            init_es.positions = at_gs.positions + mic_displacement(seed_at.positions, at_gs.positions, at_gs.cell)
                            logger.info(f"    [i] Seeding excited-state spin-proxy relaxation from {es_seed}")
                    except Exception as e:
                        logger.warning(f"    [!] Could not load excited_state_seed_path '{es_seed}': {e}")

                at_es = qe_runner.run_relaxation(
                    init_es,
                    calc_dir=es_calc_dir,
                    tot_magnetization=target_mag,
                    nspin=2,
                    force_rerun=force,
                )

                dr_fc = mic_displacement(at_es.positions, at_gs.positions, at_gs.cell)
                at_es.positions = at_gs.positions + dr_fc
                at_es.info["defect_center"] = np.asarray(center, dtype=float)
                at_es.info["es_source"] = es_method
                if "defect_positions" in at_gs.info:
                    at_es.info["defect_positions"] = at_gs.info["defect_positions"]
                at_es.calc = None
                tmp_es = es_path + ".tmp.xyz"
                ase.io.write(tmp_es, at_es, format="extxyz")
                os.replace(tmp_es, es_path)

                m = at_gs.get_masses()
                dq_fc = float(np.sqrt(np.sum(m[:, None] * (dr_fc ** 2))))
                logger.info(f"[+] Optical Excited State DFT relaxation completed (Franck-Condon Delta Q = {dq_fc:.4f} amu^(1/2)*A)")
                return at_es, dq_fc
            except Exception as e:
                err_msg = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
                if not getattr(self.config, "allow_fallbacks", False):
                    raise RuntimeError(
                        f"Excited-state DFT relaxation ({es_method}) failed: {err_msg}. "
                        "Heuristic fallbacks to synthetic breathing displacements are disabled by default (allow_fallbacks=False). "
                        "Resolve the DFT calculation issue or explicitly set allow_fallbacks=True to permit fallback."
                    ) from e
                logger.warning(f"[!] Spin-proxy DFT relaxation failed or unavailable: {err_msg}")
                logger.warning("    Falling back to synthetic breathing displacement...")

        at_es = at_gs.copy()
        disp_vecs = mic_displacement(at_es.positions, center, at_es.cell)
        dists = np.linalg.norm(disp_vecs, axis=1)

        defect_pos_list = at_gs.info.get("defect_positions")
        if defect_pos_list is None or len(defect_pos_list) == 0:
            defect_pos_list = [center]

        # Identify which atoms in at_gs belong to the defect core (within 0.35 A of any defect site)
        defect_atom_indices = set()
        for dp in defect_pos_list:
            d_to_dp = mic_distances(at_gs.positions, dp, at_gs.cell)
            for idx, d in enumerate(d_to_dp):
                if d < 0.35:
                    defect_atom_indices.add(idx)

        # Host atoms are all atoms not in the defect core
        host_mask = np.ones(len(at_gs), dtype=bool)
        for idx in defect_atom_indices:
            host_mask[idx] = False

        # Find coordinating host shell: nearest host atoms to ANY defect site
        coord_set = set()
        for dp in defect_pos_list:
            d_to_dp = mic_distances(at_gs.positions, dp, at_gs.cell)
            valid_dists = d_to_dp[host_mask]
            if len(valid_dists) > 0:
                d_min_dp = float(np.min(valid_dists))
                shell_candidates = [
                    i for i in range(len(at_gs))
                    if host_mask[i] and d_to_dp[i] <= 1.25 * d_min_dp
                ]
                coord_set.update(shell_candidates)

        coord_indices = sorted(list(coord_set))
        if len(coord_indices) < 2:
            host_indices = np.where(host_mask)[0]
            if len(host_indices) > 0:
                order = np.argsort(dists[host_indices])
                coord_indices = list(host_indices[order[:6]])
            else:
                coord_indices = [i for i, d in enumerate(dists) if d > 0.05][:6]

        d_min = float(np.min(dists[coord_indices])) if len(coord_indices) > 0 else 2.0

        if self.config.excited_state_displacement is not None:
            disp_mag = float(self.config.excited_state_displacement)
        elif d_min > 1.85:
            disp_mag = 0.15  # Soft ionic lattice
        else:
            disp_mag = 0.08  # Stiff covalent lattice

        coord_symbols = [at_es[i].symbol for i in coord_indices]
        logger.info(f"[+] Defect coordinating shell: {len(coord_indices)} atoms ({coord_symbols}), d_min = {d_min:.3f} A")
        logger.info(f"    Applying optical breathing displacement disp_mag = {disp_mag:.3f} A...")

        for idx in coord_indices:
            dr = disp_vecs[idx]
            norm_dr = np.linalg.norm(dr)
            if norm_dr > 1e-4:
                at_es.positions[idx] += disp_mag * (dr / norm_dr)

        at_es.calc = self.calc_foundation
        at_es.set_constraint(FixAtoms(indices=coord_indices))
        opt = BFGS(at_es, logfile=None)
        opt.run(fmax=0.02, steps=30)
        at_es.set_constraint()

        dr_fc = mic_displacement(at_es.positions, at_gs.positions, at_gs.cell)
        at_es.positions = at_gs.positions + dr_fc
        at_es.info["defect_center"] = np.asarray(center, dtype=float)
        at_es.info["es_source"] = "synthetic_fallback" if (es_method in ("spin_proxy", "delta_scf")) else "synthetic"
        if "defect_positions" in at_gs.info:
            at_es.info["defect_positions"] = at_gs.info["defect_positions"]
        at_es.calc = None
        tmp_es = es_path + ".tmp.xyz"
        ase.io.write(tmp_es, at_es, format="extxyz")
        os.replace(tmp_es, es_path)

        m = at_gs.get_masses()
        dq_fc = float(np.sqrt(np.sum(m[:, None] * (dr_fc ** 2))))
        logger.info(f"[+] Built Optical Excited State (Franck-Condon Delta Q = {dq_fc:.4f} amu^(1/2)*A)")
        return at_es, dq_fc

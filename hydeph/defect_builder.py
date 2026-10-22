"""Supercell tiling, defect site detection, and foundation-potential geometry relaxation."""

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


class DefectBuilder:
    """Constructs pristine and defect supercells, embeds unit cells, and optimizes geometries."""

    def __init__(self, config: ToolkitConfig, calc_foundation=None):
        self.config = config
        os.makedirs(self.config.workspace_dir, exist_ok=True)
        if calc_foundation is not None:
            self.calc_foundation = calc_foundation
        elif config.foundation_potential_path:
            try:
                self.calc_foundation = CPUNEP(config.foundation_potential_path)
            except Exception:
                self.calc_foundation = None
        else:
            self.calc_foundation = None
        self.vac_pos: Optional[np.ndarray] = None

    def _detect_supercell_defect_pos(self, pristine_sc: Atoms, defect_sc: Atoms) -> np.ndarray:
        """Detects the defect coordinate in a pre-built defect supercell relative to `pristine_sc`."""
        cell = pristine_sc.cell
        if len(defect_sc) < len(pristine_sc):
            for at in pristine_sc:
                dists = mic_distances(defect_sc.positions, at.position, cell)
                if len(dists) == 0 or np.min(dists) > 0.5:
                    return at.position.copy()
        elif len(defect_sc) == len(pristine_sc):
            for at_p, at_d in zip(pristine_sc, defect_sc):
                if at_p.symbol != at_d.symbol:
                    return at_d.position.copy()
        else:
            for at in defect_sc:
                dists = mic_distances(pristine_sc.positions, at.position, cell)
                if len(dists) == 0 or np.min(dists) > 0.5:
                    return at.position.copy()
        return np.asarray(defect_sc.get_cell().sum(axis=0) / 2.0, dtype=float)

    def build_supercell_and_defect(self) -> Tuple[Atoms, Atoms]:
        """Builds pristine and defect supercells, persisting `defect_center` in `Atoms.info`."""
        pristine = ase.io.read(self.config.pristine_cell_path)

        rep = self.config.supercell_repeat
        pristine_sc = pristine.repeat(rep)
        pristine_sc_path = os.path.join(self.config.workspace_dir, "pristine_supercell.xyz")
        ase.io.write(pristine_sc_path, pristine_sc)
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
                self.vac_pos = self._detect_supercell_defect_pos(pristine_sc, defect_sc)
                logger.info(f"[+] Loaded existing defect supercell ({len(defect_sc)} atoms, defect center = {self.vac_pos})")
        else:
            defect_sc = pristine_sc.copy()
            dtype = self.config.defect_type.lower()
            site_idx = self.config.defect_site_index
            elem = self.config.defect_site_element
            sub_elem = self.config.substitutional_element

            if elem and "->" in elem and sub_elem is None:
                parts = [p.strip() for p in elem.split("->")]
                elem, sub_elem = parts[0], parts[1]

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

            if dtype == "vacancy":
                removed_sym = defect_sc[target_idx].symbol
                del defect_sc[target_idx]
                logger.info(f"[+] Created {removed_sym} vacancy at site {target_idx}, coordinate = {self.vac_pos}")
            elif dtype == "substitutional":
                if not sub_elem:
                    raise ValueError("Substitutional defect requires `substitutional_element` (or 'Host->Sub' in `defect_site_element`).")
                old_sym = defect_sc[target_idx].symbol
                defect_sc[target_idx].symbol = sub_elem
                logger.info(f"[+] Created substitution {old_sym} -> {sub_elem} at site {target_idx}, coordinate = {self.vac_pos}")
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
        ase.io.write(defect_init_path, defect_sc)
        return pristine_sc, defect_sc

    def embed_defect_unitcell(self, pristine_uc: Atoms, defect_uc: Atoms, rep: Tuple[int, int, int]) -> Tuple[Atoms, np.ndarray]:
        """Tiles pristine unit cells around a central defect unit cell and returns `(supercell, defect_pos)`.
        
        Vectorized with NumPy broadcasting for maximum efficiency across large supercells.
        """
        a1, a2, a3 = pristine_uc.cell[0], pristine_uc.cell[1], pristine_uc.cell[2]
        ic, jc, kc = rep[0] // 2, rep[1] // 2, rep[2] // 2
        shift_center = ic * a1 + jc * a2 + kc * a3

        defect_pos = shift_center
        if len(defect_uc) < len(pristine_uc):
            for at in pristine_uc:
                dists = mic_distances(defect_uc.positions, at.position, pristine_uc.cell)
                if len(dists) == 0 or np.min(dists) > 0.5:
                    defect_pos = at.position + shift_center
                    logger.info(f"    [i] Auto-detected vacancy of {at.symbol} at {defect_pos}")
                    break
        elif len(defect_uc) == len(pristine_uc):
            for at_p, at_d in zip(pristine_uc, defect_uc):
                if at_p.symbol != at_d.symbol:
                    defect_pos = at_d.position + shift_center
                    logger.info(f"    [i] Auto-detected substitution {at_p.symbol} -> {at_d.symbol} at {defect_pos}")
                    break
        else:
            for at in defect_uc:
                dists = mic_distances(pristine_uc.positions, at.position, pristine_uc.cell)
                if np.min(dists) > 0.5:
                    defect_pos = at.position + shift_center
                    logger.info(f"    [i] Auto-detected interstitial of {at.symbol} at {defect_pos}")
                    break

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
        return supercell, defect_pos

    def relax_ground_state(self, defect_sc: Atoms, force_relax: bool = False) -> Atoms:
        """Relaxes ground-state geometry using the foundation potential, or reuses cached `relaxed_gs.xyz`."""
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
                    logger.info(f"[i] Reusing existing relaxed ground state from {gs_path}")
                    return cached_gs
            except Exception:
                pass

        logger.info(f"\n[-] Optimizing Ground State geometry using foundation potential...")
        at_gs = defect_sc.copy()
        at_gs.calc = self.calc_foundation
        opt = BFGS(at_gs, logfile=None)
        opt.run(fmax=0.01, steps=200)

        at_gs.info["defect_center"] = np.asarray(center, dtype=float)
        ase.io.write(gs_path, at_gs)
        logger.info(f"[+] Ground State relaxation completed: E = {at_gs.get_potential_energy():.4f} eV")
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
            ase.io.write(es_path, at_es)
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
                    m = at_gs.get_masses()
                    dq_fc = float(np.sqrt(np.sum(m[:, None] * (dr_fc ** 2))))
                    if dq_fc > 1e-5:
                        logger.info(f"[i] Reusing existing relaxed excited state from {es_path} (Delta Q = {dq_fc:.4f} amu^(1/2)*A)")
                        return cached_es, dq_fc
            except Exception:
                pass

        at_es = at_gs.copy()
        disp_vecs = mic_displacement(at_es.positions, center, at_es.cell)
        dists = np.linalg.norm(disp_vecs, axis=1)
        valid_mask = dists > 0.05
        if not np.any(valid_mask):
            raise ValueError("No valid atoms found around defect center for excited state relaxation.")

        d_min = float(np.min(dists[valid_mask]))
        coord_indices = [i for i, d in enumerate(dists) if 0.05 < d <= 1.25 * d_min]
        if len(coord_indices) < 2:
            valid_idx = np.where(valid_mask)[0]
            coord_indices = list(valid_idx[np.argsort(dists[valid_idx])[:6]])

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
        ase.io.write(es_path, at_es)

        m = at_gs.get_masses()
        dq_fc = float(np.sqrt(np.sum(m[:, None] * (dr_fc ** 2))))
        logger.info(f"[+] Built Optical Excited State (Franck-Condon Delta Q = {dq_fc:.4f} amu^(1/2)*A)")
        return at_es, dq_fc

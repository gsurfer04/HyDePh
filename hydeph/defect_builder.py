"""Supercell tiling, defect site detection, and foundation-potential geometry relaxation."""

import os
import numpy as np
import ase.io
from ase import Atoms
from ase.geometry import find_mic
from ase.optimize import BFGS
from calorine.calculators import CPUNEP


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
    def __init__(self, config, calc_foundation=None):
        self.config = config
        os.makedirs(self.config.workspace_dir, exist_ok=True)
        if calc_foundation is not None:
            self.calc_foundation = calc_foundation
        elif getattr(config, "foundation_potential_path", None):
            try:
                self.calc_foundation = CPUNEP(config.foundation_potential_path)
            except Exception:
                self.calc_foundation = None
        else:
            self.calc_foundation = None
        self.vac_pos = None

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

    def build_supercell_and_defect(self):
        """Builds pristine and defect supercells, persisting `defect_center` in `Atoms.info`."""
        pristine = ase.io.read(self.config.pristine_cell_path)

        rep = self.config.supercell_repeat
        pristine_sc = pristine.repeat(rep)
        pristine_sc_path = os.path.join(self.config.workspace_dir, "pristine_supercell.xyz")
        ase.io.write(pristine_sc_path, pristine_sc)
        print(f"[+] Built pristine supercell ({len(pristine_sc)} atoms, repeat={rep})")

        if self.config.defect_cell_path and os.path.isfile(self.config.defect_cell_path):
            defect_input = ase.io.read(self.config.defect_cell_path)
            v_input = defect_input.get_volume()
            v_pristine_uc = pristine.get_volume()

            if abs(v_input - v_pristine_uc) / v_pristine_uc < 0.25:
                print(f"[+] Pre-made defect unit cell provided ({len(defect_input)} atoms).")
                print(f"    Surrounding with pristine unit cells to build {rep} supercell...")
                defect_sc, detected_pos = self.embed_defect_unitcell(pristine, defect_input, rep)
                self.vac_pos = np.asarray(detected_pos, dtype=float)
                print(f"[+] Defect embedded at {self.vac_pos}; resulting supercell has {len(defect_sc)} atoms.")
            else:
                defect_sc = defect_input
                self.vac_pos = self._detect_supercell_defect_pos(pristine_sc, defect_sc)
                print(f"[+] Loaded existing defect supercell ({len(defect_sc)} atoms, defect center = {self.vac_pos})")
        else:
            defect_sc = pristine_sc.copy()
            dtype = getattr(self.config, "defect_type", "vacancy").lower()
            site_idx = getattr(self.config, "defect_site_index", None)
            elem = getattr(self.config, "defect_site_element", None)
            sub_elem = getattr(self.config, "substitutional_element", None)

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
                print(f"[+] Created {removed_sym} vacancy at site {target_idx}, coordinate = {self.vac_pos}")
            elif dtype == "substitutional":
                if not sub_elem:
                    raise ValueError("Substitutional defect requires `substitutional_element` (or 'Host->Sub' in `defect_site_element`).")
                old_sym = defect_sc[target_idx].symbol
                defect_sc[target_idx].symbol = sub_elem
                print(f"[+] Created substitution {old_sym} -> {sub_elem} at site {target_idx}, coordinate = {self.vac_pos}")
            else:
                raise NotImplementedError(f"Defect type '{self.config.defect_type}' not supported.")

        defect_sc.info["defect_center"] = np.asarray(self.vac_pos, dtype=float)
        defect_init_path = os.path.join(self.config.workspace_dir, "defect_unrelaxed.xyz")
        ase.io.write(defect_init_path, defect_sc)
        return pristine_sc, defect_sc

    def embed_defect_unitcell(self, pristine_uc, defect_uc, rep):
        """Tiles pristine unit cells around a central defect unit cell and returns `(supercell, defect_pos)`."""
        a1, a2, a3 = pristine_uc.cell[0], pristine_uc.cell[1], pristine_uc.cell[2]
        ic, jc, kc = rep[0] // 2, rep[1] // 2, rep[2] // 2
        shift_center = ic * a1 + jc * a2 + kc * a3

        defect_pos = shift_center
        if len(defect_uc) < len(pristine_uc):
            for at in pristine_uc:
                dists = mic_distances(defect_uc.positions, at.position, pristine_uc.cell)
                if len(dists) == 0 or np.min(dists) > 0.5:
                    defect_pos = at.position + shift_center
                    print(f"    [i] Auto-detected vacancy of {at.symbol} at {defect_pos}")
                    break
        elif len(defect_uc) == len(pristine_uc):
            for at_p, at_d in zip(pristine_uc, defect_uc):
                if at_p.symbol != at_d.symbol:
                    defect_pos = at_d.position + shift_center
                    print(f"    [i] Auto-detected substitution {at_p.symbol} -> {at_d.symbol} at {defect_pos}")
                    break
        else:
            for at in defect_uc:
                dists = mic_distances(pristine_uc.positions, at.position, pristine_uc.cell)
                if np.min(dists) > 0.5:
                    defect_pos = at.position + shift_center
                    print(f"    [i] Auto-detected interstitial of {at.symbol} at {defect_pos}")
                    break

        all_positions = []
        all_symbols = []
        for i in range(rep[0]):
            for j in range(rep[1]):
                for k in range(rep[2]):
                    shift = i * a1 + j * a2 + k * a3
                    uc = defect_uc if (i, j, k) == (ic, jc, kc) else pristine_uc
                    for at in uc:
                        all_positions.append(at.position + shift)
                        all_symbols.append(at.symbol)

        cell = [rep[0] * a1, rep[1] * a2, rep[2] * a3]
        supercell = Atoms(symbols=all_symbols, positions=all_positions, cell=cell, pbc=True)
        supercell.info["defect_center"] = np.asarray(defect_pos, dtype=float)
        return supercell, defect_pos

    def relax_ground_state(self, defect_sc, force_relax=False):
        """Relaxes ground-state geometry using the foundation potential, or reuses cached `relaxed_gs.xyz`."""
        gs_path = os.path.join(self.config.workspace_dir, "relaxed_gs.xyz")
        force = force_relax or getattr(self.config, "force_resample", False)
        center = self.vac_pos if self.vac_pos is not None else get_defect_center(defect_sc)

        if not force and os.path.isfile(gs_path) and os.path.getsize(gs_path) > 0:
            try:
                cached_gs = ase.io.read(gs_path)
                if (
                    len(cached_gs) == len(defect_sc)
                    and cached_gs.get_chemical_symbols() == defect_sc.get_chemical_symbols()
                ):
                    cached_gs.info["defect_center"] = np.asarray(center, dtype=float)
                    print(f"[i] Reusing existing relaxed ground state from {gs_path}")
                    return cached_gs
            except Exception:
                pass

        at = defect_sc.copy()
        at.calc = self.calc_foundation
        opt = BFGS(at, logfile=None)
        opt.run(fmax=0.01, steps=100)
        e_gs = at.get_potential_energy()
        f_max = np.abs(at.get_forces()).max()

        at.info["defect_center"] = np.asarray(center, dtype=float)
        ase.io.write(gs_path, at)
        print(f"[+] Relaxed Ground State (nep89 E = {e_gs:.4f} eV, max F = {f_max:.4e} eV/A)")
        return at

    def generate_excited_state_geometry(self, at_gs, force_relax=False):
        """Loads `excited_cell_path` if supplied, or applies PBC-safe outward breathing to the first coordination shell."""
        es_path = os.path.join(self.config.workspace_dir, "relaxed_es.xyz")
        force = force_relax or getattr(self.config, "force_resample", False)
        center = self.vac_pos if self.vac_pos is not None else get_defect_center(at_gs)

        if getattr(self.config, "excited_cell_path", None) and os.path.isfile(self.config.excited_cell_path):
            excited_input = ase.io.read(self.config.excited_cell_path)
            v_input = excited_input.get_volume()
            pristine = ase.io.read(self.config.pristine_cell_path)
            v_pristine_uc = pristine.get_volume()

            if abs(v_input - v_pristine_uc) / v_pristine_uc < 0.25:
                print(f"[+] Pre-made excited state unit cell provided ({len(excited_input)} atoms).")
                print(f"    Surrounding with pristine unit cells to build {self.config.supercell_repeat} supercell...")
                at_es, _ = self.embed_defect_unitcell(pristine, excited_input, self.config.supercell_repeat)
            else:
                at_es = excited_input
                print(f"[+] Loaded existing excited state supercell ({len(at_es)} atoms)")

            # Unwrap excited-state positions relative to at_gs using minimum-image convention
            dr_fc = mic_displacement(at_es.positions, at_gs.positions, at_gs.cell)
            at_es.positions = at_gs.positions + dr_fc
            at_es.info["defect_center"] = np.asarray(center, dtype=float)
            ase.io.write(es_path, at_es)
            m = at_gs.get_masses()
            dq_fc = float(np.sqrt(np.sum(m[:, None] * (dr_fc ** 2))))
            print(f"[+] Loaded Optical Excited State from file (Franck-Condon Delta Q = {dq_fc:.4f} amu^(1/2)*A)")
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
                        print(f"[i] Reusing existing relaxed excited state from {es_path} (Delta Q = {dq_fc:.4f} amu^(1/2)*A)")
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

        if getattr(self.config, "excited_state_displacement", None) is not None:
            disp_mag = float(self.config.excited_state_displacement)
        elif d_min > 1.85:
            disp_mag = 0.15  # Soft ionic lattice
        else:
            disp_mag = 0.08  # Stiff covalent lattice

        coord_symbols = [at_es[i].symbol for i in coord_indices]
        print(f"[+] Defect coordinating shell: {len(coord_indices)} atoms ({coord_symbols}), d_min = {d_min:.3f} A")
        print(f"    Applying optical breathing displacement disp_mag = {disp_mag:.3f} A...")

        for idx in coord_indices:
            dr = disp_vecs[idx]
            norm_dr = np.linalg.norm(dr)
            if norm_dr > 1e-4:
                at_es.positions[idx] += disp_mag * (dr / norm_dr)

        from ase.constraints import FixAtoms
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
        print(f"[+] Built Optical Excited State (Franck-Condon Delta Q = {dq_fc:.4f} amu^(1/2)*A)")
        return at_es, dq_fc


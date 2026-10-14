"""
hydeph.defect_builder
=====================
Handles supercell expansion, color center defect creation (vacancies,
substitutions, complexes), coordination shell identification, and initial
geometry optimization using the foundation potential (nep89).
"""

import os
import numpy as np
import ase.io
from ase import Atoms
from ase.optimize import BFGS
from calorine.calculators import CPUNEP


class DefectBuilder:
    def __init__(self, config):
        self.config = config
        os.makedirs(self.config.workspace_dir, exist_ok=True)
        self.calc_foundation = CPUNEP(config.foundation_potential_path)

    def build_supercell_and_defect(self):
        """Builds pristine supercell and defect supercell."""
        # 1. Load pristine cell
        pristine = ase.io.read(self.config.pristine_cell_path)
        
        # 2. Expand supercell
        rep = self.config.supercell_repeat
        pristine_sc = pristine.repeat(rep)
        pristine_sc_path = os.path.join(self.config.workspace_dir, "pristine_supercell.xyz")
        ase.io.write(pristine_sc_path, pristine_sc)
        print(f"[+] Built pristine supercell ({len(pristine_sc)} atoms, repeat={rep})")

        # 3. Create or load defect cell
        if self.config.defect_cell_path and os.path.isfile(self.config.defect_cell_path):
            defect_input = ase.io.read(self.config.defect_cell_path)
            v_input = defect_input.get_volume()
            v_pristine_uc = pristine.get_volume()
            
            # Check if defect_input is a defect unit cell that should be surrounded by pristine cells
            if abs(v_input - v_pristine_uc) / v_pristine_uc < 0.25:
                print(f"[+] Pre-made defect unit cell provided ({len(defect_input)} atoms).")
                print(f"    Surrounding with pristine unit cells to build {rep} supercell...")
                defect_sc, detected_pos = self.embed_defect_unitcell(pristine, defect_input, rep)
                self.vac_pos = detected_pos
                print(f"[+] Defect embedded at {self.vac_pos}; resulting supercell has {len(defect_sc)} atoms.")
            else:
                defect_sc = defect_input
                print(f"[+] Loaded existing defect supercell ({len(defect_sc)} atoms)")
        else:
            defect_sc = pristine_sc.copy()
            if self.config.defect_type == "vacancy":
                elem = self.config.defect_site_element
                cand_indices = [i for i, s in enumerate(defect_sc.get_chemical_symbols()) if s == elem]
                if not cand_indices:
                    raise ValueError(f"Element {elem} not found in pristine supercell for vacancy creation.")
                
                # Pick candidate closest to cell geometric center
                center = defect_sc.get_cell().sum(axis=0) / 2.0
                dists = [np.linalg.norm(defect_sc.positions[i] - center) for i in cand_indices]
                vac_idx = cand_indices[np.argmin(dists)]
                vac_pos = defect_sc.positions[vac_idx].copy()
                self.vac_pos = vac_pos
                del defect_sc[vac_idx]
                print(f"[+] Created {elem} vacancy (F-center) at site {vac_idx}, vacancy coordinate = {vac_pos}")
            else:
                raise NotImplementedError(f"Defect type {self.config.defect_type} not yet implemented.")

        defect_init_path = os.path.join(self.config.workspace_dir, "defect_unrelaxed.xyz")
        ase.io.write(defect_init_path, defect_sc)
        return pristine_sc, defect_sc

    def embed_defect_unitcell(self, pristine_uc, defect_uc, rep):
        """Embeds a pre-made defect unit cell into the center block of a pristine supercell."""
        a1, a2, a3 = pristine_uc.cell[0], pristine_uc.cell[1], pristine_uc.cell[2]
        ic, jc, kc = rep[0] // 2, rep[1] // 2, rep[2] // 2
        shift_center = ic * a1 + jc * a2 + kc * a3

        defect_pos = shift_center
        # Detect defect site by comparing pristine_uc and defect_uc
        if len(defect_uc) < len(pristine_uc):
            for at in pristine_uc:
                dists = np.linalg.norm(defect_uc.positions - at.position, axis=1)
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
                dists = np.linalg.norm(pristine_uc.positions - at.position, axis=1)
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

        cell = [rep[0]*a1, rep[1]*a2, rep[2]*a3]
        supercell = Atoms(symbols=all_symbols, positions=all_positions, cell=cell, pbc=True)
        return supercell, defect_pos

    def relax_ground_state(self, defect_sc):
        """Relaxes ground-state geometry using foundation potential."""
        at = defect_sc.copy()
        at.calc = self.calc_foundation
        opt = BFGS(at, logfile=None)
        opt.run(fmax=0.01, steps=100)
        e_gs = at.get_potential_energy()
        f_max = np.abs(at.get_forces()).max()
        
        gs_path = os.path.join(self.config.workspace_dir, "relaxed_gs.xyz")
        ase.io.write(gs_path, at)
        print(f"[+] Relaxed Ground State (nep89 E = {e_gs:.4f} eV, max F = {f_max:.4e} eV/A)")
        return at

    def generate_excited_state_geometry(self, at_gs):
        """
        Generates or loads the optical excited-state equilibrium geometry.
        Supports explicit user-supplied excited-state structure (excited_cell_path)
        or procedural defect coordination shell breathing/Jahn-Teller displacement.
        """
        # Option A: User-supplied excited state cell/supercell
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
                
            es_path = os.path.join(self.config.workspace_dir, "relaxed_es.xyz")
            ase.io.write(es_path, at_es)
            dr_fc = at_es.positions - at_gs.positions
            m = at_gs.get_masses()
            dq_fc = np.sqrt(np.sum(np.repeat(m, 3) * dr_fc.flatten()**2))
            print(f"[+] Loaded Optical Excited State from file (Franck-Condon Delta Q = {dq_fc:.4f} amu^(1/2)*A)")
            return at_es, dq_fc

        # Option B: Automated procedural defect core relaxation
        at_es = at_gs.copy()
        center = self.vac_pos if hasattr(self, 'vac_pos') and self.vac_pos is not None else (at_gs.get_cell().sum(axis=0) / 2.0)
        
        # Identify coordinating atoms in first coordination shell around defect center
        dists = np.linalg.norm(at_es.positions - center, axis=1)
        valid_mask = dists > 0.05
        if not np.any(valid_mask):
            raise ValueError("No valid atoms found around defect center for excited state relaxation.")
        
        d_min = np.min(dists[valid_mask])
        # First coordination shell: all atoms within 1.25 * d_min (e.g. 6 cations in rock-salt, 4 C/N in diamond)
        coord_indices = [i for i, d in enumerate(dists) if 0.05 < d <= 1.25 * d_min]
        if len(coord_indices) < 2:
            coord_indices = list(np.argsort(dists[valid_mask])[:6])
            
        # Physical displacement magnitude:
        # Default ~0.08 A for stiff covalent hosts (diamond, SiC), ~0.15 A for softer ionic lattices (alkali halides)
        if getattr(self.config, "excited_state_displacement", None) is not None:
            disp_mag = float(self.config.excited_state_displacement)
        elif d_min > 1.85:
            disp_mag = 0.15  # Soft ionic lattice (e.g. LiCl, LiF, NaCl)
        else:
            disp_mag = 0.08  # Stiff covalent lattice (e.g. Diamond NV, SiV, SiC)

        coord_symbols = [at_es[i].symbol for i in coord_indices]
        print(f"[+] Defect coordinating shell: {len(coord_indices)} atoms ({coord_symbols}), d_min = {d_min:.3f} A")
        print(f"    Applying optical breathing displacement disp_mag = {disp_mag:.3f} A...")
        
        for idx in coord_indices:
            dr = at_es.positions[idx] - center
            norm_dr = np.linalg.norm(dr)
            if norm_dr > 1e-4:
                unit_dr = dr / norm_dr
                at_es.positions[idx] += disp_mag * unit_dr

        # Refine outer shell with foundation potential while constraining the breathing core
        from ase.constraints import FixAtoms
        at_es.calc = self.calc_foundation
        at_es.set_constraint(FixAtoms(indices=coord_indices))
        opt = BFGS(at_es, logfile=None)
        opt.run(fmax=0.02, steps=30)
        at_es.set_constraint()
        
        e_es = at_es.get_potential_energy()
        es_path = os.path.join(self.config.workspace_dir, "relaxed_es.xyz")
        ase.io.write(es_path, at_es)
        
        dr_fc = at_es.positions - at_gs.positions
        m = at_gs.get_masses()
        dq_fc = np.sqrt(np.sum(np.repeat(m, 3) * dr_fc.flatten()**2))
        print(f"[+] Built Optical Excited State (Franck-Condon Delta Q = {dq_fc:.4f} amu^(1/2)*A)")
        return at_es, dq_fc


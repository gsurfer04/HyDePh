"""
hydeph.frame_sampler
====================
Implements the 4-Phase representative frame generation protocol:
  1. Static Franck-Condon Reaction Coordinate Scan (lambda in [-0.2, 1.2])
  2. Defect-Core Local Rattling & Frozen Phonon Displacements
  3. Foundation-Potential Molecular Dynamics (NVT) + Farthest Point Sampling (FPS)
"""

import os
import time
import numpy as np
import ase.io
from ase import Atoms
from calorine.calculators import CPUNEP


def farthest_point_sampling(descriptor_matrix, n_select):
    """
    Selects n_select maximally diverse configurations using Farthest Point Sampling.
    descriptor_matrix: shape (N_frames, N_features)
    """
    n_frames = descriptor_matrix.shape[0]
    if n_select >= n_frames:
        return list(range(n_frames))
    
    # Start with frame having maximum norm
    norms = np.linalg.norm(descriptor_matrix, axis=1)
    selected = [np.argmax(norms)]
    min_dists = np.linalg.norm(descriptor_matrix - descriptor_matrix[selected[0]], axis=1)
    
    for _ in range(1, n_select):
        next_idx = np.argmax(min_dists)
        selected.append(next_idx)
        new_dists = np.linalg.norm(descriptor_matrix - descriptor_matrix[next_idx], axis=1)
        min_dists = np.minimum(min_dists, new_dists)
        
    return selected


class FrameSampler:
    def __init__(self, config):
        self.config = config
        if getattr(config, "foundation_potential_path", None) and os.path.isfile(config.foundation_potential_path):
            self.calc_foundation = CPUNEP(config.foundation_potential_path)
        else:
            self.calc_foundation = None

    def sample_franck_condon_path(self, at_gs, at_es, n_points=7):
        """Phase 1: Generates structures along linear optical coordinate."""
        dr = at_es.positions - at_gs.positions
        # Guarantee 0.0 and 1.0 are present
        base_lams = [0.0, 1.0, 0.5, -0.2, 1.2]
        extra_lams = list(np.linspace(-0.1, 1.1, max(0, n_points - len(base_lams))))
        lambdas = base_lams + extra_lams
        
        fc_frames = []
        for i, lam in enumerate(lambdas[:n_points]):
            at = at_gs.copy()
            at.positions = at_gs.positions + lam * dr
            label = f"fc_{i:02d}"
            at.info["sample_type"] = "franck_condon"
            at.info["frame_id"] = label
            at.info["label"] = label
            at.info["lambda"] = float(lam)
            fc_frames.append(at)
        print(f"[+] Phase 1: Generated {len(fc_frames)} Franck-Condon path frames (lambda in [-0.2, 1.2])")
        return fc_frames

    def sample_defect_core_rattling(self, at_gs, n_frames=12, r_cutoff=6.0):
        """Phase 2: Generates defect-core localized rattle configurations."""
        center = at_gs.get_cell().sum(axis=0) / 2.0
        dists = np.linalg.norm(at_gs.positions - center, axis=1)
        core_mask = dists < r_cutoff
        
        seed = getattr(self.config, "random_seed", 42)
        rng = np.random.RandomState(seed)
        
        rattle_frames = []
        for i in range(n_frames):
            at = at_gs.copy()
            sigma = 0.08 if i % 2 == 0 else 0.14
            displacements = rng.normal(0.0, sigma, at.positions.shape)
            displacements[~core_mask] *= 0.15
            at.positions += displacements
            label = f"rattle_{i:02d}"
            at.info["sample_type"] = "core_rattle"
            at.info["frame_id"] = label
            at.info["label"] = label
            at.info["rattle_sigma"] = float(sigma)
            rattle_frames.append(at)
            
        print(f"[+] Phase 2: Generated {len(rattle_frames)} defect-core rattle frames (r < {r_cutoff} A)")
        return rattle_frames

    def sample_foundation_md_with_fps(self, at_gs, n_select=20, n_steps=2000, dt_fs=1.0):
        """
        Phase 3: Fast NVT MD with foundation potential (nep89) followed by
        Farthest Point Sampling (FPS) in atomic descriptor space.
        """
        scratch_dir = os.path.join(self.config.workspace_dir, "sampler_md")
        os.makedirs(scratch_dir, exist_ok=True)
        
        from ase.md.langevin import Langevin
        from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
        from ase import units
        
        seed = getattr(self.config, "random_seed", 42) + 101
        at = at_gs.copy()
        at.calc = self.calc_foundation
        temp_k = self.config.md_temp_exploration
        MaxwellBoltzmannDistribution(at, temperature_K=temp_k, rng=np.random.RandomState(seed))
        dyn = Langevin(at, dt_fs * units.fs, temperature_K=temp_k, friction=0.01 / units.fs, logfile=None)
        
        trajectory = []
        def record_frame():
            trajectory.append(at.copy())
            
        dyn.attach(record_frame, interval=10)
        dyn.run(n_steps)
        
        print(f"[+] Phase 3: Propagated {len(trajectory)} MD trajectory frames at {temp_k} K using nep89")
        
        center = at_gs.get_cell().sum(axis=0) / 2.0
        n_frames = len(trajectory)
        features = []
        for frame in trajectory:
            dists = np.linalg.norm(frame.positions - center, axis=1)
            core_dists = np.sort(dists)[:18] # 18 nearest neighbors
            features.append(core_dists)
        feature_matrix = np.array(features)
        
        selected_indices = farthest_point_sampling(feature_matrix, n_select)
        selected_frames = []
        for i, idx in enumerate(selected_indices):
            fr = trajectory[idx]
            label = f"md_fps_{i:02d}"
            fr.info["sample_type"] = "foundation_md_fps"
            fr.info["frame_id"] = label
            fr.info["label"] = label
            selected_frames.append(fr)
            
        print(f"[+] Phase 3: Farthest Point Sampling selected {len(selected_frames)} maximally diverse frames")
        return selected_frames

    def assemble_candidate_dataset(self, at_gs, at_es, force_resample=False):
        """Assembles the complete candidate dataset across all sampling phases, reusing existing pool if present."""
        out_path = os.path.join(self.config.workspace_dir, "candidate_frames.xyz")
        force = force_resample or getattr(self.config, "force_resample", False)
        
        if not force and os.path.isfile(out_path) and os.path.getsize(out_path) > 0:
            try:
                cached_candidates = ase.io.read(out_path, ":")
                if len(cached_candidates) > 0:
                    for idx, fr in enumerate(cached_candidates):
                        if "frame_id" not in fr.info:
                            stype = fr.info.get("sample_type", "frame")
                            fr.info["frame_id"] = f"{stype}_{idx:02d}"
                        if "label" not in fr.info:
                            fr.info["label"] = fr.info["frame_id"]
                        fr.info["candidate_index"] = idx
                        fr.info["frame_name"] = f"frame_{idx:03d}_{fr.info['frame_id']}"
                    print(f"[i] Found existing candidate dataset ({len(cached_candidates)} frames) in {out_path}.")
                    print("    Reusing candidate pool to preserve geometry and labels across runs.")
                    return cached_candidates
            except Exception as e:
                print(f"    [!] Could not read existing candidate_frames.xyz ({e}). Generating fresh pool.")

        fc_frames = self.sample_franck_condon_path(at_gs, at_es, self.config.n_fc_frames)
        rattle_frames = self.sample_defect_core_rattling(at_gs, self.config.n_phonon_frames)
        md_frames = self.sample_foundation_md_with_fps(at_gs, self.config.n_md_frames)
        
        # Interleave to ensure balanced representation from the very first frame
        all_candidates = []
        max_len = max(len(fc_frames), len(rattle_frames), len(md_frames))
        for i in range(max_len):
            if i < len(fc_frames):
                all_candidates.append(fc_frames[i])
            if i < len(rattle_frames):
                all_candidates.append(rattle_frames[i])
            if i < len(md_frames):
                all_candidates.append(md_frames[i])
                
        for idx, fr in enumerate(all_candidates):
            fr.info["candidate_index"] = idx
            fr.info["frame_name"] = f"frame_{idx:03d}_{fr.info['frame_id']}"
            
        ase.io.write(out_path, all_candidates, format="extxyz")
        print(f"[+] Assembled candidate dataset: {len(all_candidates)} total representative frames -> {out_path}")
        return all_candidates

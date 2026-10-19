"""Four-phase candidate frame sampling: Franck-Condon path, core rattling, and foundation MD with FPS."""

import os
import numpy as np
import ase.io
from ase import Atoms
from calorine.calculators import CPUNEP

from .defect_builder import mic_displacement, mic_distances, get_defect_center


def farthest_point_sampling(descriptor_matrix, n_select):
    """Selects `n_select` rows from `descriptor_matrix` (N_frames, N_features) via Farthest Point Sampling."""
    descriptor_matrix = np.asarray(descriptor_matrix, dtype=float)
    n_frames = descriptor_matrix.shape[0]
    if n_select >= n_frames:
        return list(range(n_frames))

    sq_norms = np.einsum("ij,ij->i", descriptor_matrix, descriptor_matrix)
    selected = [int(np.argmax(sq_norms))]
    diff0 = descriptor_matrix - descriptor_matrix[selected[0]]
    min_sq_dists = np.einsum("ij,ij->i", diff0, diff0)

    for _ in range(1, n_select):
        next_idx = int(np.argmax(min_sq_dists))
        selected.append(next_idx)
        diff = descriptor_matrix - descriptor_matrix[next_idx]
        min_sq_dists = np.minimum(min_sq_dists, np.einsum("ij,ij->i", diff, diff))

    return selected


class FrameSampler:
    def __init__(self, config, calc_foundation=None):
        self.config = config
        if calc_foundation is not None:
            self.calc_foundation = calc_foundation
        elif getattr(config, "foundation_potential_path", None):
            try:
                self.calc_foundation = CPUNEP(config.foundation_potential_path)
            except Exception:
                self.calc_foundation = None
        else:
            self.calc_foundation = None

    def _compute_core_descriptors(self, frames, at_ref, r_cutoff=6.0):
        """Computes defect-core feature vectors for FPS using NEP descriptors or PBC-safe local geometry."""
        center = get_defect_center(at_ref)
        ref_dists = mic_distances(at_ref.positions, center, at_ref.cell)
        core_mask = ref_dists < r_cutoff
        if not np.any(core_mask):
            core_mask = np.ones(len(at_ref), dtype=bool)

        pot_path = getattr(self.config, "foundation_potential_path", None)
        if pot_path and os.path.isfile(pot_path) and os.path.getsize(pot_path) > 256:
            try:
                from calorine.nep import get_descriptors
                desc_list = []
                for fr in frames:
                    d_atom = get_descriptors(fr, potential_filename=pot_path)
                    d_core = d_atom[core_mask]
                    desc_list.append(np.concatenate([np.mean(d_core, axis=0), np.std(d_core, axis=0)]))
                return np.asarray(desc_list, dtype=float)
            except Exception:
                pass

        features = []
        n_keep = min(18, len(at_ref))
        for fr in frames:
            disp_vecs = mic_displacement(fr.positions, center, fr.cell)
            dists = np.linalg.norm(disp_vecs, axis=1)
            order = np.argsort(dists)[:n_keep]
            sorted_d = dists[order]
            dr_ref = mic_displacement(fr.positions[core_mask], at_ref.positions[core_mask], fr.cell)
            rms_disp = np.array([
                float(np.sqrt(np.mean(dr_ref ** 2))),
                float(np.max( np.linalg.norm(dr_ref, axis=1) )),
            ])
            features.append(np.concatenate([sorted_d, rms_disp]))
        return np.asarray(features, dtype=float)

    def sample_franck_condon_path(self, at_gs, at_es, n_points=7):
        """Phase 1: Structures along the minimum-image linear optical coordinate."""
        dr = mic_displacement(at_es.positions, at_gs.positions, at_gs.cell)
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
        """Phase 2: Defect-core localized Gaussian rattle configurations."""
        center = get_defect_center(at_gs)
        dists = mic_distances(at_gs.positions, center, at_gs.cell)
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
        """Phase 3: NVT MD with the foundation potential followed by Farthest Point Sampling."""
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

        feature_matrix = self._compute_core_descriptors(trajectory, at_gs)
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
        """Builds or loads `candidate_frames.xyz` with interleaved frames from Phases 1-3."""
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

    def generate_replacement_frame(self, sample_type, at_gs, type_index, candidate_index=None, r_cutoff=6.0):
        """Generates a replacement `core_rattle` or `foundation_md_fps` frame when a candidate fails twice."""
        base_seed = getattr(self.config, "random_seed", 42)
        at = at_gs.copy()
        at.arrays.pop("forces", None)
        at.info.pop("energy", None)

        if sample_type == "core_rattle":
            rng = np.random.RandomState(base_seed + 1000 + int(type_index))
            center = get_defect_center(at)
            dists = mic_distances(at.positions, center, at.cell)
            core_mask = dists < r_cutoff
            sigma = 0.08 if type_index % 2 == 0 else 0.12
            displacements = rng.normal(0.0, sigma, at.positions.shape)
            displacements[~core_mask] *= 0.15
            at.positions += displacements
            label = f"rattle_{type_index:02d}"
            at.info["sample_type"] = "core_rattle"
            at.info["rattle_sigma"] = float(sigma)
        elif sample_type == "foundation_md_fps":
            rng = np.random.RandomState(base_seed + 2000 + int(type_index))
            if self.calc_foundation is not None:
                from ase.md.langevin import Langevin
                from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
                from ase import units

                at.calc = self.calc_foundation
                temp_k = getattr(self.config, "md_temp_exploration", 500.0)
                MaxwellBoltzmannDistribution(at, temperature_K=temp_k, rng=rng)
                dyn = Langevin(at, 1.0 * units.fs, temperature_K=temp_k, friction=0.01 / units.fs, logfile=None)
                dyn.run(100)
                at = at.copy()
            else:
                displacements = rng.normal(0.0, 0.10, at.positions.shape)
                at.positions += displacements
            label = f"md_fps_{type_index:02d}"
            at.info["sample_type"] = "foundation_md_fps"
        else:
            raise ValueError(f"Replacement generation unsupported for sample_type='{sample_type}'.")

        at.info["frame_id"] = label
        at.info["label"] = label
        if candidate_index is not None:
            at.info["candidate_index"] = int(candidate_index)
            at.info["frame_name"] = f"frame_{int(candidate_index):03d}_{label}"
        return at


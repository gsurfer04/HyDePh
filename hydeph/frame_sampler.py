"""Four-phase candidate frame sampling: Franck-Condon path, core rattling, and foundation MD with FPS."""

from __future__ import annotations

import os
import logging
from itertools import zip_longest
from typing import List, Optional
import numpy as np
import ase.io
from ase import Atoms
from ase.geometry import get_distances
from calorine.calculators import CPUNEP  # noqa: F401

from .config import ToolkitConfig
from .calculator import load_foundation_calculator
from .defect_builder import mic_displacement, mic_distances, get_defect_center

from tqdm import tqdm

logger = logging.getLogger(__name__)

DEFAULT_MIN_BOND_RATIO = 0.85
DEFAULT_MAX_DISP_RATIO = 0.20
DEFAULT_MAX_SIGMA_BOND_RATIO = 0.055
DEFAULT_DUPLICATE_TOL_A = 1e-3


def farthest_point_sampling(descriptor_matrix: np.ndarray, n_select: int) -> List[int]:
    """Selects `n_select` rows from `descriptor_matrix` (N_frames, N_features) via Farthest Point Sampling.
    
    Maintains an incremental running minimum distance vector to avoid redundant matrix operations.
    """
    D = np.asarray(descriptor_matrix, dtype=float)
    n_total = D.shape[0]
    if n_select >= n_total:
        return list(range(n_total))
    if n_select <= 0 or n_total == 0:
        return []

    sq_norms = np.einsum("ij,ij->i", D, D)
    selected = [int(np.argmax(sq_norms))]
    min_dists = np.full(n_total, np.inf)

    for _ in range(n_select - 1):
        last_pt = D[selected[-1]]
        d = np.sum((D - last_pt) ** 2, axis=1)
        np.minimum(min_dists, d, out=min_dists)
        selected.append(int(np.argmax(min_dists)))

    return selected


def clean_candidate_atoms(atoms: Atoms) -> Atoms:
    """Detaches any attached calculator and strips evaluation properties (energy, forces, stress) from info and arrays.
    
    Candidate frames represent unlabelled candidate geometries destined for DFT single-point
    calculations. Stripping calculator results prevents ASE `save_calc_results` KeyError conflicts
    during extended XYZ serialization.
    """
    at = atoms.copy()
    at.calc = None
    for k in ("energy", "forces", "stress", "magmom", "free_energy"):
        at.info.pop(k, None)
    for k in ("forces", "stress", "magmoms"):
        at.arrays.pop(k, None)
    return at


def get_rattle_sigma(index: int) -> float:
    """Returns standardized defect-core rattle sigma (0.08 A for even indices, 0.14 A for odd indices)."""
    return 0.08 if int(index) % 2 == 0 else 0.14


def min_interatomic_distance(atoms: Atoms) -> float:
    """Returns the minimum pairwise minimum-image distance (in A), or inf if fewer than 2 atoms."""
    if len(atoms) < 2:
        return float("inf")
    _, dists = get_distances(atoms.positions, cell=atoms.cell, pbc=atoms.pbc)
    np.fill_diagonal(dists, np.inf)
    return float(np.min(dists))


def is_duplicate_geometry(
    atoms_a: Atoms,
    atoms_b: Atoms,
    tol: float = DEFAULT_DUPLICATE_TOL_A,
) -> bool:
    """Returns True when two periodic structures match within `tol` (in A) modulo rigid translation."""
    if len(atoms_a) != len(atoms_b) or atoms_a.get_chemical_symbols() != atoms_b.get_chemical_symbols():
        return False
    if len(atoms_a) == 0:
        return True
    if np.max(np.abs(atoms_a.cell.array - atoms_b.cell.array)) > tol:
        return False
    dr = mic_displacement(atoms_a.positions, atoms_b.positions, atoms_a.cell, pbc=atoms_a.pbc)
    if len(atoms_a) > 1:
        dr = dr - np.mean(dr, axis=0, keepdims=True)
    return bool(np.max(np.linalg.norm(dr, axis=1)) < tol)


def _cap_rattle_sigma(
    raw_sigma: float,
    d_min_ref: float,
    max_sigma_ratio: float = DEFAULT_MAX_SIGMA_BOND_RATIO,
) -> float:
    """Caps Gaussian rattle sigma proportionally to shortest equilibrium bond length."""
    if np.isfinite(d_min_ref) and d_min_ref > 0.0:
        return float(min(raw_sigma, max_sigma_ratio * d_min_ref))
    return float(raw_sigma)


def apply_displacement_guard(
    ref_atoms: Atoms,
    displacements: np.ndarray,
    min_bond_ratio: float = DEFAULT_MIN_BOND_RATIO,
    max_disp_ratio: float = DEFAULT_MAX_DISP_RATIO,
    d_min_ref: Optional[float] = None,
) -> tuple[np.ndarray, float]:
    """Scales displacements so per-atom shifts and min pair distance stay within physical bounds.

    Returns ``(guarded_displacements, scale_factor)``.
    """
    disp = np.asarray(displacements, dtype=float)
    d_ref = min_interatomic_distance(ref_atoms) if d_min_ref is None else float(d_min_ref)
    if len(ref_atoms) < 2 or not np.isfinite(d_ref) or d_ref <= 0.0:
        return disp, 1.0

    max_norm = float(np.max(np.linalg.norm(disp, axis=1)))
    if max_norm <= 0.0:
        return disp, 1.0

    scale = 1.0
    max_allowed_disp = max_disp_ratio * d_ref
    if max_norm > max_allowed_disp:
        scale = min(scale, max_allowed_disp / max_norm)

    d_floor = min_bond_ratio * d_ref
    trial = ref_atoms.copy()
    trial.positions = ref_atoms.positions + scale * disp
    if min_interatomic_distance(trial) < d_floor:
        lo, hi = 0.0, scale
        for _ in range(18):
            mid = 0.5 * (lo + hi)
            trial.positions = ref_atoms.positions + mid * disp
            if min_interatomic_distance(trial) >= d_floor:
                lo = mid
            else:
                hi = mid
        scale = lo

    return disp * scale, float(scale)


class FrameSampler:
    """Samples candidate defect supercells across Franck-Condon, rattling, and MD spaces."""

    def __init__(self, config: ToolkitConfig, calc_foundation=None):
        self.config = config
        if calc_foundation is not None:
            self.calc_foundation = calc_foundation
        elif config.foundation_potential_path and os.path.isfile(config.foundation_potential_path):
            self.calc_foundation = load_foundation_calculator(
                potential_path=config.foundation_potential_path,
                existing_calc=None,
                allow_missing=False,
            )
        else:
            self.calc_foundation = None

    def _compute_core_descriptors(self, frames: List[Atoms], at_ref: Atoms, r_cutoff: float = 6.0) -> np.ndarray:
        """Computes defect-core feature vectors for FPS using NEP descriptors or PBC-safe local geometry."""
        center = get_defect_center(at_ref)
        ref_dists = mic_distances(at_ref.positions, center, at_ref.cell)
        core_mask = ref_dists < r_cutoff
        if not np.any(core_mask):
            core_mask = np.ones(len(at_ref), dtype=bool)

        pot_path = self.config.foundation_potential_path
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
                float(np.max(np.linalg.norm(dr_ref, axis=1))),
            ])
            features.append(np.concatenate([sorted_d, rms_disp]))
        return np.asarray(features, dtype=float)

    def sample_franck_condon_path(self, at_gs: Atoms, at_es: Atoms, n_points: int = 7) -> List[Atoms]:
        """Phase 1: Structures along the minimum-image linear optical coordinate."""
        dr = mic_displacement(at_es.positions, at_gs.positions, at_gs.cell)
        base_lams = [0.0, 1.0, 0.5, -0.2, 1.2]
        extra_lams = list(np.linspace(-0.1, 1.1, max(0, n_points - len(base_lams))))
        lambdas = base_lams + extra_lams

        fc_frames = []
        for i, lam in enumerate(lambdas[:n_points]):
            at = clean_candidate_atoms(at_gs)
            at.positions = at_gs.positions + lam * dr
            label = f"fc_{i:02d}"
            at.info["sample_type"] = "franck_condon"
            at.info["frame_id"] = label
            at.info["label"] = label
            at.info["lambda"] = float(lam)
            fc_frames.append(at)
        logger.info(f"[+] Phase 1: Generated {len(fc_frames)} Franck-Condon path frames (lambda in [-0.2, 1.2])")
        return fc_frames

    def sample_defect_core_rattling(
        self, at_gs: Atoms, at_es: Optional[Atoms] = None, n_frames: int = 12, r_cutoff: Optional[float] = None
    ) -> List[Atoms]:
        """Phase 2: Defect-core and/or supercell Gaussian rattle configurations."""
        if n_frames <= 0:
            return []
        cutoff = r_cutoff if r_cutoff is not None else getattr(self.config, "rattle_cutoff", 6.0)
        mode = getattr(self.config, "rattle_mode", "hybrid").lower()

        seed = self.config.random_seed
        rng = np.random.RandomState(seed)

        n_es = (n_frames // 2) if (at_es is not None and n_frames >= 2) else 0
        n_gs = n_frames - n_es
        d_min_gs = min_interatomic_distance(at_gs)
        d_min_es = min_interatomic_distance(at_es) if at_es is not None else d_min_gs

        rattle_frames = []
        n_guarded = 0
        for i in range(n_frames):
            use_es = (i >= n_gs and at_es is not None)
            ref_atoms = at_es if use_es else at_gs
            ref_state = "excited" if use_es else "ground"
            d_ref = d_min_es if use_es else d_min_gs

            at = clean_candidate_atoms(ref_atoms)
            center = get_defect_center(ref_atoms)
            dists = mic_distances(ref_atoms.positions, center, ref_atoms.cell)
            core_mask = dists < cutoff

            raw_sigma = get_rattle_sigma(i)
            sigma = _cap_rattle_sigma(raw_sigma, d_ref)
            displacements = rng.normal(0.0, sigma, at.positions.shape)

            if mode == "whole":
                pass
            elif mode == "core":
                displacements[~core_mask] *= 0.15
            else:  # "hybrid"
                if i % 3 == 0:
                    displacements *= 0.70
                else:
                    displacements[~core_mask] *= 0.15

            displacements, guard_scale = apply_displacement_guard(ref_atoms, displacements, d_min_ref=d_ref)
            if sigma < raw_sigma - 1e-6 or guard_scale < 1.0 - 1e-6:
                n_guarded += 1

            at.positions += displacements
            label = f"rattle_{i:02d}"
            at.info["sample_type"] = "core_rattle"
            at.info["frame_id"] = label
            at.info["label"] = label
            at.info["rattle_sigma"] = float(sigma)
            if guard_scale < 1.0 - 1e-6:
                at.info["displacement_guard_scale"] = float(guard_scale)
            at.info["reference_state"] = ref_state
            rattle_frames.append(at)

        logger.info(
            f"[+] Phase 2: Generated {len(rattle_frames)} rattle frames "
            f"(mode='{mode}', cutoff={cutoff} A, GS={n_gs}, ES={n_es}"
            + (f", displacement-guarded={n_guarded}" if n_guarded else "")
            + ")"
        )
        return rattle_frames

    def sample_hessian_rattling(
        self, at_gs: Atoms, at_es: Optional[Atoms] = None, n_frames: int = 4, r_cutoff: Optional[float] = None
    ) -> List[Atoms]:
        """Phase 2b: Small-amplitude Hessian-regime rattle configurations (sigma in [0.01, 0.03] A)."""
        if n_frames <= 0:
            return []
        cutoff = r_cutoff if r_cutoff is not None else getattr(self.config, "rattle_cutoff", 6.0)
        seed = self.config.random_seed + 500
        rng = np.random.RandomState(seed)
        sigmas = [0.015, 0.025, 0.015, 0.025, 0.02, 0.03]

        n_es = (n_frames // 2) if (at_es is not None and n_frames >= 2) else 0
        n_gs = n_frames - n_es
        d_min_gs = min_interatomic_distance(at_gs)
        d_min_es = min_interatomic_distance(at_es) if at_es is not None else d_min_gs

        frames = []
        for i in range(n_frames):
            use_es = (i >= n_gs and at_es is not None)
            ref_atoms = at_es if use_es else at_gs
            ref_state = "excited" if use_es else "ground"
            d_ref = d_min_es if use_es else d_min_gs

            at = clean_candidate_atoms(ref_atoms)
            center = get_defect_center(ref_atoms)
            dists = mic_distances(ref_atoms.positions, center, ref_atoms.cell)
            core_mask = dists < cutoff
            sigma = sigmas[i % len(sigmas)]

            displacements = rng.normal(0.0, sigma, at.positions.shape)
            if i % 3 != 0:
                displacements[~core_mask] *= 0.15

            displacements, guard_scale = apply_displacement_guard(ref_atoms, displacements, d_min_ref=d_ref)
            at.positions += displacements
            label = f"hessian_rattle_{i:02d}"
            at.info["sample_type"] = "core_rattle"
            at.info["is_hessian"] = True
            at.info["frame_id"] = label
            at.info["label"] = label
            at.info["rattle_sigma"] = float(sigma)
            if guard_scale < 1.0 - 1e-6:
                at.info["displacement_guard_scale"] = float(guard_scale)
            at.info["reference_state"] = ref_state
            frames.append(at)

        logger.info(
            f"[+] Phase 2b: Generated {len(frames)} small-amplitude Hessian rattle frames "
            f"(sigma in [0.01, 0.03] A, GS={n_gs}, ES={n_es})"
        )
        return frames

    def sample_pristine_rattling(self, n_frames: int = 0, sigma: float = 0.02) -> List[Atoms]:
        """Optional Phase 2c: Small-amplitude rattle frames of the pristine host supercell."""
        if n_frames <= 0 or not self.config.pristine_cell_path or not os.path.isfile(self.config.pristine_cell_path):
            return []
        try:
            pristine_uc = ase.io.read(self.config.pristine_cell_path)
            supercell_rep = getattr(self.config, "supercell_repeat", (2, 2, 2))
            pristine_sc = pristine_uc.repeat(supercell_rep)
            d_min_prist = min_interatomic_distance(pristine_sc)
            eff_sigma = _cap_rattle_sigma(sigma, d_min_prist)

            seed = self.config.random_seed + 900
            rng = np.random.RandomState(seed)
            pristine_frames = []
            for i in range(n_frames):
                at = clean_candidate_atoms(pristine_sc)
                displacements = rng.normal(0.0, eff_sigma, at.positions.shape)
                displacements, guard_scale = apply_displacement_guard(pristine_sc, displacements, d_min_ref=d_min_prist)
                at.positions += displacements
                label = f"pristine_rattle_{i:02d}"
                at.info["sample_type"] = "pristine_rattle"
                at.info["frame_id"] = label
                at.info["label"] = label
                at.info["rattle_sigma"] = float(eff_sigma)
                if guard_scale < 1.0 - 1e-6:
                    at.info["displacement_guard_scale"] = float(guard_scale)
                at.info["is_pristine"] = True
                pristine_frames.append(at)

            logger.info(
                f"[+] Phase 2c: Generated {len(pristine_frames)} rattled pristine supercell frames (sigma = {eff_sigma:.3f} A)"
            )
            return pristine_frames
        except Exception as e:
            logger.warning(f"Could not generate pristine rattle frames: {e}")
            return []

    def sample_foundation_md_with_fps(
        self, at_gs: Atoms, at_es: Optional[Atoms] = None, n_select: int = 20, n_steps: int = 2000, dt_fs: float = 1.0
    ) -> List[Atoms]:
        """Phase 3: NVT MD with the foundation potential followed by Farthest Point Sampling."""
        if n_select <= 0:
            return []
        scratch_dir = os.path.join(self.config.workspace_dir, "sampler_md")
        os.makedirs(scratch_dir, exist_ok=True)

        from ase.md.langevin import Langevin
        from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
        from ase import units

        n_es_select = (n_select // 2) if (at_es is not None and n_select >= 2) else 0
        n_gs_select = n_select - n_es_select

        targets = [("ground", at_gs, n_gs_select, 101, "sampling_trajectory_gs.traj")]
        if n_es_select > 0 and at_es is not None:
            targets.append(("excited", at_es, n_es_select, 201, "sampling_trajectory_es.traj"))

        reference_frames: List[Atoms] = [at_gs]
        if at_es is not None:
            reference_frames.append(at_es)

        selected_frames = []
        global_md_idx = 0

        for state_name, init_atoms, n_pick, seed_offset, traj_fname in targets:
            if n_pick <= 0:
                continue
            seed = self.config.random_seed + seed_offset
            at = clean_candidate_atoms(init_atoms)
            at.calc = self.calc_foundation
            temp_k = self.config.md_temp_exploration
            MaxwellBoltzmannDistribution(at, temperature_K=temp_k, rng=np.random.RandomState(seed))
            dyn = Langevin(
                at,
                dt_fs * units.fs,
                temperature_K=temp_k,
                friction=0.01 / units.fs,
                logfile=None,
                rng=np.random.RandomState(seed + 1),
            )

            traj_file = os.path.join(scratch_dir, traj_fname)
            traj_writer = ase.io.Trajectory(traj_file, "w", at)
            dyn.attach(traj_writer.write, interval=10)

            steps_target = max(50, n_steps // (2 if len(targets) > 1 else 1))
            pbar = tqdm(
                total=steps_target,
                desc=f"MD candidate sampling ({state_name})",
                unit="step",
                leave=False,
            )
            step_chunk = 50
            steps_done = 0
            while steps_done < steps_target:
                this_chunk = min(step_chunk, steps_target - steps_done)
                dyn.run(this_chunk)
                steps_done += this_chunk
                pbar.update(this_chunk)
            pbar.close()
            traj_writer.close()

            trajectory = ase.io.read(traj_file, ":")
            # Step 0 written by dyn.attach is identical to init_atoms (already in FC phase).
            raw_pool = list(trajectory[1:]) if len(trajectory) > 1 else list(trajectory)
            seen_external = reference_frames + selected_frames
            traj_pool = [
                fr for fr in raw_pool
                if not any(is_duplicate_geometry(fr, ref) for ref in seen_external)
            ]
            n_filtered = len(raw_pool) - len(traj_pool)
            logger.info(
                f"[+] Phase 3 ({state_name}): Propagated {len(trajectory)} MD trajectory frames at {temp_k} K "
                f"({len(traj_pool)} unique post-step-0 candidates"
                + (f", {n_filtered} duplicate(s) skipped" if n_filtered else "")
                + ")"
            )

            if len(traj_pool) < n_pick:
                logger.warning(
                    f"[!] Phase 3 ({state_name}): Only {len(traj_pool)} unique MD frames available for {n_pick} requested picks."
                )
                for fr in raw_pool:
                    if len(traj_pool) >= n_pick:
                        break
                    if all(fr is not kept for kept in traj_pool):
                        traj_pool.append(fr)

            feature_matrix = self._compute_core_descriptors(traj_pool, init_atoms)
            ranked_count = len(traj_pool) if n_pick >= len(traj_pool) else min(len(traj_pool), max(n_pick * 3, n_pick + 10))
            ranked_indices = farthest_point_sampling(feature_matrix, ranked_count)

            chosen_for_state: List[Atoms] = []
            used_indices = set()
            for idx in ranked_indices:
                cand = traj_pool[idx]
                if not any(is_duplicate_geometry(cand, prev) for prev in chosen_for_state):
                    chosen_for_state.append(cand)
                    used_indices.add(idx)
                    if len(chosen_for_state) >= n_pick:
                        break

            # Scan remaining non-selected pool frames if internal trajectory duplicates reduced picks
            if len(chosen_for_state) < n_pick:
                for idx, cand in enumerate(traj_pool):
                    if idx in used_indices:
                        continue
                    if not any(is_duplicate_geometry(cand, prev) for prev in chosen_for_state):
                        chosen_for_state.append(cand)
                        used_indices.add(idx)
                        if len(chosen_for_state) >= n_pick:
                            break

            for cand in chosen_for_state:
                fr = clean_candidate_atoms(cand)
                label = f"md_fps_{global_md_idx:02d}"
                fr.info["sample_type"] = "foundation_md_fps"
                fr.info["reference_state"] = state_name
                fr.info["frame_id"] = label
                fr.info["label"] = label
                selected_frames.append(fr)
                global_md_idx += 1

        logger.info(f"[+] Phase 3: Farthest Point Sampling selected {len(selected_frames)} maximally diverse frames")
        return selected_frames

    def assemble_candidate_dataset(self, at_gs: Atoms, at_es: Atoms, force_resample: bool = False) -> List[Atoms]:
        """Builds or loads `candidate_frames.xyz` with interleaved frames from Phases 1-3."""
        out_path = os.path.join(self.config.workspace_dir, "candidate_frames.xyz")
        force = force_resample or self.config.force_resample

        from .provenance import compute_atoms_hash, write_stage_manifest, verify_stage_manifest
        stage_cfg = {
            "n_fc_frames": self.config.n_fc_frames,
            "n_phonon_frames": self.config.n_phonon_frames,
            "n_md_frames": self.config.n_md_frames,
            "n_hessian_frames": getattr(self.config, "n_hessian_frames", 0),
            "n_pristine_rattle_frames": getattr(self.config, "n_pristine_rattle_frames", 0),
            "rattle_mode": getattr(self.config, "rattle_mode", "hybrid"),
            "rattle_cutoff": getattr(self.config, "rattle_cutoff", 6.0),
            "random_seed": self.config.random_seed,
        }
        input_hashes = {
            "at_gs": compute_atoms_hash(at_gs),
            "at_es": compute_atoms_hash(at_es),
        }

        manifest_path = os.path.join(self.config.workspace_dir, "stage_2_manifest.json")
        has_manifest = os.path.isfile(manifest_path)
        manifest_ok = (
            verify_stage_manifest(
                self.config.workspace_dir,
                stage_idx=2,
                config_dict=stage_cfg,
                input_hashes=input_hashes,
                required_outputs=["candidate_frames.xyz"],
            )
            if has_manifest
            else True
        )

        is_cache_valid = (
            not force
            and os.path.isfile(out_path)
            and os.path.getsize(out_path) > 0
            and manifest_ok
        )

        if is_cache_valid:
            try:
                cached_candidates = ase.io.read(out_path, ":")
                if len(cached_candidates) > 0:
                    cleaned_cached = []
                    for idx, fr in enumerate(cached_candidates):
                        clean_fr = clean_candidate_atoms(fr)
                        if "frame_id" not in clean_fr.info:
                            stype = clean_fr.info.get("sample_type", "frame")
                            clean_fr.info["frame_id"] = f"{stype}_{idx:02d}"
                        if "label" not in clean_fr.info:
                            clean_fr.info["label"] = clean_fr.info["frame_id"]
                        clean_fr.info["candidate_index"] = idx
                        clean_fr.info["frame_name"] = f"frame_{idx:03d}_{clean_fr.info['frame_id']}"
                        cleaned_cached.append(clean_fr)
                    if not has_manifest:
                        try:
                            write_stage_manifest(
                                self.config.workspace_dir,
                                stage_idx=2,
                                config_dict=stage_cfg,
                                input_hashes=input_hashes,
                                output_files=["candidate_frames.xyz"],
                            )
                        except Exception:
                            pass
                    logger.info(f"[i] Found valid cached candidate dataset ({len(cleaned_cached)} frames) in {out_path}.")
                    logger.info("    Reusing candidate pool to preserve geometry and labels across runs.")
                    return cleaned_cached
            except Exception as e:
                logger.warning(f"    [!] Could not read existing candidate_frames.xyz ({e}). Generating fresh pool.")
        elif os.path.isfile(out_path):
            logger.info("    [i] Candidate frames cache invalidated: input GS/ES geometries or sampling parameters changed.")

        fc_frames = self.sample_franck_condon_path(at_gs, at_es, self.config.n_fc_frames)
        rattle_frames = self.sample_defect_core_rattling(at_gs, at_es, self.config.n_phonon_frames)
        n_hess = getattr(self.config, "n_hessian_frames", 0)
        hessian_frames = self.sample_hessian_rattling(at_gs, at_es, n_hess) if n_hess > 0 else []
        md_frames = self.sample_foundation_md_with_fps(at_gs, at_es, self.config.n_md_frames)
        n_prist = getattr(self.config, "n_pristine_rattle_frames", 0)
        pristine_frames = self.sample_pristine_rattling(n_prist) if n_prist > 0 else []

        # Interleave using zip_longest
        raw_candidates = [
            fr for group in zip_longest(fc_frames, rattle_frames, hessian_frames, md_frames, pristine_frames)
            for fr in group if fr is not None
        ]

        all_candidates = []
        for idx, fr in enumerate(raw_candidates):
            cfr = clean_candidate_atoms(fr)
            cfr.info["candidate_index"] = idx
            cfr.info["frame_name"] = f"frame_{idx:03d}_{cfr.info['frame_id']}"
            all_candidates.append(cfr)

        tmp_out = out_path + ".tmp"
        ase.io.write(tmp_out, all_candidates, format="extxyz")
        os.replace(tmp_out, out_path)
        write_stage_manifest(
            self.config.workspace_dir,
            stage_idx=2,
            config_dict=stage_cfg,
            input_hashes=input_hashes,
            output_files=[out_path],
        )
        logger.info(f"[+] Total candidate pool assembled: {len(all_candidates)} frames (written to {out_path})")
        return all_candidates

    def generate_replacement_frame(
        self,
        sample_type: str,
        at_gs: Atoms,
        type_index: int,
        candidate_index: Optional[int] = None,
        r_cutoff: float = 6.0,
    ) -> Atoms:
        """Generates a replacement `core_rattle` or `foundation_md_fps` frame when a candidate fails twice."""
        base_seed = self.config.random_seed
        at = clean_candidate_atoms(at_gs)

        d_ref = min_interatomic_distance(at)

        if sample_type == "core_rattle":
            rng = np.random.RandomState(base_seed + 1000 + int(type_index))
            center = get_defect_center(at)
            dists = mic_distances(at.positions, center, at.cell)
            core_mask = dists < r_cutoff
            sigma = _cap_rattle_sigma(get_rattle_sigma(type_index), d_ref)
            displacements = rng.normal(0.0, sigma, at.positions.shape)
            displacements[~core_mask] *= 0.15
            displacements, guard_scale = apply_displacement_guard(at, displacements, d_min_ref=d_ref)
            at.positions += displacements
            label = f"rattle_{type_index:02d}"
            at.info["sample_type"] = "core_rattle"
            at.info["rattle_sigma"] = float(sigma)
            if guard_scale < 1.0 - 1e-6:
                at.info["displacement_guard_scale"] = float(guard_scale)
        elif sample_type == "foundation_md_fps":
            rng = np.random.RandomState(base_seed + 2000 + int(type_index))
            if self.calc_foundation is not None:
                from ase.md.langevin import Langevin
                from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
                from ase import units

                at.calc = self.calc_foundation
                temp_k = self.config.md_temp_exploration
                MaxwellBoltzmannDistribution(at, temperature_K=temp_k, rng=rng)
                dyn = Langevin(
                    at,
                    1.0 * units.fs,
                    temperature_K=temp_k,
                    friction=0.01 / units.fs,
                    logfile=None,
                    rng=rng,
                )
                dyn.run(100)
                at = clean_candidate_atoms(at)
            else:
                sigma_fb = _cap_rattle_sigma(0.10, d_ref)
                displacements = rng.normal(0.0, sigma_fb, at.positions.shape)
                displacements, _ = apply_displacement_guard(at, displacements, d_min_ref=d_ref)
                at.positions += displacements
            label = f"md_fps_{type_index:02d}"
            at.info["sample_type"] = "foundation_md_fps"
        else:
            raise ValueError(f"Replacement generation unsupported for sample_type='{sample_type}'.")

        at = clean_candidate_atoms(at)
        at.info["frame_id"] = label
        at.info["label"] = label
        if candidate_index is not None:
            at.info["candidate_index"] = int(candidate_index)
            at.info["frame_name"] = f"frame_{int(candidate_index):03d}_{label}"
        return at


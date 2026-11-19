"""Property-driven host classification, Alkauskas Hessian splicing, and Franck-Condon force routing."""

from __future__ import annotations

import copy
import hashlib
import logging
import math
import os
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Optional, Sequence, Tuple

import ase.io
import numpy as np
from ase import Atoms

logger = logging.getLogger(__name__)


def _unwrap_bulk_calculator(calc: Any) -> Any:
    """Return the pristine bulk foundation calculator when a HybridDefectCalculator is provided."""
    return getattr(calc, "calc_base", calc)


def is_bulk_phonon_stable(
    pristine_uc: Atoms,
    calc_bulk: Any,
    delta: float = 0.01,
    imag_tol_meV: float = -1.0,
) -> bool:
    """Check whether calc_bulk has no significant imaginary modes on the pristine lattice."""
    from ase.phonons import Phonons

    lengths = np.asarray(pristine_uc.cell.lengths(), dtype=float)
    bulk_rep = tuple(max(2, min(3, int(math.ceil(9.0 / max(1e-6, float(L)))))) for L in lengths)
    while int(len(pristine_uc) * np.prod(bulk_rep)) > 64 and max(bulk_rep) > 1:
        idx = int(np.argmax(bulk_rep))
        lst = list(bulk_rep)
        lst[idx] -= 1
        bulk_rep = tuple(lst)

    with tempfile.TemporaryDirectory(prefix="hydeph_stab_") as tmpdir:
        ph = Phonons(
            pristine_uc,
            calc_bulk,
            supercell=bulk_rep,
            delta=float(delta),
            name=os.path.join(tmpdir, "ph"),
        )
        ph.run()
        ph.read(acoustic=True)
        kpts = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [0.0, 0.5, 0.0],
                [0.0, 0.0, 0.5],
                [0.5, 0.5, 0.5],
            ],
            dtype=float,
        )
        energies_eV = np.asarray(ph.band_structure(kpts), dtype=float)
        # ASE returns negative real values for imaginary modes in band_structure()
        hw_min_meV = float(np.min(np.real(energies_eV)) * 1000.0)
    return hw_min_meV >= float(imag_tol_meV)


def estimate_host_phonon_bandwidth(
    pristine_uc: Atoms,
    calc_bulk: Any,
    delta: float = 0.01,
) -> float:
    """Estimate maximum host optical phonon energy (meV) using ASE Phonons on the pristine unit cell."""
    from ase.phonons import Phonons

    lengths = np.asarray(pristine_uc.cell.lengths(), dtype=float)
    bulk_rep = tuple(max(1, min(3, int(math.ceil(8.0 / max(1e-6, float(L)))))) for L in lengths)
    # Keep quick bandwidth probe tiny (<= 64 atoms)
    while int(len(pristine_uc) * np.prod(bulk_rep)) > 64 and max(bulk_rep) > 1:
        idx = int(np.argmax(bulk_rep))
        lst = list(bulk_rep)
        lst[idx] -= 1
        bulk_rep = tuple(lst)

    with tempfile.TemporaryDirectory(prefix="hydeph_bw_") as tmpdir:
        ph = Phonons(
            pristine_uc,
            calc_bulk,
            supercell=bulk_rep,
            delta=float(delta),
            name=os.path.join(tmpdir, "ph"),
        )
        ph.run()
        ph.read(acoustic=True)
        kpts = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [0.0, 0.5, 0.0],
                [0.0, 0.0, 0.5],
                [0.5, 0.5, 0.5],
            ],
            dtype=float,
        )
        energies_eV = np.asarray(ph.band_structure(kpts), dtype=float)
        hw_max_meV = float(np.max(np.real(energies_eV)) * 1000.0)
    return hw_max_meV


def resolve_phonon_supercell_repeat(
    pristine_uc: Atoms,
    calc_bulk: Any = None,
    min_repeat: Sequence[int] = (1, 1, 1),
    hw_max_meV: Optional[float] = None,
    max_atoms: int = 5000,
) -> Tuple[int, int, int]:
    """Select an embedded phonon supercell repeat automatically from host stiffness and unit-cell lengths.

    Stiff covalent hosts (hw_max >= 140 meV, e.g., diamond) require L >= 27 A (~8x8x8) to suppress
    Brillouin-zone boundary peak splitting; intermediate polar-covalent hosts (75-140 meV, e.g., 4H-SiC)
    require L >= 24 A (~8x8x3); soft ionic hosts (< 75 meV, e.g., NaCl) converge cleanly at L >= 20 A (~4x4x4).
    """
    if hw_max_meV is None and calc_bulk is not None:
        try:
            hw_max_meV = estimate_host_phonon_bandwidth(pristine_uc, _unwrap_bulk_calculator(calc_bulk))
        except Exception as exc:
            logger.warning(f"    [!] Host bandwidth probe failed ({exc}); falling back to intermediate length target 24 A.")
            hw_max_meV = 100.0
    elif hw_max_meV is None:
        hw_max_meV = 100.0

    if hw_max_meV >= 140.0:
        l_target = 27.0
        host_class = "stiff covalent"
    elif hw_max_meV >= 75.0:
        l_target = 24.0
        host_class = "polar covalent"
    else:
        l_target = 20.0
        host_class = "soft ionic"

    lengths = np.asarray(pristine_uc.cell.lengths(), dtype=float)
    min_rep = tuple(max(1, int(x)) for x in min_repeat)
    rep = [max(min_rep[i], int(math.ceil(l_target / max(1e-6, float(lengths[i]))))) for i in range(3)]

    n_uc = max(1, len(pristine_uc))
    while int(n_uc * rep[0] * rep[1] * rep[2]) > int(max_atoms):
        # Trim the axis with the largest supercell edge first while respecting min_repeat
        edge_lens = [rep[i] * float(lengths[i]) if rep[i] > min_rep[i] else -1.0 for i in range(3)]
        best_ax = int(np.argmax(edge_lens))
        if edge_lens[best_ax] < 0.0:
            break
        rep[best_ax] -= 1

    rep_tup = (int(rep[0]), int(rep[1]), int(rep[2]))
    logger.info(
        f"    [+] Auto-selected phonon_supercell_repeat={rep_tup} "
        f"({host_class}: hw_max={hw_max_meV:.1f} meV, L_min_target={l_target:.1f} A, "
        f"box={tuple(round(rep_tup[i] * float(lengths[i]), 2) for i in range(3))} A)"
    )
    return rep_tup


def _map_supercell_sites(
    pristine_uc: Atoms,
    at_large: Atoms,
    rep_large: Tuple[int, int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Map each atom in an embedded supercell to its pristine (ix, iy, iz, basis_idx) slot."""
    nx, ny, nz = (int(rep_large[0]), int(rep_large[1]), int(rep_large[2]))
    n_uc = len(pristine_uc)
    pristine_large = pristine_uc.repeat((nx, ny, nz))
    n_L = len(at_large)

    site_to_atom = np.full((nx, ny, nz, n_uc), -1, dtype=int)
    atom_to_site = np.full((n_L, 4), -1, dtype=int)

    cell = np.asarray(at_large.cell, dtype=float)
    inv_cell = np.linalg.inv(cell)
    pos_L = np.asarray(at_large.positions, dtype=float)
    pos_pr = np.asarray(pristine_large.positions, dtype=float)

    # Chunked MIC distance matching keeps memory < ~30 MB even at 5000+ atoms
    chunk = 256
    used_pr = set()
    for start in range(0, n_L, chunk):
        end = min(n_L, start + chunk)
        diff = pos_pr[None, :, :] - pos_L[start:end, None, :]
        frac = diff @ inv_cell
        frac -= np.round(frac)
        cart = frac @ cell
        dists = np.linalg.norm(cart, axis=2)
        nearest = np.argmin(dists, axis=1)
        for loc, p_idx in enumerate(nearest):
            j = start + loc
            if dists[loc, p_idx] < 1.2 and int(p_idx) not in used_pr:
                used_pr.add(int(p_idx))
                ib = int(p_idx % n_uc)
                cell_idx = int(p_idx // n_uc)
                iz = int(cell_idx % nz)
                iy = int((cell_idx // nz) % ny)
                ix = int(cell_idx // (ny * nz))
                site_to_atom[ix, iy, iz, ib] = j
                atom_to_site[j] = (ix, iy, iz, ib)

    return site_to_atom, atom_to_site


def tile_bulk_hessian(
    pristine_uc: Atoms,
    at_large: Atoms,
    rep_large: Tuple[int, int, int],
    calc_bulk: Any,
    delta: float = 0.01,
    cache_dir: Optional[str] = None,
) -> np.ndarray:
    """Compute pristine crystal force constants with ASE Phonons and tile onto at_large."""
    from ase.phonons import Phonons

    nx, ny, nz = (int(rep_large[0]), int(rep_large[1]), int(rep_large[2]))
    lengths = np.asarray(pristine_uc.cell.lengths(), dtype=float)
    bulk_rep = (
        min(nx, max(2 if nx >= 2 else 1, int(math.ceil(10.0 / max(1e-6, float(lengths[0])))))),
        min(ny, max(2 if ny >= 2 else 1, int(math.ceil(10.0 / max(1e-6, float(lengths[1])))))),
        min(nz, max(2 if nz >= 2 else 1, int(math.ceil(10.0 / max(1e-6, float(lengths[2])))))),
    )

    def _run_ph(ph_name: str) -> Tuple[np.ndarray, np.ndarray]:
        ph = Phonons(pristine_uc, calc_bulk, supercell=bulk_rep, delta=float(delta), name=ph_name)
        ph.run()
        ph.read(acoustic=True)
        fc = np.asarray(ph.get_force_constant(), dtype=float)
        R_vecs = np.asarray(ph._lattice_vectors_array, dtype=int)
        if R_vecs.ndim == 2 and R_vecs.shape[0] == 3 and R_vecs.shape[1] != 3:
            R_vecs = R_vecs.T
        return fc, R_vecs

    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        ph_name = os.path.join(cache_dir, f"bulk_ph_{bulk_rep[0]}x{bulk_rep[1]}x{bulk_rep[2]}")
        fc, R_vecs = _run_ph(ph_name)
    else:
        with tempfile.TemporaryDirectory(prefix="hydeph_ph_") as tmpdir:
            fc, R_vecs = _run_ph(os.path.join(tmpdir, "bulk_ph"))

    n_uc = len(pristine_uc)
    n_L = len(at_large)
    site_to_atom, atom_to_site = _map_supercell_sites(pristine_uc, at_large, rep_large)
    H_4d = np.zeros((n_L, 3, n_L, 3), dtype=float)

    mapped_atoms = np.flatnonzero(atom_to_site[:, 0] >= 0)
    if len(mapped_atoms) == 0:
        return H_4d

    sites_m = atom_to_site[mapped_atoms]
    for bi in range(n_uc):
        mask_bi = sites_m[:, 3] == bi
        if not np.any(mask_bi):
            continue
        atoms_i = mapped_atoms[mask_bi]
        ix_i = sites_m[mask_bi, 0]
        iy_i = sites_m[mask_bi, 1]
        iz_i = sites_m[mask_bi, 2]
        for k, dR in enumerate(R_vecs):
            dx, dy, dz = int(dR[0]), int(dR[1]), int(dR[2])
            ix_j = (ix_i + dx) % nx
            iy_j = (iy_i + dy) % ny
            iz_j = (iz_i + dz) % nz
            for bj in range(n_uc):
                blk = fc[k, 3 * bi : 3 * bi + 3, 3 * bj : 3 * bj + 3]
                if np.max(np.abs(blk)) < 1e-10:
                    continue
                atoms_j = site_to_atom[ix_j, iy_j, iz_j, bj]
                valid = atoms_j >= 0
                if np.any(valid):
                    H_4d[atoms_i[valid], :, atoms_j[valid], :] += blk[None, :, :]

    return H_4d


def compute_core_hessian_columns(
    at_large: Atoms,
    calc_gs: Any,
    core_indices: np.ndarray,
    delta: float = 0.01,
    n_workers: int = 8,
) -> np.ndarray:
    """Compute central finite-difference Hessian columns ONLY for core_indices on at_large."""
    n_L = len(at_large)
    n_core = len(core_indices)
    H_core_cols = np.zeros((n_core, 3, n_L, 3), dtype=float)
    if n_core == 0:
        return H_core_cols

    tasks: List[Tuple[int, int, int]] = [
        (loc, int(atom_idx), ax)
        for loc, atom_idx in enumerate(core_indices)
        for ax in range(3)
    ]

    _local = threading.local()
    _serial_lock = threading.Lock()
    use_threads = int(n_workers) > 1 and len(tasks) > 1

    def _get_worker_calc():
        if not use_threads:
            return calc_gs
        if not hasattr(_local, "calc") or _local.calc is None:
            try:
                _local.calc = copy.deepcopy(calc_gs)
            except Exception:
                _local.calc = calc_gs
                _local.shared = True
            else:
                _local.shared = False
        return _local.calc

    def _eval_pair(item: Tuple[int, int, int]) -> Tuple[int, int, np.ndarray]:
        loc, atom_idx, ax = item
        wcalc = _get_worker_calc()
        at_p = at_large.copy()
        at_p.positions[atom_idx, ax] += float(delta)
        at_p.calc = wcalc

        at_m = at_large.copy()
        at_m.positions[atom_idx, ax] -= float(delta)
        at_m.calc = wcalc

        if getattr(_local, "shared", False):
            with _serial_lock:
                f_plus = np.asarray(at_p.get_forces(), dtype=float)
                f_minus = np.asarray(at_m.get_forces(), dtype=float)
        else:
            f_plus = np.asarray(at_p.get_forces(), dtype=float)
            f_minus = np.asarray(at_m.get_forces(), dtype=float)

        col = (f_minus - f_plus) / (2.0 * float(delta))
        return loc, ax, col

    if use_threads:
        with ThreadPoolExecutor(max_workers=int(n_workers)) as ex:
            for loc, ax, col in ex.map(_eval_pair, tasks):
                H_core_cols[loc, ax] = col
    else:
        for item in tasks:
            loc, ax, col = _eval_pair(item)
            H_core_cols[loc, ax] = col

    return H_core_cols


def build_spliced_hessian(
    pristine_uc: Atoms,
    at_large: Atoms,
    rep_large: Tuple[int, int, int],
    calc_gs: Any,
    calc_bulk: Any = None,
    core_radius_A: Optional[float] = None,
    delta: float = 0.01,
    n_workers: int = 8,
    cache_dir: Optional[str] = None,
    fingerprint: Optional[str] = None,
) -> np.ndarray:
    """Build an Alkauskas-spliced 2D Hessian (pristine bulk FC + exact defect-core columns + ASR)."""
    from .defect_builder import mic_distances
    from .lineshape import apply_acoustic_sum_rule_and_projection

    n_L = len(at_large)
    lengths = np.asarray(at_large.cell.lengths(), dtype=float)
    max_safe_r = 0.42 * float(np.min(lengths))
    if core_radius_A is None:
        r_ws_dft = float(at_large.info.get("dft_wigner_seitz_radius_A", 0.0))
        core_radius_A = min(max_safe_r, max(5.5, r_ws_dft))
    else:
        core_radius_A = min(max_safe_r, float(core_radius_A))

    c_center = np.asarray(at_large.info.get("defect_center", at_large.cell.sum(axis=0) / 2.0), dtype=float)
    dists = mic_distances(at_large.positions, c_center, at_large.cell)
    core_indices = np.flatnonzero(dists <= core_radius_A)
    if len(core_indices) < min(12, n_L):
        core_indices = np.argsort(dists)[: min(12, n_L)]

    fp_str = fingerprint or ""
    cache_npz = None
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        key = hashlib.sha256(f"{fp_str}|{len(at_large)}|{core_radius_A:.4f}".encode("utf-8")).hexdigest()[:16]
        cache_npz = os.path.join(cache_dir, f"spliced_hessian_{key}.npz")
        if os.path.isfile(cache_npz):
            try:
                dat = np.load(cache_npz)
                H_cached = dat["H_2d"]
                if H_cached.shape == (3 * n_L, 3 * n_L):
                    logger.info(f"    [+] Loaded cached Alkauskas-spliced Hessian: {cache_npz}")
                    return H_cached
            except Exception:
                pass

    eff_bulk = _unwrap_bulk_calculator(calc_gs) if calc_bulk is None else calc_bulk
    logger.info(
        f"    [-] Building Alkauskas bulk+core spliced Hessian for {n_L} atoms "
        f"({len(core_indices)} core atoms within r <= {core_radius_A:.2f} A; {6 * len(core_indices)} core disps vs {6 * n_L} full)..."
    )
    H_4d = tile_bulk_hessian(pristine_uc, at_large, rep_large, eff_bulk, delta=delta, cache_dir=cache_dir)
    H_core_cols = compute_core_hessian_columns(at_large, calc_gs, core_indices, delta=delta, n_workers=n_workers)

    H_spliced = H_4d.copy()
    for loc, i_c in enumerate(core_indices):
        H_spliced[:, :, i_c, :] = np.transpose(H_core_cols[loc], (1, 2, 0))
    H_spliced[core_indices, :, :, :] = H_core_cols

    H_2d = H_spliced.reshape(3 * n_L, 3 * n_L)
    H_2d = 0.5 * (H_2d + H_2d.T)
    H_clean = apply_acoustic_sum_rule_and_projection(H_2d, at_large)

    if cache_npz is not None:
        try:
            np.savez_compressed(cache_npz, H_2d=H_clean)
        except Exception:
            pass

    return H_clean


def assess_relaxation_asymmetry(
    at_gs: Atoms,
    at_es: Atoms,
    calc_gs: Any,
    calc_es: Any,
) -> Tuple[float, float, float]:
    """Return (Delta_E_rel_ES, Delta_E_rel_GS, asymmetry_ratio) across GS and ES minima."""
    a_gs = at_gs.copy()
    a_gs.calc = calc_gs
    e_gs_gs = float(a_gs.get_potential_energy())
    a_gs.calc = calc_es
    e_es_gs = float(a_gs.get_potential_energy())

    a_es = at_es.copy()
    a_es.calc = calc_es
    e_es_es = float(a_es.get_potential_energy())
    a_es.calc = calc_gs
    e_gs_es = float(a_es.get_potential_energy())

    drel_es = max(0.0, e_es_gs - e_es_es)
    drel_gs = max(0.0, e_gs_es - e_gs_gs)
    if drel_es > 1e-5 and drel_gs > 1e-5:
        ratio = max(drel_es / drel_gs, drel_gs / drel_es)
    else:
        ratio = 1.0
    return drel_es, drel_gs, ratio


def _pbc_rmsd(pos_a: np.ndarray, pos_b: np.ndarray, cell: np.ndarray) -> float:
    diff = pos_a - pos_b
    inv = np.linalg.inv(cell)
    frac = diff @ inv
    frac -= np.round(frac)
    cart = frac @ cell
    return float(np.sqrt(np.mean(np.sum(cart ** 2, axis=1))))


def load_dft_optical_forces(
    at_gs: Atoms,
    at_es: Atoms,
    workspace_dir: str,
    force_state: str = "absorption",
) -> Optional[np.ndarray]:
    """Load exact DFT optical force differences for GS->ES ('absorption') or ES->GS ('emission') if available."""
    state = str(force_state).lower()
    info_key = "dft_optical_forces_em" if state == "emission" else "dft_optical_forces_abs"
    ref_at = at_es if state == "emission" else at_gs
    if info_key in ref_at.arrays:
        arr = np.asarray(ref_at.arrays[info_key], dtype=float)
        if arr.shape == (len(ref_at), 3):
            return arr
    if info_key in ref_at.info:
        arr = np.asarray(ref_at.info[info_key], dtype=float)
        if arr.shape == (len(ref_at), 3):
            return arr

    if not workspace_dir or not os.path.isdir(workspace_dir):
        return None

    nat = len(ref_at)
    syms = ref_at.get_chemical_symbols()
    cell = np.asarray(ref_at.cell, dtype=float)

    def _pbc_max_disp(pos_a: np.ndarray, pos_b: np.ndarray) -> float:
        diff = pos_a - pos_b
        frac = diff @ np.linalg.inv(cell)
        frac -= np.round(frac)
        return float(np.max(np.linalg.norm(frac @ cell, axis=1)))

    # 1. Prefer exact relaxation endpoints + conjugate single-point SCF at R_ES / R_GS
    qe_es_p = os.path.join(workspace_dir, "qe_es_relax", "pw.out")
    qe_gs_p = os.path.join(workspace_dir, "qe_gs_relax", "pw.out")
    if state == "emission":
        qe_gs_es_p = os.path.join(workspace_dir, "qe_gs_on_es", "pw.out")
        if os.path.isfile(qe_gs_es_p) and os.path.isfile(qe_es_p):
            try:
                gs_es = ase.io.read(qe_gs_es_p, index=-1)
                es_last = ase.io.read(qe_es_p, index=-1)
                if (
                    len(gs_es) == nat
                    and len(es_last) == nat
                    and gs_es.get_chemical_symbols() == syms
                    and es_last.get_chemical_symbols() == syms
                    and _pbc_rmsd(gs_es.positions, es_last.positions, cell) < 0.02
                    and _pbc_rmsd(gs_es.positions, at_es.positions, cell) < 0.08
                    and _pbc_max_disp(gs_es.positions, at_es.positions) < 0.08
                ):
                    return np.asarray(gs_es.get_forces() - es_last.get_forces(), dtype=float)
            except Exception:
                pass
    elif state == "absorption":
        if os.path.isfile(qe_es_p) and os.path.isfile(qe_gs_p):
            try:
                es0 = ase.io.read(qe_es_p, index=0)
                gs_last = ase.io.read(qe_gs_p, index=-1)
                if (
                    len(es0) == nat
                    and len(gs_last) == nat
                    and es0.get_chemical_symbols() == syms
                    and gs_last.get_chemical_symbols() == syms
                    and _pbc_rmsd(es0.positions, gs_last.positions, cell) < 0.02
                    and _pbc_rmsd(es0.positions, at_gs.positions, cell) < 0.08
                ):
                    return np.asarray(es0.get_forces() - gs_last.get_forces(), dtype=float)
            except Exception:
                pass

    # 2. Fallback to exact paired labeled DFT frames (smallest RMSD within strict max atom displacement < 0.035 A)
    f_gs_xyz = os.path.join(workspace_dir, "labeled_dft_frames.xyz")
    f_es_xyz = os.path.join(workspace_dir, "labeled_frames_es.xyz")
    if os.path.isfile(f_gs_xyz) and os.path.isfile(f_es_xyz):
        try:
            frames_g = ase.io.read(f_gs_xyz, index=":")
            frames_e = ase.io.read(f_es_xyz, index=":")
            best_rmsd = float("inf")
            best_diff = None
            for fg, fe in zip(frames_g, frames_e):
                if len(fg) != nat or len(fe) != nat or fg.get_chemical_symbols() != syms:
                    continue
                if _pbc_rmsd(fg.positions, fe.positions, cell) > 0.01:
                    continue
                rmsd_ref = _pbc_rmsd(fg.positions, ref_at.positions, cell)
                max_ref = _pbc_max_disp(fg.positions, ref_at.positions)
                if rmsd_ref < 0.025 and max_ref < 0.035 and rmsd_ref < best_rmsd:
                    best_rmsd = rmsd_ref
                    diff = fe.get_forces() - fg.get_forces()
                    best_diff = np.asarray(-diff if state == "emission" else diff, dtype=float)
            if best_diff is not None:
                return best_diff
        except Exception:
            pass

    return None

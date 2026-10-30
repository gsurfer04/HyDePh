"""Parallel finite-displacement Hessian evaluator for ASE Vibrations."""

from __future__ import annotations

import os
import sys
import copy
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
from ase import Atoms
from ase.vibrations import Vibrations
from ase.parallel import world

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
    def tqdm(iterable=None, *args, **kwargs):
        return _DummyPbar(iterable, *args, **kwargs)

logger = logging.getLogger(__name__)


class ParallelVibrations(Vibrations):
    """High-throughput multi-threaded finite-displacement Hessian evaluator.

    Extends ASE `Vibrations` to compute missing atomic displacements concurrently
    across available CPU cores or GPU calculator instances, significantly accelerating
    supercell phonon evaluations (e.g. 2388 displacements on a 5x5x2 supercell).

    Parameters
    ----------
    atoms : Atoms
        Equilibrium atomic structure.
    name : str
        Prefix for vibration cache files.
    delta : float
        Displacement magnitude in Angstroms (default: 0.01).
    nfree : int
        Number of displacements per coordinate (2 for +/- delta).
    n_workers : int
        Number of concurrent workers for displacement evaluation (default: 4).
    """

    def __init__(
        self,
        atoms: Atoms,
        name: str = "vib",
        delta: float = 0.01,
        nfree: int = 2,
        n_workers: int = 4,
        **kwargs,
    ):
        super().__init__(atoms, name=name, delta=delta, nfree=nfree, **kwargs)
        self.n_workers = max(1, int(n_workers))

    def run_parallel(self, show_progress: bool = True) -> None:
        """Executes finite displacement calculations concurrently across workers.

        Parameters
        ----------
        show_progress : bool
            Whether to display an interactive progress bar.
        """
        if self.n_workers <= 1:
            super().run()
            return

        if not self.cache.writable:
            raise RuntimeError(
                "Cannot run calculation. Cache must be writable."
            )

        self._check_old_pickles()

        # Gather all pending displacements
        pending: List[Tuple[Any, Atoms]] = []
        for disp, at in self.iterdisplace(inplace=False):
            # Check if already present in cache
            if disp.name in self.cache:
                continue
            pending.append((disp, at))

        total_disps = 6 * len(self.atoms)
        already_cached = total_disps - len(pending)

        logger.info(
            f"[ParallelVibrations] Total displacements: {total_disps} | "
            f"Already cached: {already_cached} | Pending: {len(pending)} | "
            f"Workers: {self.n_workers}"
        )

        if not pending:
            return

        pbar = tqdm(
            total=len(pending),
            desc=f"Normal modes ({self.n_workers} workers)",
            unit="disp",
            disable=not show_progress,
            leave=False,
        )

        import threading

        _local = threading.local()
        _calc_lock = threading.Lock()

        def _get_worker_calc(base_calc):
            if not hasattr(_local, "calc") or _local.calc is None:
                try:
                    _local.calc = copy.deepcopy(base_calc)
                except Exception:
                    _local.calc = base_calc
            return _local.calc

        def _compute_single_displacement(item: Tuple[Any, Atoms]) -> str:
            disp, at_disp = item
            worker_calc = _get_worker_calc(self.calc)
            at_eval = at_disp.copy()
            at_eval.calc = worker_calc

            with self.cache.lock(disp.name) as handle:
                if handle is None:
                    return disp.name

                results = {}
                if worker_calc is self.calc:
                    with _calc_lock:
                        results["forces"] = worker_calc.get_forces(at_eval)
                        if getattr(self, "ir", False):
                            results["dipole"] = worker_calc.get_dipole_moment(at_eval)
                else:
                    results["forces"] = worker_calc.get_forces(at_eval)
                    if getattr(self, "ir", False):
                        results["dipole"] = worker_calc.get_dipole_moment(at_eval)

                if world.rank == 0:
                    handle.save(results)
            return disp.name

        try:
            with ThreadPoolExecutor(max_workers=self.n_workers) as pool:
                futures = [pool.submit(_compute_single_displacement, item) for item in pending]
                for fut in as_completed(futures):
                    fut.result()
                    pbar.update(1)
        finally:
            pbar.close()

    def run(self) -> None:
        """Overrides `run` to invoke high-throughput parallel displacement evaluation."""
        self.run_parallel(show_progress=True)

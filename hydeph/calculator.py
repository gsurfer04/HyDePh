"""ASE calculator combining a foundation NEP potential with a defect Delta-NEP:
  V_hybrid(R) = V_base(R) + Delta_V_nep(R)
  F_hybrid(R) = F_base(R) + Delta_F_nep(R)
"""

from __future__ import annotations

import os
import logging
from typing import Optional, List, Union
import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from calorine.calculators import CPUNEP

logger = logging.getLogger(__name__)


def load_foundation_calculator(
    potential_path: Optional[str] = None,
    existing_calc: Optional[Calculator] = None,
    allow_missing: bool = False,
) -> Optional[Calculator]:
    """Loads a CPUNEP foundation potential calculator with fail-loud error reporting.

    Parameters
    ----------
    potential_path : str, optional
        Path to the GPUMD/NEP foundation potential file (e.g. nep89_20250409.txt).
    existing_calc : Calculator, optional
        An existing calculator instance to reuse, avoiding redundant model re-initialization.
    allow_missing : bool
        If False (default), raises FileNotFoundError or RuntimeError if the model cannot be loaded.
        If True, logs a warning and returns None.

    Returns
    -------
    Optional[Calculator]
        An initialized CPUNEP calculator instance, or None if potential_path is empty and allow_missing is True.
    """
    if existing_calc is not None:
        return existing_calc
    if not potential_path:
        if allow_missing:
            return None
        raise ValueError("Foundation potential path was not specified.")
    if not os.path.isfile(potential_path):
        if allow_missing:
            logger.warning(f"Foundation potential file not found: {potential_path}")
            return None
        raise FileNotFoundError(
            f"Foundation potential file not found: {potential_path}. "
            "Please provide a valid path to a GPUMD NEP potential file."
        )
    try:
        return CPUNEP(potential_path)
    except Exception as exc:
        if allow_missing:
            logger.warning(f"Could not initialize foundation potential from '{potential_path}': {exc}")
            return None
        raise RuntimeError(
            f"Failed to initialize foundation potential from '{potential_path}': {exc}"
        ) from exc


class HybridDefectCalculator(Calculator):
    """Hybrid defect potential evaluator combining a foundation model and a defect Delta-NEP."""

    implemented_properties = ['energy', 'forces']

    def __init__(
        self,
        foundation_pot_path: Optional[str] = None,
        delta_pot_path: Optional[Union[str, List[str]]] = None,
        calc_base: Optional[Calculator] = None,
        calc_delta: Optional[Calculator] = None,
        delta_pot_paths: Optional[List[str]] = None,
        energy_offset_per_atom: Optional[float] = None,
    ):
        super().__init__()
        self.calc_base = load_foundation_calculator(
            potential_path=foundation_pot_path,
            existing_calc=calc_base,
            allow_missing=False,
        )

        raw_paths: List[str] = []
        if delta_pot_paths:
            raw_paths = list(delta_pot_paths)
        elif isinstance(delta_pot_path, (list, tuple)):
            raw_paths = list(delta_pot_path)
        elif delta_pot_path is not None:
            raw_paths = [delta_pot_path]

        self.delta_pot_paths = raw_paths
        self.calc_deltas: List[Calculator] = []

        if calc_delta is not None:
            self.calc_delta = calc_delta
            self.calc_deltas = [calc_delta]
        elif raw_paths:
            for p in raw_paths:
                if not os.path.isfile(p):
                    raise FileNotFoundError(f"Delta-NEP potential file not found: {p}")
                self.calc_deltas.append(CPUNEP(p))
            self.calc_delta = self.calc_deltas[0]
        else:
            raise ValueError("Either `delta_pot_path` or `calc_delta` must be provided.")

        # Baseline mean energy offset per atom
        if energy_offset_per_atom is not None:
            self.energy_offset_per_atom = float(energy_offset_per_atom)
        elif raw_paths:
            import json
            primary_path = raw_paths[0]
            offset_cand1 = primary_path + ".offset.json"
            offset_cand2 = os.path.join(os.path.dirname(os.path.abspath(primary_path)), "delta_offset.json")
            loaded_offset = 0.0
            for cand in (offset_cand1, offset_cand2):
                if os.path.isfile(cand):
                    try:
                        with open(cand, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        loaded_offset = float(data.get("energy_offset_per_atom_eV", 0.0))
                        break
                    except Exception:
                        pass
            self.energy_offset_per_atom = loaded_offset
        else:
            self.energy_offset_per_atom = 0.0

    @property
    def is_committee(self) -> bool:
        """Returns True if this calculator uses a committee ensemble."""
        return len(self.calc_deltas) > 1

    def calculate(
        self,
        atoms: Optional[Atoms] = None,
        properties: List[str] = ['energy', 'forces'],
        system_changes=all_changes,
    ):
        """Calculates energy and/or forces gated on requested properties."""
        super().calculate(atoms, properties, system_changes)

        req_props = [p for p in properties if p in self.implemented_properties]
        if not req_props:
            return

        self.calc_base.calculate(atoms, req_props, system_changes)
        e_base = float(self.calc_base.results.get('energy', 0.0))
        f_base = np.asarray(self.calc_base.results.get('forces', np.zeros((len(atoms), 3))), dtype=float)

        e_deltas: List[float] = []
        f_deltas: List[np.ndarray] = []

        for calc_d in self.calc_deltas:
            calc_d.calculate(atoms, req_props, system_changes)
            if 'energy' in req_props:
                e_deltas.append(float(calc_d.results['energy']))
            if 'forces' in req_props:
                f_deltas.append(np.asarray(calc_d.results['forces'], dtype=float))

        offset_tot = float(len(atoms) * self.energy_offset_per_atom)

        if 'energy' in req_props:
            mean_e_delta = float(np.mean(e_deltas))
            self.results['energy'] = float(e_base + mean_e_delta + offset_tot)
            if len(e_deltas) > 1:
                self.results['energy_std'] = float(np.std(e_deltas))

        if 'forces' in req_props:
            mean_f_delta = np.mean(f_deltas, axis=0)
            self.results['forces'] = f_base + mean_f_delta
            if len(f_deltas) > 1:
                self.results['forces_std'] = np.std(f_deltas, axis=0)


class PairedDefectCalculator:
    """Paired ground- and excited-state hybrid defect potential.
    
    Evaluates separate ground state V_GS(R) and optical excited state V_ES(R) potential
    energy surfaces, and provides the instantaneous electronic transition gap:
      Delta U(R) = V_ES(R) - V_GS(R)
    """

    def __init__(
        self,
        calc_gs: HybridDefectCalculator,
        calc_es: HybridDefectCalculator,
    ):
        self.calc_gs = calc_gs
        self.calc_es = calc_es

    def get_potential_energy_gs(self, atoms: Atoms) -> float:
        """Returns the ground state potential energy V_GS(R)."""
        return float(self.calc_gs.get_potential_energy(atoms))

    def get_potential_energy_es(self, atoms: Atoms) -> float:
        """Returns the optical excited state potential energy V_ES(R)."""
        return float(self.calc_es.get_potential_energy(atoms))

    def get_energy_gap(self, atoms: Atoms) -> float:
        """Returns the instantaneous electronic transition energy gap Delta U(R) = V_ES(R) - V_GS(R)."""
        return float(self.get_potential_energy_es(atoms) - self.get_potential_energy_gs(atoms))

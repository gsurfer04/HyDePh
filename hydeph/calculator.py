"""ASE calculator combining a foundation NEP potential with a defect Delta-NEP:
  V_hybrid(R) = V_base(R) + Delta_V_nep(R)
  F_hybrid(R) = F_base(R) + Delta_F_nep(R)
"""

from __future__ import annotations

import logging
from typing import Optional, List
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from calorine.calculators import CPUNEP

logger = logging.getLogger(__name__)


class HybridDefectCalculator(Calculator):
    """Hybrid defect potential evaluator combining a foundation model and a defect Delta-NEP."""

    implemented_properties = ['energy', 'forces']

    def __init__(
        self,
        foundation_pot_path: Optional[str] = None,
        delta_pot_path: Optional[str] = None,
        calc_base: Optional[Calculator] = None,
        calc_delta: Optional[Calculator] = None,
    ):
        super().__init__()
        if calc_base is not None:
            self.calc_base = calc_base
        elif foundation_pot_path is not None:
            self.calc_base = CPUNEP(foundation_pot_path)
        else:
            raise ValueError("Either `foundation_pot_path` or `calc_base` must be provided.")

        if calc_delta is not None:
            self.calc_delta = calc_delta
        elif delta_pot_path is not None:
            self.calc_delta = CPUNEP(delta_pot_path)
        else:
            raise ValueError("Either `delta_pot_path` or `calc_delta` must be provided.")

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
        self.calc_delta.calculate(atoms, req_props, system_changes)

        if 'energy' in req_props:
            self.results['energy'] = float(
                self.calc_base.results['energy'] + self.calc_delta.results['energy']
            )
        if 'forces' in req_props:
            self.results['forces'] = (
                self.calc_base.results['forces'] + self.calc_delta.results['forces']
            )

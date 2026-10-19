"""
ASE calculator combining a foundation NEP potential with a defect Delta-NEP:
  V_hybrid(R) = V_base(R) + Delta_V_nep(R)
  F_hybrid(R) = F_base(R) + Delta_F_nep(R)
"""

from ase.calculators.calculator import Calculator, all_changes
from calorine.calculators import CPUNEP


class HybridDefectCalculator(Calculator):
    implemented_properties = ['energy', 'forces']

    def __init__(self, foundation_pot_path, delta_pot_path, calc_base=None):
        super().__init__()
        self.calc_base = calc_base if calc_base is not None else CPUNEP(foundation_pot_path)
        self.calc_delta = CPUNEP(delta_pot_path)

    def calculate(self, atoms=None, properties=['energy', 'forces'], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)

        self.calc_base.calculate(atoms, properties, system_changes)
        self.calc_delta.calculate(atoms, properties, system_changes)

        self.results['energy'] = self.calc_base.results['energy'] + self.calc_delta.results['energy']
        self.results['forces'] = self.calc_base.results['forces'] + self.calc_delta.results['forces']

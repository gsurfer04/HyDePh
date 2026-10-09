"""
defect_delta_nep.calculator
===========================
Defines the HybridDefectCalculator combining the universal foundation potential
(nep89) with the defect-specific Delta-NEP.

  V_hybrid(R) = V_base(R) + Delta_V_nep(R)
  F_hybrid(R) = F_base(R) + Delta_F_nep(R)
"""

from ase.calculators.calculator import Calculator, all_changes
from calorine.calculators import CPUNEP


class HybridDefectCalculator(Calculator):
    implemented_properties = ['energy', 'forces']

    def __init__(self, foundation_pot_path, delta_pot_path):
        super().__init__()
        self.calc_base = CPUNEP(foundation_pot_path)
        self.calc_delta = CPUNEP(delta_pot_path)

    def calculate(self, atoms=None, properties=['energy', 'forces'], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        
        # 1. Base foundation potential (nep89)
        self.calc_base.calculate(atoms, properties, system_changes)
        e_base = self.calc_base.results['energy']
        f_base = self.calc_base.results['forces']
        
        # 2. Delta-NEP potential
        self.calc_delta.calculate(atoms, properties, system_changes)
        e_delta = self.calc_delta.results['energy']
        f_delta = self.calc_delta.results['forces']
        
        # 3. Hybrid sum
        self.results['energy'] = e_base + e_delta
        self.results['forces'] = f_base + f_delta

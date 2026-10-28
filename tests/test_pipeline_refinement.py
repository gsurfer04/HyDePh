"""
Test end-to-end pipeline hybrid refinement and mic_displacement execution.
"""

import numpy as np
from unittest.mock import patch
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from hydeph.config import ToolkitConfig
from hydeph.pipeline import run_pipeline


class MockHarmonicCalc(Calculator):
    """Simple harmonic oscillator calculator for testing BFGS and vibrations."""
    implemented_properties = ["energy", "forces"]

    def __init__(self, k=2.0, r0=None):
        super().__init__()
        self.k = k
        self.r0 = r0

    def calculate(self, atoms=None, properties=["energy", "forces"], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        pos = atoms.get_positions()
        if self.r0 is None:
            self.r0 = np.zeros_like(pos)
        dr = pos - self.r0
        # E = 0.5 * k * dr^2
        energy = float(0.5 * self.k * np.sum(dr ** 2))
        # F = -k * dr
        forces = -self.k * dr
        self.results["energy"] = energy
        self.results["forces"] = forces


def test_pipeline_hybrid_refinement_executes_cleanly(tmp_path):
    """Verify that run_pipeline executes Step 5 refinement and mic_displacement without NameError."""
    # Create simple bulk pristine cell
    pristine_file = tmp_path / "pristine.xyz"
    pristine = Atoms("C2", positions=[[0, 0, 0], [1.5, 1.5, 1.5]], cell=[4, 4, 4], pbc=True)
    import ase.io
    ase.io.write(str(pristine_file), pristine)

    pot_file = tmp_path / "nep89.txt"
    pot_file.write_text("nep model header\n")

    cfg = ToolkitConfig(
        pristine_cell_path=str(pristine_file),
        foundation_potential_path=str(pot_file),
        workspace_dir=str(tmp_path / "ws"),
        temperatures=[10.0],
        enable_anharmonic=False,
        plot_configuration_coordinate=True,
    )

    at_gs = Atoms("C2", positions=[[0, 0, 0], [1.5, 1.5, 1.5]], cell=[4, 4, 4], pbc=True)
    at_gs.info["dft_gap_eV"] = 2.0
    at_es = Atoms("C2", positions=[[0, 0, 0], [1.6, 1.6, 1.6]], cell=[4, 4, 4], pbc=True)

    # Calculator with minimum at [0.02, 0.02, 0.02], which forces max_f > 0.015 on initial at_gs
    mock_calc = MockHarmonicCalc(k=5.0, r0=np.array([[0.0, 0.0, 0.0], [1.55, 1.55, 1.55]]))

    labeled_frame = at_gs.copy()
    labeled_frame.info["sample_type"] = "franck_condon"
    labeled_frame.info["lambda"] = 0.0
    labeled_frame.info["energy"] = -100.0
    labeled_frame.info["dft_gap_eV"] = 2.0

    with patch("hydeph.pipeline.DefectBuilder") as mock_builder_cls, \
         patch("hydeph.pipeline.FrameSampler") as mock_sampler_cls, \
         patch("hydeph.pipeline.QERunner") as mock_qe_cls, \
         patch("hydeph.pipeline.DeltaTrainer") as mock_trainer_cls, \
         patch("hydeph.pipeline.HybridDefectCalculator", return_value=mock_calc):

        builder_instance = mock_builder_cls.return_value
        builder_instance.calc_foundation = mock_calc
        builder_instance.build_supercell_and_defect.return_value = (at_gs, at_gs)
        builder_instance.relax_ground_state.return_value = at_gs
        builder_instance.generate_excited_state_geometry.return_value = (at_es, 0.5)

        sampler_instance = mock_sampler_cls.return_value
        sampler_instance.assemble_candidate_dataset.return_value = [labeled_frame]

        qe_instance = mock_qe_cls.return_value
        qe_instance.label_all_frames.return_value = [labeled_frame]
        qe_instance.label_fc_excited_state.return_value = [labeled_frame]
        qe_instance.label_excited_state_frames.return_value = [labeled_frame]
        qe_instance.last_failed_entries = []

        trainer_instance = mock_trainer_cls.return_value
        trainer_instance.compute_delta_dataset.return_value = [labeled_frame]
        trainer_instance.train_delta_nep.return_value = str(tmp_path / "mock_delta.txt")

        summary = run_pipeline(cfg)

        assert summary is not None
        assert "delta_potential" in summary
        assert "lineshapes" in summary
        assert "stokes_analysis" in summary
        assert summary["stokes_analysis"] is not None
        assert "e_zpl_eV" in summary["stokes_analysis"]
        assert summary["stokes_analysis"]["e_zpl_eV"] > 0

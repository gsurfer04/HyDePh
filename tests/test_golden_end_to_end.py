"""Golden end-to-end integration test verifying Stages 1 to 6 with mock pw.x and nep binaries."""

import json
from unittest.mock import patch
import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from hydeph.config import ToolkitConfig
from hydeph.pipeline import run_pipeline


class MockGoldenCalc(Calculator):
    """Deterministic toy calculator providing stable energies and forces."""
    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, properties=["energy", "forces"], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        if len(atoms) >= 2:
            dr = atoms.positions[1] - atoms.positions[0] - np.array([1.5, 0, 0])
            dist = np.linalg.norm(dr)
            energy = -20.0 + 5.0 * (dist ** 2)
            forces = np.zeros((len(atoms), 3))
            forces[0] = 5.0 * dr
            forces[1] = -5.0 * dr
        else:
            energy = -10.0
            forces = np.zeros((len(atoms), 3))
        self.results["energy"] = energy
        self.results["forces"] = forces


def test_golden_end_to_end_pipeline(tmp_path):
    """Executes Stages 1-6 end-to-end on a minimal 2-atom system and verifies all artifacts."""
    ws = tmp_path / "ws_golden"
    ws.mkdir()

    pristine_file = tmp_path / "pristine.xyz"
    pristine = Atoms("C2", positions=[[0, 0, 0], [1.5, 0, 0]], cell=[5, 5, 5], pbc=True)
    pristine.write(str(pristine_file))

    pot_file = tmp_path / "foundation_nep.txt"
    pot_file.write_text("dummy foundation nep potential", encoding="utf-8")

    pseudo_dir = tmp_path / "pseudo"
    pseudo_dir.mkdir()
    (pseudo_dir / "C.upf").write_text('<PP_HEADER z_valence="4.0" />\n', encoding="utf-8")

    cfg = ToolkitConfig(
        pristine_cell_path=str(pristine_file),
        foundation_potential_path=str(pot_file),
        workspace_dir=str(ws),
        pseudo_dir=str(pseudo_dir),
        pseudopotentials={"C": "C.upf"},
        spin_multiplicity=1,
        tot_charge=0.0,
        supercell_repeat=(1, 1, 1),
        phonon_supercell_repeat=(1, 1, 1),
        n_fc_frames=3,
        n_phonon_frames=2,
        n_md_frames=0,
        max_dft_frames=5,
        temperatures=[10.0, 77.0, 300.0],
        allow_defect_vc_relax=True,
        relaxation_calculation="relax",
        phonon_smearing_meV=5.0,
    )

    toy_calc = MockGoldenCalc()

    # Mock DFT execution in QERunner
    def mock_label_single(atoms, calc_dir, **kwargs):
        at_lab = atoms.copy()
        at_lab.calc = None
        if len(at_lab) >= 2:
            dist = np.linalg.norm(at_lab.positions[1] - at_lab.positions[0] - np.array([1.5, 0, 0]))
            at_lab.info["energy"] = -25.0 + 3.0 * (dist ** 2)
            forces = np.zeros((len(at_lab), 3))
            forces[0] = [0.01, 0.0, 0.0]
            forces[1] = [-0.01, 0.0, 0.0]
        else:
            at_lab.info["energy"] = -12.0
            forces = np.zeros((len(at_lab), 3))
        at_lab.arrays["forces"] = forces
        at_lab.info["qe_status"] = "converged"
        at_lab.info["sample_type"] = atoms.info.get("sample_type", "franck_condon")
        at_lab.info["dft_gap_eV"] = 2.15
        at_lab.info["dft_homo_eV"] = -4.50
        at_lab.info["dft_lumo_eV"] = -2.35
        return at_lab

    # Mock DeltaTrainer
    def mock_train(self, frames, pot_filename="nep_delta_gs.txt", subdir="delta_nep_training_gs", **kwargs):
        nep_out = ws / pot_filename
        nep_out.write_text("mock trained delta nep", encoding="utf-8")
        parity_path = ws / "delta_nep_parity.json"
        parity_data = {
            "train_rmse_energy_meV_atom": 0.8,
            "train_rmse_force_eV_A": 0.035,
            "test_rmse_energy_meV_atom": 1.1,
            "test_rmse_force_eV_A": 0.042,
            "test_rmse_virial_eV_atom": 0.015,
            "pre_refinement_force_rmse_eV_A": 0.15,
        }
        parity_path.write_text(json.dumps(parity_data), encoding="utf-8")
        qc_path = ws / "delta_nep_qc_report.json"
        qc_data = {
            "all_passed": True,
            "test_energy_rmse_meV_atom": 1.1,
            "test_force_rmse_eV_A": 0.042,
            "gates": {
                "energy_rmse": {"value": 1.1, "threshold": 2.0, "passed": True},
                "force_rmse": {"value": 0.042, "threshold": 0.08, "passed": True},
            },
        }
        qc_path.write_text(json.dumps(qc_data), encoding="utf-8")
        return str(nep_out)

    def mock_run_relaxation(atoms, calc_dir, **kwargs):
        at_rel = atoms.copy()
        at_rel.calc = None
        if "es" in str(calc_dir).lower():
            at_rel.positions[0, 0] += 0.05
            at_rel.info["energy"] = -23.0
        else:
            at_rel.info["energy"] = -25.0
        at_rel.arrays["forces"] = np.zeros((len(atoms), 3))
        at_rel.info["dft_gap_eV"] = 2.15
        at_rel.info["dft_homo_eV"] = -4.50
        at_rel.info["dft_lumo_eV"] = -2.35
        at_rel.info["qe_status"] = "converged"
        return at_rel

    with patch("hydeph.defect_builder.CPUNEP", return_value=toy_calc), \
         patch("hydeph.calculator.CPUNEP", return_value=toy_calc), \
         patch("hydeph.frame_sampler.CPUNEP", return_value=toy_calc), \
         patch("hydeph.delta_trainer.CPUNEP", return_value=toy_calc), \
         patch("hydeph.qe.runner.QERunner.run_relaxation", side_effect=mock_run_relaxation), \
         patch("hydeph.qe.runner.QERunner.label_single_frame", side_effect=mock_label_single), \
         patch("hydeph.delta_trainer.DeltaTrainer.train_delta_nep", mock_train), \
         patch("hydeph.lineshape.LineshapeEngine.plot_all"):

        summary = run_pipeline(cfg, from_stage=1, to_stage=6)

        assert summary is not None
        assert "lineshapes" in summary
        assert summary.get("workspace_dir") == str(ws)

        # Verify all stage manifests exist
        for s in range(1, 7):
            m_path = ws / f"stage_{s}_manifest.json"
            assert m_path.is_file(), f"Missing manifest for Stage {s}"
            with open(m_path, "r", encoding="utf-8") as f:
                mdata = json.load(f)
            assert mdata["stage"] == s

        # Verify Stage 1 artifacts
        assert (ws / "relaxed_gs.xyz").is_file()
        assert (ws / "relaxed_es.xyz").is_file()

        # Verify Stage 2 artifacts
        assert (ws / "configuration_coordinate_data.json").is_file()
        assert (ws / "configuration_coordinate_curve.csv").is_file()

        # Verify Stage 3 artifacts
        assert (ws / "labeled_dft_frames.xyz").is_file()

        # Verify Stage 4 artifacts
        assert (ws / "nep_delta_gs.txt").is_file() or (ws / "nep_delta.txt").is_file()
        assert (ws / "delta_nep_parity.json").is_file()
        assert (ws / "delta_nep_qc_report.json").is_file()

        # Verify Stage 6 artifacts
        assert (ws / "optical_lineshapes.csv").is_file()
        assert (ws / "spectral_density.csv").is_file()
        assert (ws / "lineshape_summary.json").is_file()

        # Verify incremental provenance
        assert (ws / "incremental_provenance.json").is_file()
        with open(ws / "incremental_provenance.json", "r", encoding="utf-8") as f:
            prov = json.load(f)
        assert len(prov["stages"]) == 6
        assert "git_commit" in prov
        assert prov.get("hydeph_version") is not None

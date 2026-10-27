"""Tests for Batch 2: Fail-loud scientific robustness and provenance tracking."""

import os
import json
import pytest
import numpy as np
from ase import Atoms
from hydeph.config import ToolkitConfig
from hydeph.defect_builder import DefectBuilder
from hydeph.provenance import calculate_file_sha256, generate_provenance_manifest, write_provenance


class DummyCalc:
    """Mock ASE calculator."""
    def __init__(self):
        self.atoms = None

    def get_potential_energy(self, atoms=None, *args, **kwargs):
        return -100.0

    def get_forces(self, atoms=None, *args, **kwargs):
        if atoms is not None:
            return np.zeros((len(atoms), 3))
        return np.zeros((len(self.atoms), 3))


class FailingQERunner:
    """Mock QE runner that always raises a DFT failure."""
    def run_relaxation(self, *args, **kwargs):
        raise RuntimeError("Simulated Quantum ESPRESSO convergence crash")


def test_ground_state_dft_failure_raises_by_default(tmp_path):
    """When allow_fallbacks=False, ground state DFT relaxation failure raises RuntimeError."""
    pristine_path = str(tmp_path / "pristine.xyz")
    pristine = Atoms("C8", positions=np.eye(8, 3) * 1.5, cell=[6, 6, 6], pbc=True)
    pristine.write(pristine_path)

    cfg = ToolkitConfig(
        pristine_cell_path=pristine_path,
        workspace_dir=str(tmp_path / "ws_gs_fail"),
        ground_state_relax_method="dft",
        allow_fallbacks=False,
    )

    builder = DefectBuilder(cfg, calc_foundation=DummyCalc(), calc_qe=FailingQERunner())
    _, defect_sc = builder.build_supercell_and_defect()

    with pytest.raises(RuntimeError, match="Ground-state DFT relaxation failed"):
        builder.relax_ground_state(defect_sc)


def test_ground_state_dft_failure_allowed_with_flag(tmp_path):
    """When allow_fallbacks=True, ground state DFT relaxation failure falls back to foundation."""
    pristine_path = str(tmp_path / "pristine.xyz")
    pristine = Atoms("C8", positions=np.eye(8, 3) * 1.5, cell=[6, 6, 6], pbc=True)
    pristine.write(pristine_path)

    cfg = ToolkitConfig(
        pristine_cell_path=pristine_path,
        workspace_dir=str(tmp_path / "ws_gs_fb"),
        ground_state_relax_method="dft",
        allow_fallbacks=True,
    )

    builder = DefectBuilder(cfg, calc_foundation=DummyCalc(), calc_qe=FailingQERunner())
    _, defect_sc = builder.build_supercell_and_defect()

    at_gs = builder.relax_ground_state(defect_sc)
    assert at_gs.info.get("gs_source") == "foundation_fallback"


def test_excited_state_dft_failure_raises_by_default(tmp_path):
    """When allow_fallbacks=False, excited state DFT relaxation failure raises RuntimeError."""
    pristine_path = str(tmp_path / "pristine.xyz")
    pristine = Atoms("C8", positions=np.eye(8, 3) * 1.5, cell=[6, 6, 6], pbc=True)
    pristine.write(pristine_path)

    cfg = ToolkitConfig(
        pristine_cell_path=pristine_path,
        workspace_dir=str(tmp_path / "ws_es_fail"),
        excited_state_method="spin_proxy",
        ground_state_relax_method="foundation",
        allow_fallbacks=False,
    )

    builder = DefectBuilder(cfg, calc_foundation=DummyCalc(), calc_qe=FailingQERunner())
    _, defect_sc = builder.build_supercell_and_defect()
    at_gs = builder.relax_ground_state(defect_sc)

    with pytest.raises(RuntimeError, match="Excited-state DFT relaxation"):
        builder.generate_excited_state_geometry(at_gs)


def test_provenance_sha256_and_manifest(tmp_path):
    """Tests file hash calculation, manifest assembly, and JSON persistence."""
    sample_file = tmp_path / "test_file.txt"
    sample_file.write_text("HyDePh test provenance content", encoding="utf-8")

    file_hash = calculate_file_sha256(str(sample_file))
    assert file_hash is not None
    assert len(file_hash) == 64

    non_existent_hash = calculate_file_sha256(str(tmp_path / "does_not_exist.txt"))
    assert non_existent_hash is None

    cfg = ToolkitConfig(
        pristine_cell_path=str(sample_file),
        workspace_dir=str(tmp_path / "ws_prov"),
        zpl_energy_eV=1.945,
        allow_fallbacks=False,
    )

    at_gs = Atoms("C2", positions=[[0, 0, 0], [1.5, 0, 0]], cell=[5, 5, 5], pbc=True)
    at_gs.info["gs_source"] = "dft"
    at_gs.info["energy"] = -150.0

    at_es = at_gs.copy()
    at_es.info["es_source"] = "delta_scf"

    cc_results = {
        "zpl_source": "user_configured",
        "e_zpl_eV": 1.945,
    }

    manifest = generate_provenance_manifest(
        config=cfg,
        at_gs=at_gs,
        at_es=at_es,
        cc_results=cc_results,
        total_time_s=42.5,
    )

    assert manifest["allow_fallbacks"] is False
    assert manifest["origins"]["gs_source"] == "dft"
    assert manifest["origins"]["es_source"] == "delta_scf"
    assert manifest["origins"]["zpl_source"] == "user_configured"
    assert manifest["origins"]["zpl_energy_eV"] == 1.945
    assert manifest["input_files"]["pristine_cell_path"]["sha256"] == file_hash

    prov_path = write_provenance(str(tmp_path / "ws_prov"), manifest)
    assert os.path.isfile(prov_path)

    with open(prov_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["origins"]["gs_source"] == "dft"
    assert loaded["total_walltime_s"] == 42.5

"""Tests for CINECA Leonardo HPC integration, SLURM dispatch, GPU pinning, and ParallelVibrations."""

import os
import json
import pytest
import numpy as np
from ase import Atoms
from ase.calculators.emt import EMT
from ase.vibrations import Vibrations

from hydeph.config import ToolkitConfig
from hydeph.hpc.cluster_profiles import (
    ClusterProfile,
    LEONARDO_BOOSTER,
    LEONARDO_BOOSTER_DEBUG,
    LEONARDO_DCGP,
    GENERIC_SLURM,
    get_cluster_profile,
    detect_cluster_environment,
)
from hydeph.hpc.slurm_dispatcher import (
    is_slurm_active,
    get_slurm_job_id,
    get_array_task_id,
    get_pinned_worker_env,
    mark_frame_done,
    is_frame_done,
    SlurmDispatcher,
)
from hydeph.hpc.parallel_vib import ParallelVibrations


def test_cluster_profiles():
    """Validates predefined cluster profiles and profile lookup."""
    booster = get_cluster_profile("leonardo_booster")
    assert booster.name == "leonardo_booster"
    assert booster.gpus_per_node == 4
    assert booster.cpus_per_node == 32
    assert booster.partition == "boost_usr_prod"
    assert booster.qos is None
    assert booster.has_gpu is True
    assert "q-e/7.2" in booster.default_modules
    assert "cuda/11.8" in booster.default_modules

    dcgp = get_cluster_profile("leonardo_dcgp")
    assert dcgp.name == "leonardo_dcgp"
    assert dcgp.gpus_per_node == 0
    assert dcgp.cpus_per_node == 112
    assert dcgp.partition == "dcgp_usr_prod"
    assert dcgp.has_gpu is False

    # Alias check
    alias_booster = get_cluster_profile("leonardo")
    assert alias_booster.name == "leonardo_booster"

    # Fallback check
    fallback = get_cluster_profile("nonexistent_profile")
    assert fallback.name == "leonardo_booster"


def test_cluster_environment_detection(monkeypatch):
    """Tests environment variable detection on Leonardo login/compute nodes."""
    monkeypatch.setenv("HOSTNAME", "login01.leonardo.cineca.it")
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "leonardo")
    prof = detect_cluster_environment()
    assert prof is not None
    assert prof.name == "leonardo_booster"

    monkeypatch.setenv("SLURM_JOB_PARTITION", "dcgp_usr_prod")
    prof_dcgp = detect_cluster_environment()
    assert prof_dcgp is not None
    assert prof_dcgp.name == "leonardo_dcgp"


def test_slurm_environment_and_pinning(monkeypatch):
    """Tests SLURM environment queries and multi-GPU pinning logic."""
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_ARRAY_TASK_ID", raising=False)
    assert is_slurm_active() is False
    assert get_slurm_job_id() is None
    assert get_array_task_id() is None

    monkeypatch.setenv("SLURM_JOB_ID", "1234567")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "7")
    assert is_slurm_active() is True
    assert get_slurm_job_id() == "1234567"
    assert get_array_task_id() == 7

    # Test GPU pinning for 4-GPU Booster node
    env0 = get_pinned_worker_env(worker_idx=0, gpus_per_node=4, cpus_per_worker=8)
    assert env0["CUDA_VISIBLE_DEVICES"] == "0"
    assert env0["OMP_NUM_THREADS"] == "8"

    env1 = get_pinned_worker_env(worker_idx=1, gpus_per_node=4, cpus_per_worker=8)
    assert env1["CUDA_VISIBLE_DEVICES"] == "1"

    env3 = get_pinned_worker_env(worker_idx=3, gpus_per_node=4, cpus_per_worker=8)
    assert env3["CUDA_VISIBLE_DEVICES"] == "3"

    # Worker 4 wraps around to GPU 0
    env4 = get_pinned_worker_env(worker_idx=4, gpus_per_node=4, cpus_per_worker=8)
    assert env4["CUDA_VISIBLE_DEVICES"] == "0"


def test_atomic_frame_markers(tmp_path):
    """Validates atomic .done marker creation and querying on Lustre scratch."""
    calc_dir = str(tmp_path / "frame_0001")
    assert is_frame_done(calc_dir) is False

    mark_frame_done(calc_dir, metadata={"energy_eV": -123.456, "status": "converged"})
    assert is_frame_done(calc_dir) is True

    done_file = os.path.join(calc_dir, ".done")
    assert os.path.isfile(done_file)
    with open(done_file, "r") as f:
        data = json.load(f)
    assert data["metadata"]["energy_eV"] == -123.456
    assert "timestamp" in data
    assert "hostname" in data


def test_slurm_job_array_script_generation(tmp_path):
    """Tests production of valid SLURM job array submission scripts."""
    dispatcher = SlurmDispatcher(
        profile=LEONARDO_BOOSTER,
        account="cin_test_01",
        workspace_dir=str(tmp_path / "run"),
    )
    cfg_file = str(tmp_path / "config.json")
    with open(cfg_file, "w") as f:
        json.dump({"dummy": True}, f)

    out_script = str(tmp_path / "submit_array.slurm")
    script = dispatcher.generate_job_array_script(
        config_path=cfg_file,
        n_frames=40,
        chunk_size=1,
        max_concurrent=16,
        time_limit="02:00:00",
        script_output_path=out_script,
        conda_env_name="hydeph_env",
    )

    assert os.path.isfile(out_script)
    assert "#SBATCH --array=0-39%16" in script
    assert "#SBATCH --account=cin_test_01" in script
    assert "#SBATCH --partition=boost_usr_prod" in script
    assert "--qos" not in script
    assert "#SBATCH --gres=gpu:1" in script
    assert "module load q-e/7.2" in script
    assert "module load cuda/11.8" in script
    assert "conda activate hydeph_env" in script
    assert "python -m hydeph.hpc.array_worker" in script


def test_salloc_command_generation():
    """Tests salloc command generation for interactive Leonardo sessions."""
    from hydeph.hpc.slurm_dispatcher import generate_salloc_command

    cmd_booster = generate_salloc_command(
        profile=LEONARDO_BOOSTER,
        account="cin_interactive_01",
        nodes=1,
        time_limit="02:00:00",
        debug=False,
    )
    assert cmd_booster.startswith("salloc -A cin_interactive_01 -p boost_usr_prod")
    assert "--qos" not in cmd_booster
    assert "--gres=gpu:4" in cmd_booster
    assert "--nodes=1" in cmd_booster
    assert "--ntasks-per-node=4" in cmd_booster
    assert "--cpus-per-task=8" in cmd_booster
    assert "--time=02:00:00" in cmd_booster

    cmd_dbg = generate_salloc_command(
        profile=LEONARDO_BOOSTER,
        account="cin_interactive_01",
        nodes=1,
        time_limit="00:30:00",
        debug=True,
    )
    assert "--qos=boost_qos_dbg" in cmd_dbg

    cmd_dcgp = generate_salloc_command(
        profile=LEONARDO_DCGP,
        account="cin_interactive_01",
        nodes=1,
        time_limit="01:00:00",
    )
    assert "-p dcgp_usr_prod" in cmd_dcgp
    assert "--gres=gpu" not in cmd_dcgp
    assert "--cpus-per-task=16" in cmd_dcgp


def test_parallel_vibrations_numerical_equivalence(tmp_path):
    """Tests that ParallelVibrations produces identical eigenfrequencies to standard ASE Vibrations."""
    # Create simple dimer system with EMT calculator
    atoms_seq = Atoms("N2", positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 1.10)])
    atoms_seq.calc = EMT()

    atoms_par = Atoms("N2", positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 1.10)])
    atoms_par.calc = EMT()

    # 1. Sequential Vibrations
    vib_seq_dir = str(tmp_path / "vib_seq")
    os.makedirs(vib_seq_dir, exist_ok=True)
    vib_seq = Vibrations(atoms_seq, name=os.path.join(vib_seq_dir, "vib"), delta=0.01)
    vib_seq.run()
    energies_seq = vib_seq.get_energies()

    # 2. ParallelVibrations with 2 concurrent workers
    vib_par_dir = str(tmp_path / "vib_par")
    os.makedirs(vib_par_dir, exist_ok=True)
    vib_par = ParallelVibrations(
        atoms_par, name=os.path.join(vib_par_dir, "vib"), delta=0.01, n_workers=2
    )
    vib_par.run()
    energies_par = vib_par.get_energies()

    # Verify identical eigenvalues to machine precision
    np.testing.assert_allclose(energies_seq, energies_par, atol=1e-6)


def test_toolkit_config_hpc_extensions(tmp_path):
    """Validates HPC configuration fields in ToolkitConfig, serialization, and deserialization."""
    pristine_xyz = str(tmp_path / "pristine.xyz")
    Atoms("Si", positions=[(0, 0, 0)]).write(pristine_xyz)

    cfg = ToolkitConfig(
        pristine_cell_path=pristine_xyz,
        hpc_scheduler="slurm",
        cluster_profile="leonardo_booster",
        slurm_account="cin_project_123",
        parallel_hessian=True,
        parallel_hessian_workers=8,
    )

    assert cfg.hpc_scheduler == "slurm"
    assert cfg.cluster_profile == "leonardo_booster"
    assert cfg.slurm_account == "cin_project_123"
    assert cfg.slurm_partition == "boost_usr_prod"
    assert cfg.slurm_qos is None
    assert "q-e/7.2" in cfg.slurm_modules
    assert cfg.parallel_hessian is True
    assert cfg.parallel_hessian_workers == 8

    # Serialization test
    d = cfg.to_dict()
    assert d["hpc_scheduler"] == "slurm"
    assert d["slurm_partition"] == "boost_usr_prod"

    # Deserialization test
    cfg_restored = ToolkitConfig.from_dict(d)
    assert cfg_restored.hpc_scheduler == "slurm"
    assert cfg_restored.slurm_account == "cin_project_123"
    assert cfg_restored.slurm_partition == "boost_usr_prod"


def test_prepare_all_dft_inputs_and_slurm_scripts(tmp_path):
    """Tests that prepare_all_dft_inputs generates pw.in decks and standalone SLURM runner scripts."""
    from hydeph.qe_runner import QERunner

    ws = str(tmp_path / "ws_dft_prep")
    os.makedirs(ws, exist_ok=True)

    pristine_xyz = str(tmp_path / "pristine.xyz")
    Atoms("C2", positions=[(0, 0, 0), (0, 0, 1.4)], cell=[10, 10, 10], pbc=True).write(pristine_xyz)

    pseudo_dir = str(tmp_path / "pseudo")
    os.makedirs(pseudo_dir, exist_ok=True)
    c_upf = os.path.join(pseudo_dir, "C.upf")
    with open(c_upf, "w") as f:
        f.write('<PP_HEADER z_valence="4.0" />\n')

    cfg = ToolkitConfig(
        pristine_cell_path=pristine_xyz,
        workspace_dir=ws,
        pseudo_dir=pseudo_dir,
        pseudopotentials={"C": "C.upf"},
        cluster_profile="leonardo_booster",
        slurm_account="cin_staff",
        spin_multiplicity=1,
    )
    qe = QERunner(cfg, pw_exe="pw.x")

    # Create 2 candidate frames
    at1 = Atoms("C2", positions=[(0, 0, 0), (0, 0, 1.4)], cell=[10, 10, 10], pbc=True)
    at1.info["sample_type"] = "franck_condon"
    at1.info["frame_id"] = "fc_00"
    at1.info["candidate_index"] = 0
    at1.info["lambda"] = 0.0

    at2 = Atoms("C2", positions=[(0, 0, 0), (0, 0, 1.5)], cell=[10, 10, 10], pbc=True)
    at2.info["sample_type"] = "franck_condon"
    at2.info["frame_id"] = "fc_01"
    at2.info["candidate_index"] = 1
    at2.info["lambda"] = 1.0

    frames = [at1, at2]
    prep = qe.prepare_all_dft_inputs(frames, at_gs=at1)

    assert prep["n_gs_frames"] == 2
    assert os.path.isfile(os.path.join(ws, "dft_calculations", "frame_000_fc_00", "pw.in"))
    assert os.path.isfile(os.path.join(ws, "dft_calculations", "frame_001_fc_01", "pw.in"))

    # Verify standalone SLURM script
    slurm_script = os.path.join(ws, "submit_dft_pure_qe.slurm")
    assert os.path.isfile(slurm_script)
    with open(slurm_script, "r") as f:
        content = f.read()
    assert "module load q-e/7.2" in content
    assert "if module is-avail" not in content
    assert "CUDA_VISIBLE_DEVICES" in content
    assert "pw.x" in content
    # No python invocation required for execution on compute node!
    assert not any(line.strip().startswith("python ") for line in content.splitlines())

    # Verify array script and list
    array_script = os.path.join(ws, "submit_dft_array_pure_qe.slurm")
    assert os.path.isfile(array_script)
    list_file = os.path.join(ws, "dft_frames_list.txt")
    assert os.path.isfile(list_file)


def test_ingest_completed_calculations(tmp_path):
    """Tests ingesting completed pw.out calculations from disk without executing pw.x."""
    from hydeph.qe_runner import QERunner

    ws = str(tmp_path / "ws_ingest")
    os.makedirs(ws, exist_ok=True)

    pristine_xyz = str(tmp_path / "pristine_ingest.xyz")
    Atoms("C2", positions=[(0, 0, 0), (0, 0, 1.4)], cell=[10, 10, 10], pbc=True).write(pristine_xyz)

    pseudo_dir = str(tmp_path / "pseudo")
    os.makedirs(pseudo_dir, exist_ok=True)
    c_upf = os.path.join(pseudo_dir, "C.upf")
    with open(c_upf, "w") as f:
        f.write('<PP_HEADER z_valence="4.0" />\n')

    pot_file = str(tmp_path / "nep.txt")
    with open(pot_file, "w") as f:
        f.write("dummy nep foundation potential")

    cfg = ToolkitConfig(
        pristine_cell_path=pristine_xyz,
        foundation_potential_path=pot_file,
        workspace_dir=ws,
        pseudo_dir=pseudo_dir,
        pseudopotentials={"C": "C.upf"},
        spin_multiplicity=1,
        pw_exe="pw.x",
    )
    qe = QERunner(cfg, pw_exe="pw.x")

    at1 = Atoms("C2", positions=[(0, 0, 0), (0, 0, 1.4)], cell=[10, 10, 10], pbc=True)
    at1.info["sample_type"] = "franck_condon"
    at1.info["frame_id"] = "fc_00"
    at1.info["candidate_index"] = 0
    at1.info["lambda"] = 0.0

    # Write pw.in and mock converged pw.out
    calc_dir = os.path.join(ws, "dft_calculations", "frame_000_fc_00")
    os.makedirs(calc_dir, exist_ok=True)
    pw_in = os.path.join(calc_dir, "pw.in")
    from hydeph.qe_runner import write_qe_input
    write_qe_input(at1, pw_in, cfg)

    pw_out = os.path.join(calc_dir, "pw.out")
    mock_pw_out = """
     Program PWSCF v.7.2 starts
     !    total energy              =     -22.50000000 Ry
     Forces acting on atoms (cartesian axes, Ry/au):
     atom    1 type  1   force =     0.00000000    0.00000000    0.01000000
     atom    2 type  1   force =     0.00000000    0.00000000   -0.01000000
     Total force =     0.014142     Total SCF correction =     0.000001
     the Fermi energy is    -3.5000 ev
     JOB DONE.
"""
    with open(pw_out, "w") as f:
        f.write(mock_pw_out)

    labeled_gs, labeled_es = qe.ingest_completed_calculations([at1], at_gs=at1)
    assert len(labeled_gs) == 1
    assert labeled_gs[0].info["qe_status"] == "converged"
    assert pytest.approx(labeled_gs[0].info["energy"], 1e-3) == -22.5 * 13.6056980659
    assert labeled_gs[0].arrays["forces"].shape == (2, 3)

    out_xyz = os.path.join(ws, "labeled_dft_frames.xyz")
    assert os.path.isfile(out_xyz)


def test_run_pipeline_to_stage(tmp_path):
    """Tests that run_pipeline stops cleanly at requested stage."""
    from unittest.mock import patch
    from hydeph.pipeline import run_pipeline

    ws = str(tmp_path / "ws_pipe_stage")
    os.makedirs(ws, exist_ok=True)

    pristine_xyz = str(tmp_path / "pristine_p.xyz")
    Atoms("C2", positions=[(0, 0, 0), (0, 0, 1.4)], cell=[10, 10, 10], pbc=True).write(pristine_xyz)

    pseudo_dir = str(tmp_path / "pseudo")
    os.makedirs(pseudo_dir, exist_ok=True)
    c_upf = os.path.join(pseudo_dir, "C.upf")
    with open(c_upf, "w") as f:
        f.write('<PP_HEADER z_valence="4.0" />\n')

    pot_file = str(tmp_path / "nep.txt")
    with open(pot_file, "w") as f:
        f.write("dummy nep foundation potential")

    cfg = ToolkitConfig(
        pristine_cell_path=pristine_xyz,
        foundation_potential_path=pot_file,
        workspace_dir=ws,
        pseudo_dir=pseudo_dir,
        pseudopotentials={"C": "C.upf"},
        spin_multiplicity=1,
        pw_exe="pw.x",
    )

    with patch("hydeph.pipeline.DefectBuilder") as mock_builder_cls:
        mock_builder = mock_builder_cls.return_value
        at_gs = Atoms("C2", positions=[(0, 0, 0), (0, 0, 1.4)], cell=[10, 10, 10], pbc=True)
        at_es = Atoms("C2", positions=[(0, 0, 0), (0, 0, 1.5)], cell=[10, 10, 10], pbc=True)
        mock_builder.build_supercell_and_defect.return_value = (at_gs, at_gs)
        mock_builder.relax_ground_state.return_value = at_gs
        mock_builder.generate_excited_state_geometry.return_value = (at_es, 0.5)

        res1 = run_pipeline(cfg, to_stage=1)
        assert res1["stage"] == 1
        assert "candidate_frames" not in res1


def test_prepare_relaxation_inputs(tmp_path):
    """Verify prepare_relaxation_inputs generates gs_relax, es_relax, and SLURM runner script."""
    from hydeph.qe_runner import QERunner
    from hydeph.defect_builder import DefectBuilder

    ws = str(tmp_path / "ws_relax_prep")
    os.makedirs(ws, exist_ok=True)
    pseudo_dir = str(tmp_path / "pseudo")
    os.makedirs(pseudo_dir, exist_ok=True)
    with open(os.path.join(pseudo_dir, "C.upf"), "w") as f:
        f.write('<PP_HEADER z_valence="4.0" />\n')

    pristine_p = str(tmp_path / "pristine.xyz")
    at = Atoms("C3", positions=[(0, 1, 0), (np.sqrt(3)/2, -0.5, 0), (-np.sqrt(3)/2, -0.5, 0)], cell=[10, 10, 10], pbc=True)
    at.write(pristine_p)

    cfg = ToolkitConfig(
        pristine_cell_path=pristine_p,
        workspace_dir=ws,
        pseudo_dir=pseudo_dir,
        pseudopotentials={"C": "C.upf"},
        excited_state_method="delta_scf",
        delta_scf_homo_band=5,
        delta_scf_lumo_band=6,
        delta_scf_lumo_degeneracy=1,
        delta_scf_spin_channel=2,
        auto_jahn_teller_distortion=True,
        jahn_teller_amplitude=0.04,
        spin_multiplicity=3,
    )

    builder = DefectBuilder(cfg)
    defect_sc = at.copy()
    defect_sc.info["defect_center"] = [0.0, 0.0, 0.0]

    res = builder.prepare_relaxation_inputs(defect_sc)

    assert os.path.isfile(res["gs_relax_in"])
    assert os.path.isfile(res["gs_scf_in"])
    assert os.path.isfile(res["es_relax_in"])
    assert os.path.isfile(res["slurm_script"])

    # Verify SLURM script contains srun, modules, and extraction
    with open(res["slurm_script"], "r") as f:
        content = f.read()
    assert "qe_gs_relax" in content
    assert "qe_es_relax" in content
    assert "srun" in content
    assert "relaxed_gs.xyz" in content


def test_resumption_from_cluster_relaxation(tmp_path):
    """Verify pipeline and DefectBuilder resume from cluster pw.out without re-running relaxation."""
    from hydeph.pipeline import run_pipeline

    ws = str(tmp_path / "ws_resume")
    os.makedirs(ws, exist_ok=True)
    pristine_p = str(tmp_path / "pristine.xyz")
    at = Atoms("C2", positions=[(0, 0, 0), (0, 0, 1.4)], cell=[10, 10, 10], pbc=True)
    at.write(pristine_p)

    pseudo_dir = str(tmp_path / "pseudo")
    os.makedirs(pseudo_dir, exist_ok=True)
    with open(os.path.join(pseudo_dir, "C.upf"), "w") as f:
        f.write('<PP_HEADER z_valence="4.0" />\n')

    pot_file = str(tmp_path / "nep.txt")
    with open(pot_file, "w") as f:
        f.write("dummy nep")

    cfg = ToolkitConfig(
        pristine_cell_path=pristine_p,
        defect_cell_path=pristine_p,
        supercell_repeat=(1, 1, 1),
        foundation_potential_path=pot_file,
        workspace_dir=ws,
        pseudo_dir=pseudo_dir,
        pseudopotentials={"C": "C.upf"},
        spin_multiplicity=1,
    )

    # Mock converged pw.out in qe_gs_relax
    gs_dir = os.path.join(ws, "qe_gs_relax")
    os.makedirs(gs_dir, exist_ok=True)
    # Write mock pw.in so template_atoms can be inferred
    from hydeph.qe_runner import write_qe_input
    write_qe_input(at, os.path.join(gs_dir, "pw.in"), cfg, calculation="relax")
    with open(os.path.join(gs_dir, "pw.out"), "w") as f:
        f.write("""
!    total energy              =     -150.12345678 Ry
Begin final coordinates
ATOMIC_POSITIONS (angstrom)
C        0.000000   0.000000   0.000000
C        0.000000   0.000000   1.420000
End final coordinates
bfgs converged in 5 iterations
""")

    # Mock converged pw.out in qe_es_relax
    es_dir = os.path.join(ws, "qe_es_relax")
    os.makedirs(es_dir, exist_ok=True)
    write_qe_input(at, os.path.join(es_dir, "pw.in"), cfg, calculation="relax")
    with open(os.path.join(es_dir, "pw.out"), "w") as f:
        f.write("""
!    total energy              =     -148.12345678 Ry
Begin final coordinates
ATOMIC_POSITIONS (angstrom)
C        0.000000   0.000000   0.000000
C        0.000000   0.000000   1.460000
End final coordinates
bfgs converged in 7 iterations
""")

    # Resume pipeline from Stage 1: should discover both pw.out and write relaxed_gs.xyz, relaxed_es.xyz
    from unittest.mock import patch
    with patch("hydeph.calculator.CPUNEP") as mock_nep:
        calc_inst = mock_nep.return_value
        calc_inst.get_potential_energy.return_value = -100.0
        calc_inst.get_forces.return_value = np.zeros((2, 3))
        res = run_pipeline(cfg, to_stage=1)
        assert os.path.isfile(os.path.join(ws, "relaxed_gs.xyz"))
        assert os.path.isfile(os.path.join(ws, "relaxed_es.xyz"))
        assert res["stage"] == 1


def test_prepare_all_dft_inputs_and_array_script(tmp_path):
    """Verify QERunner.prepare_all_dft_inputs creates GS and ES inputs, dft_frames_list.txt, and SLURM array."""
    from hydeph.qe_runner import QERunner

    ws = str(tmp_path / "ws_dft_prep")
    os.makedirs(ws, exist_ok=True)
    pseudo_dir = str(tmp_path / "pseudo")
    os.makedirs(pseudo_dir, exist_ok=True)
    with open(os.path.join(pseudo_dir, "C.upf"), "w") as f:
        f.write('<PP_HEADER z_valence="4.0" />\n')

    cfg = ToolkitConfig(
        pristine_cell_path=str(tmp_path / "pristine.xyz"),
        workspace_dir=ws,
        pseudo_dir=pseudo_dir,
        pseudopotentials={"C": "C.upf"},
        paired_delta_nep=True,
        excited_state_method="delta_scf",
        delta_scf_homo_band=4,
        delta_scf_lumo_band=5,
        delta_scf_lumo_degeneracy=1,
        delta_scf_spin_channel=2,
        spin_multiplicity=1,
    )
    qe = QERunner(cfg)

    # 3 candidate frames
    at = Atoms("C2", positions=[(0, 0, 0), (0, 0, 1.4)], cell=[10, 10, 10], pbc=True)
    frames = []
    for i in range(3):
        fr = at.copy()
        fr.info["candidate_index"] = i
        fr.info["frame_id"] = f"fc_{i:02d}"
        fr.info["sample_type"] = "franck_condon"
        fr.info["lambda"] = float(i) / 2.0
        frames.append(fr)

    res = qe.prepare_all_dft_inputs(frames, at_gs=at)

    assert res["n_gs_frames"] == 3
    assert res["n_es_frames"] == 3
    assert os.path.isfile(res["slurm_script"])
    assert os.path.isfile(res["array_script"])
    assert os.path.isfile(res["list_file"])

    # Verify dft_frames_list.txt has 6 lines with forward slashes
    with open(res["list_file"], "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    assert len(lines) == 6
    assert all("\\" not in l for l in lines)
    assert any("dft_calculations/frame_" in l for l in lines)
    assert any("dft_calculations_es/frame_es_" in l for l in lines)

    # Verify array script contains exact bounds
    with open(res["array_script"], "r") as f:
        arr_content = f.read()
    assert "--array=0-5%16" in arr_content
    assert "pw.x" in arr_content


def test_pipeline_prepare_dft_stops_cleanly(tmp_path):
    """Verify run_pipeline stops cleanly at Stage 2 when prepare_dft_inputs is requested."""
    from hydeph.pipeline import run_pipeline

    ws = str(tmp_path / "ws_pipe_prep")
    os.makedirs(ws, exist_ok=True)
    pristine_p = str(tmp_path / "pristine.xyz")
    at = Atoms("C2", positions=[(0, 0, 0), (0, 0, 1.4)], cell=[10, 10, 10], pbc=True)
    at_es = Atoms("C2", positions=[(0, 0, 0), (0, 0, 1.45)], cell=[10, 10, 10], pbc=True)
    at.write(pristine_p)
    at.write(os.path.join(ws, "relaxed_gs.xyz"))
    at_es.write(os.path.join(ws, "relaxed_es.xyz"))

    pseudo_dir = str(tmp_path / "pseudo")
    os.makedirs(pseudo_dir, exist_ok=True)
    with open(os.path.join(pseudo_dir, "C.upf"), "w") as f:
        f.write('<PP_HEADER z_valence="4.0" />\n')

    pot_file = str(tmp_path / "nep.txt")
    with open(pot_file, "w") as f:
        f.write("dummy nep")

    cfg = ToolkitConfig(
        pristine_cell_path=pristine_p,
        defect_cell_path=pristine_p,
        supercell_repeat=(1, 1, 1),
        foundation_potential_path=pot_file,
        workspace_dir=ws,
        pseudo_dir=pseudo_dir,
        pseudopotentials={"C": "C.upf"},
        spin_multiplicity=1,
        prepare_dft_inputs=True,
        n_fc_frames=3,
        n_phonon_frames=0,
        n_md_frames=0,
    )

    from unittest.mock import patch
    with patch("hydeph.calculator.CPUNEP") as mock_nep:
        calc_inst = mock_nep.return_value
        calc_inst.get_potential_energy.return_value = -100.0
        calc_inst.get_forces.return_value = np.zeros((2, 3))
        res = run_pipeline(cfg)

        assert res["stage"] == 2
        assert "dft_inputs" in res
        assert res["dft_inputs"] is not None
        assert os.path.isfile(os.path.join(ws, "submit_dft_array_pure_qe.slurm"))
        assert os.path.isfile(os.path.join(ws, "dft_frames_list.txt"))


def test_ingest_completed_calculations_with_exact_energetics(tmp_path):
    """Verify ingest_completed_calculations parses GS & ES outputs and calculates exact adiabatic ZPL."""
    from hydeph.qe_runner import QERunner, write_qe_input

    ws = str(tmp_path / "ws_ingest")
    os.makedirs(ws, exist_ok=True)
    pseudo_dir = str(tmp_path / "pseudo")
    os.makedirs(pseudo_dir, exist_ok=True)
    with open(os.path.join(pseudo_dir, "C.upf"), "w") as f:
        f.write('<PP_HEADER z_valence="4.0" />\n')

    cfg = ToolkitConfig(
        pristine_cell_path=str(tmp_path / "pristine.xyz"),
        workspace_dir=ws,
        pseudo_dir=pseudo_dir,
        pseudopotentials={"C": "C.upf"},
        spin_multiplicity=1,
    )
    qe = QERunner(cfg)

    at_gs = Atoms("C2", positions=[(0, 0, 0), (0, 0, 1.4)], cell=[10, 10, 10], pbc=True)
    at_es = Atoms("C2", positions=[(0, 0, 0), (0, 0, 1.45)], cell=[10, 10, 10], pbc=True)

    # Prepare candidate frames: lambda=0 (GS geometry) and lambda=1 (ES geometry)
    f0 = at_gs.copy()
    f0.info.update({"candidate_index": 0, "frame_id": "fc_00", "sample_type": "franck_condon", "lambda": 0.0})
    f1 = at_es.copy()
    f1.info.update({"candidate_index": 1, "frame_id": "fc_01", "sample_type": "franck_condon", "lambda": 1.0})
    cand_frames = [f0, f1]

    # Setup GS calculations: E_GS(0) = -150.0 eV, E_GS(1) = -149.5 eV
    gs_dir0 = os.path.join(ws, "dft_calculations", "frame_000_fc_00")
    os.makedirs(gs_dir0, exist_ok=True)
    write_qe_input(f0, os.path.join(gs_dir0, "pw.in"), cfg)
    with open(os.path.join(gs_dir0, "pw.out"), "w") as f:
        f.write("""
!    total energy              =     -11.02500000 Ry
     Forces acting on atoms (cartesian axes, Ry/au):
     atom    1 type  1   force =     0.00000000    0.00000000    0.00000000
     atom    2 type  1   force =     0.00000000    0.00000000    0.00000000
     JOB DONE.
""")

    gs_dir1 = os.path.join(ws, "dft_calculations", "frame_001_fc_01")
    os.makedirs(gs_dir1, exist_ok=True)
    write_qe_input(f1, os.path.join(gs_dir1, "pw.in"), cfg)
    with open(os.path.join(gs_dir1, "pw.out"), "w") as f:
        f.write("""
!    total energy              =     -10.98800000 Ry
     Forces acting on atoms (cartesian axes, Ry/au):
     atom    1 type  1   force =     0.00000000    0.00000000    0.00000000
     atom    2 type  1   force =     0.00000000    0.00000000    0.00000000
     JOB DONE.
""")

    # Setup ES calculations: E_ES(0) = -10.878 Ry, E_ES(1) = -10.900 Ry
    es_dir0 = os.path.join(ws, "dft_calculations_es", "frame_es_000_fc_00")
    os.makedirs(es_dir0, exist_ok=True)
    write_qe_input(f0, os.path.join(es_dir0, "pw.in"), cfg)
    with open(os.path.join(es_dir0, "pw.out"), "w") as f:
        f.write("""
!    total energy              =     -10.87800000 Ry
     Forces acting on atoms (cartesian axes, Ry/au):
     atom    1 type  1   force =     0.00000000    0.00000000    0.00000000
     atom    2 type  1   force =     0.00000000    0.00000000    0.00000000
     JOB DONE.
""")

    es_dir1 = os.path.join(ws, "dft_calculations_es", "frame_es_001_fc_01")
    os.makedirs(es_dir1, exist_ok=True)
    write_qe_input(f1, os.path.join(es_dir1, "pw.in"), cfg)
    with open(os.path.join(es_dir1, "pw.out"), "w") as f:
        f.write("""
!    total energy              =     -10.90000000 Ry
     Forces acting on atoms (cartesian axes, Ry/au):
     atom    1 type  1   force =     0.00000000    0.00000000    0.00000000
     atom    2 type  1   force =     0.00000000    0.00000000    0.00000000
     JOB DONE.
""")

    labeled_gs, labeled_es = qe.ingest_completed_calculations(cand_frames, at_gs=at_gs, at_es=at_es)

    assert len(labeled_gs) == 2
    assert len(labeled_es) == 2
    assert os.path.isfile(os.path.join(ws, "labeled_dft_frames.xyz"))
    assert os.path.isfile(os.path.join(ws, "labeled_frames_es.xyz"))

    # E_ZPL = E_ES(lambda=1) - E_GS(lambda=0) = (-10.900 - (-11.025)) * 13.605698066 = 0.125 * 13.605698066 = 1.7007 eV
    assert "e_zpl_exact_eV" in at_gs.info
    assert at_gs.info["zpl_source"] == "delta_scf_adiabatic"
    assert pytest.approx(at_gs.info["e_zpl_exact_eV"], rel=1e-3) == 1.7007


def test_ingest_without_candidate_frames_auto_discovery(tmp_path):
    """Verify ingest_completed_calculations auto-discovers frames from disk when candidate_frames is None."""
    from hydeph.qe_runner import QERunner, write_qe_input

    ws = str(tmp_path / "ws_auto_ingest")
    os.makedirs(ws, exist_ok=True)
    pseudo_dir = str(tmp_path / "pseudo")
    os.makedirs(pseudo_dir, exist_ok=True)
    with open(os.path.join(pseudo_dir, "C.upf"), "w") as f:
        f.write('<PP_HEADER z_valence="4.0" />\n')

    cfg = ToolkitConfig(
        pristine_cell_path=str(tmp_path / "pristine.xyz"),
        workspace_dir=ws,
        pseudo_dir=pseudo_dir,
        pseudopotentials={"C": "C.upf"},
        spin_multiplicity=1,
    )
    qe = QERunner(cfg)

    at = Atoms("C2", positions=[(0, 0, 0), (0, 0, 1.4)], cell=[10, 10, 10], pbc=True)
    gs_dir = os.path.join(ws, "dft_calculations", "frame_000_fc_00")
    os.makedirs(gs_dir, exist_ok=True)
    write_qe_input(at, os.path.join(gs_dir, "pw.in"), cfg)
    with open(os.path.join(gs_dir, "pw.out"), "w") as f:
        f.write("""
!    total energy              =     -11.00000000 Ry
     Forces acting on atoms (cartesian axes, Ry/au):
     atom    1 type  1   force =     0.00000000    0.00000000    0.00000000
     atom    2 type  1   force =     0.00000000    0.00000000    0.00000000
     JOB DONE.
""")

    labeled_gs, labeled_es = qe.ingest_completed_calculations(None)
    assert len(labeled_gs) == 1
    assert os.path.isfile(os.path.join(ws, "labeled_dft_frames.xyz"))


def test_manual_cluster_module_configuration_and_order(tmp_path):
    """Verify that cluster modules are manually configured in exact order without heuristic if loops."""
    ws = str(tmp_path / "ws_manual_modules")
    os.makedirs(ws, exist_ok=True)
    pseudo_dir = str(tmp_path / "pseudo")
    os.makedirs(pseudo_dir, exist_ok=True)
    c_upf = os.path.join(pseudo_dir, "C.upf")
    with open(c_upf, "w") as f:
        f.write('<PP_HEADER z_valence="4.0" />\n')

    pristine_xyz = str(tmp_path / "pristine.xyz")
    at = Atoms("C2", positions=[(0, 0, 0), (0, 0, 1.4)], cell=[10, 10, 10], pbc=True)
    import ase.io
    ase.io.write(pristine_xyz, at)

    custom_modules = [
        "profile/chem-phys",
        "quantum-espresso/7.4.1--hpcx-mpi--2.19--nvhpc--24.5-openblas",
        "cuda/12.6",
    ]

    # Test dictionary / yaml construction using 'modules' alias
    cfg = ToolkitConfig.from_dict({
        "pristine_cell_path": pristine_xyz,
        "workspace_dir": ws,
        "pseudo_dir": pseudo_dir,
        "pseudopotentials": {"C": "C.upf"},
        "modules": custom_modules,
        "slurm_account": "IscrC_Test",
        "slurm_partition": "boost_usr_prod",
        "spin_multiplicity": 1,
    })

    assert cfg.slurm_modules == custom_modules

    qe = QERunner(cfg, pw_exe="pw.x")

    # 1. Test relaxation launcher script
    rel_res = qe.prepare_relaxation_inputs(at)
    rel_slurm = rel_res["slurm_script"]
    assert os.path.isfile(rel_slurm)
    with open(rel_slurm, "r") as f:
        rel_content = f.read()

    assert "if module is-avail" not in rel_content
    assert "module purge" in rel_content
    assert "module load profile/chem-phys" in rel_content
    assert "module load quantum-espresso/7.4.1--hpcx-mpi--2.19--nvhpc--24.5-openblas" in rel_content
    assert "module load cuda/12.6" in rel_content

    # Strict ordering check: profile/chem-phys must be loaded before quantum-espresso, and QE before cuda
    idx_purge = rel_content.index("module purge")
    idx_prof = rel_content.index("module load profile/chem-phys")
    idx_qe = rel_content.index("module load quantum-espresso/7.4.1")
    idx_cuda = rel_content.index("module load cuda/12.6")
    assert idx_purge < idx_prof < idx_qe < idx_cuda

    # 2. Test candidate DFT scripts (standalone and array)
    at.info["sample_type"] = "franck_condon"
    at.info["candidate_index"] = 0
    dft_res = qe.prepare_candidate_dft_inputs([at])
    dft_slurm = dft_res["slurm_script"]
    array_slurm = dft_res["array_script"]

    for script_path in (dft_slurm, array_slurm):
        assert os.path.isfile(script_path)
        with open(script_path, "r") as f:
            content = f.read()
        assert "if module is-avail" not in content
        assert "module purge" in content
        p_idx = content.index("module load profile/chem-phys")
        q_idx = content.index("module load quantum-espresso/7.4.1")
        c_idx = content.index("module load cuda/12.6")
        assert p_idx < q_idx < c_idx





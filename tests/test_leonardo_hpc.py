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
    assert booster.qos == "boost_qos_prod"
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
    assert "#SBATCH --qos=boost_qos_prod" in script
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
    assert "--qos=boost_qos_prod" in cmd_booster
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
    assert cfg.slurm_qos == "boost_qos_prod"
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

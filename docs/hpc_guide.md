# HPC & Cluster Execution Guide

HyDePh supports parallelized execution across SLURM-managed GPU and CPU clusters. The HPC subsystem (`hydeph.hpc`) provides hardware profiles, deterministic environment module loading, multi-GPU worker pinning, Lustre-safe atomic completion markers, and decoupled pure-Quantum ESPRESSO (`pw.x`) script generation.

---

## 1. Cluster Configuration & Environment Modules

Cluster parameters are configured in `config.yaml` (or via `ToolkitConfig`):

```yaml
# Scheduler & hardware profile
hpc_scheduler: slurm                # "local" or "slurm"
cluster_profile: generic_slurm      # "generic_slurm", "leonardo_booster", "leonardo_dcgp", or "auto"

# SLURM resource allocation
slurm_account: my_project_account   # Defaults to $SLURM_ACCOUNT or $CINECA_ACCOUNT
slurm_partition: gpu_prod           # Partition name
slurm_qos: null                     # Optional QoS string (omitted from #SBATCH when null)
slurm_nodes: 1
slurm_gpus_per_node: 4
slurm_cpus_per_task: 8
gpu_pinning: true

# Relaxation job allocation (Stage 1 pure-QE script; defaults tuned for rapid scheduling)
slurm_relax_gpus: 1
slurm_relax_time: "01:00:00"
slurm_relax_qos: null               # e.g. "boost_qos_dbg" for fast debug queues

# Environment modules (loaded in exact specified order)
modules:
  - profile/chem-phys
  - autoload
  - quantum-espresso/7.2--openmpi--4.1.4--cuda--11.8

# Parallel supercell Hessian evaluation (Stage 6)
parallel_hessian: true
parallel_hessian_workers: 16
```

### Deterministic Module Loading
HPC software stacks frequently require environment modules to be loaded in a strict dependency sequence (for example, loading a domain profile module before the compiler or Quantum ESPRESSO module becomes visible).
- Specify modules under `modules:` (or aliases `slurm_modules:`, `cluster_modules:`, `hpc_modules:`).
- HyDePh emits `module purge` followed by `module load <name>` in the **exact order listed**, without heuristic availability loops.
- No default modules are assumed. If cluster script generation is invoked with an empty module list, HyDePh logs a warning so you can verify that `pw.x` is available on the compute nodes' `PATH`.

---

## 2. Execution Workflows

### Workflow 1: Decoupled Staged Execution (Recommended)

Activating Python or Conda environments inside non-interactive SLURM compute jobs can be fragile on restricted cluster filesystems. In the decoupled workflow, **only standalone Quantum ESPRESSO (`pw.x`) runs on compute nodes**, while lightweight structure generation, candidate sampling, $\Delta$-NEP training, and spectroscopy run on the login node or a local GPU workstation.

#### Phase A: Stage 1 Defect Relaxations (Optional Pure-QE Offload)
If ground- and excited-state geometry optimizations use DFT (`ground_state_relax_method: dft`), pre-generate the relaxation input decks and SLURM batch script without running `pw.x` locally:

```bash
hydeph config.yaml --prepare-relax
```

This creates:
- `qe_gs_relax/pw.in`: Ground-state BFGS relaxation deck (`occupations = 'fixed'`, supporting fixed-cell `relax` or variable-cell `vc-relax` via `relaxation_calculation`).
- `qe_es_relax/pw.in`: Excited-state $\Delta\text{SCF}$ relaxation deck (`occupations = 'from_input'`), including symmetry-breaking Jahn-Teller distortion when `auto_jahn_teller_distortion: true`.
- `submit_relaxations_pure_qe.slurm`: Standalone SLURM script (defaulting to 1 GPU and 1 hour via `slurm_relax_gpus: 1` and `slurm_relax_time: "01:00:00"` for rapid queue scheduling) that executes ground-state relaxation, extracts frontier Kohn-Sham eigenvalues to update the lowest-energy $\Delta\text{SCF}$ occupation card if needed, and runs excited-state relaxation.

Submit on the cluster (from either the project root or `workspace_dir`):
```bash
sbatch submit_relaxations_pure_qe.slurm

# Or submit to a fast debug queue (e.g. <= 30 min on Leonardo Booster):
sbatch --qos=boost_qos_dbg --time=00:30:00 submit_relaxations_pure_qe.slurm
```

#### Phase B: Stage 2 Candidate Sampling & Stage 3 Input Preparation
Once `relaxed_gs.xyz` and `relaxed_es.xyz` (or `qe_gs_relax/pw.out` and `qe_es_relax/pw.out`) are available, generate candidate configurations and pre-build all single-point DFT directories:

```bash
hydeph config.yaml --to-stage 2
```
*(Or explicitly pass `--prepare-dft`.)*

HyDePh ingests completed relaxations, runs the foundation bulk stability check, generates the candidate dataset (`candidate_frames.xyz`), writes `pw.in` files with resolved relative `pseudo_dir` paths into `dft_calculations/frame_*` and `dft_calculations_es/frame_es_*`, and outputs two standalone SLURM launchers (written to both `workspace_dir` and the parent directory):
1. `submit_dft_array_pure_qe.slurm` + `dft_frames_list.txt`: SLURM Job Array script pre-configured with `#SBATCH --array=0-(N-1)%16` where each array task labels one frame on 1 GPU and skips frames that already contain `JOB DONE.`.
2. `submit_dft_pure_qe.slurm`: Single-node multi-GPU batch script that schedules concurrent `pw.x` processes across all GPUs on the node (`CUDA_VISIBLE_DEVICES=0..G-1`).

#### Phase C: Submit Candidate Frame DFT Calculations
Submit either the job array (recommended for high throughput across many nodes) or the single-node batch script:

```bash
# High-throughput SLURM Job Array (uses embedded #SBATCH --array=0-(N-1)%16 directive):
sbatch submit_dft_array_pure_qe.slurm

# Or override array range / concurrency explicitly on the command line:
sbatch --array=0-79%16 submit_dft_array_pure_qe.slurm

# Or single-node multi-GPU batch job:
sbatch submit_dft_pure_qe.slurm
```

#### Phase D: Ingest Outputs, Train $\Delta$-NEP & Compute Spectra (Stages 4–6)
After the SLURM jobs finish, resume the pipeline from Stage 4:

```bash
hydeph config.yaml --from-stage 4
```

HyDePh automatically:
1. Parses all converged `pw.out` files from `dft_calculations/` and `dft_calculations_es/` into `labeled_dft_frames.xyz` and `labeled_dft_frames_es.xyz`.
2. Extracts adiabatic $E_{\mathrm{ZPL}}$, vertical absorption/emission energies, and Stokes shift.
3. Trains paired $\Delta$-NEP models (`nep_delta_gs.txt`, `nep_delta_es.txt`) and verifies quality control gates (Stage 4–5).
4. Evaluates the supercell Hessian in parallel (`ParallelVibrations`) and computes multi-temperature harmonic and anharmonic optical lineshapes (Stage 6).

---

### Workflow 2: End-to-End Multi-GPU Batch Job

When Python, GPUMD (`nep`), and Quantum ESPRESSO (`pw.x`) are all available on the compute node, the entire pipeline (Stages 1–6) can run in a single unattended SLURM allocation.

With `n_workers_qe: 4`, `n_procs_qe: 1`, and `gpu_pinning: true`, `get_pinned_worker_env` pins each concurrent worker $w \in \{0, \dots, G-1\}$ to a dedicated GPU (`CUDA_VISIBLE_DEVICES = w % G`, `GPU_DEVICE_ORDINAL = w % G`) with `OMP_NUM_THREADS = slurm_cpus_per_task`:

```bash
hydeph config.yaml --workers 4
```

---

### Workflow 3: Python Array Worker Dispatch (`SlurmDispatcher`)

For workflows managed programmatically from Python, `SlurmDispatcher` generates a SLURM Job Array script that invokes `python -m hydeph.hpc.array_worker` per frame or frame chunk:

```python
from hydeph.hpc import SlurmDispatcher, get_cluster_profile

dispatcher = SlurmDispatcher(
    profile=get_cluster_profile("generic_slurm"),
    account="my_account",
    workspace_dir="./hydeph_run",
    slurm_modules=["profile/chem-phys", "quantum-espresso/7.2"],
)

dispatcher.generate_job_array_script(
    config_path="config.yaml",
    n_frames=40,
    chunk_size=1,
    max_concurrent=16,
    time_limit="02:00:00",
    script_output_path="submit_array.slurm",
)
```

Each array worker writes an atomic `.done` JSON marker (`os.replace`) upon completion, preventing race conditions on shared Lustre or GPFS scratch filesystems.

---

### Workflow 4: Interactive Multi-GPU Sessions (`salloc` + `tmux`)

For interactive debugging or short test runs, launch `salloc` inside a persistent `tmux` session on the login node so SSH drops do not terminate the allocation:

```bash
# 1. Start persistent tmux session on login node
tmux new -s hydeph

# 2. Request interactive GPU node allocation
salloc -A $SLURM_ACCOUNT -p <partition> \
    --nodes=1 --ntasks-per-node=4 --cpus-per-task=8 --gres=gpu:4 --time=02:00:00

# 3. Load modules, activate environment, and run HyDePh
hydeph config.yaml --workers 4
```

---

### Parallel Supercell Hessian Evaluation (Stage 6)

In Stage 6, finite-difference phonon calculations on large supercells (e.g., 398 atoms $\to$ 2,388 displacements) are parallelized across CPU threads via `ParallelVibrations`:

```yaml
parallel_hessian: true
parallel_hessian_workers: 16
```

Each worker evaluates displaced structures with thread-local copies of `PairedDefectCalculator`, reducing supercell Hessian walltime from ~20 minutes to under 1 minute.

---

## 3. Reference Example: CINECA Leonardo Supercomputer

HyDePh includes built-in hardware profiles (`hydeph/hpc/cluster_profiles.py`) and helper scripts (`scripts/leonardo/`) for the **CINECA Leonardo** Tier-0 system.

### Built-in Profiles

| Profile Name | Partition | QoS | Node Hardware | `gpus_per_node` | `default_cpus_per_task` |
|---|---|---|---|---|---|
| `leonardo_booster` | `boost_usr_prod` | `None` (account default) | 1x Xeon 8358 (32 cores) + 4x NVIDIA A100 64GB | `4` | `8` |
| `leonardo_booster_debug` | `boost_usr_prod` | `boost_qos_dbg` | 1x Xeon 8358 (32 cores) + 4x NVIDIA A100 64GB (max 30 min) | `4` | `8` |
| `leonardo_dcgp` | `dcgp_usr_prod` | `None` | 2x Xeon Platinum 8480+ (112 cores, CPU-only) | `0` | `14` |
| `generic_slurm` | `gpu` | `None` | Generic 4-GPU node (32 cores) | `4` | `8` |

### Example `config.yaml` for CINECA Leonardo Booster

```yaml
pristine_cell_path: ./4H_SiC_unitcell.xyz
defect_cell_path: ./4H_SiC_kk_divacancy.xyz
foundation_potential_path: ./nep89_20250409.txt
pseudo_dir: ./pseudo
workspace_dir: ./output_sic_kk

supercell_repeat: [3, 3, 2]
spin_polarized: true
spin_multiplicity: 3
excited_state_method: delta_scf
paired_delta_nep: true
auto_jahn_teller_distortion: true
ground_state_relax_method: dft

# Leonardo Booster HPC settings
hpc_scheduler: slurm
cluster_profile: leonardo_booster
slurm_account: <your_cineca_account>
slurm_partition: boost_usr_prod
slurm_gpus_per_node: 4
slurm_cpus_per_task: 8

# Explicit CINECA module loading order
modules:
  - profile/chem-phys
  - autoload
  - quantum-espresso/7.2--openmpi--4.1.4--cuda--11.8

parallel_hessian: true
parallel_hessian_workers: 16
```

### Filesystem & Quota Guidelines on Leonardo
- **Use `$SCRATCH` or `$WORK` for Workspaces**: Place `workspace_dir` on `$SCRATCH` (high-bandwidth Lustre parallel filesystem) rather than `$HOME`, which enforces a strict 50 GB quota and is unsuitable for Quantum ESPRESSO `.wfc` scratch files.
- **Account Export**: Export `export CINECA_ACCOUNT="<your_account>"` in `~/.bashrc` so `ToolkitConfig` and helper scripts in `scripts/leonardo/` resolve `#SBATCH --account` automatically.

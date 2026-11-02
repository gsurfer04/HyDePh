# Running HyDePh on CINECA Leonardo Supercomputer

This guide details the procedures for executing large-scale, parallelized HyDePh defect workflows on the **Leonardo Tier-0 Supercomputer** hosted at CINECA (Bologna, Italy).

---

## 1. System Specifications & Partitions

Leonardo consists of two primary compute modules:

### A. Leonardo Booster Module (Primary Target)
* **Architecture**: Atos BullSequana X2135 blade servers.
* **Nodes**: 1x Intel Xeon Platinum 8358 (32 cores @ 2.60 GHz) + **4x custom NVIDIA A100-SXM4 (64 GB)**.
* **Memory**: 512 GB RAM per node.
* **Interconnect**: Dual 200 Gb/s Mellanox HDR Infiniband.
* **SLURM Partitions**:
  * `boost_usr_prod`: Standard production partition (walltime up to 24 hours).
  * `boost_qos_dbg`: Debug queue for short interactive tests (max 2 nodes, 30 min).

### B. Leonardo DCGP Module (CPU Partition)
* **Architecture**: Intel Sapphire Rapids.
* **Nodes**: 2x Intel Xeon Platinum 8480+ (56 cores each = **112 cores/node**), no GPUs.
* **SLURM Partition**: `dcgp_usr_prod`.

---

## 2. Quickstart: Environment Setup

1. **Log in to Leonardo**:
   ```bash
   ssh <username>@login.leonardo.cineca.it
   ```

2. **Clone the repository on `$WORK`**:
   ```bash
   cd $WORK
   git clone -b feature/cineca-leonardo https://github.com/gsurfer04/HyDePh.git
   cd HyDePh
   ```

3. **Configure the environment**:
   Run the automated setup script to load CINECA modules and install a dedicated Python environment:
   ```bash
   source scripts/leonardo/env_leonardo.sh
   ```

---

## 3. Execution & Parallelization Modes

### Mode 1: Decoupled Staged HPC Execution (Recommended — Zero Python on Compute Nodes)
Getting Python virtual environments or Conda activated reliably across non-interactive compute node Slurm jobs can be awkward. 

In this recommended workflow, **only native Quantum ESPRESSO (`pw.x`) runs on compute nodes**. The pre-processing (geometry & frame sampling) and post-processing ($\Delta$-NEP training, hybrid potential assembly, lineshapes) run directly on the login node!

#### Phase 1: Pre-Processing on Login Node
Run stages 1 and 2 directly on the login node (fast, takes ~20 seconds, no GPU allocation needed):
```bash
python -m hydeph run config.yaml --to-stage 2
```
This automatically:
* Constructs defect supercells and relaxes ground/excited states with the foundation potential.
* Samples candidate snapshot configurations along the optical coordinate.
* Pre-generates all Quantum ESPRESSO `pw.in` input files in `dft_calculations/` and `dft_calculations_es/`.
* Writes the pure bash/SLURM runner script `submit_dft_pure_qe.slurm`.

#### Phase 2: Pure Quantum ESPRESSO on Compute Nodes
Submit the heavy DFT calculations to the Leonardo Booster compute nodes:
```bash
# Standard 4-GPU Booster batch job:
sbatch submit_dft_pure_qe.slurm

# Or fast debug queue (up to 30 min walltime):
sbatch --qos=boost_qos_dbg submit_dft_pure_qe.slurm
```
* **Pure native CINECA modules**: Uses `module load profile/chem-phys` and `module load q-e/7.2`.
* **Zero Python required**: Compute nodes do not need Python or Conda!
* **Multi-GPU pinning**: Automatically runs 4 concurrent calculations pinned across the 4x A100 GPUs (`CUDA_VISIBLE_DEVICES=0,1,2,3`).
* Disconnect safely; monitor with `squeue -u $USER`.

#### Phase 3: Post-Processing & Optical Lineshapes on Login Node
Once the DFT batch job completes:
```bash
python -m hydeph run config.yaml --from-stage 4
```
HyDePh will automatically:
* Ingest all converged `pw.out` files from disk into `labeled_dft_frames.xyz`.
* Train the $\Delta$-NEP models (Stage 4).
* Assemble the hybrid potential (Stage 5).
* Compute supercell vibrations and multi-temperature optical lineshapes (Stage 6).

---

### Mode 2: Unattended Single-Sweep Batch Submission via `sbatch`
When the cluster queue is congested or wait times are unpredictable, **batch submission is the most reliable approach**. You submit the job and can immediately close your SSH connection or shut down your laptop. Slurm handles queue waiting, allocation, and execution completely in the background.

```bash
# Using the helper script:
bash scripts/leonardo/submit_booster_batch.sh

# Or directly with sbatch:
sbatch scripts/leonardo/submit_booster_workflow.slurm

# Or submit to the fast debug queue (up to 30 min walltime):
bash scripts/leonardo/submit_booster_batch.sh --debug
```

A single Booster node contains 4x NVIDIA A100 GPUs and 32 CPU cores. By specifying `--n-workers-qe 4 --n-procs-qe 1`, HyDePh launches 4 concurrent Quantum ESPRESSO processes, pinning each worker to an isolated GPU (`CUDA_VISIBLE_DEVICES=0,1,2,3`) with 8 OpenMP threads each. This accelerates candidate frame DFT labeling by **$4\times$** without any multi-node overhead.

To monitor or manage your batch job:
```bash
# Check queue status
squeue -u $USER

# Stream live output
tail -f hydeph_*.out

# Cancel job if needed
scancel <JOB_ID>
```

> [!TIP]
> **Email Notifications**: Add `#SBATCH --mail-type=ALL` and `#SBATCH --mail-user=your.email@domain.com` to `submit_booster_workflow.slurm` to receive automatic notifications when your job starts and completes.

---

### Mode 2: Interactive Execution via `salloc` + `tmux` (Disconnect Protection)
Official CINECA documentation ([CINECA HPC Scheduler](https://docs.hpc.cineca.it/hpc/hpc_scheduler.html) and [Leonardo User Guide](https://docs.hpc.cineca.it/hpc/leonardo.html)) recommends `salloc` over `srun` for interactive sessions.

However, standard interactive allocations require keeping the SSH connection open. If the queue wait is long or your connection drops, Slurm cancels the request or kills the allocation.

To prevent this, **run `salloc` inside a persistent `tmux` session on the login node**:

1. **Start `tmux` on the login node**:
   ```bash
   tmux new -s hydeph
   ```

2. **Request an Interactive Booster Allocation**:
   ```bash
   # Standard production allocation (e.g. 2 hours):
   bash scripts/leonardo/interactive_booster.sh 02:00:00

   # Or for rapid testing on the debug queue (30 min limit, near-zero wait time):
   bash scripts/leonardo/interactive_booster.sh --debug
   ```
   *Equivalent raw Slurm command (omitting --qos so Slurm applies your account default):*
   ```bash
   salloc -A $CINECA_ACCOUNT -p boost_usr_prod \
       --nodes=1 --ntasks-per-node=4 --cpus-per-task=8 --gres=gpu:4 --time=02:00:00
   ```

3. **Detach safely while waiting in queue**:
   Press `Ctrl` + `b`, then press `d`.
   You can now safely log out or close your laptop. Slurm keeps your position in the queue.

4. **Reattach anytime later**:
   ```bash
   ssh <username>@login.leonardo.cineca.it
   tmux attach -t hydeph
   ```

5. **Execute HyDePh Inside the `salloc` Allocation**:
   Once granted, you are placed in an interactive subshell:
   ```bash
   # 1. Activate environment
   source scripts/leonardo/env_leonardo.sh

   # 2. Launch HyDePh across the 4 allocated GPUs
   python -m hydeph \
       --config config.yaml \
       --n-workers-qe 4 \
       --n-procs-qe 1
   ```

6. **Release Allocation**:
   When finished, type `exit` to terminate the allocation and stop billing your project budget.

---

### Mode 3: Rapid Verification via the Debug Queue (`boost_qos_dbg`)
Leonardo maintains dedicated nodes for the debug queue (`boost_qos_dbg`). If you want to verify your environment or test your workflow without waiting in the production queue:
```bash
# Interactive debug session (max 30 min):
bash scripts/leonardo/interactive_booster.sh --debug

# Or unattended debug batch job:
bash scripts/leonardo/submit_booster_batch.sh --debug
```

---

### Mode 3: SLURM Job Array for Candidate Frame Labeling (Stage 3)
For large frame pools (e.g. 40–100+ frames), candidate frame labeling can be parallelized across up to 16 concurrent GPU tasks using a SLURM Job Array.

1. **Generate Candidate Frames** (Stages 1–2):
   ```bash
   python -m hydeph --config config.yaml --to-stage 2
   ```

2. **Launch the Job Array**:
   ```bash
   sbatch --array=0-39%16 scripts/leonardo/submit_booster_array.slurm
   ```
   Each array task labels one candidate frame with GPU acceleration into `$SCRATCH/hydeph_run/dft_runs/frame_XXXX/`. Atomic `.done` markers ensure zero race conditions on Lustre scratch.

3. **Assemble and Complete the Workflow** (Stages 4–6):
   Once the array completes, simply run:
   ```bash
   python -m hydeph --config config.yaml --from-stage 4
   ```
   HyDePh automatically discovers all completed DFT frames, trains the $\Delta$-NEP, computes the supercell Hessian in parallel, and evaluates the multi-temperature optical lineshapes.

### Mode 3: Parallel Supercell Hessian Evaluation (Stage 6)
On a $5\times 5\times 2$ supercell (398 atoms), computing the full Hessian involves **2388 atomic displacements**. With `ParallelVibrations`:
```python
parallel_hessian = True
parallel_hessian_workers = 16  # Evaluates 16 displacements concurrently
```
The displacement evaluation walltime decreases from **~20 minutes** down to **~45 seconds**.

---

## 4. Best Practices for CINECA Leonardo

* **Use `$SCRATCH` for Workspaces**:
  Always place large run directories inside `$SCRATCH` (e.g. `$SCRATCH/hydeph_run_${SLURM_JOB_ID}`) to leverage Leonardo's high-bandwidth Lustre parallel filesystem.
* **Keep `$HOME` Clean**:
  `$HOME` has a strict quota (50 GB) and is not intended for heavy I/O or temporary QE wavefunctions.
* **Account Specification**:
  Ensure your project account is exported:
  ```bash
  export CINECA_ACCOUNT="<your_project_account>"
  ```

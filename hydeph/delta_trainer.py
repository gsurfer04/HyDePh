"""Extracts Delta E and Delta F relative to the foundation potential and trains Delta-NEP in GPUMD."""

import os
import sys
import time
import shutil
import subprocess
import numpy as np
import ase.io
from ase import Atoms
from calorine.calculators import CPUNEP


def resolve_nep_executable(nep_exe=None):
    """Resolves the GPUMD `nep` executable path."""
    if nep_exe:
        if os.path.isfile(nep_exe):
            return os.path.abspath(nep_exe)
        found = shutil.which(nep_exe)
        if found:
            return found
        return nep_exe

    env_nep = os.environ.get("NEP_EXE") or os.environ.get("GPUMD_NEP")
    if env_nep:
        if os.path.isfile(env_nep):
            return os.path.abspath(env_nep)
        found = shutil.which(env_nep)
        if found:
            return found
        return env_nep

    if sys.platform.startswith("win"):
        for name in ["nep.exe", "nep"]:
            found = shutil.which(name)
            if found:
                return found
        for p in [r"C:\GPUMD\bin\nep.exe", r"C:\Program Files\GPUMD\bin\nep.exe"]:
            if os.path.isfile(p):
                return os.path.abspath(p)
        return "nep.exe"
    else:
        found = shutil.which("nep") or shutil.which("nep.exe")
        if found:
            return found
        for p in ["/usr/local/bin/nep", "/usr/bin/nep"]:
            if os.path.isfile(p):
                return p
        return "/usr/local/bin/nep"


def write_delta_nep_in(out_path, unique_elements, neurons=30, batch=2000, steps=30000):
    """Writes `nep.in` for Delta-NEP training."""
    elem_str = " ".join(unique_elements)
    content = f"""type {len(unique_elements)} {elem_str}
version 4
cutoff 4.5 3.5
n_max 6 4
basis_size 8 8
l_max 2 1
neuron {neurons}
batch {batch}
population 50
generation {steps}
"""
    with open(out_path, "w") as f:
        f.write(content)


class DeltaTrainer:
    def __init__(self, config):
        self.config = config
        self.calc_foundation = CPUNEP(config.foundation_potential_path)

    def compute_delta_dataset(self, labeled_frames):
        """Computes Delta E and Delta F for each DFT-labeled frame relative to the foundation potential."""
        delta_frames = []
        print("\n[-] Evaluating foundation potential (nep89) to extract Delta E and Delta F...")

        for fr in labeled_frames:
            e_dft = fr.info["energy"]
            f_dft = fr.arrays["forces"]

            at_base = fr.copy()
            at_base.calc = self.calc_foundation
            e_base = at_base.get_potential_energy()
            f_base = at_base.get_forces()

            fr_delta = fr.copy()
            fr_delta.info["energy"] = e_dft - e_base
            fr_delta.arrays["forces"] = f_dft - f_base
            delta_frames.append(fr_delta)

        print(f"[+] Computed Delta properties for {len(delta_frames)} frames.")
        return delta_frames

    def train_delta_nep(self, delta_frames, test_fraction=0.20):
        """Writes train/test splits and runs GPUMD `nep` training."""
        train_dir = os.path.join(self.config.workspace_dir, "delta_nep_training")
        os.makedirs(train_dir, exist_ok=True)

        n_total = len(delta_frames)
        n_test = max(1, int(n_total * test_fraction))
        n_train = n_total - n_test

        rng = np.random.RandomState(42)
        indices = rng.permutation(n_total)
        train_frames = [delta_frames[i] for i in indices[:n_train]]
        test_frames = [delta_frames[i] for i in indices[n_train:]]

        ase.io.write(os.path.join(train_dir, "train.xyz"), train_frames, format="extxyz")
        ase.io.write(os.path.join(train_dir, "test.xyz"), test_frames, format="extxyz")

        unique_elems = sorted(list(set(delta_frames[0].get_chemical_symbols())))
        nep_in = os.path.join(train_dir, "nep.in")
        effective_batch = min(self.config.nep_batch, len(train_frames))
        write_delta_nep_in(nep_in, unique_elems, neurons=self.config.nep_neurons,
                           batch=effective_batch, steps=self.config.nep_steps)

        print(f"\n[-] Training Delta-NEP on GPU (train={len(train_frames)}, test={len(test_frames)}, steps={self.config.nep_steps})...")
        t0 = time.time()
        nep_cmd = resolve_nep_executable(getattr(self.config, "nep_exe", None))
        res = subprocess.run([nep_cmd], cwd=train_dir, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"NEP training failed using '{nep_cmd}':\n{res.stderr}\n{res.stdout}")

        print(f"[+] Delta-NEP training completed in {time.time()-t0:.2f} s")

        pot_src = os.path.join(train_dir, "nep.txt")
        pot_dest = os.path.join(self.config.workspace_dir, "nep_delta.txt")
        if os.path.isfile(pot_src):
            shutil.copy2(pot_src, pot_dest)
        else:
            raise FileNotFoundError(f"Trained NEP potential '{pot_src}' was not found after training.")
        print(f"[+] Trained Delta-NEP saved: {pot_dest}")

        loss_path = os.path.join(train_dir, "loss.out")
        if os.path.isfile(loss_path):
            d_loss = np.loadtxt(loss_path)
            if len(d_loss.shape) > 1 and len(d_loss) > 0:
                final_loss = d_loss[-1]
                print(f"    - Final Loss: Total={final_loss[0]:.4e}, Energy={final_loss[1]:.4e}, Force={final_loss[2]:.4e}")

        return pot_dest

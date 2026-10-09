"""
defect_delta_nep.delta_trainer
==============================
Computes Delta E and Delta F between target DFT and foundation potential (nep89),
assembles train.xyz and test.xyz, writes nep.in, and trains the Delta-NEP in GPUMD.
"""

import os
import time
import subprocess
import numpy as np
import ase.io
from ase import Atoms
from calorine.calculators import CPUNEP


def write_delta_nep_in(out_path, unique_elements, neurons=30, batch=2000, steps=30000):
    """Writes a clean nep.in configuration for Delta-NEP training."""
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
        """Computes Delta E and Delta F for each frame relative to nep89."""
        delta_frames = []
        print("\n[-] Evaluating foundation potential (nep89) to extract Delta E and Delta F...")
        
        for idx, fr in enumerate(labeled_frames):
            e_dft = fr.info["energy"]
            f_dft = fr.arrays["forces"]
            
            # Evaluate nep89
            at_base = fr.copy()
            at_base.calc = self.calc_foundation
            e_base = at_base.get_potential_energy()
            f_base = at_base.get_forces()
            
            delta_e = e_dft - e_base
            delta_f = f_dft - f_base
            
            fr_delta = fr.copy()
            fr_delta.info["energy"] = delta_e
            fr_delta.arrays["forces"] = delta_f
            delta_frames.append(fr_delta)
            
        print(f"[+] Computed Delta properties for {len(delta_frames)} frames.")
        return delta_frames

    def train_delta_nep(self, delta_frames, test_fraction=0.20):
        """Prepares train/test split and runs GPUMD nep training."""
        train_dir = os.path.join(self.config.workspace_dir, "delta_nep_training")
        os.makedirs(train_dir, exist_ok=True)
        
        # Split train / test
        n_total = len(delta_frames)
        n_test = max(1, int(n_total * test_fraction))
        n_train = n_total - n_test
        
        # Shuffle with fixed seed for reproducibility
        rng = np.random.RandomState(42)
        indices = rng.permutation(n_total)
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]
        
        train_frames = [delta_frames[i] for i in train_idx]
        test_frames = [delta_frames[i] for i in test_idx]
        
        train_xyz = os.path.join(train_dir, "train.xyz")
        test_xyz = os.path.join(train_dir, "test.xyz")
        ase.io.write(train_xyz, train_frames, format="extxyz")
        ase.io.write(test_xyz, test_frames, format="extxyz")
        
        # Unique chemical elements
        unique_elems = sorted(list(set(delta_frames[0].get_chemical_symbols())))
        nep_in = os.path.join(train_dir, "nep.in")
        effective_batch = min(self.config.nep_batch, len(train_frames))
        write_delta_nep_in(nep_in, unique_elems, neurons=self.config.nep_neurons,
                           batch=effective_batch, steps=self.config.nep_steps)
        
        print(f"\n[-] Training Delta-NEP on GPU (train={len(train_frames)}, test={len(test_frames)}, steps={self.config.nep_steps})...")
        t0 = time.time()
        res = subprocess.run(["/usr/local/bin/nep"], cwd=train_dir, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"NEP training failed:\n{res.stderr}\n{res.stdout}")
            
        print(f"[+] Delta-NEP training completed in {time.time()-t0:.2f} s")
        
        # Copy trained potential
        pot_src = os.path.join(train_dir, "nep.txt")
        pot_dest = os.path.join(self.config.workspace_dir, "nep_delta.txt")
        os.system(f"cp {pot_src} {pot_dest}")
        print(f"[+] Trained Delta-NEP saved: {pot_dest}")
        
        # Parity analysis
        loss_path = os.path.join(train_dir, "loss.out")
        if os.path.isfile(loss_path):
            d_loss = np.loadtxt(loss_path)
            if len(d_loss.shape) > 1 and len(d_loss) > 0:
                final_loss = d_loss[-1]
                print(f"    - Final Loss: Total={final_loss[0]:.4e}, Energy={final_loss[1]:.4e}, Force={final_loss[2]:.4e}")
                
        return pot_dest

"""
hydeph.qe_runner
================
Automates Quantum ESPRESSO pw.x single-point calculations to label candidate
frames with total DFT energies and atomic forces using user-provided pseudopotentials.
"""

import os
import re
import time
import subprocess
import numpy as np
import ase.io
from ase import Atoms
from ase.units import Ry, Bohr


def write_qe_input(atoms, pw_in_path, config, calculation="scf"):
    """Writes a clean Quantum ESPRESSO pw.x input card."""
    cell = atoms.get_cell()
    pos = atoms.positions
    symbols = atoms.get_chemical_symbols()
    
    unique_symbols = sorted(list(set(symbols)))
    mass_dict = {"Li": 6.941, "Cl": 35.453, "C": 12.011, "Si": 28.085, "N": 14.007}

    header = f"""&CONTROL
  calculation = '{calculation}',
  restart_mode = 'from_scratch',
  pseudo_dir = '{config.pseudo_dir}',
  outdir = './tmp',
  prefix = 'calc',
  tprnfor = .true.,
  tstress = .false.,
  disk_io = 'none'
/
&SYSTEM
  ibrav = 0,
  nat = {len(atoms)},
  ntyp = {len(unique_symbols)},
  ecutwfc = {config.ecutwfc},
  ecutrho = {config.ecutrho},
  nspin = {2 if config.spin_polarized else 1},
  tot_magnetization = {config.spin_multiplicity - 1 if config.spin_polarized else 0},
  tot_charge = {config.tot_charge},
  occupations = 'smearing',
  smearing = 'gaussian',
  degauss = 0.005
/
&ELECTRONS
  conv_thr = 1.0d-5,
  mixing_beta = 0.4
/
ATOMIC_SPECIES
"""
    for s in unique_symbols:
        upf_name = os.path.basename(config.pseudopotentials[s])
        mass = mass_dict.get(s, 20.0)
        header += f"  {s:<3} {mass:8.4f}  {upf_name}\n"

    header += "\nCELL_PARAMETERS (angstrom)\n"
    for vec in cell:
        header += f"  {vec[0]:14.8f} {vec[1]:14.8f} {vec[2]:14.8f}\n"

    header += "\nATOMIC_POSITIONS (angstrom)\n"
    for s, p in zip(symbols, pos):
        header += f"  {s:<3} {p[0]:14.8f} {p[1]:14.8f} {p[2]:14.8f}\n"

    header += f"\nK_POINTS (automatic)\n  {config.kpoints[0]} {config.kpoints[1]} {config.kpoints[2]} 0 0 0\n"

    with open(pw_in_path, "w") as f:
        f.write(header)


def parse_qe_output(pw_out_path):
    """Parses total energy (eV) and forces (eV/A) from QE pw.x output."""
    if not os.path.isfile(pw_out_path):
        return None, None
    
    energy_ev = None
    forces_list = []
    
    with open(pw_out_path, "r") as f:
        lines = f.readlines()
        
    for i, line in enumerate(lines):
        if "!    total energy" in line:
            # '!    total energy              =     -47.99240624 Ry'
            parts = line.split("=")
            if len(parts) > 1:
                e_ry = float(parts[1].replace("Ry", "").strip())
                energy_ev = e_ry * Ry  # 1 Ry = 13.6056980659 eV
                
        if "Forces acting on atoms (cartesian axes, Ry/au):" in line:
            forces_list = []
            for j in range(i + 2, len(lines)):
                fline = lines[j].strip()
                if not fline or "total force" in fline:
                    break
                m = re.search(r"force =\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)", fline)
                if m:
                    fx = float(m.group(1)) * (Ry / Bohr)
                    fy = float(m.group(2)) * (Ry / Bohr)
                    fz = float(m.group(3)) * (Ry / Bohr)
                    forces_list.append([fx, fy, fz])
                    
    forces_ev_a = np.array(forces_list) if forces_list else None
    return energy_ev, forces_ev_a


class QERunner:
    def __init__(self, config, n_procs=2):
        self.config = config
        self.n_procs = n_procs
        self.pw_exe = "/usr/local/bin/pw.x"

    def label_single_frame(self, atoms, calc_dir):
        """Runs single-point QE calculation for one frame with live output streaming and caching."""
        os.makedirs(calc_dir, exist_ok=True)
        pw_in = os.path.join(calc_dir, "pw.in")
        pw_out = os.path.join(calc_dir, "pw.out")
        
        # Check cache
        if os.path.isfile(pw_out):
            with open(pw_out, "r") as f:
                content = f.read()
            if "JOB DONE." in content:
                e_ev, f_ev_a = parse_qe_output(pw_out)
                if e_ev is not None and f_ev_a is not None and len(f_ev_a) == len(atoms):
                    print(f"      [Cache hit] Reusing converged calculation in {calc_dir}", flush=True)
                    at_labeled = atoms.copy()
                    at_labeled.info["energy"] = e_ev
                    at_labeled.arrays["forces"] = f_ev_a
                    return at_labeled

        write_qe_input(atoms, pw_in, self.config)
        
        cmd = f"mpirun -np {self.n_procs} {self.pw_exe} -i pw.in > pw.out 2>&1"
        proc = subprocess.Popen(cmd, shell=True, cwd=calc_dir)
        
        last_iter = 0
        while proc.poll() is None:
            time.sleep(4.0)
            if os.path.isfile(pw_out):
                try:
                    with open(pw_out, "r") as f:
                        lines = f.readlines()
                    for line in reversed(lines[-30:]):
                        if "iteration #" in line:
                            m = re.search(r"iteration #\s+(\d+)", line)
                            if m:
                                cur_iter = int(m.group(1))
                                if cur_iter != last_iter:
                                    last_iter = cur_iter
                                    print(f"      [pw.x] SCF iteration {cur_iter}...", flush=True)
                            break
                        elif "Forces acting on atoms" in line:
                            print("      [pw.x] SCF converged! Calculating atomic forces...", flush=True)
                            break
                except Exception:
                    pass

        proc.wait()
        
        e_ev, f_ev_a = parse_qe_output(pw_out)
        if e_ev is None or f_ev_a is None or len(f_ev_a) != len(atoms):
            tail_msg = ""
            if os.path.isfile(pw_out):
                with open(pw_out, "r") as f:
                    tail_msg = "".join(f.readlines()[-20:])
            raise RuntimeError(f"QE calculation failed in {calc_dir}.\n{tail_msg}")
            
        at_labeled = atoms.copy()
        at_labeled.info["energy"] = e_ev
        at_labeled.arrays["forces"] = f_ev_a
        return at_labeled

    def label_all_frames(self, frames, max_frames=None):
        """Labels all frames in the dataset."""
        frames_to_run = frames[:max_frames] if max_frames else frames
        labeled_frames = []
        dft_dir = os.path.join(self.config.workspace_dir, "dft_calculations")
        os.makedirs(dft_dir, exist_ok=True)
        
        print(f"\n[-] Running Quantum ESPRESSO on {len(frames_to_run)} representative frames (np={self.n_procs})...")
        for idx, fr in enumerate(frames_to_run):
            cdir = os.path.join(dft_dir, f"frame_{idx:03d}")
            t0 = time.time()
            labeled_fr = self.label_single_frame(fr, cdir)
            labeled_frames.append(labeled_fr)
            print(f"    [+] Frame {idx+1:2d}/{len(frames_to_run)} labeled | E = {labeled_fr.info['energy']:.4f} eV | Max F = {np.abs(labeled_fr.arrays['forces']).max():.4f} eV/A ({time.time()-t0:.1f} s)")
            
        out_xyz = os.path.join(self.config.workspace_dir, "labeled_dft_frames.xyz")
        ase.io.write(out_xyz, labeled_frames)
        print(f"[+] All frames successfully labeled -> {out_xyz}")
        return labeled_frames

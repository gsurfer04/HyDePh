"""Quantum ESPRESSO Slurm script generation and subprocess execution monitoring."""

from __future__ import annotations

import os
import re
import time
import subprocess
import logging
from typing import Optional, List, Dict
from ase import Atoms

from .io import RY_TO_EV
from .parsers import parse_qe_relaxation_output

logger = logging.getLogger(__name__)


def format_slurm_module_lines(modules: Optional[List[str]] = None, purge: bool = True) -> str:
    """Formats deterministic bash module purge and load lines in exact specified order without heuristic loops.

    Preserves exact user-defined library loading order (e.g. profile/chem-phys before quantum-espresso).
    """
    lines = []
    if purge:
        lines.append("module purge")
    if modules:
        for m in modules:
            m_clean = str(m).strip()
            if m_clean:
                lines.append(f"module load {m_clean}")
    else:
        lines.append("# Note: No cluster environment modules specified in configuration.")
    return "\n".join(lines)


def run_monitored_qe_subprocess(
    cmd_args: List[str],
    calc_dir: str,
    pw_out: str,
    env: Dict[str, str],
    timeout: Optional[float] = None,
    mode: str = "scf",
    allow_fallbacks: bool = False,
    atoms_template: Optional[Atoms] = None,
) -> Optional[Atoms]:
    """Executes a Quantum ESPRESSO subprocess, continuously monitoring progress and enforcing timeouts.

    Parameters
    ----------
    cmd_args : List[str]
        Command line arguments for the subprocess.
    calc_dir : str
        Working directory for the calculation.
    pw_out : str
        Path to output file (pw.out).
    env : Dict[str, str]
        Environment variables for subprocess execution.
    timeout : float, optional
        Maximum execution walltime in seconds.
    mode : str
        "scf" (logs SCF iterations) or "relax" (logs BFGS steps).
    allow_fallbacks : bool
        If True and mode == "relax", attempts to recover latest partial BFGS geometry upon timeout.
    atoms_template : Atoms, optional
        Reference Atoms object for parsing partial relaxation coordinates.

    Returns
    -------
    Optional[Atoms]
        Partial relaxed geometry if timed out under allow_fallbacks=True, otherwise None.

    Raises
    ------
    RuntimeError
        If the process execution fails or times out.
    """
    t_start = time.time()
    t_last_log = t_start
    last_iter = 0
    cur_step = 0
    last_logged_bfgs_step = -1
    last_logged_iter = (0, 0)
    last_logged_scf_converged_step = -1
    bfgs_converged_logged = False
    scf_converged_logged = False
    sleep_interval = min(0.5, max(0.05, float(timeout) / 5.0 if timeout else 0.5))
    is_relax = mode in ("relax", "vc-relax")
    calc_type = "relaxation" if is_relax else "calculation"

    def _poll_log() -> None:
        nonlocal last_iter, cur_step, last_logged_bfgs_step, last_logged_iter
        nonlocal last_logged_scf_converged_step, bfgs_converged_logged, scf_converged_logged
        if not os.path.isfile(pw_out):
            return
        try:
            with open(pw_out, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            if not lines:
                return

            if not is_relax:
                for line in reversed(lines[-30:]):
                    if "iteration #" in line:
                        m = re.search(r"iteration #\s+(\d+)", line)
                        if m:
                            cur_i = int(m.group(1))
                            if cur_i != last_iter:
                                last_iter = cur_i
                                logger.info(f"      [pw] SCF iteration {cur_i}...")
                        break
                    elif "Forces acting on atoms" in line or "convergence has been achieved" in line:
                        if not scf_converged_logged:
                            scf_converged_logged = True
                            logger.info("      [pw] SCF converged! Calculating atomic forces...")
                        break
            else:
                tail_lines = lines[-120:]
                tail_text = "".join(tail_lines)

                # 1. BFGS final convergence check
                m_conv = re.search(r"bfgs converged in\s+(\d+)\s+scf cycles and\s+(\d+)\s+bfgs steps", tail_text, re.IGNORECASE)
                if m_conv and not bfgs_converged_logged:
                    bfgs_converged_logged = True
                    logger.info(f"      [pw relax] BFGS converged in {m_conv.group(1)} SCF cycles and {m_conv.group(2)} BFGS steps!")

                # 2. BFGS step finished check
                for m_step in re.finditer(r"number of bfgs steps\s*=\s*(\d+)", tail_text, re.IGNORECASE):
                    step_num = int(m_step.group(1))
                    if step_num > last_logged_bfgs_step:
                        start_idx = max(0, m_step.start() - 600)
                        end_idx = min(len(tail_text), m_step.end() + 600)
                        win = tail_text[start_idx:end_idx]

                        e_new = re.findall(r"energy\s+new\s*=\s*([-\d\.]+)", win)
                        e_fin = re.findall(r"Final energy\s*=\s*([-\d\.]+)", win)
                        e_tot = re.findall(r"!    total energy\s*=\s*([-\d\.]+)", win)
                        e_val = float(e_new[-1]) if e_new else (float(e_fin[-1]) if e_fin else (float(e_tot[-1]) if e_tot else None))

                        grad_m = re.findall(r"Gradient error\s*=\s*([\d\.E+-]+)", win)
                        grad_val = grad_m[-1] if grad_m else None

                        e_str = f" (E = {e_val * RY_TO_EV:.4f} eV" if e_val is not None else ""
                        if grad_val:
                            e_str += f", Grad = {grad_val} Ry/Bohr)" if e_str else f" (Grad = {grad_val} Ry/Bohr)"
                        elif e_str:
                            e_str += ")"

                        logger.info(f"      [pw relax] BFGS Step {step_num} finished!{e_str}")
                        last_logged_bfgs_step = step_num
                        cur_step = max(cur_step, step_num + 1)

                # 3. Step SCF convergence check
                m_scf_conv = re.findall(r"convergence has been achieved in\s+(\d+)\s+iterations", tail_text, re.IGNORECASE)
                if m_scf_conv and last_logged_scf_converged_step < cur_step:
                    last_logged_scf_converged_step = cur_step
                    logger.info(f"      [pw relax] Step {cur_step} | SCF converged in {m_scf_conv[-1]} iterations. Calculating forces...")

                # 4. Step SCF iteration check
                for line in reversed(tail_lines):
                    if "iteration #" in line:
                        m_it = re.search(r"iteration #\s+(\d+)", line)
                        if m_it:
                            it_num = int(m_it.group(1))
                            if (cur_step, it_num) != last_logged_iter and cur_step > last_logged_scf_converged_step:
                                last_logged_iter = (cur_step, it_num)
                                logger.info(f"      [pw relax] Step {cur_step} | SCF iteration {it_num}...")
                        break
        except Exception:
            pass

    try:
        with open(pw_out, "wb") as f_out:
            proc = subprocess.Popen(
                cmd_args,
                stdout=f_out,
                stderr=subprocess.STDOUT,
                cwd=calc_dir,
                env=env,
            )

            while proc.poll() is None:
                now = time.time()
                if timeout is not None and (now - t_start) > float(timeout):
                    try:
                        proc.kill()
                        proc.wait()
                    except Exception:
                        pass
                    if is_relax and allow_fallbacks and atoms_template is not None:
                        at_part, e_part = parse_qe_relaxation_output(pw_out, atoms_template, require_converged=False)
                        if at_part is not None:
                            logger.warning(
                                f"      [pw relax] Relaxation reached timeout ({float(timeout):.1f}s) but recovered latest BFGS geometry "
                                f"({at_part.info.get('energy', 0.0):.4f} eV) under allow_fallbacks=True."
                            )
                            return at_part
                    raise RuntimeError(f"QE {calc_type} timed out after {float(timeout):.1f}s in {calc_dir}")

                time.sleep(sleep_interval)
                now = time.time()
                if (now - t_last_log) >= 2.0:
                    t_last_log = now
                    _poll_log()

            proc.wait()
            _poll_log()
    except OSError as exc:
        raise RuntimeError(f"QE {calc_type} failed in {calc_dir}: {exc}") from exc

    return None

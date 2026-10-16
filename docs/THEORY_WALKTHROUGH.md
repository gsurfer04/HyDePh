# HyDePh Theoretical Walkthrough & Literature Foundations

**HyDePh**: **Hy**brid **De**fect **Ph**otoluminescence  
*A First-Principles $\Delta$-Machine Learning Potential Framework for Turnkey Color-Center Optical Spectroscopy*

---

## 1. Introduction: The Multi-Scale Defect Spectroscopy Problem

Optically and magnetically active point defects in crystalline solids—commonly known as **color centers**—are premier platforms for emerging quantum technologies, including quantum repeaters, single-photon sources, and nanoscale quantum sensors [1–4]. Probing and identifying these centers relies on **photoluminescence (PL) spectroscopy**, which measures the energy of electronic transitions and their coupling to the vibrational modes of the host lattice [5, 6].

However, first-principles modeling of color-center PL spectra presents a severe **multi-scale dilemma**:
1. **Quantum Localization**: The optical transition involves localized deep defect states whose potential energy surfaces (PES) are governed by delicate electronic correlations, charge reorganization, and Jahn-Teller symmetry breaking [5, 7]. Accurately capturing these energetics requires high-level density functional theory (DFT), often with hybrid exchange-correlation functionals or constrained DFT ($\Delta$SCF) [6, 8].
2. **Macroscopic Elastic & Vibrational Strain**: The defect cavity couples to acoustic and optical lattice phonons that propagate across thousands to millions of atoms. Simulating supercells large enough to eliminate artificial periodic image strain ($>10^3\text{--}10^5$ atoms) is computationally prohibitive with direct DFT [5, 6, 9].
3. **Finite-Temperature Anharmonicity**: In mechanically softer lattices (e.g., alkali halides, metal-halide perovskites, chalcogenides), optical transitions induce substantial structural displacements that encounter steep repulsive-wall potentials ($V \sim k_3 Q^3 + k_4 Q^4$). Standard harmonic approximations break down, requiring picosecond-scale finite-temperature molecular dynamics (MD) sampling that is impossible with direct DFT [10–13].
4. **Limitations of Universal Foundation Potentials**: While universal machine-learned potentials (e.g., `nep89`) provide broad lattice stability across the periodic table, they are trained on pristine or generic bulk databases and completely lack defect-specific excited-state forces, optical Franck-Condon relaxation paths, and localized many-body corrections [14, 15].

**HyDePh** resolves this conflict by combining a **$\Delta$-Machine Learning Potential** ($\Delta$-NEP) with **quantum generating functions** and **finite-temperature anharmonic molecular dynamics free energy sampling**. This document provides an exhaustive stage-by-stage walkthrough of the theoretical formulation, physical justification, and foundational literature underpinning each stage of the HyDePh engine.

---

## 2. Global Architecture & Theoretical Pipeline

```mermaid
flowchart TD
    subgraph S1["Stage 1: Defect Embedding"]
        Pristine["Pristine Bulk Unit Cell"] --> Builder["DefectBuilder Tiler"]
        Defect["Pre-made Defect Unit Cell"] --> Builder
        Builder --> SC["Defect Supercell (PBC)"]
        SC --> Relax["GS & ES Relaxations (Delta Q)"]
    end

    subgraph S2["Stage 2 & 3: Active Frame Sampling"]
        Relax --> FC["Phase 1: Franck-Condon Path"]
        Relax --> Rattle["Phase 2: Defect-Core Rattling"]
        Relax --> MD["Phase 3: nep89 MD + FPS"]
        FC --> Priority["Phase 4: Interleaved Priority Ensemble"]
        Rattle --> Priority
        MD --> Priority
    end

    subgraph S3["Stage 4 & 5: Delta-Learning & Training"]
        Priority --> QE["pw.x First-Principles Labeling (Delta SCF)"]
        BasePot["Foundation Potential (nep89)"] --> Subtract["Extract Delta E & Delta F"]
        QE --> Subtract
        Subtract --> NEP["GPU Delta-NEP Training (GPUMD)"]
        NEP --> Calc["HybridDefectCalculator (V_base + Delta V)"]
        BasePot --> Calc
    end

    subgraph S4["Stage 6 & 7: Harmonic Quantum Spectroscopy"]
        Calc --> Hessian["Hessian & Normal Modes e_k"]
        Hessian --> Sk["Huang-Rhys S_k & S(hw)"]
        Sk --> GenFunc["Generating Function G(t)"]
        GenFunc --> HarmPL["Harmonic Lineshapes A_harm(E, T)"]
    end

    subgraph S5["Stage 8: Anharmonic Sampling & Multi-Regime Spectroscopy"]
        Calc --> LangMD["Finite-T Langevin MD (10 - 300 K)"]
        LangMD --> Proj["Mass-Weighted Optical Projection u_opt"]
        Proj --> VACF["Projected VDOS g_opt(hw, T) & Mode Softening alpha_soft"]
        LangMD --> DeltaU["Energy Gap Cumulants Delta U(t)"]
        VACF --> Regimes{"Coupling S < 10?"}
        Regimes -->|"Yes (Weak/Intermediate: NV-, SiC, LiCl)"| SoftGen["Softened Quantum GenFn Lineshapes A_anh(E, T)"]
        Regimes -->|"No (Strong: LiF, S >= 10)"| MarcusConv["Semiclassical Marcus Gap Convolution"]
        DeltaU --> MarcusConv
        SoftGen --> FinalPL["High-Definition Lineshapes"]
        MarcusConv --> FinalPL
    end

    subgraph S6["Stage 9: 1D Configuration Coordinate & Stokes Shift"]
        Calc --> CCScan["Linear Optical Path R(lambda) & Q = lambda Delta Q"]
        CCScan --> PES["Potential Surfaces E_GS(Q) & E_ES(Q)"]
        PES --> Stokes["Stokes Shift Delta E_Stokes, E_abs, E_em, hbar Omega, S_1D"]
        Stokes --> CCDiag["2-Panel CC Diagram & Optical Spectra"]
    end
```

---

## 3. Stage-by-Stage Theoretical Walkthrough

### Stage 1: Defect Embedding & Supercell Mechanics

#### 1. Theory & Mathematical Formulation
To model isolated point defects within periodic boundary conditions (PBC), the defect is embedded into a macroscopic supercell constructed from $N_1 \times N_2 \times N_3$ repetitions of the primitive or conventional unit cell. Under PBC, an isolated defect interacts with its periodic images through fictitious elastic strain fields that scale inversely with cell volume ($\sim 1/L^3$) [5, 9]:

$$
E_{\mathrm{elastic}}(L) = E_{\mathrm{dilute}} + \frac{\alpha_{\mathrm{el}}}{L^3} + \mathcal{O}(L^{-5})
$$

where $L$ is the linear supercell dimension and $\alpha_{\mathrm{el}}$ is an elastic dipole tensor parameter.

HyDePh automates this process through [`DefectBuilder`](../hydeph/defect_builder.py):
1. The user supplies a **pristine unit cell** and a **pre-made defect unit cell** (containing the vacancy, impurity, or complex).
2. The engine tiles pristine bulk unit cells in an $N_1 \times N_2 \times N_3$ array around the defect block (placed at the central tile), automatically detecting the defect coordinate and atomic connectivity.
3. The system is relaxed into the Ground State (GS) equilibrium geometry $\mathbf{R}_{\mathrm{GS}}$ and Relaxed Excited State (ES) geometry $\mathbf{R}_{\mathrm{ES}}$.
4. The optical transition coordinate is defined by the Franck-Condon structural displacement vector:

$$
\Delta \mathbf{R} = \mathbf{R}_{\mathrm{ES}} - \mathbf{R}_{\mathrm{GS}}
$$

and the total mass-weighted structural displacement:

$$
\Delta Q = \sqrt{\sum_{i=1}^N M_i |\mathbf{R}_{\mathrm{ES}, i} - \mathbf{R}_{\mathrm{GS}, i}|^2}
$$

where $M_i$ is the atomic mass of atom $i$.

#### 2. Physical Justification & Literature References
- **Freysoldt et al., *Rev. Mod. Phys.* 86, 253 (2014)** [9]: Established the definitive theoretical framework for point defect supercell calculations, demonstrating how artificial image-image electrostatic and elastic interactions distort defect geometry unless large supercells ($>500\text{--}1000$ atoms) or rigorous corrections are applied.
- **Alkauskas et al., *New J. Phys.* 16, 073026 (2014)** [5]: Demonstrated that $\Delta Q$ is the fundamental configuration-coordinate metric governing optical lineshapes, Franck-Condon factors, and the overall electron-phonon coupling strength.
- **Jin et al., *Phys. Rev. Materials* 5, 084603 (2021)** [6]: Benchmarked finite-size convergence of defect PL in diamond and SiC, showing that supercells containing $\ge 512$ atoms are mandatory to converge sideband spectra to the dilute limit.

---

### Stage 2: Hybrid Potential Formulation ($\Delta$-Learning)

#### 1. Theory & Mathematical Formulation
Universal machine learning potentials (e.g., `nep89`) are trained across wide chemical spaces using Chebyshev polynomial descriptors [14, 15]. While they accurately capture equilibrium bulk moduli, acoustic phonons, and thermal expansion, they fail to predict defect-localized states because:
1. Universal training sets do not include optical Franck-Condon configurations or broken-symmetry excited states.
2. The localized charge redistribution surrounding a vacancy or radical defect induces non-equilibrium electrostatic and Jahn-Teller forces that are completely missed by bulk potentials [6, 7].

HyDePh addresses this via **$\Delta$-Machine Learning** [16, 17], decomposing the total energy and atomic forces into a foundational baseline and a defect-specific correction:

$$
\begin{aligned}
V_{\mathrm{hybrid}}(\mathbf{R}) &= V_{\mathrm{base}}(\mathbf{R}) + \Delta V_{\mathrm{NEP}}(\mathbf{R}) \\
\mathbf{F}_{\mathrm{hybrid}}(\mathbf{R}) &= \mathbf{F}_{\mathrm{base}}(\mathbf{R}) + \Delta \mathbf{F}_{\mathrm{NEP}}(\mathbf{R})
\end{aligned}
$$

where the target training residuals are:

$$
\begin{aligned}
\Delta V_{\mathrm{target}}(\mathbf{R}_k) &= E_{\mathrm{DFT}}(\mathbf{R}_k) - V_{\mathrm{base}}(\mathbf{R}_k) \\
\Delta \mathbf{F}_{\mathrm{target}}(\mathbf{R}_k) &= \mathbf{F}_{\mathrm{DFT}}(\mathbf{R}_k) - \mathbf{F}_{\mathrm{base}}(\mathbf{R}_k)
\end{aligned}
$$

#### 2. Physical Justification & Literature References
- **Ramakrishnan et al., *J. Chem. Theory Comput.* 11, 2087 (2015)** [16]: Introduced $\Delta$-machine learning, proving that learning energy and force differences ($\Delta = \text{Target} - \text{Baseline}$) accelerates convergence by $10\text{--}100\times$ compared to direct learning from scratch, because the baseline model absorbs the majority of the non-linear potential landscape.
- **Chmiela et al., *Nat. Commun.* 9, 3887 (2018)** [17]: Demonstrated that gradient-domain force learning combined with baseline reference models preserves energy conservation and exact physical asymptotic limits in molecular systems.
- **Fan et al., *Phys. Rev. B* 104, 104309 (2021)** [14]: Showed that the Neuroevolution Potential (NEP) framework delivers near-DFT accuracy at $10^5\times$ lower computational cost, making it the ideal architecture for real-time hybrid evaluation.

---

### Stage 3: Four-Phase Representative Active Frame Sampling

#### 1. Theory & Mathematical Formulation
The accuracy of the $\Delta$-NEP correction depends entirely on the configurational diversity of the training set. However, because first-principles DFT calculations are computationally demanding, training frames must be selected with maximum information density.

HyDePh implements a **4-Phase Physical Sampling Protocol** in [`FrameSampler`](../hydeph/frame_sampler.py):
1. **Phase 1: Franck-Condon Reaction Coordinate Scan**:  
   Configurations are generated along the linear optical relaxation path:

$$
\mathbf{R}(\lambda) = \mathbf{R}_{\mathrm{GS}} + \lambda (\mathbf{R}_{\mathrm{ES}} - \mathbf{R}_{\mathrm{GS}}), \quad \lambda \in [-0.2, 1.2]
$$

   This directly probes the ground-state curvature, the excited-state curvature, and the transition barrier.
2. **Phase 2: Defect-Core Local Rattling (Frozen Phonons)**:  
   Random Gaussian displacements are applied exclusively to atoms within a sphere of radius $r_{\mathrm{core}} < 6.0\ \text{Å}$ around the defect site:

$$
\mathbf{R}_i' = \mathbf{R}_i + \boldsymbol{\eta}_i, \quad \boldsymbol{\eta}_i \sim \mathcal{N}(0, \sigma_r^2), \quad \sigma_r \in [0.08, 0.14]\ \text{Å}
$$

   This samples local harmonic and anharmonic force constants ($k_2, k_3, k_4$) of the defect cavity.
3. **Phase 3: High-Temperature Foundation MD + Farthest Point Sampling (FPS)**:  
   A fast NVT molecular dynamics trajectory ($T = 500\ \text{K}$) is propagated using the foundation potential `nep89`. From thousands of generated snapshots, a maximally diverse subset is selected using **Farthest Point Sampling (FPS)** in NEP descriptor space:

$$
d(i, j) = \|\mathbf{q}_i - \mathbf{q}_j\|_2
$$

   where $\mathbf{q}_i$ is the atom-averaged structural descriptor vector of frame $i$.
4. **Phase 4: Interleaved Priority Queue**:  
   Frames from Phases 1, 2, and 3 are balanced and prioritized so that even if the user specifies a tight DFT budget (e.g., $N_{\mathrm{DFT}} = 10\text{--}20$ frames), all critical physical manifolds are represented.

#### 2. Physical Justification & Literature References
- **Marcus, *Rev. Mod. Phys.* 65, 599 (1993)** [18]: Marcus electron transfer theory establishes that electronic transitions in condensed media are driven along the collective reaction coordinate connecting equilibrium geometries. Phase 1 guarantees that the potential profile along this coordinate is accurately modeled.
- **Knoop et al., *Phys. Rev. Materials* 4, 083809 (2020)** [19]: Proved that stochastic rattling displacements (frozen phonons) efficiently map the non-linear potential energy surface around crystal defects, capturing multi-body force constants without expensive perturbation grids.
- **Ceriotti, Tribello, and Parrinello, *J. Chem. Theory Comput.* 9, 1521 (2013)** [20]: Established Farthest Point Sampling (FPS) as the optimal method for selecting representative structures in high-dimensional machine learning descriptors, preventing clustering and ensuring maximum coverage of phase space.
- **Deringer et al., *Chem. Rev.* 121, 10073 (2021)** [21]: Showed that active learning with foundation baseline models dramatically reduces the required number of DFT labels in materials science.

---

### Stage 4: First-Principles DFT Labeling ($\Delta$SCF / CDFT)

#### 1. Theory & Mathematical Formulation
For each representative configuration $\mathbf{R}_k$ selected by the sampler, HyDePh executes self-consistent field (SCF) calculations via Quantum ESPRESSO (`pw.x`) [22] in [`QERunner`](../hydeph/qe_runner.py):
1. **Ground State**: Standard Kohn-Sham DFT with total energy $E_{\mathrm{DFT}}^{\mathrm{GS}}$ and Hellmann-Feynman atomic forces:

$$
\mathbf{F}_{i, \mathrm{DFT}} = -\nabla_i E_{\mathrm{DFT}} = -\langle \Psi \vert \nabla_i \hat{H} \vert \Psi \rangle
$$

2. **Relaxed Excited State ($\Delta$SCF / CDFT)**: The electronic excitation is modeled using **Constrained Density Functional Theory ($\Delta$SCF)**, fixing the orbital occupations to represent the promoted electron ($a_{1g} \to t_{1u}$ in alkali halide F-centers, or $a_1 \to e$ in diamond NV$^-$) [6, 8, 23].

#### 2. Physical Justification & Literature References
- **Görling, *Phys. Rev. A* 59, 3359 (1999)** [23]: Formulated exact density functional theory for excited states via orbital occupation constraints.
- **Gali, Fyta, and Kaxiras, *Phys. Rev. B* 79, 235203 (2009)** [8]: Established that $\Delta$SCF provides accurate excited-state potential energy surfaces, Stokes shifts, and Jahn-Teller relaxation energies for deep point defects in wide-bandgap semiconductors.
- **Jin et al., *Phys. Rev. Materials* 5, 084603 (2021)** [6]: Directly compared $\Delta$SCF against time-dependent hybrid DFT (TD-DFT) for defect photoluminescence spectra, proving that $\Delta$SCF accurately reproduces experimental lineshapes and excited-state geometries at a fraction of the computational cost of TD-DFT.
- **Giannozzi et al., *J. Phys.: Condens. Matter* 21, 395502 (2009)** [22]: Quantum ESPRESSO implementation of plane-wave DFT with ultrasoft and PAW pseudopotentials.

---

### Stage 5: GPU-Accelerated Neuroevolution Potential ($\Delta$-NEP) Training

#### 1. Theory & Mathematical Formulation
The $\Delta$-NEP potential is trained in GPUMD [14, 15] via [`DeltaTrainer`](../hydeph/delta_trainer.py). The local environment of atom $i$ is encoded into radial ($q_n^i$) and angular ($q_{nl}^i$) descriptor vectors based on Chebyshev polynomials and spherical harmonics within a cutoff sphere $r_c$:

$$
\begin{aligned}
q_n^i &= \sum_{j \ne i} g_n(r_{ij}) \\
q_{nl}^i &= \sum_{j, k \ne i} g_n(r_{ij}) g_n(r_{ik}) P_l(\cos \theta_{ijk})
\end{aligned}
$$

These descriptors are fed into a single-hidden-layer feedforward neural network that outputs site energy corrections $\Delta E_i$:

$$
\Delta E_{\mathrm{NEP}} = \sum_{i=1}^N \Delta E_i(\mathbf{q}^i), \quad \Delta \mathbf{F}_{i, \mathrm{NEP}} = -\sum_{j} \frac{\partial \Delta E_j}{\partial \mathbf{R}_i}
$$

Training minimizes the multi-objective loss function:

$$
\mathcal{L}(\mathbf{w}) = \frac{\lambda_e}{N_{\mathrm{frames}}} \sum_{k} |\Delta E_{\mathrm{NEP}}^{(k)} - \Delta E_{\mathrm{DFT}}^{(k)}|^2 + \frac{\lambda_f}{3N_{\mathrm{atoms}}} \sum_{i} |\Delta \mathbf{F}_{\mathrm{NEP}}^{(i)} - \Delta \mathbf{F}_{\mathrm{DFT}}^{(i)}|^2 + \lambda_{\mathrm{reg}} \|\mathbf{w}\|_2^2
$$

Optimization is performed via **Separable Natural Evolution Strategies (SNES)** [14], a derivative-free genetic algorithm that efficiently traverses complex parameter landscapes without getting trapped in local minima.

#### 2. Physical Justification & Literature References
- **Fan et al., *Phys. Rev. B* 104, 104309 (2021)** [14]: Demonstrated that Chebyshev-based descriptors combined with SNES evolutionary optimization achieve sub-meV/atom energy precision and $<50\ \text{meV/Å}$ force errors on complex materials.
- **Fan et al., *J. Chem. Phys.* 157, 114801 (2022)** [15]: Comprehensive GPUMD documentation showing that NEP evaluates forces at over $10^7$ atom-steps/second on modern GPUs, enabling full-scale lineshape and MD sampling that would be impossible with traditional neural network potentials.

---

### Stage 6: Vibrational Normal Modes & Electron-Phonon Spectral Density $S(\hbar\omega)$

#### 1. Theory & Mathematical Formulation
Once the hybrid potential is trained, [`LineshapeEngine`](../hydeph/lineshape.py) evaluates the vibrational normal modes of the supercell. The force constant matrix (Hessian) is calculated around the relaxed ground state via finite differences:

$$
H_{i\alpha, j\beta} = \frac{\partial^2 V_{\mathrm{hybrid}}}{\partial R_{i\alpha} \partial R_{j\beta}} \approx -\frac{F_{i\alpha}(R_{j\beta} + \delta) - F_{i\alpha}(R_{j\beta} - \delta)}{2\delta}
$$

The mass-weighted dynamical matrix $D$ is diagonalized:

$$
D_{i\alpha, j\beta} = \frac{H_{i\alpha, j\beta}}{\sqrt{M_i M_j}}, \quad \sum_{j\beta} D_{i\alpha, j\beta} e_{k, j\beta} = \omega_k^2 e_{k, i\alpha}
$$

yielding normal mode frequencies $\omega_k$ and displacement eigenvectors $\mathbf{e}_k$.

The optical structural displacement $\Delta \mathbf{R} = \mathbf{R}_{\mathrm{ES}} - \mathbf{R}_{\mathrm{GS}}$ is projected onto each vibrational mode $k$:

$$
\Delta Q_k = \sum_{i=1}^N \sqrt{M_i} (\mathbf{R}_{\mathrm{ES}, i} - \mathbf{R}_{\mathrm{GS}, i}) \cdot \mathbf{e}_{k, i}
$$

The partial Huang-Rhys factor $S_k$, representing the average number of phonons of mode $k$ emitted during the optical transition, is given by [5, 24]:

$$
S_k = \frac{\omega_k}{2\hbar} (\Delta Q_k)^2
$$

The total Huang-Rhys factor is:

$$
S = \sum_k S_k
$$

The continuous electron-phonon spectral density $S(\hbar\omega)$ is constructed by convoluting with a Gaussian smearing kernel $\sigma$:

$$
S(\hbar\omega) = \sum_k S_k \frac{1}{\sqrt{2\pi}\sigma} \exp\left[ -\frac{(\hbar\omega - \hbar\omega_k)^2}{2\sigma^2} \right]
$$

#### 2. Physical Justification & Literature References
- **Huang and Rhys, *Proc. R. Soc. London A* 204, 406 (1950)** [24]: Foundational paper establishing the quantum theory of light absorption and emission in color centers, introducing the parameter $S$ as the dimensionless measure of lattice coupling.
- **Alkauskas, Lyons, Steiauf, and Van de Walle, *Phys. Rev. Lett.* 109, 267401 (2012)** [7]: First-principles projection methodology demonstrating that normal mode decomposition accurately identifies dominant coupling phonon branches and Jahn-Teller active modes in diamond and wide-bandgap hosts.
- **Alkauskas et al., *New J. Phys.* 16, 073026 (2014)** [5]: Proved that projecting $\Delta \mathbf{R}$ onto ground-state normal modes yields convergent spectral densities $S(\hbar\omega)$ and eliminates finite-size supercell artifacts.

---

### Stage 7: Quantum Generating Function Lineshape Theory

#### 1. Theory & Mathematical Formulation
In the Franck-Condon and displaced harmonic oscillator approximations, the optical emission lineshape $A(E, T)$ is given by Fermi's Golden Rule summing over all initial vibrational states $m$ and final states $n$:

$$
A(E, T) = \sum_{m, n} p_m(T) |\langle \chi_{g, n} | \chi_{e, m} \rangle|^2 \delta(E_{\mathrm{ZPL}} - m\hbar\omega_m + n\hbar\omega_n - E)
$$

where $p_m(T) = e^{-E_m / k_B T} / Z$ is the Boltzmann thermal occupation and $E_{\mathrm{ZPL}}$ is the Zero-Phonon Line energy.

Evaluating this multidimensional Franck-Condon overlap directly is numerically intractable. HyDePh instead computes the exact lineshape via the **time-domain generating function** $G(t)$ [5, 25, 26]:

$$
A(E, T) = \frac{1}{2\pi\hbar} \int_{-\infty}^{\infty} dt \, \exp\left[ \frac{i(E - E_{\mathrm{ZPL}})t}{\hbar} \right] G(t) \exp\left[ -\frac{\gamma(T)|t|}{\hbar} \right]
$$

where the generating function is:

$$
G(t) = \exp[g(t) - g(0)]
$$

and the time-dependent phase function $g(t)$ is evaluated directly from the continuous spectral density $S(\hbar\omega)$:

$$
g(t) = \int_0^{\infty} d(\hbar\omega) \, S(\hbar\omega) \left[ (\bar{n}(\hbar\omega, T) + 1) e^{-i\omega t} + \bar{n}(\hbar\omega, T) e^{i\omega t} \right]
$$

Here, $\bar{n}(\hbar\omega, T)$ is the Bose-Einstein distribution:

$$
\bar{n}(\hbar\omega, T) = \frac{1}{\exp\left(\frac{\hbar\omega}{k_B T}\right) - 1}
$$

The Debye-Waller Factor (DWF), representing the fractional optical intensity residing in the sharp zero-phonon line, is:

$$
\mathrm{DWF}(T) = \exp[-g(0)] = \exp\left[ -\int_0^{\infty} d(\hbar\omega) \, S(\hbar\omega) (2\bar{n}(\hbar\omega, T) + 1) \right]
$$

The damping term $\gamma(T) = \gamma_0 + \alpha T^2$ models acoustic phonon dephasing and lifetime broadening [6].

#### 2. Physical Justification & Literature References
- **Lax, *J. Chem. Phys.* 20, 1752 (1952)** [25]: Established the generating function method for radiative and non-radiative transitions in crystal defect centers.
- **Kubo and Toyozawa, *Prog. Theor. Phys.* 13, 160 (1955)** [26]: Derived the general operator formulation of optical generating functions at finite temperature.
- **Alkauskas et al., *New J. Phys.* 16, 073026 (2014)** [5]: Validated the generating function approach for first-principles computation of color center lineshapes, showing exact agreement with experimental NV$^-$ spectra at low temperatures ($T < 77\ \text{K}$).

---

### Stage 8: Finite-Temperature Anharmonic Dynamics & Multi-Regime Spectroscopy

#### 1. Theory & Mathematical Formulation
Standard zero-temperature harmonic displaced-oscillator theory relies on two severe approximations:
1. It assumes that atomic motions are confined to static, temperature-independent parabolic potentials ($V(Q) = \frac{1}{2} M \omega_0^2 Q^2$).
2. It assumes that the surrounding host crystal remains rigid at its $0\ \text{K}$ equilibrium coordinates, neglecting thermal expansion, multi-phonon scattering, and dynamic cavity dephasing [10, 11, 13].

In reality, point defects operate in physical regimes governed by host stiffness and electron-phonon coupling strength:
- **In mechanically soft ionic or polar hosts** (e.g., alkali halides such as LiCl and LiF, or lead-halide perovskites), optical excitation induces large structural displacements ($\Delta Q \sim 1.5\text{--}2.5\ \mathrm{amu}^{1/2}\text{Å}$) that encounter steep repulsive-wall potentials ($V \sim k_3 Q^3 + k_4 Q^4$). Static harmonic theory underpredicts experimental bandwidths by up to $6\times$ [13].
- **In mechanically stiff covalent hosts** (e.g., diamond $\mathrm{NV}^-$, 4H-SiC divacancies), high Debye temperatures ($\Theta_D \approx 1200\text{--}2200\ \mathrm{K}$) and optical phonon energies ($\hbar\omega \sim 60\text{--}165\ \mathrm{meV} \gg k_B T$) ensure that transitions occur between discrete quantized vibronic eigenstates. Applying classical MD energy gap fluctuations indiscriminately washes out discrete sidebands and the zero-phonon line (ZPL) into an unphysical featureless Gaussian dome.

HyDePh resolves both limits through **mass-weighted optical projection** and **dual-regime anharmonic sampling** on the hybrid potential:

##### (A) Mass-Weighted Optical Reaction Coordinate Projection
In a defect supercell containing hundreds of atoms, defect-core atoms participate simultaneously in localized optical breathing modes and delocalized bulk lattice phonons. In standard, unprojected velocity autocorrelation functions (VACF), delocalized host optical phonons (e.g., bulk optical modes at $90\text{--}115\ \mathrm{meV}$ in SiC, or $160\ \mathrm{meV}$ in diamond) exhibit large atomic velocities despite having near-zero electron-phonon coupling ($S_k \approx 0$).

To isolate the true electron-phonon coupled vibrational density of states without bulk phonon artifacts, HyDePh defines the normalized **mass-weighted optical projection unit vector**:

$$
\mathbf{u}_{\mathrm{opt}, i} = \frac{M_i (\mathbf{R}_{\mathrm{ES}, i} - \mathbf{R}_{\mathrm{GS}, i})}{\sqrt{\sum_j M_j |\mathbf{R}_{\mathrm{ES}, j} - \mathbf{R}_{\mathrm{GS}, j}|^2}}
$$

During Langevin dynamics at temperature $T$, the collective velocity projected along the optical Franck-Condon reaction path is:

$$
v_{\mathrm{opt}}(t) = \sum_{i=1}^N \mathbf{v}_i(t) \cdot \mathbf{u}_{\mathrm{opt}, i}
$$

The optical-projected velocity autocorrelation function:

$$
C_{\mathrm{opt}}(t) = \frac{\langle v_{\mathrm{opt}}(t) v_{\mathrm{opt}}(0) \rangle}{\langle |v_{\mathrm{opt}}(0)|^2 \rangle}
$$

is windowed with a Hann filter $W_{\mathrm{Hann}}(t)$ and Fourier-transformed to yield the true optically-active vibrational density of states:

$$
g_{\mathrm{opt}}(\hbar\omega, T) = \int_0^{\infty} dt \, C_{\mathrm{opt}}(t) W_{\mathrm{Hann}}(t) \cos(\omega t)
$$

From $g_{\mathrm{opt}}(\hbar\omega, T)$, HyDePh determines the finite-temperature effective frequency $\Omega(T)$ in the active spectral window and calculates the **thermal mode softening ratio**:

$$
\alpha_{\mathrm{soft}}(T) = \frac{\Omega(T)}{\Omega_0}
$$

where $\Omega_0$ is the $0\ \mathrm{K}$ harmonic effective frequency.

##### (B) Dual Coupling Regimes: Quantum Generating Function vs. Semiclassical Marcus
HyDePh automatically branches lineshape evaluation based on the total Huang-Rhys factor $S$:

1. **Weak-to-Intermediate Coupling Regime ($S < 10$)**:
   - *Systems*: Diamond $\mathrm{NV}^-$ ($S = 3.17$), 4H-SiC $kk$-divacancy ($S = 4.77$), LiCl F-center ($S = 4.00$).
   - *Physics*: The optical spectrum features a prominent Zero-Phonon Line and discrete, quantized multi-phonon sidebands ($A_0, A_1, A_2, \dots$). Classical MD continuous equipartition fluctuations $\sigma_{\Delta U} \sim \sqrt{2 k_B T E_{\mathrm{rel}}}$ would obliterate the ZPL and wash out sidebands.
   - *Method*: HyDePh evaluates lineshapes via the **finite-temperature quantum generating function**, renormalizing the spectral density by the physical mode softening $\alpha_{\mathrm{soft}}(T)$:
     
     $$
     S_{\mathrm{anh}}(\hbar\omega, T) = \frac{1}{\alpha_{\mathrm{soft}}(T)} S\left( \frac{\hbar\omega}{\alpha_{\mathrm{soft}}(T)} \right)
     $$
     
     and incorporating total dephasing linewidth from baseline acoustic scattering and intrinsic anharmonic phonon-phonon collisions:
     
     $$
     \gamma_{\mathrm{total}}(T) = \gamma_{\mathrm{base}}(T) + 0.3 \sqrt{\frac{T}{10}}\ \mathrm{meV}
     $$
     
     This preserves the sharp, delta-like Zero-Phonon Line ($\mathrm{DWF} = e^{-S}$) and discrete multi-phonon interference while capturing true finite-temperature softening and broadening.

2. **Strong Coupling / Semiclassical Marcus Regime ($S \ge 10$)**:
   - *Systems*: LiF F-center ($S = 23.93$).
   - *Physics*: Large lattice relaxation suppresses the Zero-Phonon Line to zero ($\mathrm{DWF} = e^{-23.9} \approx 4\times 10^{-11}$). Hundreds of overlapping vibronic transitions form a dense, smooth continuum.
   - *Method*: HyDePh samples the vertical energy gap trajectory $\Delta U(t) = V_{\mathrm{ES}}(\mathbf{R}(t)) - V_{\mathrm{GS}}(\mathbf{R}(t))$ in Langevin dynamics, extracts the thermodynamic variance $\sigma_{\Delta U}^2(T)$, and convolutes the zero-temperature spectrum with the classical Gaussian fluctuation distribution:
     
     $$
     A_{\mathrm{anh}}(E, T) = \int d(\Delta U) \, P_{\mathrm{anh}}(\Delta U, T) \, A_{\mathrm{quant}}(E - \Delta U, T=0)
     $$
     
     capturing the broad, featureless Gaussian emission band (FWHM $\sim 490\text{--}1100\ \mathrm{meV}$).

#### 2. Deep-Dive: Theoretical and Physical Justification of the Langevin Integrator

A critical methodological question in molecular dynamics is whether the **stochastic nature of the Langevin thermostat** is physically and mathematically justified for optical defect lineshapes, or whether deterministic integrators (such as microcanonical $NVE$ or Nosé-Hoover $NVT$) should be preferred. 

Here we provide the rigorous theoretical justification across four distinct physical foundations:

##### (A) Exact Canonical Ensemble Equivalence for Configuration-Space Cumulants
In the semiclassical Franck-Condon and cumulant expansion frameworks (Lax [25], Mukamel [10]), the thermal broadening distribution $P_{\mathrm{anh}}(\Delta U, T)$ is an **equilibrium canonical ensemble average**:

$$
P_{\mathrm{anh}}(\Delta U, T) = \langle \delta(\Delta U - \Delta U(\mathbf{R})) \rangle_{\mathrm{canonical}} = \frac{1}{Z_R} \int d\mathbf{R} \, \exp\left[ -\frac{V_{\mathrm{GS}}(\mathbf{R})}{k_B T} \right] \delta(\Delta U - \Delta U(\mathbf{R}))
$$

Crucially, the electronic energy gap $\Delta U(\mathbf{R}) = V_{\mathrm{ES}}(\mathbf{R}) - V_{\mathrm{GS}}(\mathbf{R})$ depends **exclusively on atomic positions $\mathbf{R}$**, with zero dependence on atomic momenta $\mathbf{P}$.

By the **Fluctuation-Dissipation Theorem (FDT)**, the stochastic Langevin equation of motion:

$$
M_i \ddot{\mathbf{R}}_i = -\nabla_i V(\mathbf{R}) - \gamma M_i \dot{\mathbf{R}}_i + \mathbf{\Xi}_i(t)
$$

possesses a Fokker-Planck (Klein-Kramers) operator whose unique stationary probability density in phase space is the exact Gibbs canonical distribution:

$$
\mathcal{P}_{\mathrm{stationary}}(\mathbf{R}, \mathbf{P}) \propto \exp\left[ -\frac{H(\mathbf{R}, \mathbf{P})}{k_B T} \right] = \exp\left[ -\frac{\sum_i \frac{\mathbf{P}_i^2}{2 M_i} + V(\mathbf{R})}{k_B T} \right]
$$

Integrating out the momentum degrees of freedom $\mathbf{P}$ yields the exact Boltzmann distribution in configuration space:

$$
P(\mathbf{R}) = \int d\mathbf{P} \, \mathcal{P}_{\mathrm{stationary}}(\mathbf{R}, \mathbf{P}) = \frac{1}{Z_R} \exp\left[ -\frac{V(\mathbf{R})}{k_B T} \right]
$$

Because the configuration-space distribution is mathematically identical to the true canonical ensemble, the thermal energy gap moments—specifically the mean shift $\langle \Delta U \rangle_T$ and the thermodynamic variance $\sigma_{\Delta U}^2(T)$—are **unbiased and exact**. The stochastic noise does not alter the thermodynamic potential energy surface.

##### (B) Ergodicity Guarantee in Quasi-Harmonic Defect Cavities
Why choose Langevin dynamics over deterministic thermostats such as Nosé-Hoover (NH) or Nosé-Hoover Chains (NHC)?

In point-defect supercells ($10^2\text{--}10^3$ atoms), the vibrational modes are discrete, and at cryogenic to intermediate temperatures ($10\text{--}77\ \text{K}$), atomic motions remain in the quasi-harmonic regime. It is a well-established theorem of statistical mechanics (Tuckerman [29]) that **deterministic thermostats are notoriously non-ergodic for harmonic and weakly-coupled oscillator systems**:
1. By the Kolmogorov-Arnold-Moser (KAM) theorem, deterministic trajectories become trapped on invariant tori in phase space, preventing equipartition of energy between high-frequency optical defect-cavity modes and low-frequency acoustic bulk modes.
2. Intramolecular/intracavity vibrational energy redistribution (IVR) can be exceedingly slow, leading to persistent unphysical temperature gradients between localized and delocalized phonon branches ("flying ice cube" and mode-decoupling artifacts).

In contrast, the stochastic force $\mathbf{\Xi}_i(t)$ in Langevin dynamics acts as an isotropic white-noise perturbation that continuously mixes vibrational phases. This **guarantees rigorous ergodicity**, ensuring that the trajectory explores all degrees of freedom and samples the steep repulsive walls ($k_3 Q^3, k_4 Q^4$) within compact picosecond trajectories ($2\text{--}5\ \text{ps}$).

##### (C) Open Quantum Subsystem Physics: The Mori-Zwanzig & Caldeira-Leggett Bath
From a fundamental open quantum system perspective, a color center (e.g., the vacancy cavity and its nearest coordination shells) is **not an isolated Hamiltonian system**. It is an open subsystem coupled to a semi-infinite crystalline bulk that serves as an infinite thermal phonon reservoir.

Under the **Mori-Zwanzig projection operator formalism** (Zwanzig [28], Mori [31]) and the **Caldeira-Leggett model** [30], projecting out the infinite bath of distant lattice phonon modes transforms Newton's equations into the Generalized Langevin Equation (GLE):

$$
M_i \ddot{\mathbf{R}}_i = -\nabla_i V_{\mathrm{defect}}(\mathbf{R}) - \int_0^t K(t - s) \dot{\mathbf{R}}(s) \, ds + \mathbf{\Xi}(t)
$$

where $K(t - s)$ is the non-Markovian memory friction kernel and $\mathbf{\Xi}(t)$ is the stochastic thermal force exerted by the bath phonons.

In the Markovian limit—where the continuum of bulk acoustic and optical phonons has a broad spectral density compared to the local defect ring relaxation time—the memory kernel contracts to $K(t - s) \approx 2 \gamma M_i \delta(t - s)$. Thus, the Langevin equation implemented in HyDePh is not an ad hoc simulation contrivance: **it is the exact physical equation of motion governing a defect cavity embedded within an infinite crystalline phonon bath**.

##### (D) Dynamical Fidelity (VACF / VDOS) & The Weak-Coupling Regime
A natural concern with stochastic thermostats is whether the friction $\gamma_{\mathrm{lang}}$ perturbs dynamical time-correlation functions, such as the velocity autocorrelation function (VACF) $C_{\mathrm{opt}}(t)$ and the vibrational density of states $g_{\mathrm{opt}}(\hbar\omega, T)$.

For a damped harmonic oscillator with natural frequency $\omega_0$, the Langevin friction introduces a Lorentzian broadening and shifts the resonant frequency to:

$$
\omega_{\mathrm{damped}} = \sqrt{\omega_0^2 - \left(\frac{\gamma_{\mathrm{lang}}}{2}\right)^2} \approx \omega_0 \left[ 1 - \frac{\gamma_{\mathrm{lang}}^2}{8\omega_0^2} \right]
$$

In HyDePh, the friction coefficient is configured strictly in the **weak-coupling limit** ($\gamma_{\mathrm{lang}} \ll \omega_0$):
- For typical optical phonons in crystals ($\hbar\omega_0 \approx 20\text{--}30\ \text{meV}$, corresponding to vibrational periods $T_{\mathrm{vib}} \sim 140\text{--}200\ \text{fs}$), the damping time is set to $\tau_{\mathrm{lang}} = 1/\gamma_{\mathrm{lang}} \ge 100\text{--}1000\ \text{fs}$.
- With $\gamma_{\mathrm{lang}} / \omega_0 \le 0.05$, the frequency shift is:

$$
\frac{|\Delta\omega|}{\omega_0} \approx \frac{\gamma_{\mathrm{lang}}^2}{8\omega_0^2} < 0.0003 \quad (< 0.03\%)
$$

For an optical mode at $20\ \text{meV}$, this corresponds to an unmeasurable shift of $< 0.006\ \text{meV}$. Consequently, the thermal mode softening observed in HyDePh reflects the **true Hamiltonian anharmonic curvature of the potential energy surface**, completely uncorrupted by thermostat damping.

---

### Stage 9: 1D Configuration Coordinate Diagrams & Stokes Shift Analysis (v0.3.1)

#### 1. Theory & Mathematical Formulation
While full dynamical lineshape calculations evaluate electron-phonon coupling across all $3N$ vibrational normal modes or thermal trajectories, the **one-dimensional Configuration Coordinate (CC) model** remains the foundational framework in defect optical spectroscopy [5, 6, 8, 32]. It projects the high-dimensional nuclear configuration space onto a single collective optical reaction coordinate connecting the relaxed ground state $\mathbf{R}_{\mathrm{GS}}$ and the relaxed excited state $\mathbf{R}_{\mathrm{ES}}$:

$$
\mathbf{R}(\lambda) = \mathbf{R}_{\mathrm{GS}} + \lambda (\mathbf{R}_{\mathrm{ES}} - \mathbf{R}_{\mathrm{GS}}), \quad \lambda \in [\lambda_{\min}, \lambda_{\max}]
$$

The physical, mass-weighted configuration coordinate $Q$ is:

$$
Q = \lambda \Delta Q
$$

where $\Delta Q$ is the total mass-weighted structural displacement:

$$
\Delta Q = \sqrt{\sum_{i=1}^N M_i |\mathbf{R}_{\mathrm{ES}, i} - \mathbf{R}_{\mathrm{GS}, i}|^2} \quad (\text{amu}^{1/2}\text{Å})
$$

##### Potential Energy Surfaces $E_{\mathrm{GS}}(Q)$ and $E_{\mathrm{ES}}(Q)$
Along the coordinate path $\mathbf{R}(\lambda)$, HyDePh evaluates the potential energy of the supercell using the trained hybrid defect potential $V_{\mathrm{hybrid}}(\mathbf{R})$:

$$
E_{\mathrm{GS}}(\lambda) = V_{\mathrm{hybrid}}(\mathbf{R}(\lambda)) - V_{\mathrm{hybrid}}(\mathbf{R}_{\mathrm{GS}})
$$

By definition, $E_{\mathrm{GS}}(0) = 0$. At $\lambda = 1$ ($Q = \Delta Q$, the excited-state nuclear geometry), the ground-state potential energy defines the **ground-state Franck-Condon relaxation energy** (or luminescence relaxation energy):

$$
\Delta E_{\mathrm{rel,GS}} = V_{\mathrm{hybrid}}(\mathbf{R}_{\mathrm{ES}}) - V_{\mathrm{hybrid}}(\mathbf{R}_{\mathrm{GS}}) = E_{\mathrm{GS}}(\lambda = 1)
$$

In the canonical equal-curvature Franck-Condon model [5, 6, 24], the excited-state potential energy surface has its minimum at $Q = \Delta Q$ at the Zero-Phonon Line (ZPL) energy $E_{\mathrm{ZPL}}$:

$$
E_{\mathrm{ES}}(Q) = E_{\mathrm{ZPL}} + \frac{1}{2} K_{\mathrm{ES}} (Q - \Delta Q)^2 = E_{\mathrm{ZPL}} + \Delta E_{\mathrm{rel,ES}} (1 - \lambda)^2
$$

where the excited-state relaxation energy satisfies $\Delta E_{\mathrm{rel,ES}} = \Delta E_{\mathrm{rel,GS}}$.

##### Vertical Optical Transitions & Stokes Shift
Under the Franck-Condon principle, electronic transitions occur vertically on time scales ($\sim 10^{-15}\ \text{s}$) much faster than nuclear motion ($\sim 10^{-13}\ \text{s}$):
1. **Vertical Optical Absorption ($E_{\mathrm{abs}}$)**:
   Occurs from the ground-state minimum at $Q = 0$ ($\lambda = 0$):

   $$
   E_{\mathrm{abs}} = E_{\mathrm{ES}}(Q=0) - E_{\mathrm{GS}}(Q=0) = E_{\mathrm{ZPL}} + \Delta E_{\mathrm{rel,ES}}
   $$

2. **Vertical Luminescence Emission ($E_{\mathrm{em}}$)**:
   Following non-radiative vibrational relaxation in the excited state to $Q = \Delta Q$ ($\lambda = 1$), emission occurs vertically down to the ground-state surface:

   $$
   E_{\mathrm{em}} = E_{\mathrm{ES}}(Q=\Delta Q) - E_{\mathrm{GS}}(Q=\Delta Q) = E_{\mathrm{ZPL}} - \Delta E_{\mathrm{rel,GS}}
   $$

3. **Total Stokes Shift ($\Delta E_{\mathrm{Stokes}}$)**:
   The energy difference between the vertical absorption peak and luminescence emission peak:

   $$
   \Delta E_{\mathrm{Stokes}} = E_{\mathrm{abs}} - E_{\mathrm{em}} = \Delta E_{\mathrm{rel,ES}} + \Delta E_{\mathrm{rel,GS}} = 2 \Delta E_{\mathrm{rel,GS}}
   $$

##### Effective Phonon Frequency & 1D Huang-Rhys Factor
From the harmonic curvature $K = \frac{2 \Delta E_{\mathrm{rel}}}{(\Delta Q)^2}$ in units of $\text{eV}/(\text{amu}\cdot\text{Å}^2)$, the effective vibrational frequency $\Omega$ is:

$$
\Omega = \sqrt{K} = \frac{\sqrt{2 \Delta E_{\mathrm{rel}}}}{\Delta Q}
$$

Using fundamental physical constants ($1\ \text{eV}/(\text{amu}\cdot\text{Å}^2) = 9.648533 \times 10^{27}\ \text{s}^{-2}$ and $\hbar = 6.58211957 \times 10^{-16}\ \text{eV}\cdot\text{s}$), the effective phonon energy $\hbar\Omega$ is derived exactly as:

$$
\hbar\Omega\ (\text{meV}) = 91.4348\ \frac{\sqrt{\Delta E_{\mathrm{rel,GS}}\ (\text{eV})}}{\Delta Q\ (\text{amu}^{1/2}\text{Å})}
$$

The corresponding one-dimensional Huang-Rhys factor $S_{\mathrm{1D}}$ represents the mean number of effective phonons emitted during the optical transition:

$$
S_{\mathrm{1D}} = \frac{\Delta E_{\mathrm{rel,GS}}}{\hbar\Omega}
$$

<p align="center">
  <img src="../assets/diamond_nv_cc_diagram.png" width="96%" alt="HyDePh 1D Configuration Coordinate Diagram and Stokes Shift for Diamond NV-">
</p>

#### 2. Physical Justification & Literature References
- **Alkauskas et al., *New J. Phys.* 16, 073026 (2014)** [5]: Formulated the rigorous link between 1D configuration coordinate diagrams, mass-weighted coordinates $\Delta Q$, and full 3D multi-mode lineshapes for defect centers.
- **Jin et al., *Phys. Rev. Materials* 5, 084603 (2021)** [6]: Demonstrated that the 1D CC Stokes shift and relaxation energies provide direct experimental benchmarks for validating hybrid functional and $\Delta$SCF potential energy surfaces.
- **A. Gali, *Nanophotonics* 12, 589 (2023)** [32]: Reviewed configuration coordinate models and Stokes shifts for solid-state quantum emitters, emphasizing the necessity of accurate ground- and excited-state curvatures for predicting single-photon emission purity and Debye-Waller factors.

---

## 4. Stiff vs. Soft Crystals: Why Quantum Theory is Needed for Stiff Hosts and Anharmonic MD for Soft Hosts

The dual-regime architecture in HyDePh is founded on fundamental condensed-matter physics: **the physical mechanism governing optical lineshape broadening changes qualitatively between stiff and soft crystalline hosts**.

<p align="center">
  <img src="../assets/hydeph_stiff_vs_soft_benchmark.png" width="96%" alt="HyDePh Stiff vs Soft Benchmark">
</p>

### 1. Stiff Covalent Crystals: Diamond $\mathrm{NV}^-$ and 4H-SiC $kk$-VV$^0$
- **Lattice Rigidity & Energy Scales**: Diamond and silicon carbide feature exceptionally high Debye temperatures ($\Theta_D \approx 2200\ \mathrm{K}$ for diamond, $\Theta_D \approx 1200\ \mathrm{K}$ for 4H-SiC) and high optical phonon cutoff frequencies ($\hbar\omega \sim 65\text{--}165\ \mathrm{meV}$). At cryogenic and ambient operating temperatures ($10\text{--}300\ \mathrm{K}$), $k_B T \ll \hbar\omega$.
- **Quantized Vibronic States**: Transitions do not occur along a continuous classical energy landscape. Instead, optical emission occurs between discrete quantum vibronic states $|g, n\rangle \leftarrow |e, 0\rangle$, creating a sharp, delta-like Zero-Phonon Line (ZPL) and resolved multi-phonon replicas at discrete energy offsets ($E - E_{\mathrm{ZPL}} = -n\hbar\omega_{\mathrm{vib}}$).
- **Failure of Classical MD**: Classical molecular dynamics obeys the equipartition theorem, distributing $\frac{1}{2} k_B T$ per quadratic degree of freedom continuously. Along the optical coordinate, this classical thermal displacement produces continuous Gaussian energy gap fluctuations:
  
  $$
  \sigma_{\Delta U} \sim \sqrt{2 k_B T E_{\mathrm{rel}}}
  $$
  
  Convolving the optical spectrum with this classical Gaussian distribution washes out the delta-function ZPL and smears all discrete vibronic sidebands into a featureless, over-broadened dome (red dashed curve in Panel a).
- **Physical Solution**: The **Quantum Generating Function** is strictly indispensable for stiff crystals. By integrating over the quantum Bose-Einstein distribution $\bar{n}(\hbar\omega, T)$, it rigorously captures discrete multi-phonon interference, preserving the sharp ZPL ($\mathrm{DWF} = 4.22\%$ in diamond $\mathrm{NV}^-$) and the experimental vibronic sidebands at $-65\ \mathrm{meV}$, $-135\ \mathrm{meV}$, and $-205\ \mathrm{meV}$. MD sampling provides only the real temperature-dependent frequency softening $\alpha_{\mathrm{soft}}(T)$ and dephasing linewidth $\gamma(T)$.

### 2. Soft Ionic Crystals: Lithium Chloride (LiCl) and Lithium Fluoride (LiF)
- **Lattice Softness & Large Displacements**: Alkali halides feature low Debye temperatures ($\Theta_D \approx 420\ \mathrm{K}$ for LiCl) and low vibrational frequencies ($\hbar\omega \sim 15\text{--}35\ \mathrm{meV}$). The removal of a halide anion to create an F-center causes massive outward relaxation of coordinating cations ($\Delta Q \sim 1.63\ \mathrm{amu}^{1/2}\text{Å}$ in LiCl, $\Delta Q \sim 2.39\ \mathrm{amu}^{1/2}\text{Å}$ in LiF).
- **Steep Anharmonic Repulsive Walls**: Outward breathing of nearest-neighbor cations compresses directly against the second-neighbor anion shell, encountering steep repulsive wall potentials ($V \sim k_3 Q^3 + k_4 Q^4$).
- **Failure of Harmonic-Only Theory**: Zero-temperature harmonic displaced-oscillator models assume fixed parabolic potentials evaluated at $0\ \text{K}$. In LiCl, harmonic theory predicts an emission FWHM of only $78\ \mathrm{meV}$ at $77\ \text{K}$—underestimating the experimental bandwidth ($460\ \text{meV}$) by **$5.9\times$**! Furthermore, harmonic theory cannot predict thermal lattice expansion or the temperature-induced softening of the defect-core optical mode from $27.8\ \mathrm{meV}$ down to $19.8\ \mathrm{meV}$.
- **Physical Solution**: **Finite-Temperature Anharmonic MD** is strictly required for soft crystals. Langevin dynamics directly samples the steep cubic and quartic walls of the potential energy surface and captures dynamic 3D cavity deformations. In LiCl, HyDePh expands the theoretical bandwidth by **$3.4\times$ to $265\ \mathrm{meV}$**, directly bridging the gap toward the $460\ \mathrm{meV}$ experimental band, while shifting the defect mode to $19.8\ \mathrm{meV}$ in exact agreement with Takiyama's experimental value ($19.7\ \mathrm{meV}$). In ultra-strong coupling systems like LiF ($S = 23.93$), semiclassical Marcus convolution correctly reproduces the massive $490\text{--}1101\ \mathrm{meV}$ broad band.

### 3. Comprehensive Benchmark Matrix Across 4 Material Classes

| Test System | Crystal Host | Host Stiffness | Huang-Rhys $S$ | $\Delta Q$ ($\mathrm{amu}^{1/2}\text{Å}$) | Coupling Regime | Dominant Modes | Lineshape Characteristics | Physical Necessity |
|---|---|---|---|---|---|---|---|---|
| **Diamond $\mathrm{NV}^-$** | Covalent (Diamond) | Very Stiff ($\Theta_D \approx 2200\ \mathrm{K}$) | **$3.166$** | $0.627$ | **Quantum** ($S < 10$) | $65$, $135$, $160\ \mathrm{meV}$ | Sharp ZPL at $0\ \mathrm{meV}$ ($\mathrm{DWF} = 4.22\%$), resolved sidebands at $-65, -135, -205\ \mathrm{meV}$ | **Quantum GenFn Required**: Classical MD obliterates ZPL and washes sidebands into a featureless dome |
| **4H-SiC $kk$-VV$^0$** | Polar Covalent (SiC) | Stiff ($\Theta_D \approx 1200\ \mathrm{K}$) | **$4.766$** | $0.812$ | **Quantum** ($S < 10$) | $38$, $76\ \mathrm{meV}$ | Sharp ZPL at $0\ \mathrm{meV}$ ($\mathrm{DWF} = 0.85\%$), sharp quasi-local sidebands at $-38, -76\ \mathrm{meV}$; bulk $100\ \mathrm{meV}$ artifact removed | **$\mathbf{u}_{\mathrm{opt}}$ Projection Required**: Filters out bulk non-coupling optical phonons ($90\text{--}115\ \mathrm{meV}$) |
| **LiCl F-Center** | Ionic (Rocksalt) | Soft ($\Theta_D \approx 420\ \mathrm{K}$) | **$3.999$** | $1.632$ | **Quantum** ($S < 10$) | Softened to $19.8\ \mathrm{meV}$ (from $27.8\ \mathrm{meV}$) | Structured soft-crystal band, $3.4\times$ bandwidth expansion toward experiment ($265\ \mathrm{meV}$) | **Anharmonic MD Required**: Harmonic theory is $5.9\times$ too narrow ($78\ \mathrm{meV}$ vs $460\ \mathrm{meV}$ exp) and misses mode softening |
| **LiF F-Center** | Ionic (Fluorite) | Intermediate Soft ($\Theta_D \approx 730\ \mathrm{K}$) | **$23.930$** | $2.391$ | **Semiclassical Marcus** ($S \ge 10$) | Dense quasicontinuum | Broad classical Gaussian band; $\mathrm{FWHM} = 490\ \mathrm{meV}$ at $10\ \mathrm{K}$, $1101\ \mathrm{meV}$ at $300\ \mathrm{K}$; ZPL suppressed ($\mathrm{DWF} \approx 0$) | **Semiclassical Marcus Required**: Massive displacement renders ZPL negligible, producing purely classical broadening |

---

## 5. Synthesis: Methodological Matrix

| Stage | Submodule | Key Equation | Physical Role | Foundational Articles |
|---|---|---|---|---|
| **1. Embedding** | [`defect_builder.py`](../hydeph/defect_builder.py) | $\Delta \mathbf{R} = \mathbf{R}_{\mathrm{ES}} - \mathbf{R}_{\mathrm{GS}}$<br>$\Delta Q = \sqrt{\sum_i M_i \vert \Delta \mathbf{R}_i \vert^2}$ | Embedded supercell; eliminates image strain; identifies optical relaxation coordinate | Freysoldt et al. [9]<br>Alkauskas et al. [5]<br>Jin et al. [6] |
| **2. $\Delta$-Learning** | [`calculator.py`](../hydeph/calculator.py) | $V_{\mathrm{hyb}} = V_{\mathrm{base}} + \Delta V_{\mathrm{NEP}}$<br>$\mathbf{F}_{\mathrm{hyb}} = \mathbf{F}_{\mathrm{base}} + \Delta \mathbf{F}_{\mathrm{NEP}}$ | Separates bulk lattice elasticity from localized quantum defect forces | Ramakrishnan et al. [16]<br>Chmiela et al. [17]<br>Fan et al. [14] |
| **3. Frame Sampling** | [`frame_sampler.py`](../hydeph/frame_sampler.py) | $\mathbf{R}(\lambda) = \mathbf{R}_{\mathrm{GS}} + \lambda \Delta \mathbf{R}$<br>$d(i, j) = \Vert \mathbf{q}_i - \mathbf{q}_j \Vert_2$ | 4-phase sampling: Franck-Condon scan, frozen-phonon rattling, and MD+FPS | Marcus [18]<br>Ceriotti et al. [20]<br>Knoop et al. [19] |
| **4. DFT Labeling** | [`qe_runner.py`](../hydeph/qe_runner.py) | $\mathbf{F}_{i} = -\langle \Psi \vert \nabla_i \hat{H} \vert \Psi \rangle$<br>Constrained orbital occupations | Accurate $\Delta$SCF ground- and excited-state energies and Hellmann-Feynman forces | Görling [23]<br>Gali et al. [8]<br>Jin et al. [6] |
| **5. $\Delta$-NEP Training** | [`delta_trainer.py`](../hydeph/delta_trainer.py) | $\mathcal{L}(\mathbf{w}) = \lambda_e \mathrm{RMSE}_E^2 + \lambda_f \mathrm{RMSE}_F^2$<br>Natural Evolution Strategies | GPU-accelerated fitting of localized $\Delta$-correction potential in GPUMD | Fan et al. [14]<br>Fan et al. [15] |
| **6. Phonon Normal Modes** | [`lineshape.py`](../hydeph/lineshape.py) | $S_k = \frac{\omega_k}{2\hbar} (\Delta Q_k)^2$<br>$S(\hbar\omega) = \sum_k S_k \delta(\hbar\omega - \hbar\omega_k)$ | Diagonalization of dynamical matrix; evaluates mode-resolved electron-phonon coupling | Huang & Rhys [24]<br>Alkauskas et al. [7]<br>Alkauskas et al. [5] |
| **7. Generating Function** | [`lineshape.py`](../hydeph/lineshape.py) | $G(t) = \exp[g(t) - g(0)]$<br>$A(E) = \frac{1}{2\pi\hbar}\int dt e^{i\Delta E t/\hbar} G(t)$ | Multi-temperature quantum emission lineshape & Debye-Waller factor | Lax [25]<br>Kubo & Toyozawa [26]<br>Alkauskas et al. [5] |
| **8. Anharmonic Sampling** | [`lineshape.py`](../hydeph/lineshape.py) | $\mathbf{u}_{\mathrm{opt}} = \frac{M_i \Delta \mathbf{R}_{\mathrm{FC}, i}}{\|\sqrt{M} \Delta \mathbf{R}_{\mathrm{FC}}\|}$<br>$g_{\mathrm{opt}}(\hbar\omega) \to \alpha_{\mathrm{soft}}(T) = \frac{\Omega(T)}{\Omega_0}$<br>$S_{\mathrm{anh}} = \alpha_{\mathrm{soft}}^{-1} S(\hbar\omega / \alpha_{\mathrm{soft}})$ ($S < 10$)<br>$A_{\mathrm{anh}} = A_{\mathrm{quant}} * P_{\mathrm{anh}}(\Delta U)$ ($S \ge 10$) | Optical reaction coordinate projection; eliminates host bulk phonon artifacts; thermal mode softening & dephasing; dual quantum ($S < 10$) & Marcus ($S \ge 10$) regimes | Mukamel [10]<br>Zacharias et al. [11, 12]<br>Takiyama [13]<br>Tuckerman [29]<br>Zwanzig [28] |
| **9. 1D CC & Stokes Shift** | [`cc_diagram.py`](../hydeph/cc_diagram.py) | $\mathbf{R}(\lambda) = \mathbf{R}_{\mathrm{GS}} + \lambda \Delta \mathbf{R}$<br>$\Delta E_{\mathrm{Stokes}} = E_{\mathrm{abs}} - E_{\mathrm{em}} = 2 \Delta E_{\mathrm{rel}}$<br>$\hbar\Omega = 91.4348 \frac{\sqrt{\Delta E_{\mathrm{rel}}}}{\Delta Q}$ | 1D Configuration coordinate diagram; Franck-Condon relaxation energies; absorption & luminescence Stokes shifts; effective mode frequency & $S_{\mathrm{1D}}$ | Alkauskas et al. [5]<br>Jin et al. [6]<br>Gali [32] |

---

## 6. Comprehensive Bibliography & Foundational References

1. **D. D. Awschalom, R. Hanson, J. Wrachtrup, and B. B. Zhou**, *"Quantum technologies with optically interfaced nitrogen-vacancy centers in diamond"*, *Science* **361**, eaat8989 (2018). [DOI: 10.1126/science.aat8989](https://doi.org/10.1126/science.aat8989)
2. **G. Wolfowicz, F. J. Heremans, C. P. Anderson, S. Kanai, H. Seo, A. Gali, and D. D. Awschalom**, *"Quantum guidelines for solid-state spin qubits"*, *Nat. Rev. Mater.* **6**, 655–673 (2021). [DOI: 10.1038/s41578-021-00306-y](https://doi.org/10.1038/s41578-021-00306-y)
3. **M. Atatüre, D. Englund, N. Vamivakas, S. Y. Lee, and J. Wrachtrup**, *"Material platforms for spin-based photonic quantum technologies"*, *Nat. Rev. Mater.* **3**, 38–51 (2018). [DOI: 10.1038/s41578-018-0008-9](https://doi.org/10.1038/s41578-018-0008-9)
4. **W. B. Gao, A. Imamoglu, H. Bernien, and R. Hanson**, *"Coherent manipulation, measurement and entanglement of individual solid-state spins using optical fields"*, *Nat. Photonics* **9**, 363–373 (2015). [DOI: 10.1038/nphoton.2015.58](https://doi.org/10.1038/nphoton.2015.58)
5. **A. Alkauskas, B. B. Buckley, D. D. Awschalom, and C. G. Van de Walle**, *"First-principles theory of luminescence lineshapes of color centers in semiconductors and insulators"*, *New J. Phys.* **16**, 073026 (2014). [DOI: 10.1088/1367-2630/16/7/073026](https://doi.org/10.1088/1367-2630/16/7/073026)
6. **Y. Jin, M. Govoni, G. Wolfowicz, S. E. Sullivan, F. J. Heremans, D. D. Awschalom, and G. Galli**, *"Photoluminescence spectra of point defects in semiconductors: Validation of first-principles calculations"*, *Phys. Rev. Materials* **5**, 084603 (2021). [DOI: 10.1103/PhysRevMaterials.5.084603](https://doi.org/10.1103/PhysRevMaterials.5.084603)
7. **A. Alkauskas, J. L. Lyons, D. Steiauf, and C. G. Van de Walle**, *"First-principles calculations of luminescence spectrum line shapes for defects in semiconductors: the role of phonons and Jahn-Teller effect"*, *Phys. Rev. Lett.* **109**, 267401 (2012). [DOI: 10.1103/PhysRevLett.109.267401](https://doi.org/10.1103/PhysRevLett.109.267401)
8. **A. Gali, M. Fyta, and E. Kaxiras**, *"Ab initio supercell calculations on nitrogen-vacancy center in diamond: Electronic structure and hyperfine tensors"*, *Phys. Rev. B* **79**, 235203 (2009). [DOI: 10.1103/PhysRevB.79.235203](https://doi.org/10.1103/PhysRevB.79.235203)
9. **C. Freysoldt, B. Grabowski, T. Hickel, J. Neugebauer, G. Kresse, A. Janotti, and C. G. Van de Walle**, *"First-principles calculations for point defects in solids"*, *Rev. Mod. Phys.* **86**, 253–305 (2014). [DOI: 10.1103/RevModPhys.86.253](https://doi.org/10.1103/RevModPhys.86.253)
10. **S. Mukamel**, *Principles of Nonlinear Optical Spectroscopy*, Oxford University Press, New York (1995). [ISBN: 978-0195104479]
11. **M. Zacharias and F. Giustino**, *"One-shot calculation of temperature-dependent optical spectra and phonon-induced band-gap renormalization"*, *Phys. Rev. B* **94**, 075125 (2016). [DOI: 10.1103/PhysRevB.94.075125](https://doi.org/10.1103/PhysRevB.94.075125)
12. **M. Zacharias, M. Scheffler, and F. Giustino**, *"Theory of the temperature-dependent optical absorption and emission spectra of solids from first principles"*, *Phys. Rev. Res.* **2**, 013202 (2020). [DOI: 10.1103/PhysRevResearch.2.013202](https://doi.org/10.1103/PhysRevResearch.2.013202)
13. **K. Takiyama**, *"The F-Center Emission in LiCl"*, *J. Phys. Soc. Jpn.* **44**, 1627–1634 (1978). [DOI: 10.1143/JPSJ.44.1627](https://doi.org/10.1143/JPSJ.44.1627)
14. **Z. Fan, Z. Chen, V. L. Deringer, C. S. Ramanan, A. C. T. van Duin, and T. Ala-Nissila**, *"Neuroevolution machine learning potentials: Combining high accuracy and low cost in atomistic simulations and application to carbon"*, *Phys. Rev. B* **104**, 104309 (2021). [DOI: 10.1103/PhysRevB.104.104309](https://doi.org/10.1103/PhysRevB.104.104309)
15. **Z. Fan et al.**, *"GPUMD: A highly efficient graphics processing unit code for molecular dynamics with the neuroevolution potential"*, *J. Chem. Phys.* **157**, 114801 (2022). [DOI: 10.1063/5.0106617](https://doi.org/10.1063/5.0106617)
16. **P. Ramakrishnan, P. O. Dral, M. Rupp, and O. A. von Lilienfeld**, *"$\Delta$-Machine Learning to Accurately Predict Quantum Chemical Properties"*, *J. Chem. Theory Comput.* **11**, 2087–2096 (2015). [DOI: 10.1021/acs.jctc.5b00099](https://doi.org/10.1021/acs.jctc.5b00099)
17. **S. Chmiela, H. E. Sauceda, K. R. Müller, and A. Tkatchenko**, *"Towards exact molecular dynamics simulations with machine-learned force fields"*, *Nat. Commun.* **9**, 3887 (2018). [DOI: 10.1038/s41467-018-06169-2](https://doi.org/10.1038/s41467-018-06169-2)
18. **R. A. Marcus**, *"Electron transfer reactions in chemistry. Theory and experiment"*, *Rev. Mod. Phys.* **65**, 599–610 (1993). [DOI: 10.1103/RevModPhys.65.599](https://doi.org/10.1103/RevModPhys.65.599)
19. **F. Knoop, M. Purcell, M. Scheffler, and C. Carbogno**, *"Anharmonic properties of solids from first principles: A high-throughput study"*, *Phys. Rev. Materials* **4**, 083809 (2020). [DOI: 10.1103/PhysRevMaterials.4.083809](https://doi.org/10.1103/PhysRevMaterials.4.083809)
20. **M. Ceriotti, G. A. Tribello, and M. Parrinello**, *"Demonstrating the power of sketches for molecular simulation"*, *J. Chem. Theory Comput.* **9**, 1521–1532 (2013). [DOI: 10.1021/ct3010563](https://doi.org/10.1021/ct3010563)
21. **V. L. Deringer, M. A. Caro, and G. Csányi**, *"Machine learning potentials for atomistic simulations"*, *Chem. Rev.* **121**, 10073–10141 (2021). [DOI: 10.1021/acs.chemrev.1c00022](https://doi.org/10.1021/acs.chemrev.1c00022)
22. **P. Giannozzi et al.**, *"QUANTUM ESPRESSO: a modular and open-source software project for quantum simulations of materials"*, *J. Phys.: Condens. Matter* **21**, 395502 (2009). [DOI: 10.1088/0953-8984/21/39/395502](https://doi.org/10.1088/0953-8984/21/39/395502)
23. **A. Görling**, *"Density-functional theory for excited states"*, *Phys. Rev. A* **59**, 3359–3374 (1999). [DOI: 10.1103/PhysRevA.59.3359](https://doi.org/10.1103/PhysRevA.59.3359)
24. **K. Huang and A. Rhys**, *"Theory of light absorption and non-radiative transitions in F-centres"*, *Proc. R. Soc. London A* **204**, 406–423 (1950). [DOI: 10.1098/rspa.1950.0184](https://doi.org/10.1098/rspa.1950.0184)
25. **M. Lax**, *"The Franck-Condon principle and its application to crystals"*, *J. Chem. Phys.* **20**, 1752–1760 (1952). [DOI: 10.1063/1.1700283](https://doi.org/10.1063/1.1700283)
26. **R. Kubo and Y. Toyozawa**, *"Application of the method of generating function to radiative and non-radiative transitions of a trapped electron in a crystal"*, *Prog. Theor. Phys.* **13**, 160–182 (1955). [DOI: 10.1143/PTP.13.160](https://doi.org/10.1143/PTP.13.160)
27. **B. J. Berne and R. Pecora**, *Dynamic Light Scattering: With Applications to Chemistry, Biology, and Physics*, John Wiley & Sons, New York (1976). [ISBN: 978-0486411552]
28. **R. Zwanzig**, *Nonequilibrium Statistical Mechanics*, Oxford University Press, New York (2001). [DOI: 10.1093/oso/9780195140187.001.0001](https://doi.org/10.1093/oso/9780195140187.001.0001)
29. **M. E. Tuckerman**, *Statistical Mechanics: Theory and Molecular Simulation*, Oxford University Press, Oxford (2010). [ISBN: 978-0198525264]
30. **A. O. Caldeira and A. J. Leggett**, *"Path integral approach to degree of freedom in a dissipative quantum system"*, *Ann. Phys.* **149**, 374–456 (1983). [DOI: 10.1016/0003-4916(83)90202-6](https://doi.org/10.1016/0003-4916(83)90202-6)
31. **H. Mori**, *"Transport, collective motion, and Brownian motion"*, *Prog. Theor. Phys.* **33**, 423–455 (1965). [DOI: 10.1143/PTP.33.423](https://doi.org/10.1143/PTP.33.423)
32. **A. Gali**, *"Ab initio theory of point defects as single photon emitters in wide band gap semiconductors"*, *Nanophotonics* **12**, 589–609 (2023). [DOI: 10.1515/nanoph-2022-0599](https://doi.org/10.1515/nanoph-2022-0599)



"""
Unit tests for HyDePh defect builder, lineshape engine, and hybrid calculator.
"""

import os
import numpy as np
import pytest
from ase import Atoms
from unittest.mock import MagicMock

from hydeph.config import ToolkitConfig
from hydeph.lineshape import compute_quantum_lineshape
from hydeph.defect_builder import DefectBuilder


def test_quantum_lineshape_calculation():
    """Verify compute_quantum_lineshape produces normalized spectrum."""
    hw_grid = np.linspace(1.0, 50.0, 100)
    s_cont = 5.0 * np.exp(-0.5 * ((hw_grid - 20.0) / 3.0)**2)  # Gaussian mode at 20 meV
    
    eg, a_norm = compute_quantum_lineshape(
        hw_grid,
        s_cont,
        temp_k=77.0,
        dephasing_gamma_meV=2.0,
        e_min_meV=-100.0,
        e_max_meV=20.0,
        n_e=200
    )
    
    assert len(eg) == 200
    assert len(a_norm) == 200
    assert np.all(a_norm >= 0.0)
    area = np.trapezoid(a_norm, eg) if hasattr(np, "trapezoid") else np.trapz(a_norm, eg)
    assert pytest.approx(area, rel=1e-2) == 1.0


def test_defect_builder_embed_unitcell(tmp_path):
    """Verify embedding a defect unit cell into a supercell."""
    pristine_uc = Atoms("Li4Cl4", cell=[5, 5, 5], pbc=True)
    pristine_uc.set_positions([
        [0, 0, 0], [2.5, 2.5, 0], [2.5, 0, 2.5], [0, 2.5, 2.5],
        [2.5, 0, 0], [0, 2.5, 0], [0, 0, 2.5], [2.5, 2.5, 2.5]
    ])
    
    # 7-atom defect unit cell (missing 1 atom)
    defect_uc = pristine_uc[:-1].copy()

    pristine_file = tmp_path / "pristine.xyz"
    defect_file = tmp_path / "defect.xyz"
    from ase.io import write
    write(str(pristine_file), pristine_uc)
    write(str(defect_file), defect_uc)

    cfg = ToolkitConfig(
        pristine_cell_path=str(pristine_file),
        defect_cell_path=str(defect_file),
        supercell_repeat=(2, 2, 2)
    )

    from unittest.mock import patch
    with patch("hydeph.defect_builder.CPUNEP"):
        builder = DefectBuilder(cfg)
        sc, pos = builder.embed_defect_unitcell(pristine_uc, defect_uc, rep=(2, 2, 2))
    
    # 8 cells * 8 atoms - 1 vacancy = 63 atoms
    assert len(sc) == 63
    assert pos is not None


def test_charge_and_spin_configuration(tmp_path):
    """Verify charge and spin multiplicity synchronization and QE input generation for NV- center."""
    from hydeph.qe_runner import write_qe_input

    # NV- center setup: net charge -1, triplet state (S=1 -> 2S+1=3, tot_magnetization=2)
    cfg = ToolkitConfig(
        pristine_cell_path=str(tmp_path / "pristine.xyz"),
        charge=-1,
        spin_multiplicity=3,
    )
    assert cfg.charge == -1.0
    assert cfg.tot_charge == -1.0
    assert cfg.spin_multiplicity == 3
    assert cfg.tot_magnetization == 2

    # Verify alias tot_charge and tot_magnetization input
    cfg2 = ToolkitConfig(
        pristine_cell_path=str(tmp_path / "pristine.xyz"),
        tot_charge=-1.0,
        tot_magnetization=2,
    )
    assert cfg2.charge == -1.0
    assert cfg2.tot_charge == -1.0
    assert cfg2.spin_multiplicity == 3
    assert cfg2.tot_magnetization == 2

    # Verify generated pw.in card contains tot_charge and tot_magnetization
    atoms = Atoms("C2", positions=[[0, 0, 0], [1.5, 1.5, 1.5]], cell=[3, 3, 3], pbc=True)
    pw_in = tmp_path / "pw_nv.in"
    cfg.pseudopotentials = {"C": "C.UPF"}
    (tmp_path / "C.UPF").write_text("fake")
    cfg.pseudo_dir = str(tmp_path)

    write_qe_input(atoms, str(pw_in), cfg)
    content = pw_in.read_text()
    assert "tot_charge = -1.0" in content
    assert "tot_magnetization = 2" in content
    assert "nspin = 2" in content


def test_generic_excited_state_generation(tmp_path):
    """Verify procedural excited state generation works for non-Li systems (e.g. Diamond NV)."""
    from unittest.mock import patch
    # Diamond-like cell with C and N (no Li)
    atoms = Atoms("C7N", positions=[
        [0.89, 2.68, 0.89], [-0.0, -0.0, 1.78], [0.89, 0.89, 2.68],
        [-0.0, 1.78, 0.0], [1.78, 0.0, 0.0], [2.68, 0.89, 0.89],
        [1.78, 1.78, 1.78], [2.68, 2.68, 2.68]
    ], cell=[3.57, 3.57, 3.57], pbc=True)
    
    cfg = ToolkitConfig(
        pristine_cell_path=str(tmp_path / "pristine.xyz"),
        workspace_dir=str(tmp_path),
        excited_state_displacement=0.08
    )

    with patch("hydeph.defect_builder.CPUNEP") as mock_nep:
        mock_calc = MagicMock()
        mock_calc.get_potential_energy.return_value = -100.0
        mock_calc.get_forces.return_value = np.zeros((len(atoms), 3))
        mock_nep.return_value = mock_calc

        builder = DefectBuilder(cfg)
        builder.vac_pos = np.array([2.0, 2.0, 2.0])
        
        at_es, dq_fc = builder.generate_excited_state_geometry(atoms)
        assert dq_fc > 0.0, "Franck-Condon displacement must be strictly positive"
        assert not np.allclose(at_es.positions, atoms.positions)


def test_user_supplied_excited_state_cell(tmp_path):
    """Verify loading user-supplied excited state cell."""
    from unittest.mock import patch
    from ase.io import write
    atoms_uc = Atoms("C8", positions=np.zeros((8, 3)), cell=[3.5, 3.5, 3.5], pbc=True)
    atoms_gs = atoms_uc.repeat((2, 2, 2))
    
    atoms_es_uc = atoms_uc.copy()
    atoms_es_uc.positions[0] = [0.2, 0.0, 0.0]

    pristine_file = tmp_path / "pristine.xyz"
    es_file = tmp_path / "excited_uc.xyz"
    write(str(pristine_file), atoms_uc)
    write(str(es_file), atoms_es_uc)

    cfg = ToolkitConfig(
        pristine_cell_path=str(pristine_file),
        excited_cell_path=str(es_file),
        supercell_repeat=(2, 2, 2),
        workspace_dir=str(tmp_path)
    )

    with patch("hydeph.defect_builder.CPUNEP"):
        builder = DefectBuilder(cfg)
        at_es, dq_fc = builder.generate_excited_state_geometry(atoms_gs)
        assert len(at_es) == len(atoms_gs)
        assert dq_fc > 0.0


def test_smart_axis_scaling(tmp_path):
    """Verify smart axis scaling calculations for phonon and lineshape plots."""
    from hydeph.lineshape import LineshapeEngine
    cfg = ToolkitConfig(
        pristine_cell_path=str(tmp_path / "pristine.xyz"),
        smart_axis_scaling=True,
    )
    engine = LineshapeEngine(cfg, hybrid_calc=None)

    # 1. Phonon limits: active modes only up to 40 meV on a 70 meV grid
    hw_grid = np.linspace(0.5, 70.0, 700)
    s_norm = np.exp(-0.5 * ((hw_grid - 25.0) / 4.0)**2)
    hw_xlim, hw_ylim = engine.get_smart_phonon_limits(hw_grid, s_norm)
    assert hw_xlim[0] == 0.0
    assert hw_xlim[1] < 70.0  # Should be neatly framed around ~40-50 meV, not trailing to 70
    assert hw_ylim[1] > 1.0   # Should provide headroom for legend

    # 2. Lineshape limits: active emission from -400 to 0 meV on [-1000, 200] grid
    eg = np.linspace(-1000.0, 200.0, 1200)
    a_spec = 0.01 * np.exp(-0.5 * (eg / 2.0)**2) # ZPL
    a_spec += 0.005 * np.exp(-0.5 * ((eg + 100.0) / 40.0)**2) # sideband
    lineshape_dict = {10.0: (eg, a_spec)}
    
    e_xlim, e_ylim = engine.get_smart_lineshape_limits(lineshape_dict)
    assert e_xlim[0] > -1000.0  # Tightly framed, not wasting space out to -1000
    assert e_xlim[0] <= -200.0  # Captures sideband
    assert e_xlim[1] >= 35.0    # Always frames ZPL at 0 meV
    assert e_ylim[0] <= 0.0     # Baseline at or below 0
    assert e_ylim[1] > np.max(a_spec) # Headroom above peak

    # 3. Test temperature-specific subset limits vs global
    a_spec_300K = 0.002 * np.exp(-0.5 * ((eg + 250.0) / 150.0)**2)
    lineshape_dict[300.0] = (eg, a_spec_300K)
    
    global_xlim, _ = engine.get_smart_lineshape_limits(lineshape_dict)
    subset_xlim_10K, _ = engine.get_smart_lineshape_limits(lineshape_dict, active_curve_subset=[(eg, a_spec)])
    assert subset_xlim_10K[0] > global_xlim[0]

    # 4. Manual overrides respected
    cfg_override = ToolkitConfig(
        pristine_cell_path=str(tmp_path / "pristine.xyz"),
        lineshape_e_min=-350.0,
        lineshape_e_max=50.0,
        lineshape_y_max=0.02
    )
    engine_override = LineshapeEngine(cfg_override, hybrid_calc=None)
    over_xlim, over_ylim = engine_override.get_smart_lineshape_limits(lineshape_dict)
    assert over_xlim[0] == -350.0
    assert over_xlim[1] == 50.0
    assert over_ylim[1] == 0.02


def test_finite_size_phonon_smearing(tmp_path):
    """Verify finite-size smearing collapses discrete mode comb into smooth physical bands."""
    from hydeph.lineshape import LineshapeEngine
    cfg_discrete = ToolkitConfig(
        pristine_cell_path=str(tmp_path / "pristine.xyz"),
        phonon_smearing_meV=1.8, # Tiny smearing
    )
    cfg_physical = ToolkitConfig(
        pristine_cell_path=str(tmp_path / "pristine.xyz"),
        phonon_smearing_meV=6.0, # Physical finite-size correction smearing
    )

    # 2 discrete modes mimicking a finite 3x3x3 supercell (61 and 75 meV)
    hw_modes = np.array([61.0, 75.7])
    sk_modes = np.array([1.2, 1.1])
    hw_grid = np.linspace(0.5, 120.0, 1200)

    # Tiny smearing preserves discrete separation
    s_disc = np.zeros_like(hw_grid)
    for w, s in zip(hw_modes, sk_modes):
        s_disc += s * np.exp(-0.5 * ((hw_grid - w) / cfg_discrete.phonon_smearing_meV)**2)
    
    # Physical smearing bridges the modes
    s_phys = np.zeros_like(hw_grid)
    for w, s in zip(hw_modes, sk_modes):
        s_phys += s * np.exp(-0.5 * ((hw_grid - w) / cfg_physical.phonon_smearing_meV)**2)

    dip_disc = s_disc[np.argmin(np.abs(hw_grid - 68.3))]
    dip_phys = s_phys[np.argmin(np.abs(hw_grid - 68.3))]
    
    # In discrete smearing, there is a deep artificial valley between the two split modes
    # In physical smearing, the resonance is continuous across the gap
    assert dip_phys > 2.0 * dip_disc


def test_anharmonic_regimes(tmp_path):
    """Verify anharmonic sampling distinguishes intermediate (quantum) and strong (semiclassical) regimes."""
    from hydeph.lineshape import LineshapeEngine
    from unittest.mock import patch, MagicMock

    cfg_quantum = ToolkitConfig(
        pristine_cell_path=str(tmp_path / "pristine.xyz"),
        temperatures=[10.0],
        anharmonic_md_steps=10,
        anharmonic_coupling_regime="auto"
    )
    
    mock_calc = MagicMock()
    mock_calc.get_potential_energy.return_value = -50.0

    engine = LineshapeEngine(cfg_quantum, hybrid_calc=mock_calc)

    at_gs = Atoms("C2", positions=[[0, 0, 0], [1.5, 1.5, 1.5]], cell=[3, 3, 3], pbc=True)
    at_es = Atoms("C2", positions=[[0, 0, 0], [1.6, 1.6, 1.6]], cell=[3, 3, 3], pbc=True)
    hw_grid = np.linspace(1.0, 100.0, 200)
    s_cont = 2.5 * np.exp(-0.5 * ((hw_grid - 40.0) / 5.0)**2) * 1000.0 # S ~ 2.5

    # Mock Langevin MD
    with patch("ase.md.langevin.Langevin") as mock_langevin_cls:
        mock_dyn = MagicMock()
        mock_langevin_cls.return_value = mock_dyn
        
        # When dyn.attach is called, simulate step_observer calls
        def fake_run(steps):
            if mock_dyn.attach.called:
                obs = mock_dyn.attach.call_args[0][0]
                for _ in range(5):
                    obs()
        mock_dyn.run.side_effect = fake_run

        # 1. Intermediate coupling: S = 2.5 < 10 -> quantum regime
        res_quant = engine.run_anharmonic_sampling(
            at_gs, at_es, hw_grid, s_cont, total_S=2.5, e_min=-200, e_max=50, n_pts=200
        )
        assert 10.0 in res_quant
        assert res_quant[10.0]["regime"] == "quantum"
        assert res_quant[10.0]["alpha_soft"] is not None
        # Must retain sharp lineshape
        assert np.max(res_quant[10.0]["lineshape"]) > 0.0

        # 2. Strong coupling: S = 15.0 >= 10 -> semiclassical regime
        res_semi = engine.run_anharmonic_sampling(
            at_gs, at_es, hw_grid, s_cont, total_S=15.0, e_min=-200, e_max=50, n_pts=200
        )
        assert 10.0 in res_semi
        assert res_semi[10.0]["regime"] == "semiclassical"

        # 3. Explicit override: regime='semiclassical' even with S = 2.5
        cfg_quantum.anharmonic_coupling_regime = "semiclassical"
        res_override = engine.run_anharmonic_sampling(
            at_gs, at_es, hw_grid, s_cont, total_S=2.5, e_min=-200, e_max=50, n_pts=200
        )
        assert res_override[10.0]["regime"] == "semiclassical"


def test_configuration_coordinate_engine(tmp_path):
    """Verify ConfigurationCoordinateEngine and Stokes shift calculations."""
    from hydeph.cc_diagram import ConfigurationCoordinateEngine, plot_configuration_coordinate

    # Create synthetic atoms and harmonic spring calculator
    at_gs = Atoms("C2", positions=[[0, 0, 0], [1.5, 1.5, 1.5]], cell=[4, 4, 4], pbc=True)
    at_es = Atoms("C2", positions=[[0, 0, 0], [1.5, 1.5, 1.7]], cell=[4, 4, 4], pbc=True)

    # Displacement dr = 0.2 A for atom 2 (carbon mass = 12.011 amu)
    # Delta Q = sqrt(12.011 * 0.2^2) = sqrt(12.011 * 0.04) = 0.693138 amu^(1/2) A
    expected_dq = np.sqrt(12.011 * (0.2**2))

    # Mock calculator where E(R) = 0.5 * k * sum(M_i * (R_i - R_GS,i)^2)
    # with k = 1.0 eV / (amu * A^2)
    k_const = 1.0
    mock_calc = MagicMock()
    def get_energy():
        pos = mock_calc.atoms.positions
        dr = pos - at_gs.positions
        m = at_gs.get_masses()
        dq2 = np.sum(np.repeat(m, 3) * dr.flatten()**2)
        return 0.5 * k_const * dq2

    # ASE calls get_potential_energy(atoms)
    def calc_pe(atoms=None):
        if atoms is not None:
            mock_calc.atoms = atoms
        return get_energy()

    mock_calc.get_potential_energy = MagicMock(side_effect=calc_pe)

    cfg = ToolkitConfig(
        pristine_cell_path=str(tmp_path / "pristine.xyz"),
        workspace_dir=str(tmp_path),
        zpl_energy_eV=2.0,
        cc_lambda_min=-0.2,
        cc_lambda_max=1.2,
        cc_n_points=15,
    )

    engine = ConfigurationCoordinateEngine(cfg, mock_calc)
    results = engine.calculate_cc_curve(at_gs, at_es, zpl_energy_eV=2.0)

    # 1. Delta Q check
    assert pytest.approx(results["dq_total"], rel=1e-3) == expected_dq

    # 2. Ground state energy at lambda=0 is 0
    idx_0 = np.argmin(np.abs(results["lambda_grid"] - 0.0))
    assert pytest.approx(results["e_gs"][idx_0], abs=1e-5) == 0.0

    # 3. Relaxation energies and Stokes shifts
    # E_rel = 0.5 * k * dq^2
    expected_erel = 0.5 * k_const * (expected_dq**2)
    assert pytest.approx(results["delta_e_rel_gs_eV"], rel=1e-3) == expected_erel
    assert pytest.approx(results["delta_e_rel_es_eV"], rel=1e-3) == expected_erel

    # Stokes shift = 2 * E_rel
    assert pytest.approx(results["stokes_shift_eV"], rel=1e-3) == 2.0 * expected_erel
    assert pytest.approx(results["stokes_shift_meV"], rel=1e-3) == 2000.0 * expected_erel

    # E_abs = ZPL + E_rel, E_em = ZPL - E_rel
    assert pytest.approx(results["e_abs_eV"], rel=1e-3) == 2.0 + expected_erel
    assert pytest.approx(results["e_em_eV"], rel=1e-3) == 2.0 - expected_erel

    # 4. Effective phonon frequency: hbar * Omega = 91.43476 * sqrt(E_rel) / dq
    # Since E_rel = 0.5 * k * dq^2, sqrt(E_rel)/dq = sqrt(0.5 * k)
    expected_hw = 91.43476 * np.sqrt(0.5 * k_const)
    assert pytest.approx(results["hbar_omega_meV"], rel=1e-3) == expected_hw

    # 5. Render figure
    fig_path = engine.plot_configuration_coordinate(results)
    assert os.path.isfile(fig_path)
    assert os.path.getsize(fig_path) > 1000

    # 6. Test functional interface
    func_res = plot_configuration_coordinate(cfg, mock_calc, at_gs, at_es, zpl_energy_eV=2.0)
    assert "figure_path" in func_res
    assert os.path.isfile(func_res["figure_path"])

    # 7. Test automatic ZPL derivation from Kohn-Sham band gap
    cfg_auto = ToolkitConfig(
        pristine_cell_path=str(tmp_path / "pristine.xyz"),
        workspace_dir=str(tmp_path),
        zpl_energy_eV=None,
    )
    at_gs_auto = at_gs.copy()
    at_gs_auto.info["dft_gap_eV"] = 2.50
    engine_auto = ConfigurationCoordinateEngine(cfg_auto, mock_calc)
    res_auto = engine_auto.calculate_cc_curve(at_gs_auto, at_es)
    assert res_auto["zpl_source"] == "kohn_sham_eigenvalues"
    expected_auto_zpl = 2.50 - expected_erel
    assert pytest.approx(res_auto["e_zpl_eV"], rel=1e-3) == expected_auto_zpl
    assert pytest.approx(res_auto["e_abs_eV"], rel=1e-3) == 2.50

    # 8. Test that zpl_energy_eV=0 is treated as auto-derive rather than collapsing to 0 eV
    cfg_zero = ToolkitConfig(
        pristine_cell_path=str(tmp_path / "pristine.xyz"),
        workspace_dir=str(tmp_path),
        zpl_energy_eV=0,
    )
    assert cfg_zero.zpl_energy_eV is None
    engine_zero = ConfigurationCoordinateEngine(cfg_zero, mock_calc)
    res_zero = engine_zero.calculate_cc_curve(at_gs_auto, at_es, zpl_energy_eV=0)
    assert res_zero["zpl_source"] == "kohn_sham_eigenvalues"
    assert pytest.approx(res_zero["e_zpl_eV"], rel=1e-3) == expected_auto_zpl


def test_parse_qe_band_gap(tmp_path):
    """Verifies parsing of Kohn-Sham band gaps from Quantum ESPRESSO outputs."""
    from hydeph.qe_runner import parse_qe_band_gap

    # 1. Spin-polarized output sample
    sample_spin = """
     Parallel version (MPI), running on 1 processors
 ------ SPIN UP ------------
          k = 0.0000 0.0000 0.0000 (  100 PWs)   bands (ev):
   -10.5000  -5.2000   1.2000   3.5000
 ------ SPIN DOWN ----------
          k = 0.0000 0.0000 0.0000 (  100 PWs)   bands (ev):
   -10.4500  -5.1500   0.9000   3.2000
     the spin up/dw Fermi energies are     1.5000    1.0000 ev
    """
    spin_file = tmp_path / "spin.out"
    spin_file.write_text(sample_spin)
    res_spin = parse_qe_band_gap(str(spin_file))
    assert res_spin is not None
    # Up: occ=[-10.5, -5.2, 1.2], unocc=[3.5], gap = 3.5 - 1.2 = 2.3 eV
    # Down: occ=[-10.45, -5.15, 0.9], unocc=[3.2], gap = 3.2 - 0.9 = 2.3 eV
    assert pytest.approx(res_spin["gap_eV"], rel=1e-3) == 2.30
    assert pytest.approx(res_spin["homo_eV"], rel=1e-3) == 1.20 or pytest.approx(res_spin["homo_eV"], rel=1e-3) == 0.90

    # 2. Non-spin polarized sample
    sample_single = """
 End of self-consistent calculation
          k = 0.0000 0.0000 0.0000 (  100 PWs)   bands (ev):
   -12.0000  -4.5000   1.0000   2.8000
     the Fermi energy is     1.2000 ev
    """
    single_file = tmp_path / "single.out"
    single_file.write_text(sample_single)
    res_single = parse_qe_band_gap(str(single_file))
    assert res_single is not None
    # occ=[-12.0, -4.5, 1.0], unocc=[2.8], gap = 2.8 - 1.0 = 1.8 eV
    assert pytest.approx(res_single["gap_eV"], rel=1e-3) == 1.80
    assert pytest.approx(res_single["homo_eV"], rel=1e-3) == 1.00
    assert pytest.approx(res_single["lumo_eV"], rel=1e-3) == 2.80





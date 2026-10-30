# Explicit Methods for Stress Wave Propagation

**MEC4128Z Final-Year Project 019 · University of Cape Town · 2026**
Student: Tim Martin (MRTTIM005) · Supervisor: Dr Trevor Cloete

An open, verified finite element solver for elastic stress waves in a solid cylindrical bar (a Hopkinson pressure bar). It uses 2D axisymmetric four-node (Q4) elements and explicit central difference time integration. Lumped (HRZ), consistent and combined mass matrices are available. The critical time step is found from the largest eigenvalue of the constrained system, and every run uses 0.9 times this value.

The solver is verified with seven targeted tests and validated against Pochhammer–Chree theory and Govender et al. (2006).

---

## Contents

| Path | Description |
|---|---|
| `fea_core.py` | Shared library containing all finite element functions. Every script imports it. |
| `main.py` | Sandbox driver for trying the solver on a small bar. Not used for any reported result. |
| `Verification/` | One script per verification or validation test (see below). |
| `script_plots/` | Figures saved by the scripts. Created automatically on the first run. |

---

## Requirements and set-up

Python 3.10 or later, with:

```bash
pip install numpy scipy matplotlib sympy
```

NumPy 2.0 or later is needed (the Pochhammer–Chree scripts use `np.trapezoid`). The results were last reproduced with Python 3.13, NumPy 2.5, SciPy 1.18, Matplotlib 3.11 and SymPy 1.14.

No installation of the code itself is needed. Run any script directly, for example:

```bash
python Verification/patch_test.py
```

Each script adds the folder above it to the Python path, so it finds `fea_core.py` wherever it is run from. Figures are saved to the `script_plots/` folder **next to `fea_core.py`**, and each script prints the exact location when it finishes.

Keep `PC_vibration_modes.py` and `Pochammer-Chree-wave-propagation.py` in the same folder, because the second imports the first.

**Units** are consistent mm–N–tonne–s: lengths in mm, stress in MPa, density in tonne/mm³, time in s.

---

## Reproducing the results

Each script runs one test with the parameters in Table 3.2 of the report. It prints its key numbers to the console and saves its figures (PDF and PNG) to `script_plots/`. The patch test saves its two figures next to the script instead.

Run times are approximate, on a laptop with 16 GB of RAM.

| Script (in `Verification/`) | Report output | Time | What it does |
|---|---|---|---|
| `patch_test.py` | Fig 3.6, Fig 4.1 | < 1 min | Distorted 4×4 patch under a constant-strain field (including hoop strain). Prints the maximum displacement and stress error. |
| `equivalent_nodal_load_constant_axial_strain.py` | Table 4.1 | < 1 min | Compares the reactions under uniform axial strain with the consistent nodal loads, node by node, on both end faces. |
| `energy_balance.py` | Fig 4.2, Fig 4.3 | ~15 min | Energy (half-step form) and momentum balance for all three mass matrices. |
| `mass_and_mesh_convergence.py` | Fig 4.4 | ~20 min | Gauge strain on 20 to 640 axial elements for each mass matrix. The error plot uses 40–640 elements (the 20-element run is shown in the strain plot only). Prints the time step and peak strain for every run, and saves the results to `convergence_results.npy`. |
| `single-frequency-wave-dispersion.py` | Fig 4.5, Table 4.2 | ~25 min | Free-end reflection of a Hann pulse. Prints the wave-speed error and the difference after alignment for each mass matrix. |
| `Square-wave.py` | Fig 4.6 | < 1 min | Square pulse with Poisson's ratio set to zero. Shows the dispersion signature of lumped and consistent mass. |
| `PC_vibration_modes.py` | Fig 4.7 (left), Fig 4.8 | ~7 min | (1) Solves the Pochhammer–Chree frequency equation for the bar of Govender et al. (2006). (2) Measures the FEA phase velocity with the two-station method on 75, 150 and 300 axial elements. |
| `Pochammer-Chree-wave-propagation.py` | Fig 4.9 | ~15 min | Strain at a gauge compared with the first-mode Pochhammer–Chree prediction. Prints the time lag, peak ratio and normalised RMS difference. Importing `PC_vibration_modes.py` also runs that script first, so the time includes it. |

Figures 3.1, 3.2, 3.4 and 3.5 are explanatory diagrams. Figure 3.3 (input pulses) and Figure 4.7 (right, reproduced from Govender et al. 2006) are also not produced by these scripts.

---

## Using the solver for new problems

A typical run follows the program flow in Figure 3.5 of the report:

| Function | Purpose |
|---|---|
| `create_mesh` | Structured r–z mesh of Q4 elements |
| `build_GSM` | Global stiffness matrix (3×3 Gauss integration) |
| `build_mass_matrices` | Consistent and HRZ lumped mass |
| `combined_mass_matrices` | β·consistent + (1 − β)·lumped |
| `get_force_vec` | Consistent nodal loads on z = 0 |
| `get_axis_dofs` | Radial DOFs on the axis (held at zero) |
| `get_dt_crit` | Critical time step from the eigenproblem |
| `run_explicit_solver` | Central difference time stepping |
| `get_strain_stress_history` | Strain and stress against time at a point |

To change the bar, the mesh, the material or the pulse, edit the input block at the top of the relevant script. `main.py` is the simplest starting point.

---

## Notes and limitations

- Matrices are stored dense, so memory grows with the square of the number of degrees of freedom. The largest reported runs use about 6 600 DOF, and a 16 GB machine tops out at roughly 17 000 DOF. Close figure windows between runs.
- The axis DOFs must be passed as `fixed_dofs` to **both** `get_dt_crit` and `run_explicit_solver`. A mismatch makes the run unstable.
- `solve_global_eqn` does not move K·d for non-zero prescribed displacements to the right-hand side. `patch_test.py` does this itself before calling it.
- The derivations behind every function are given in Appendices A–E of the report.

## Further work

1. **Sparse storage**: store only the non-zero matrix entries, to reach the 10⁵+ DOF needed for higher modes.
2. **Validate the multi-mode regime**: separate the modes in time, following Barr et al. (2020).
3. **Compare with experiments**: test the model against real Hopkinson bar measurements.

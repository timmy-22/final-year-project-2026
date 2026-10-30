======================================================================
 EXPLICIT METHODS FOR STRESS WAVE PROPAGATION
 MEC4128Z Final-Year Project 019, University of Cape Town, 2026
 Student:    Tim Martin (MRTTIM005)
 Supervisor: Dr Trevor Cloete
======================================================================

1. WHAT IS SUBMITTED
----------------------------------------------------------------------
This submission contains the final report and the Python code used to
produce every result in it.

  MRTTIM005_Report.pdf    The final report.
  readme.txt              This file.
  code/                   The finite element solver and test scripts:

    fea_core.py           Shared library containing all finite element
                          functions. Every script imports it.
    main.py               Sandbox driver for trying the solver on a
                          small bar. Not used for any reported result.
    Verification/         One script per verification or validation
                          test (Section 3 below).
    script_plots/         Figures saved by the scripts. Created
                          automatically on the first run.

The code solves elastic stress wave propagation in a solid cylindrical
bar (a Hopkinson pressure bar) using 2D axisymmetric four-node (Q4)
elements and explicit central difference time integration. Lumped
(HRZ), consistent and combined mass matrices are available. The
critical time step is found from the largest eigenvalue of the
constrained system, and every run uses 0.9 times this value.


2. REQUIREMENTS AND SET-UP
----------------------------------------------------------------------
Python 3.10 or later, with:

    pip install numpy scipy matplotlib sympy

NumPy 2.0 or later is needed (the Pochhammer-Chree scripts use
np.trapezoid). The results were last reproduced with Python 3.13,
NumPy 2.5, SciPy 1.18, Matplotlib 3.11 and SymPy 1.14.

No installation of the code itself is needed. Run any script directly,
for example:

    python code/Verification/patch_test.py

Each script adds the folder above it to the Python path, so it finds
fea_core.py wherever it is run from. Keep PC_vibration_modes.py and
Pochammer-Chree-wave-propagation.py in the same folder, because the
second imports the first.

Units are consistent mm-N-tonne-s: lengths in mm, stress in MPa,
density in tonne/mm^3, time in s.


3. REPRODUCING THE RESULTS IN THE REPORT
----------------------------------------------------------------------
Each script runs one test with the parameters in Table 3.2 of the
report. It prints its key numbers to the console and saves its figures
(PDF and PNG) to code/script_plots/. The patch test saves its two
figures next to the script instead.

Run times are approximate, on a laptop with 16 GB of RAM.

Script (in Verification/)             Report output           Time
----------------------------------------------------------------------
patch_test.py                         Fig 3.6, Fig 4.1        < 1 min
  Distorted 4x4 patch under a constant-strain field (including hoop
  strain). Prints the maximum displacement and stress error.

equivalent_nodal_load_constant_axial_strain.py
                                      Table 4.1               < 1 min
  Compares the reactions under uniform axial strain with the
  consistent nodal loads, node by node, on both end faces.

energy_balance.py                     Fig 4.2, Fig 4.3        ~15 min
  Energy (half-step form) and momentum balance for all three mass
  matrices.

mass_and_mesh_convergence.py          Fig 4.4                 ~20 min
  Gauge strain on 40 to 640 axial elements for each mass matrix.
  Prints the time step and peak strain for every run.

single-frequency-wave-dispersion.py   Fig 4.5, Table 4.2      ~25 min
  Free-end reflection of a Hann pulse. Prints the wave-speed error
  and the difference after alignment for each mass matrix.

Square-wave.py                        Fig 4.6                 < 1 min
  Square pulse with Poisson's ratio set to zero. Shows the dispersion
  signature of lumped and consistent mass.

PC_vibration_modes.py                 Fig 4.7 (left), Fig 4.8 ~7 min
  (1) Solves the Pochhammer-Chree frequency equation for the bar of
  Govender et al. (2006). (2) Measures the FEA phase velocity with
  the two-station method on 75, 150 and 300 axial elements.

Pochammer-Chree-wave-propagation.py   Fig 4.9                 ~15 min
  Strain at a gauge compared with the first-mode Pochhammer-Chree
  prediction. Prints the time lag, peak ratio and normalised RMS
  difference. Importing PC_vibration_modes.py also runs that script
  first, so the time includes it.

Figures 3.1, 3.2, 3.4 and 3.5 are explanatory diagrams. Figure 3.3
(input pulses) and Figure 4.7 (right, reproduced from Govender et al.
2006) are also not produced by these scripts.


4. USING THE SOLVER FOR NEW PROBLEMS
----------------------------------------------------------------------
A typical run in fea_core.py follows the program flow in Figure 3.5
of the report:

    create_mesh               structured r-z mesh of Q4 elements
    build_GSM                 global stiffness matrix (3x3 Gauss)
    build_mass_matrices       consistent and HRZ lumped mass
    combined_mass_matrices    beta*consistent + (1-beta)*lumped
    get_force_vec             consistent nodal loads on z = 0
    get_axis_dofs             radial DOFs on the axis (held at zero)
    get_dt_crit               critical time step from the eigenproblem
    run_explicit_solver       central difference time stepping
    get_strain_stress_history strain and stress against time at a point

To change the bar, the mesh, the material or the pulse, edit the
input block at the top of the relevant script. main.py is the simplest
starting point.


5. NOTES AND LIMITATIONS
----------------------------------------------------------------------
- Matrices are stored dense, so memory grows with the square of the
  number of degrees of freedom. The largest runs (about 6 600 DOF)
  need several GB of RAM. Close figure windows between runs.
- The axis DOFs must be passed as fixed_dofs to BOTH get_dt_crit and
  run_explicit_solver. A mismatch makes the run unstable.
- solve_global_eqn does not move K*d for non-zero prescribed
  displacements to the right-hand side. patch_test.py does this itself
  before calling it.
- The derivations behind every function are given in Appendices A-E
  of the report.

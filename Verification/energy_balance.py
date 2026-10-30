import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import importlib, fea_core
importlib.reload(fea_core)
from fea_core import (
    create_mesh, gaussPointScheme, partialOfShapeFn, partial, Shape,
    get_element_coords, get_element_dof_indices, get_element_D_matrix,
    compute_B_and_jacobian, StressFunc, get_load_matrix, q_function,
    get_force_vec, build_mass_matrices, combined_mass_matrices, get_dt_crit,
    plot_mesh, plot_stress_field, plot_mesh_and_force, compute_stresses, solve_global_eqn, 
    build_GSM, find_nearest_element, get_strain_stress_history,run_explicit_solver,
    plot_strain_history_stacked, plot_gws_input, check_mass_conservation, get_axis_dofs
)
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
import sympy as sp
from pathlib import Path
from matplotlib.ticker import NullFormatter
#----------------------------------------------------------------------
# CONFIGURATION / INPUTS
#----------------------------------------------------------------------
#pick gauss scheme (1,2,3)
Scheme = 3
ScaleFactor = 500
beta = 0.5  #beta for combined mass matrix using form (1-beta)*Lumped + beta*consistent
Tstep = 1.87e-8   #for 1mm square elements need step size less than: 6.616591513240334e-08 (if using default material properties)
Tend = 5e-5
station = (0.0, 50.0)  #coordinate to analyse 
Tcrit_ratio = 0.9
#-----------------------------------------------------------------------------
bar_radius, bar_length = 2,80            #radius and length of bar [mm]
numEle_radius, numEle_length = 12,200       #number of elements in radius and length

nodes, elements = create_mesh(bar_radius,bar_length,numEle_radius,numEle_length)

# properties (same material everywhere) 
properties = np.tile(np.array([2.00e5, 0.3, 1]), (len(elements), 1))

#-----------------------------------------------------------------------------
# BUILD ACTUAL EQUIVALENT NODAL LOADS
#-----------------------------------------------------------------------------
forces = get_force_vec(0, nodes, numEle_radius, get_load_matrix, q_function)
#-----------------------------------------------------------------------------

DoF = 2*len(nodes)
knownDisplacements = np.zeros((DoF), dtype=bool)  #starts off with array pre filled with "False". Can then adjust some nodes' DOF to True depending on the case
for d in range(numEle_length+1):
    knownDisplacements[2*d*(numEle_radius+1)] = True    #constrain DoF associated with r-direction along centerline of bar

#----------------------This loop makes end of bar fixed in z-direction(axial fixed)------------------------------------------------------   
# for e in range(numEle_radius+1):
#     knownDisplacements[DoF - e*2-1]  = True   # starts at end and makes last row of nodes in z direction have known displacement of zero
#----------------------------------------------------------------------------------------------------------------------------------------
Displacements = np.zeros(DoF)   #initial blank displacement vector thats later used to solve principle FEA equation
Fmod = forces.flatten().reshape(-1, 1)   
force_scheme = Fmod.copy()

density = 7.850e-9      #tonne/mm^3
waveVelocity = (np.sqrt(properties[0,0]/density))/1000 #assumes single material as E just taken as first elements'
print("Wave Speed: ", waveVelocity)

#----------------------------------------------------------------------
#----------------------------------------------------------------------
# MAIN SCRIPT
#----------------------------------------------------------------------

points, weight = gaussPointScheme(Scheme)
print('Points: ', points, '\n Weigth: ', weight)

GSM = build_GSM(nodes, elements, properties, DoF, Scheme, points, weight)
Deflection, F_reaction, Fmod = solve_global_eqn(GSM, forces, Displacements, knownDisplacements, DoF)

print('Reaction Forces= \n ', F_reaction)
Deflections = Deflection
Deflection = Deflection.reshape(int(DoF/2),2)
NewNodes = nodes + Deflection*ScaleFactor

#-----------------------------------------------Stress Code-----------------------------------------------------
ElemStressAvg, StressesList = compute_stresses(nodes, elements, properties, Deflections, Scheme, points, weight)
print('Stresses = \n', StressesList)
#-------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------
# EXPLICIT INTEGRATION SCHEME
#----------------------------------------------------------------------

#--------------------------VOLUME AND MASS-----------------------------

M_lumped, M_consistent, Volume, mass_total, centroids, z_centroids, masses = build_mass_matrices(
    nodes, elements, properties, density, DoF, Scheme, points, weight)

m_combined = combined_mass_matrices(M_lumped, M_consistent, beta)

Minv = np.linalg.inv(M_lumped)
m_consis_inv = np.linalg.inv(M_consistent)
m_comb_inv = np.linalg.inv(m_combined)
#-----------------------------------------------------------------------
fixed_dofs = get_axis_dofs(nodes)
dt_crit_cons= get_dt_crit(GSM, M_consistent,fixed_dofs)
dt_crit_comb= get_dt_crit(GSM, m_combined,fixed_dofs)
dt_crit_lump= get_dt_crit(GSM, M_lumped,fixed_dofs)
Tcons = Tcrit_ratio * dt_crit_cons
Tcomb = Tcrit_ratio * dt_crit_comb
Tlump = Tcrit_ratio * dt_crit_lump
print("Delta t crit lumped:", dt_crit_lump)
print("Delta t crit consistent:", dt_crit_cons)


total_mass_fem, mass_expected = check_mass_conservation(Minv, density, Volume)
print('Total Mass: ', total_mass_fem * 1000, 'kg')
print('Expected Mass: ', mass_expected * 1000, 'kg')
#-----------------------------------------------------------------------
def force_fn(t):
    return get_force_vec(t, nodes, numEle_radius, get_load_matrix, q_function)  #so when run_explicit_solver function is run, the correct time gets parsed as this function is linked
displacement_history_hrz, KE_lump, SE_lump, W_lump, E_exact_lump,Wrect_lump, P_lump,imp_lump = run_explicit_solver(M_lumped,Minv, GSM, Tlump, Tend, force_fn, DoF, fixed_dofs)
displacement_history_consistent, KE_cons, SE_cons, W_cons,E_exact_cons,Wrect_cons,P_cons,imp_cons = run_explicit_solver(M_consistent,m_consis_inv, GSM, Tcons, Tend, force_fn, DoF, fixed_dofs)
displacement_history_combined,KE_comb, SE_comb, W_comb,E_exact_comb,Wrect_comb,P_comb,imp_comb = run_explicit_solver(m_combined,m_comb_inv, GSM, Tcomb, Tend, force_fn, DoF, fixed_dofs)
print(len(displacement_history_hrz), len(displacement_history_consistent), len(displacement_history_combined))

exact_runs = {
    "Lumped":     (E_exact_lump, Wrect_lump, Tlump),
    "Consistent": (E_exact_cons, Wrect_cons, Tcons),
    "Combined":   (E_exact_comb, Wrect_comb, Tcomb),
}

momentum = {
    "Lumped":     (P_lump, imp_lump, Tlump),
    "Consistent": (P_cons, imp_cons, Tcons),
    "Combined":   (P_comb, imp_comb, Tcomb),
}
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# for label, (P, imp, dt) in momentum.items():
#     P, imp = np.asarray(P), np.asarray(imp)
#     t = (np.arange(len(P)) + 0.5) * dt * 1e6          # since uses velocity half step, the last step is +0.5
#     scale = max(np.abs(imp).max(), 1e-30)
#     rel_err = np.abs(P - imp) / scale

#     ax1.plot(t, imp, '--', label=f'{label} impulse')
#     ax1.plot(t, P, label=f'{label} momentum')
#     ax2.semilogy(t, np.maximum(rel_err, 1e-20), label=label)   #  floor to avoid log(0) error

# ax1.set_xlabel('Time [µs]')
# ax1.set_ylabel('Axial momentum / impulse [t·mm/s]')
# ax1.set_title('Momentum vs cumulative impulse')
# ax1.legend()
# ax1.grid(alpha=0.3)

# ax2.set_xlabel('Time [µs]')
# ax2.set_ylabel('|P − impulse| / max|impulse|')
# ax2.set_title('Relative error')
# ax2.legend()
# ax2.grid(True, which='both', alpha=0.3)

# plt.tight_layout()
# plt.show()


# # fig, ax = plt.subplots(figsize=(7, 4.5))
# # for name, (E, Wr, dt) in exact_runs.items():      # {"Lumped": (E_lump, Wrect_lump, Tlump), ...}
# #     t_us = np.arange(len(E)) * dt * 1e6
# #     err = np.abs(E - Wr) / np.max(E)
# #     ax.semilogy(t_us, err + 1e-20, label=name)
# # ax.set_xlabel("Time [µs]")
# # ax.set_ylabel("|E_exact − W_rect| / max(E_exact)")
# # ax.legend(); plt.tight_layout(); plt.show()
# for name, (E, Wr, dt) in exact_runs.items():
#     err = np.abs(E - Wr) / np.max(E)
#     print(f"{name}: max relative error = {err.max():.2e}")

# fig, ax = plt.subplots(figsize=(7, 4.5))
# for name, (E, Wr, dt) in exact_runs.items():
#     t_us = np.arange(len(E)) * dt * 1e6
#     ax.plot(t_us, (E - Wr) / np.max(E), label=name)   # signed, in units of machine epsilon
# ax.set_xlabel("Time [µs]")
# ax.set_ylabel("(E_exact − W_rect) / max(E_exact)  [relative error]")
# ax.legend(); plt.tight_layout(); plt.show()

# # StressHistory = []
# # for d in displacement_history_hrz:
# #     _, StressesList_t = compute_stresses(nodes, elements, properties, d, Scheme, points, weight)
# #     StressHistory.append(StressesList_t)
# # StressHistory = np.array(StressHistory)

# history_arr_hrz = np.array(displacement_history_hrz)
# history_arr_consistent = np.array(displacement_history_consistent)
# history_arr_combined = np.array(displacement_history_combined)
# time_arr = np.arange(history_arr_hrz.shape[0]) * Tstep    
# # node = 7
# # r_dof = 2*(node-1)

# # print(history_arr[:, r_dof])
# #----------------------------------------------------------------------------

# print("Proccessed Vector =\n", Fmod)
# print("deflection =\n", Deflection)
# nSteps = len(KE_lump)

runs = {
    "Lumped":     (KE_lump, SE_lump, W_lump, Tlump),
    "Consistent": (KE_cons, SE_cons, W_cons, Tcons),
    "Combined":   (KE_comb, SE_comb, W_comb, Tcomb),
}



# fig, axs = plt.subplots(2, 2, figsize=(12, 8))
# axs = axs.ravel()          # axs[0]..axs[3]
# # Plotting layout
# # [(Consistent, Lumped),(Combined,relative balance error)]
# # TOP LEFT PLOT (CONSISTENT MASS MATRIX) ENERGY 
# for ax, (name, (KE, SE, W, dt)) in zip(axs[:3], runs.items()):
#     t_us = np.arange(len(KE)) * dt * 1e6
#     ax.plot(t_us, KE, label="KE")
#     ax.plot(t_us, SE, label="SE")
#     ax.plot(t_us, KE + SE, "k--", label="KE + SE")
#     ax.plot(t_us, W, ":", label="External work")
#     ax.set_xlabel("Time [µs]")
#     ax.set_ylabel("Energy [N·mm]")
#     ax.set_title(f"Energy components ({name})")
#     ax.legend()

# for name, (KE, SE, W, dt) in runs.items():
#     t_us = np.arange(len(KE)) * dt * 1e6
#     err = (KE + SE - W) / np.max(np.abs(W))
#     axs[3].plot(t_us, err * 100, label=name)
#     print(f"{name}: max |balance error| = {np.max(np.abs(err)) * 100:.3f} % of peak external work")
# axs[3].axhline(0, color="k", lw=0.5)
# axs[3].set_xlabel("Time [µs]")
# axs[3].set_ylabel("(KE + SE − W_ext) / max(W_ext) [%]")
# axs[3].set_title("Energy balance error")
# axs[3].legend()

# plt.tight_layout()
# plt.show()

# fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))

# for name, (KE, SE, W, dt) in runs.items():
#     t_us = np.arange(len(KE)) * dt * 1e6
#     mask = np.abs(W) > 0.01 * np.max(np.abs(W))      # hide start where W = 0 and the ratio is just noise
#     frac = (KE + SE) / W
#     axs[0].plot(t_us[mask], frac[mask], label=name)

# axs[0].axhline(1.0, color="k", lw=0.5)
# axs[0].set_xlabel("Time [µs]")
# axs[0].set_ylabel("(KE + SE) / W_ext")
# axs[0].set_title("Fraction of input work accounted for")
# axs[0].legend()

# # Right: how the work splits into KE and SE (lumped)
# KE, SE, W, dt = runs["Lumped"]
# t_us = np.arange(len(KE)) * dt * 1e6
# mask = np.abs(W) > 0.01 * np.max(np.abs(W))
# axs[1].plot(t_us[mask], (KE / W)[mask], label="KE / W_ext")
# axs[1].plot(t_us[mask], (SE / W)[mask], label="SE / W_ext")
# axs[1].axhline(0.5, color="k", lw=0.5, ls="--")
# axs[1].set_xlabel("Time [µs]")
# axs[1].set_ylabel("Fraction of W_ext")
# axs[1].set_title("Kinetic vs strain split (lumped)")
# axs[1].legend()

# plt.tight_layout()
# plt.show()

#----------------------------------------------------------------------
# REPORT FIGURES
#----------------------------------------------------------------------
from matplotlib.lines import Line2D
out = Path(fea_core.__file__).resolve().parent / "script_plots"
out.mkdir(exist_ok=True)       

colours = {"Lumped": "C0", "Consistent": "C1", "Combined": "C2"}

# Figure 1: energy components, lumped mass
KE, SE, W, dt = runs["Lumped"]
KE, SE, W = map(np.asarray, (KE, SE, W))
t_us = np.arange(len(KE)) * dt * 1e6

fig1, ax = plt.subplots(figsize=(6.5, 3.6))
ax.plot(t_us, KE, lw=0.9, label="Kinetic, $T$")
ax.plot(t_us, SE, lw=0.9, label="Strain, $U$")
ax.plot(t_us, KE + SE, "k-", lw=1.4, label="$T + U$")
ax.plot(t_us, W, color="C3", ls=":", lw=2,
        label="External work, $W$")
ax.set_xlabel(r"Time [$\mu$s]")
ax.set_ylabel(r"Energy [N$\cdot$mm]")
ax.grid(alpha=0.3)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
          ncol=4, fontsize=9, frameon=False)
fig1.tight_layout()

fig1.savefig(out / "energy_components.pdf", bbox_inches="tight")
fig1.savefig(out / "energy_components.png", dpi=200,
             bbox_inches="tight")

# Figure 2: signed energy balance error, half-step velocities only
fig2, ax = plt.subplots(figsize=(6.5, 3.6))

for name, (E, Wr, dt) in exact_runs.items():
    E, Wr = np.asarray(E), np.asarray(Wr)
    t_us = np.arange(len(E)) * dt * 1e6
    err_half = (E - Wr) / np.max(np.abs(Wr))
    ax.plot(t_us, err_half * 1e14, color=colours[name], lw=0.7,
            label=name)

ax.axhline(0, color="k", lw=0.5)
ax.set_xlabel(r"Time [$\mu$s]")
ax.set_ylabel(r"$(E - W)\,/\,\max|W|$ [$\times10^{-14}$]")
ax.grid(alpha=0.3)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
          ncol=3, fontsize=9, frameon=False)
fig2.tight_layout()

fig2.savefig(out / "energy_error.pdf", bbox_inches="tight")
fig2.savefig(out / "energy_error.png", dpi=200, bbox_inches="tight")

print("Figures saved to:", out)
plt.show()
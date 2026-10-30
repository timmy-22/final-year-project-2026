import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fea_core import (
    create_mesh, gaussPointScheme, partialOfShapeFn, partial, Shape,
    get_element_coords, get_element_dof_indices, get_element_D_matrix,
    compute_B_and_jacobian, StressFunc, get_load_matrix,
    get_force_vec, build_mass_matrices, combined_mass_matrices, run_explicit_solver, get_dt_crit,
    plot_mesh, plot_stress_field, plot_mesh_and_force, compute_stresses, solve_global_eqn, 
    build_GSM, find_nearest_element, get_strain_stress_history,
    plot_strain_history_stacked, plot_gws_input, check_mass_conservation, get_axis_dofs
)
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import fea_core
from scipy.signal import correlate
from scipy.signal import hilbert
from pathlib import Path
#----------------------------------------------------------------------
# CONFIGURATION / INPUTS
#----------------------------------------------------------------------
'''KEEP CONFIGURATION CONSTANT DURING ITERATIVE VERIFICATION'''
Scheme = 3
ScaleFactor = 500
beta = 0.5  #beta for combined mass matrix using form (1-beta)*Lumped + beta*consistent
Tcrit_ratio = 0.9   # CHANGED: Tstep now set from dt_crit after the matrices are built
Tend = 3.8e-5
#-----------------------------------------------------------------------------
bar_radius, bar_length = 0.5,100            #radius and length of bar [mm]
numEle_radius, numEle_length = 5,450       #number of elements in radius and length
station = (bar_radius, 70)  #coordinate to analyse 
z_g = station[1]
material = [2.00e5, 0.3, 1]
density = 7.850e-9              # tonne/mm^3
c = np.sqrt(material[0] / density)   # [mm/s]
T_p = 5 / 5.05e5                     # Hann pulse duration [s]
#-------------------------------INPUT FORCING FUNCTION------------------------------------
def q_func(r_position, time):
    r, t = sp.symbols('r t')
    amplitude = 200
    f = 5.05e5 
    omega = 2 * np.pi * f
    n_cycles = 5
    t_cutoff = n_cycles / f   # duration = exactly n_cycles periods

    if time <= t_cutoff:
        window = 0.5 * (1 - np.cos(2*np.pi*time/t_cutoff))   # Hann taper, 0→1→0
        pulse = amplitude * window * np.cos(omega * time)
    else:
        pulse = 0.0
    q_func = sp.lambdify((r, t), pulse, 'numpy')
    return q_func(r_position,time)

fea_core.q_function = q_func
#--------------------------------------------------------------------------------------------
nodes, elements = create_mesh(bar_radius,bar_length,numEle_radius,numEle_length)
properties = np.tile(np.array([2.00e5, 0.3, 1]), (len(elements), 1))
forces = get_force_vec(0, nodes, numEle_radius, get_load_matrix, q_func)

DoF = 2*len(nodes)
knownDisplacements = np.zeros((DoF), dtype=bool)
for d in range(numEle_length+1):
    knownDisplacements[2*d*(numEle_radius+1)] = True  

Displacements = np.zeros(DoF)
Fmod = forces.flatten().reshape(-1, 1)   
force_scheme = Fmod.copy()

density = 7.850e-9      #tonne/mm^3
waveVelocity = (np.sqrt(properties[0,0]/density))/1000

points, weight = gaussPointScheme(Scheme)

GSM = build_GSM(nodes, elements, properties, DoF, Scheme, points, weight)
Deflection, F_reaction, Fmod = solve_global_eqn(GSM, forces, Displacements, knownDisplacements, DoF)
Deflections = Deflection
Deflection = Deflection.reshape(int(DoF/2),2)
NewNodes = nodes + Deflection*ScaleFactor
ElemStressAvg, StressesList = compute_stresses(nodes, elements, properties, Deflections, Scheme, points, weight)

M_lumped, M_consistent, Volume, mass_total, centroids, z_centroids, masses = build_mass_matrices(
    nodes, elements, properties, density, DoF, Scheme, points, weight)

m_combined = combined_mass_matrices(M_lumped, M_consistent, beta)

Minv = np.linalg.inv(M_lumped)
m_consis_inv = np.linalg.inv(M_consistent)
m_comb_inv = np.linalg.inv(m_combined)

fixed_dofs = get_axis_dofs(nodes)
dt_crit_consistent = get_dt_crit(GSM, M_consistent,fixed_dofs)
Tcons = Tcrit_ratio * dt_crit_consistent
dt_crit_combined = get_dt_crit(GSM, m_combined,fixed_dofs)
Tcomb = Tcrit_ratio * dt_crit_consistent
dt_crit_lumped = get_dt_crit(GSM, M_lumped,fixed_dofs)
Tlump = Tcrit_ratio * dt_crit_consistent
print(f"dt_crit (consistent) = {dt_crit_consistent:.3e} s,  Tstep = {Tcons:.3e} s")


t_start = z_g / c - 0.5e-6
t_stop  = z_g / c + T_p + 1.0e-6


def force_fn(t):
    return get_force_vec(t, nodes, numEle_radius, get_load_matrix, q_func)
displacement_history_hrz,*_ = run_explicit_solver(M_lumped,Minv, GSM, Tlump, Tend, force_fn, DoF,fixed_dofs=fixed_dofs)
displacement_history_consistent,*_ = run_explicit_solver(M_consistent,m_consis_inv, GSM, Tcons, Tend, force_fn, DoF,fixed_dofs=fixed_dofs)
displacement_history_combined,*_ = run_explicit_solver(m_combined,m_comb_inv, GSM, Tcomb, Tend, force_fn, DoF,fixed_dofs=fixed_dofs)
history_arr_hrz = np.array(displacement_history_hrz)
history_arr_consistent = np.array(displacement_history_consistent)
history_arr_combined = np.array(displacement_history_combined)
time_arr = np.arange(history_arr_hrz.shape[0]) * Tlump

#----------------------------------------------------------------------
# NEW: incident vs reflected at the gauge, for all three mass matrices
#----------------------------------------------------------------------
travel = 2 * (bar_length - z_g) / c         # extra travel time [s]
gauge = {}
for name, hist, dt in [("Lumped",     history_arr_hrz,        Tlump),
                       ("Consistent", history_arr_consistent, Tcons),
                       ("Combined",   history_arr_combined,   Tcomb)]:
    i0, i1 = int(t_start / dt), int(t_stop / dt)
    i_shift = int(round(travel / dt))

    s = get_strain_stress_history(hist, elements, nodes, properties, *station)[0][:, 1]
    inc = s[i0:i1]
    ref = -s[i0 + i_shift : i1 + i_shift]          # flip sign, no time reversal
    lag = np.argmax(correlate(inc, ref, mode='full')) - (len(ref) - 1)

    # reflected pulse re-aligned by the best-fit delay
    ref_al = -s[i0 + i_shift - lag : i1 + i_shift - lag]
    err_al = np.max(np.abs(inc - ref_al)) / np.max(np.abs(inc))

    gauge[name] = (inc, ref, dt)
    print(f"{name:10s} delay = {abs(lag)*dt*1e9:.1f} ns | "
          f"speed below c0 = {abs(lag)*dt/travel*100:.2f} % | "
          f"after alignment = {err_al*100:.2f} %")

strain_lumped = get_strain_stress_history(history_arr_hrz, elements, nodes, properties,
                                   *station)[0][:, 1]
strain_consistent, *_ = get_strain_stress_history(history_arr_consistent, elements, nodes, properties, *station)
strain_combined, *_ = get_strain_stress_history(history_arr_combined, elements, nodes, properties, *station)

#----------------------------------------------------------------------
# NEW: report figure (vector PDF, no title, 9 pt serif)
#----------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm",
    "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "pdf.fonttype": 42,
})
inc, ref, dt_l = gauge["Lumped"]
t_rel = np.arange(len(inc)) * dt_l * 1e6 + (t_start - z_g/c) * 1e6   # 0 = arrival at c0

fig, ax = plt.subplots(figsize=(6.3, 2.8))
ax.plot(t_rel, inc*1e6, lw=1.0, label="Incident")
ax.plot(t_rel, ref*1e6, "--", lw=1.0,
        label="Reflected (inverted, shifted)")
ax.set_xlabel(r"Time after arrival at gauge [$\mu$s]")
ax.set_ylabel(r"$\varepsilon_z$ [$\mu\varepsilon$]")
ax.grid(alpha=0.3)
ax.legend(frameon=False, loc="upper right")
fig.tight_layout()

out = Path(fea_core.__file__).resolve().parent / "script_plots"
out.mkdir(exist_ok=True)                # save next to this script
fig.savefig(out / "reflection.pdf", bbox_inches="tight")
fig.savefig(out / "reflection.png", dpi=200, bbox_inches="tight")
print("Figures saved to:", out)
plt.show()


# plot_gws_input(q_func, 20, ax=None, r=0.0)
# plt.show()
# n = min(len(strain_lumped), len(strain_consistent))
# time_arr_us = np.arange(n) * Tlump * 1e6

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import importlib, fea_core
importlib.reload(fea_core)
from fea_core import (
    create_mesh, gaussPointScheme, partialOfShapeFn, partial, Shape,
    get_element_coords, get_element_dof_indices, get_element_D_matrix,
    compute_B_and_jacobian, StressFunc, get_load_matrix,
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
#----------------------------------------------------------------------
# CONFIGURATION / INPUTS
#----------------------------------------------------------------------
#pick gauss scheme (1,2,3)
Scheme = 3
ScaleFactor = 500
beta = 0.5  #beta for combined mass matrix using form (1-beta)*Lumped + beta*consistent
#Tstep = 1.87e-8   #for 1mm square elements need step size less than: 6.616591513240334e-08 (if using default material properties)
Tend = 5e-5

Tcrit_ratio = 0.9
#-----------------------------------------------------------------------------
def square_force(r_position, time):
    A  = 100.0        # MPa
    Tp = 5e-6         # pulse duration [s]  (~25 mm long in the bar)
    pulse = np.where((time >= 0) & (time < Tp), A, 0.0)
    return pulse * np.ones_like(r_position, dtype=float)
#-----------------------------------------------------------------------------
bar_radius, bar_length = 2,80            #radius and length of bar [mm]
numEle_radius, numEle_length = 4,200       #number of elements in radius and length
station = (bar_radius, 50.0)  #coordinate to analyse 

nodes, elements = create_mesh(bar_radius,bar_length,numEle_radius,numEle_length)

# properties (same material everywhere) 
properties = np.tile(np.array([2.00e5, 0.0, 1]), (len(elements), 1))    # poissons number 0 makes ripples be purely numerical


#-----------------------------------------------------------------------------
# BUILD ACTUAL EQUIVALENT NODAL LOADS
#-----------------------------------------------------------------------------
forces = get_force_vec(0, nodes, numEle_radius, get_load_matrix, square_force)
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
c0   = np.sqrt(properties[0, 0] / density)             # mm/s
Tend = 0.95 * (2*bar_length - station[1]) / c0 
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

# m_combined = combined_mass_matrices(M_lumped, M_consistent, beta)

Minv = np.linalg.inv(M_lumped)
m_consis_inv = np.linalg.inv(M_consistent)
# m_comb_inv = np.linalg.inv(m_combined)
#-----------------------------------------------------------------------
fixed_dofs = get_axis_dofs(nodes)
dt_crit_cons= get_dt_crit(GSM, M_consistent,fixed_dofs)
# dt_crit_comb= get_dt_crit(GSM, m_combined,fixed_dofs)
dt_crit_lump= get_dt_crit(GSM, M_lumped,fixed_dofs)
Tcons = Tcrit_ratio * dt_crit_cons
# Tcomb = Tcrit_ratio * dt_crit_comb
Tlump = Tcrit_ratio * dt_crit_lump
print("Delta t crit lumped:", dt_crit_lump)
print("Delta t crit consistent:", dt_crit_cons)


total_mass_fem, mass_expected = check_mass_conservation(Minv, density, Volume)
print('Total Mass: ', total_mass_fem * 1000, 'kg')
print('Expected Mass: ', mass_expected * 1000, 'kg')
#-----------------------------------------------------------------------
def force_fn(t):
    return get_force_vec(t, nodes, numEle_radius, get_load_matrix, square_force)  #so when run_explicit_solver function is run, the correct time gets parsed as this function is linked
displacement_history_hrz, KE_lump, SE_lump, W_lump, E_exact_lump,Wrect_lump, P_lump,imp_lump = run_explicit_solver(M_lumped,Minv, GSM, Tlump, Tend, force_fn, DoF, fixed_dofs=fixed_dofs)
displacement_history_consistent, KE_cons, SE_cons, W_cons,E_exact_cons,Wrect_cons,P_cons,imp_cons = run_explicit_solver(M_consistent,m_consis_inv, GSM, Tcons, Tend, force_fn, DoF, fixed_dofs=fixed_dofs)
#displacement_history_combined,KE_comb, SE_comb, W_comb,E_exact_comb,Wrect_comb,P_comb,imp_comb = run_explicit_solver(m_combined,m_comb_inv, GSM, Tcomb, Tend, force_fn, DoF, fixed_dofs=fixed_dofs)
print(len(displacement_history_hrz), len(displacement_history_consistent))#, len(displacement_history_combined))

runs = {"Consistent": (displacement_history_consistent, Tcons),
        # "Combined":   (displacement_history_combined,   Tcomb),
        "Lumped":     (displacement_history_hrz,        Tlump)}


plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm",
    "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "pdf.fonttype": 42,
})

A, Tp = 100.0, 5e-6
t_arr = station[1] / c0                       # 1D arrival time at the gauge

# ideal 1D pulse at the gauge
tt = np.linspace(0, Tend, 4000)
ideal = ((tt >= t_arr) & (tt < t_arr + Tp)).astype(float)

# panel order: lumped left, consistent right
panels = {"Lumped": (displacement_history_hrz, Tlump, "C0"),
          "Consistent": (displacement_history_consistent, Tcons, "C1")}

fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8), sharey=True)

for ax, (name, (hist, dt, colour)), tag in zip(axes, panels.items(),
                                               ["(a)", "(b)"]):
    hist = np.asarray(hist)
    t = np.arange(len(hist)) * dt
    _, stress, _, _ = get_strain_stress_history(
        hist, elements, nodes, properties, station[0], station[1])
    s = stress[:, 1] / A
    s = s * np.sign(s[np.argmax(np.abs(s))])  # flip so the plateau is +1

    ax.plot(tt*1e6, ideal, "k--", lw=0.9, label="Ideal 1D")
    ax.plot(t*1e6, s, color=colour, lw=0.9, label=name)

    ax.set_xlabel(r"Time [$\mu$s]")
    ax.set_title(f"{tag} {name} mass", fontsize=9, loc="left")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", frameon=False)

    # numbers for the text
    t5 = t[np.where(np.abs(s) > 0.05)[0][0]]   # first signal above 5%
    t50 = t[np.where(s > 0.5)[0][0]]           # 50% crossing of the front
    print(f"{name:10s} first signal {(t5-t_arr)*1e6:+.3f} us | "
          f"front {(t50-t_arr)*1e6:+.3f} us | "
          f"overshoot {(s.max()-1)*100:.1f}%")

axes[0].set_ylabel(r"$\sigma_{zz}\,/\,A$")
fig.tight_layout()

out = Path(fea_core.__file__).resolve().parent / "script_plots"
out.mkdir(exist_ok=True)                # save next to this script
fig.savefig(out / "square_pulse.pdf", bbox_inches="tight")
fig.savefig(out / "square_pulse.png", dpi=200, bbox_inches="tight")
print("Figures saved to:", out)
plt.show()
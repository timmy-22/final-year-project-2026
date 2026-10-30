from fea_core import (
    create_mesh, gaussPointScheme, partialOfShapeFn, partial, Shape,
    get_element_coords, get_element_dof_indices, get_element_D_matrix,
    compute_B_and_jacobian, StressFunc, get_load_matrix, q_function,
    get_force_vec, build_mass_matrices, combined_mass_matrices, run_explicit_solver, get_dt_crit,
    plot_mesh, plot_stress_field, plot_mesh_and_force, compute_stresses, solve_global_eqn, 
    build_GSM, find_nearest_element, get_strain_stress_history,
    plot_strain_history_stacked, plot_gws_input, check_mass_conservation, get_axis_dofs
)
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
import sympy as sp

#pip install numpy, sympy, matplotlib, scipy
#----------------------------------------------------------------------
# CONFIGURATION / INPUTS
#----------------------------------------------------------------------
#pick gauss scheme (1,2,3)
Scheme = 3
ScaleFactor = 20000
beta = 0.5  #beta for combined mass matrix using form (1-beta)*Lumped + beta*consistent
#Tstep = 3.2087e-8   #for 1mm square elements need step size less than: 6.616591513240334e-08 (if using default material properties)
Tend = 8e-6
Tcrit_ratio = 0.9
station = (0.0, 2.5)  #coordinate to analyse 
#-----------------------------------------------------------------------------
bar_radius, bar_length = 1,4            #radius and length of bar [mm]
numEle_radius, numEle_length = 4,10       #number of elements in radius and length

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

# #-------------------------EQUIVALENT NODAL LOADING VERIFICATION------------------------------------------
# strain = 0.001      #setting up displacements for constant strain case to check Equivalent Nodal Loading
# for i in range(int(DoF/2)):
#     z_i = nodes[i, 1]     
#     r_i = nodes[i, 0]     
#     Displacements[2*i]   = -properties[0,1] * strain * r_i  
#     knownDisplacements[2*i] = True
#     Displacements[2*i+1] =  strain * z_i 
#     knownDisplacements[2*i+1] = True
# #--------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------
# EXPLICIT SCHEME CONFIGURATION
#----------------------------------------------------------------------

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
print("Minv diag range:", np.diag(Minv).min(), np.diag(Minv).max())
print("GSM range:", GSM.min(), GSM.max())

total_mass_fem, mass_expected = check_mass_conservation(Minv, density, Volume)
print('Total Mass: ', total_mass_fem * 1000, 'kg')
print('Expected Mass: ', mass_expected * 1000, 'kg')
#-----------------------------------------------------------------------

def force_fn(t):
    return get_force_vec(t, nodes, numEle_radius, get_load_matrix, q_function)  #so when run_explicit_solver function is run, the correct time gets parsed as this function is linked
displacement_history_hrz, *_ = run_explicit_solver(M_lumped, Minv, GSM, Tlump, Tend, force_fn, DoF)
displacement_history_consistent,*_ = run_explicit_solver(M_consistent, m_consis_inv, GSM, Tcons, Tend, force_fn, DoF)
displacement_history_combined,*_ = run_explicit_solver(m_combined, m_comb_inv, GSM, Tcomb, Tend, force_fn, DoF)
print(len(displacement_history_hrz), len(displacement_history_consistent), len(displacement_history_combined))

# StressHistory = []
# for d in displacement_history_hrz:
#     _, StressesList_t = compute_stresses(nodes, elements, properties, d, Scheme, points, weight)
#     StressHistory.append(StressesList_t)
# StressHistory = np.array(StressHistory)

history_arr_hrz = np.asarray(displacement_history_hrz)  # Convert to numpy array
history_arr_consistent = np.asarray(displacement_history_consistent)
history_arr_combined = np.asarray(displacement_history_combined)
time_arr = np.arange(history_arr_hrz.shape[0]) * Tlump    
# node = 7
# r_dof = 2*(node-1)

# print(history_arr[:, r_dof])
#----------------------------------------------------------------------------

print("Proccessed Vector =\n", Fmod)
print("deflection =\n", Deflection)

fig, ax = plt.subplots(figsize=(8, 6))

plot_mesh(nodes, elements, ax, color='green', style='--', label='initial position')
plot_mesh(NewNodes, elements, ax, color='red', style='-', label='displaced position')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('MEC4128Z - Axial Symmetric Code')
ax.legend(loc='lower right')
ax.set_aspect('equal')
ax.grid(False)
plt.show()

# plt.plot(centroids, masses, 'o-')
# plt.xlabel('Element Radial Position (mm)')
# plt.ylabel('Element Lumped Mass (tonnes)')
# plt.title('Lumped Mass vs. Radial Position')
# plt.show()

plot_gws_input(q_function, 20, ax=None, r=0.0)
plt.show()

# plt.plot(time_arr, von_mises)
# plt.xlabel('Time (s)')
# plt.ylabel('von Mises Stress (MPa)')
# plt.title(f'von Mises Stress at Element {elem}, GP {gp} vs. Time')
# plt.show()


# centerline_elems = np.argsort(z_centroids)

# gp = 0
# sxx = StressHistory[:, centerline_elems, gp, 0]
# szz = StressHistory[:, centerline_elems, gp, 1]
# txz = StressHistory[:, centerline_elems, gp, 2]
# shh = StressHistory[:, centerline_elems, gp, 3]

# vm_history = np.sqrt(0.5*((sxx-szz)**2 + (szz-shh)**2 + (shh-sxx)**2) + 3*txz**2)
# vm_history = StressHistory[:, centerline_elems, 3]
# z_sorted = z_centroids[centerline_elems]
# vm_history = StressHistory[:, centerline_elems, 3]

# plt.figure(figsize=(8, 6))
# plt.imshow(vm_history, aspect='auto', origin='lower',
#            extent=[z_sorted.min(), z_sorted.max(), time_arr.min(), time_arr.max()],
#            cmap='viridis')
# plt.colorbar(label='von Mises Stress (MPa)')
# plt.xlabel('Axial Position z (mm)')
# plt.ylabel('Time (s)')
# plt.title('Stress Wave Propagation: Space-Time Diagram')
# plt.show()
#stations = [(0.0, 25.0), (0.0, 40.0)]   ## (r, z) pairs — pick your gauge locations
# plot_strain_history_stacked(displacement_history, elements, nodes, properties,
#                               stations, Tstep, t_max_us=600, component=1)    
# #component number 0,2,3 give radial, shear, and hoop respectively where 1 gives axial strain
# plt.show()

strain_lumped, _, _, _ = get_strain_stress_history(history_arr_hrz, elements, nodes, properties, *station)
strain_consistent, _, _, _ = get_strain_stress_history(history_arr_consistent, elements, nodes, properties, *station)
strain_combined, _, _, _ = get_strain_stress_history(history_arr_combined, elements, nodes, properties, *station)

n = min(len(strain_lumped), len(strain_consistent))
time_arr_us = np.arange(n) * Tlump * 1e6

# fig, ax = plt.subplots(figsize=(9,5))
fig, axes = plt.subplots(3, 1, figsize=(9,8), sharex=True)
for ax, hist, label in zip(axes, [history_arr_hrz, history_arr_consistent, history_arr_combined],
                             ['Lumped', 'Consistent', 'Combined']):
    strain, _, _, _ = get_strain_stress_history(hist, elements, nodes, properties, 0.0, 25.0)
    ax.plot(np.arange(len(strain))*Tlump*1e6, strain[:,1])
    ax.set_ylabel(label)
plt.tight_layout()
fig.suptitle(f'Location={station[1]}mm, r={bar_radius}, z={bar_length}, mesh={numEle_radius}x{numEle_length}, dt={Tlump}')
# ax.plot(time_arr_us, strain_lumped[:n,1], label='Lumped (HRZ) mass', lw=1.0)
# ax.plot(time_arr_us, strain_consistent[:n,1], label='Consistent mass', lw=1.0, linestyle='--')
# ax.plot(time_arr_us, strain_combined[:n,1], label= f"Combined mass(beta = {beta})", lw=1.0, linestyle='-.')
# ax.set_xlabel('Time (us)')
# ax.set_ylabel('Axial strain')
# ax.set_title(f'Lumped vs consistent mass at z={station[1]}mm')
# ax.legend()
plt.show()


print("max |d| HRZ:       ", max(np.abs(d).max() for d in displacement_history_hrz))
print("max |d| consistent:", max(np.abs(d).max() for d in displacement_history_consistent))

ax.plot(time_arr_us, strain_lumped[:n,1], label='Lumped (HRZ) mass', lw=1.0)
ax.plot(time_arr_us, strain_consistent[:n,1], label='Consistent mass', lw=1.0, linestyle='--')
ax.plot(time_arr_us, strain_combined[:n,1], label= f"Combined mass(beta = {beta})", lw=1.0, linestyle='-.')
ax.set_xlabel('Time (us)')
ax.set_ylabel('Axial strain')
ax.set_title(f'Lumped vs consistent mass at z={station[1]}mm')
ax.legend()
plt.show()

# print("Are Minv and m_consis_inv the same object?", Minv is m_consis_inv)
# print("Are they numerically identical?", np.allclose(Minv, m_consis_inv))
# print("Max abs difference between the two Minv matrices:", np.abs(Minv - m_consis_inv).max())

# diff_hrz_consistent = np.abs(history_arr_hrz - history_arr_consistent).max()
# print("Max abs difference in displacement histories:", diff_hrz_consistent)
# diff = np.abs(np.array(displacement_history_hrz) - np.array(displacement_history_consistent)).max()
# print(diff)


# dt_crit_lumped = get_dt_crit(GSM, M_lumped)
# dt_crit_consistent = get_dt_crit(GSM, M_consistent)
# dt_crit_combined = get_dt_crit(GSM, m_comb_inv)

# print(dt_crit_lumped, dt_crit_consistent, dt_crit_combined)
print("min Consistent mass step size:",dt_crit_cons)
print("min Lumped mass step size:",dt_crit_lump)
# print("min lumped nodal mass step:", min(masses))


import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import importlib, fea_core
importlib.reload(fea_core)
from fea_core import (
    create_mesh, gaussPointScheme, get_element_D_matrix,
    build_GSM, solve_global_eqn, compute_stresses, plot_mesh
)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import LogNorm

#----------------------------------------------------------------------
# CONFIGURATION / INPUTS
#----------------------------------------------------------------------
Scheme = 3
bar_radius, bar_length = 2, 2              # small square patch [mm]
numEle_radius, numEle_length = 4, 4        # 16 elements, 9 interior nodes
distortion = 0.3                            # max interior node shift as fraction of element size
seed = 1                                    # fixed seed so the distorted mesh is repeatable

# constant strain field: u_r = a*r, u_z = b*z + c  ->  eps_r = eps_theta = a, eps_z = b, gamma_rz = 0
# (a shear term u_z = c*r is NOT used: in axisymmetry tau_rz/r needs a body force to balance it,
#  so a constant tau_rz is not a valid patch-test field)
a = 0.5e-3
b = 1.0e-3
c = 0.01                                    # axial rigid-body translation [mm]

#----------------------------------------------------------------------
# MESH + DISTORTION
#----------------------------------------------------------------------
nodes, elements = create_mesh(bar_radius, bar_length, numEle_radius, numEle_length)
DoF = 2*len(nodes)
h = bar_radius/numEle_radius
l = bar_length/numEle_length

tol = 1e-12
is_boundary = ((np.abs(nodes[:, 0]) < tol) | (np.abs(nodes[:, 0] - bar_radius) < tol) |
               (np.abs(nodes[:, 1]) < tol) | (np.abs(nodes[:, 1] - bar_length) < tol))
interior = ~is_boundary

rng = np.random.default_rng(seed)
shift = rng.uniform(-distortion, distortion, size=(interior.sum(), 2)) * np.array([h, l])
nodes[interior] += shift        # only interior nodes move, so the outer boundary stays a rectangle

properties = np.tile(np.array([2.00e5, 0.3, 1]), (len(elements), 1))

#----------------------------------------------------------------------
# EXACT DISPLACEMENTS ON BOUNDARY, SOLVE FOR INTERIOR
#----------------------------------------------------------------------
Displacements = np.zeros(DoF)
Displacements[0::2] = a * nodes[:, 0]
Displacements[1::2] = b * nodes[:, 1] + c

knownDisplacements = np.zeros(DoF, dtype=bool)
knownDisplacements[0::2] = is_boundary
knownDisplacements[1::2] = is_boundary

# no external load on interior nodes. solve_global_eqn zeroes the prescribed columns of K but does not
# move their contribution to the RHS, so for NON-zero prescribed displacements it has to be done here:
# F_free = 0 - K_free,prescribed @ u_prescribed
u_known = np.where(knownDisplacements, Displacements, 0.0)
forces = np.zeros(DoF)

points, weight = gaussPointScheme(Scheme)
GSM = build_GSM(nodes, elements, properties, DoF, Scheme, points, weight)
forces = forces - GSM @ u_known
Deflection, F_reaction, Fmod = solve_global_eqn(GSM, forces, Displacements, knownDisplacements, DoF)

disp_err = np.abs(Deflection[:, 0] - Displacements)[~knownDisplacements].max() / np.abs(Displacements).max()
print(f"Max interior displacement error (relative): {disp_err:.2e}")

#----------------------------------------------------------------------
# STRESS ERROR PER ELEMENT
#----------------------------------------------------------------------
_, StressesList = compute_stresses(nodes, elements, properties, Deflection, Scheme, points, weight)

strain_exact = np.array([a, b, 0.0, a])     # [eps_r, eps_z, gamma_rz, eps_theta]
sigma_exact = get_element_D_matrix(properties, 0) @ strain_exact
print("Exact stress [sigma_r, sigma_z, tau_rz, sigma_theta] (MPa):", sigma_exact)

# norm-based relative error at every Gauss point (tau_rz exact = 0, so a per-component ratio is undefined),
# then the worst Gauss point in each element
gp_err = np.linalg.norm(StressesList - sigma_exact, axis=2) / np.linalg.norm(sigma_exact)
elem_err = gp_err.max(axis=1)
elem_err_plot = np.maximum(elem_err, 1e-17)     # floor so an exact zero can sit on the log scale
print(f"Element stress error: max = {elem_err.max():.2e}, mean = {elem_err.mean():.2e}")

#----------------------------------------------------------------------
#FIGURE fig:patchmesh  (methodology) - distorted patch
#----------------------------------------------------------------------
out_dir = os.path.dirname(os.path.abspath(__file__))

fig1, ax1 = plt.subplots(figsize=(4.5, 4.5))
plot_mesh(nodes, elements, ax1, color='0.4', style='-', label='_nolegend_')
ax1.plot(nodes[is_boundary, 0], nodes[is_boundary, 1], 's', color='k', markersize=6,
         label='Boundary node (prescribed)')
ax1.plot(nodes[interior, 0], nodes[interior, 1], 'o', mfc='white', mec='tab:red', mew=1.5,
         markersize=7, label='Interior node (solved)')
ax1.axvline(0, color='k', lw=0.8, ls='-.')           # axis of symmetry
ax1.text(0.03, bar_length*1.04, 'centerline/axis of sym', fontsize=8, va='bottom')
ax1.set_aspect('equal')
ax1.set_xlabel('$r$ [mm]')
ax1.set_ylabel('$z$ [mm]')
ax1.set_ylim(-0.15, bar_length*1.12)
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=8, frameon=False)
fig1.tight_layout()
fig1.savefig(os.path.join(out_dir, 'patch_mesh.pdf'), format='pdf', bbox_inches='tight')

#----------------------------------------------------------------------
#FIGURE fig:patcherr  (verification) - element relative stress error
#----------------------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(5, 4.5))
polys = [nodes[elements[e]] for e in range(len(elements))]
vmin = 10**np.floor(np.log10(elem_err_plot.min()))
vmax = 10**np.ceil(np.log10(elem_err_plot.max()))
pc = PolyCollection(polys, array=elem_err_plot, cmap='viridis', norm=LogNorm(vmin=vmin, vmax=vmax),
                    edgecolors='k', linewidths=0.6)
ax2.add_collection(pc)
ax2.autoscale()
ax2.set_aspect('equal')
ax2.set_xlabel('$r$ [mm]')
ax2.set_ylabel('$z$ [mm]')
cbar = fig2.colorbar(pc, ax=ax2, fraction=0.046, pad=0.04)
cbar.set_label(r'$\|\sigma - \sigma_\mathrm{exact}\| \, / \, \|\sigma_\mathrm{exact}\|$')
fig2.tight_layout()
fig2.savefig(os.path.join(out_dir, 'patch_error.pdf'), format='pdf', bbox_inches='tight')

print("Saved: patch_mesh.pdf, patch_error.pdf in", out_dir)
plt.show()
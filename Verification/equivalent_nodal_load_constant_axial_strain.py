import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from fea_core import (
    create_mesh, gaussPointScheme, build_GSM, analytical_z0_loads
)

#create necessary variables/vectors/arrays to check equivalent nodal load code
bar_radius, bar_length = 2, 10
numEle_radius, numEle_length = 5, 20
h = bar_radius / numEle_radius

nodes, elements = create_mesh(bar_radius, bar_length, numEle_radius, numEle_length)
DoF = len(nodes) * 2
properties = np.tile(np.array([2.00e5, 0.3, 1]), (len(elements), 1))
E = properties[0, 0]

points, weight = gaussPointScheme(3)
GSM = build_GSM(nodes, elements, properties, DoF, 3, points, weight)

# ---prescribe constant-strain field everywhere---
strain_test = 0.001
sigma_z = E * strain_test          # axial stress from the uniaxial field
Displacements = np.zeros(DoF)
for i in range(int(DoF/2)):
    z_i, r_i = nodes[i, 1], nodes[i, 0]
    Displacements[2*i]   = -properties[0, 1] * strain_test * r_i
    Displacements[2*i+1] =  strain_test * z_i

F_check = GSM @ Displacements      # reaction force at every DOF

F_analytical = analytical_z0_loads(nodes, numEle_radius, h, sigma_z)
total_analytical = sigma_z * np.pi * bar_radius**2

# ---node numbers on each face---
# nodes are numbered row by row from z=0, (numEle_radius+1) nodes per row
nodes_per_row = numEle_radius + 1
loaded_face = list(range(0, nodes_per_row))                      # z = 0
far_face    = list(range(len(nodes) - nodes_per_row, len(nodes)))  # z = L
other_nodes = [i for i in range(len(nodes))
               if i not in loaded_face and i not in far_face]

#----------------------------------------------------------------------
# 1) LOADED FACE (z = 0): axial reactions vs consistent loads
#----------------------------------------------------------------------
F_fea_z0 = np.array([abs(F_check[2*i+1]) for i in loaded_face])
rel_err_z0 = np.abs(F_fea_z0 - F_analytical) / F_analytical * 100

print("LOADED FACE (z = 0)")
print(f"{'Node':<6}{'r (mm)':<10}{'FEM (N)':<15}{'Analytical (N)':<18}{'Error (%)':<12}")
for k, i in enumerate(loaded_face):
    print(f"{i:<6}{nodes[i,0]:<10.2f}{F_fea_z0[k]:<15.4f}"
          f"{F_analytical[k]:<18.4f}{rel_err_z0[k]:<12.2e}")

total_z0 = F_fea_z0.sum()
print(f"Total force: FEM = {total_z0:.6f} N, Analytical = {total_analytical:.6f} N, "
      f"Error = {abs(total_z0-total_analytical)/total_analytical*100:.6e} %")

#----------------------------------------------------------------------
# 2) FAR FACE (z = L): same loads, opposite direction
#----------------------------------------------------------------------
# radial position on the far face matches the loaded face node for node,
# so the same analytical loads apply
F_fea_zL = np.array([abs(F_check[2*i+1]) for i in far_face])
rel_err_zL = np.abs(F_fea_zL - F_analytical) / F_analytical * 100

print("\nFAR FACE (z = L)")
print(f"{'Node':<6}{'r (mm)':<10}{'FEM (N)':<15}{'Analytical (N)':<18}{'Error (%)':<12}")
for k, i in enumerate(far_face):
    print(f"{i:<6}{nodes[i,0]:<10.2f}{F_fea_zL[k]:<15.4f}"
          f"{F_analytical[k]:<18.4f}{rel_err_zL[k]:<12.2e}")

total_zL = F_fea_zL.sum()
print(f"Total force: FEM = {total_zL:.6f} N, Analytical = {total_analytical:.6f} N, "
      f"Error = {abs(total_zL-total_analytical)/total_analytical*100:.6e} %")

# check the two faces push in opposite directions
sum_z0 = sum(F_check[2*i+1] for i in loaded_face)
sum_zL = sum(F_check[2*i+1] for i in far_face)
print(f"Signed totals: z=0 -> {sum_z0:.6f} N, z=L -> {sum_zL:.6f} N")

#----------------------------------------------------------------------
# 3) ALL OTHER NODES: reactions should be zero
#----------------------------------------------------------------------
# interior and lateral-surface nodes, both directions
other_forces = []
for i in other_nodes:
    other_forces.append(F_check[2*i])      # radial
    other_forces.append(F_check[2*i+1])    # axial
max_other = np.max(np.abs(other_forces))

# radial reactions on the two end faces should also be zero
end_radial = [F_check[2*i] for i in loaded_face + far_face]
max_end_radial = np.max(np.abs(end_radial))

print("\nZERO-REACTION CHECK (relative to total end force)")
print(f"Max |reaction| at interior/lateral nodes: {max_other:.3e} N "
      f"({max_other/total_analytical*100:.3e} %)")
print(f"Max |radial reaction| on end faces:       {max_end_radial:.3e} N "
      f"({max_end_radial/total_analytical*100:.3e} %)")
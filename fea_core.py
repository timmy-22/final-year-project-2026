import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import matplotlib.patches as patches
import sympy as sp
from scipy.linalg import eigh

def create_mesh(radius,length,el_radius,el_length):
    h = radius/el_radius                #element width
    l = length/el_length     

    index = 0           #element height
    nodes = np.zeros(((el_length+1)*(el_radius+1), 2))
    for i in range(el_length+1):
        z = i*l       #changes z position based on row        
        for j in range(el_radius+1):
            r = j*h
            nodes[index] = [r,z]
            index += 1
    DoF = len(nodes) * 2

    index = 0
    elements = np.zeros((el_length*el_radius, 4))
    for k in range(el_length):
        for p in range(el_radius):
            elements[index] = [p+k*(el_radius+1),p+1+k*(el_radius+1),(el_radius+1)*(1+k)+p+1,(el_radius+1)*(1+k)+p]
            index += 1
    return nodes, elements.astype(int)

#FORCE (STARTING AT CENTERLINE OF BAR AND MOVING RADIALLY OUTWARDS)
def  q_function(r_position,time):
#     r, t = sp.symbols('r t')
#     # F_expr = 50*sp.sin(200*t) * (1 - r/10)**2 + 20*sp.exp(-t/1e-4)*r    #input time and position varying force function
#     fs = 200e4       # carrier frequency, Hz 
#     T = 1.0 / fs
#     n_cycles = 12
#     half_window = (n_cycles / 2) * T
#     t0 = 25e-6       # window center, s — ~5 periods in, so the pulse ramps up smoothly from ~0
#     t0 = half_window
#     kappa = 1.1e7    # Gaussian envelope width, tuned so the pulse spans ~10 cycles (t=0 to t=2*t0)
#     decay_fraction = 0.02   #% decay as fraction of peak
#     kappa = -np.log(decay_fraction) / t0**2
#     A = 100          # amplitude, MPa (traction) 

#     F_expr = A * sp.sin(2*sp.pi*fs*(t - t0)) * sp.exp(-kappa*(t - t0)**2)
#     # F_expr = 0
# # spatially uniform — no r term at all, matching a real striker impact
#     q_func = sp.lambdify((r, t), F_expr, 'numpy')  #Convert to real python(Numpy) function
#     return q_func(r_position,time)

    """
    Axial traction [MPa] on the z=0 face: a Gaussian-windowed sine tone burst,
    spatially uniform in r.
      A              peak amplitude [MPa]
      fs             carrier frequency [Hz]
      n_cycles       number of carrier cycles in the burst (integer)
      edge_fraction  envelope value at the start and end of the burst,
                     as a fraction of its peak
    """
    fs = 500e3
    A = 100
    n_cycles = 4
    edge_fraction = 0.02
    t0 = 0.5 * n_cycles / fs                     # burst centre; burst spans t = 0 to 2*t0
    kappa = -np.log(edge_fraction) / t0**2       # fixes the envelope width
    pulse = A * np.sin(2*np.pi*fs*(time - t0)) * np.exp(-kappa*(time - t0)**2)
    pulse = np.where((time >= 0) & (time <= 2*t0), pulse, 0.0)   # finite duration
    return pulse * np.ones_like(r_position, dtype=float) 
    

def get_load_matrix(r1, r2):
    h = r2 - r1
    return (np.pi*h/6) * np.array([
        [4*r1+h,   2*r1+h  ],
        [2*r1+h,   4*r1+3*h]
    ])
    

def gaussPointScheme(pointscheme):
    if pointscheme == 1: 
        points = np.array([0])
        weight = np.array([2])
    elif pointscheme == 2:
        points = np.array([np.sqrt(1/3),-np.sqrt(1/3)])
        weight = np.array([1,1])
    else:
        points = np.array([0,np.sqrt(3/5),-np.sqrt(3/5)])
        weight = np.array([8/9,5/9,5/9])              
    return(points,weight)

def partialOfShapeFn(Xi,Eta):
    NPartial = np.array([
    [-1/4 *(1-Eta), 0, 1/4 *(1-Eta), 0, 1/4 *(1+Eta), 0, -1/4 *(1+Eta), 0],
    [-1/4 *(1-Xi), 0, -1/4 *(1+Xi), 0, 1/4 *(1+Xi), 0, 1/4 *(1-Xi), 0],
    [0, -1/4*(1-Eta), 0, 1/4*(1-Eta), 0, 1/4*(1+Eta), 0, -1/4*(1+Eta)],
    [0, -1/4 *(1-Xi), 0, -1/4 *(1+Xi), 0, 1/4 *(1+Xi), 0, 1/4 *(1-Xi)]                        
    ])
    return NPartial

def partial(Xi,Eta):
    return np.array([
        [-1/4 *(1-Eta), 1/4 *(1-Eta), 1/4 *(1+Eta), -1/4 *(1+Eta)],
        [-1/4 *(1-Xi), -1/4 *(1+Xi), 1/4 *(1+Xi), 1/4 *(1-Xi)]
        ])

def Shape(Xi,Eta):
    N1 = 0.25*(1-Xi)*(1-Eta)
    N2 = 0.25*(1+Xi)*(1-Eta)
    N3 = 0.25*(1+Xi)*(1+Eta)
    N4 = 0.25*(1-Xi)*(1+Eta)
    return N1, N2, N3, N4

def get_element_coords(nodes, elements, i):
    """4x2 [r, z] coords for element i's 4 nodes."""
    n1, n2, n3, n4 = elements[i]
    r1, z1 = nodes[n1]
    r2, z2 = nodes[n2]
    r3, z3 = nodes[n3]
    r4, z4 = nodes[n4]
    return np.array([[r1, z1], [r2, z2], [r3, z3], [r4, z4]])

def get_element_dof_indices(elements, i):
    """8 global DoF indices for element i (r,z pair per node)."""
    n1, n2, n3, n4 = elements[i]
    return np.array([2*n1,2*n1+1, 2*n2,2*n2+1, 2*n3,2*n3+1, 2*n4,2*n4+1])

def get_element_D_matrix(properties, i):
    """Axisymmetric D matrix, ordered [sigma_r, sigma_z, tau_rz, sigma_theta]."""
    v = properties[i,1]
    E_mod = properties[i,0]
    f = v/(1-v)
    g = (1-2*v)/(2*(1-v))
    return (((1-v)*E_mod)/((1-v*2)*(1+v))) * np.array([
        [1, f, 0, f],
        [f, 1, 0, f],
        [0, 0, g, 0],
        [f, f, 0, 1]
    ])

def compute_B_and_jacobian(Xi, Eta, coords):
    zeros = np.zeros((2,2)) # 2x2 of zeros to build up Gamma matrix
    H = np.array([
    [1, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 1, 1, 0]
    ])
    r1, r2, r3, r4 = coords[0,0], coords[1,0], coords[2,0], coords[3,0]
    N1,N2,N3,N4 = Shape(Xi,Eta)
    r = r1*N1 + r2*N2 + r3*N3 + r4*N4
    B4 = 1/r * np.array([N1,0,N2,0,N3,0,N4,0])

    ShapeFnPartial = partialOfShapeFn(Xi,Eta)
    J = partial(Xi,Eta) @ coords
    Jinv = np.linalg.inv(J)
    Jdet = np.linalg.det(J)
    Gamma = np.block([
        [Jinv, zeros],
        [zeros, Jinv]
    ])

    B = H @ Gamma @ ShapeFnPartial
    B = np.vstack([B, B4])
    return B, Jdet, r

def build_mass_matrices(nodes, elements, properties, density, DoF, Scheme, points, weight):
    Volume = 0.0
    M_lumped = np.zeros((DoF, DoF))
    M_consistent = np.zeros((DoF, DoF))
    centroids, z_centroids, masses = [], [], []
    
    for i in range(len(elements)):
        coords = get_element_coords(nodes, elements, i)
        z_avg = coords[:, 1].mean()
        z_centroids.append(z_avg)
        Me = np.zeros((8, 8))
        Ve = 0
        r_el = 0
        for j in range(Scheme):
            for k in range(Scheme):
                # coords = get_element_coords(nodes, elements, j)
                _, Jdet, r = compute_B_and_jacobian(points[j], points[k], coords)
                Volume += weight[j] * weight[k] * 2 * np.pi * r * Jdet
                Ve += weight[j] * weight[k] * 2 * np.pi * r * Jdet
                N1, N2, N3, N4 = Shape(points[j], points[k])
                N = np.zeros((2,8))
                N[0,0] = N1
                N[1,1] = N1
                N[0,2] = N2
                N[1,3] = N2
                N[0,4] = N3
                N[1,5] = N3
                N[0,6] = N4
                N[1,7] = N4
                Me += weight[j] * weight[k] * density * (N.T @ N) * 2*np.pi*r * Jdet   
            
                r_el += r

        MIndex = get_element_dof_indices(elements, i)   #gets correct index of current element to build mass matrix that has same indexing as stiffness matrix
        for row in range(8):
            for col in range(8):
                M_consistent[MIndex[row], MIndex[col]] += Me[row, col]         
        r_avg = r_el/(Scheme**2)
        centroids.append(r_avg)
        # HRZ lump: keep diagonal, rescale to preserve element mass
        Me_lumped_diag = np.diag(Me).copy() # returns flat 1D array of diagonal elements

        raw_diag = np.diag(Me)  #since the mass matrix is 8x8 due to r and z DOF per node, N1,N2,N3,N4 are all doubled as they appear twice in the mass matrix

        diag_r = raw_diag[0::2] # [D1_r, D2_r, D3_r, D4_r]
        diag_z = raw_diag[1::2] # [D1_z, D2_z, D3_z, D4_z]


        trueElementMass = density * Ve 
  
        scale_r = trueElementMass / np.sum(diag_r)
        scale_z = trueElementMass / np.sum(diag_z)


        Me_lumped_diag = np.zeros(8)
        Me_lumped_diag[0::2] = diag_r * scale_r
        Me_lumped_diag[1::2] = diag_z * scale_z


        Me_lumped = np.diag(Me_lumped_diag) 
        totalElementMass = Me_lumped_diag.sum()          
        element_mass = Me_lumped_diag.sum() / 2 # divided by 2 cause r and z DOF actually duplicate each nodes mass as they have mass in more than 1 "direction"
        masses.append(element_mass)
        #print('Centroids: ', centroids[i], '\n Element Volume: ', Ve, '\n masses: ', masses[i], '\n Masses Calculation ', density * Ve , '\n')

        MIndex = get_element_dof_indices(elements, i)   # same fn, reused as-is
        for row in range(8):
            for col in range(8):
                # M_global[MIndex[row], MIndex[col]] = M_global[MIndex[row], MIndex[col]] + Me_lumped[row,col]
                if row == col:  #for lum
                    M_lumped[MIndex[row], MIndex[col]] += Me_lumped[row, col]

    mass_total = Volume * density
    return M_lumped, M_consistent, Volume, mass_total, centroids, z_centroids, masses


def combined_mass_matrices(M_lumped, M_consistent, beta):
    return (1 - beta) * M_lumped + beta * M_consistent

def build_GSM(nodes, elements, properties, DoF, Scheme, points, weight):
    GSM = np.zeros([DoF, DoF])
    for i in range(len(elements)):
        Kel = np.zeros([8, 8])
        E = get_element_D_matrix(properties, i)
        coords = get_element_coords(nodes, elements, i)

        for j in range(Scheme):
            for k in range(Scheme):
                B, Jdet, r = compute_B_and_jacobian(points[j], points[k], coords)
                Kel += weight[j]*weight[k]*np.transpose(B) @ E @ B * 2*np.pi*r * Jdet

        GSMIndex = get_element_dof_indices(elements, i)
        for row in range(8):
            for col in range(8):
                GSM[GSMIndex[row], GSMIndex[col]] += Kel[row, col]
    return GSM

def solve_global_eqn(GSM, forces, Displacements, knownDisplacements, DoF):
    Fmod = np.zeros((DoF, 1))
    GSM_processed = GSM.copy()
    for l in range(DoF):
        if knownDisplacements[l]:
            Fmod[l] = Displacements[l]
            GSM_processed[l, :] = 0
            GSM_processed[:, l] = 0
            GSM_processed[l, l] = 1
        else:
            Fmod[l] = forces[l]

    Deflection = np.linalg.solve(GSM_processed, Fmod)
    F_reaction = GSM @ Deflection
    return Deflection, F_reaction, Fmod

def StressFunc(NodeStressArr):
    r = NodeStressArr[0]
    z = NodeStressArr[1]
    rz = NodeStressArr[2]
    hoopstress = NodeStressArr[3]

    Sigma1 = (r+z)/2 + np.sqrt(((r-z)/2)**2 +rz**2)
    Sigma2 = (r+z)/2 - np.sqrt(((r-z)/2)**2 +rz**2)
    MaxShear = np.sqrt(((r-z)/2)**2 +rz**2)

    VonMises = np.sqrt(0.5*((r-z)**2 +(z-hoopstress)**2 +(hoopstress - r)**2 + 6 * rz**2))
    return  Sigma1, Sigma2, MaxShear, VonMises, hoopstress

def plot_mesh(nodes, elements, ax, color='green', style='--', label='initial position'):
    plotted_label = False
    drawn_edges = set()
    
    for elem in elements:
        for i in range(4):
            n1 = elem[i]
            n2 = elem[(i + 1) % 4]
            edge = frozenset((n1, n2))
            if edge in drawn_edges:
                continue
            drawn_edges.add(edge)
            
            x_vals = [nodes[n1, 0], nodes[n2, 0]]
            y_vals = [nodes[n1, 1], nodes[n2, 1]]
            ax.plot(x_vals, y_vals, linestyle=style, color=color,
                     label=label if not plotted_label else "")
            plotted_label = True
    
    ax.plot(nodes[:, 0], nodes[:, 1], 'o', color=color, markersize=5)
    return ax

def compute_stresses(nodes, elements, properties, Deflections, Scheme, points, weight):
    ElemStressAvg = np.zeros((len(elements), 5))
    StressesList = np.zeros((len(elements), Scheme**2, 4))
    Def = np.zeros((8,))

    for i in range(len(elements)):
        E = get_element_D_matrix(properties, i)
        coords = get_element_coords(nodes, elements, i)

        DefIndex = get_element_dof_indices(elements, i)
        for p in range(8):
            Def[p] = Deflections[DefIndex[p], 0]

        index = 0
        for j in range(Scheme):
            for k in range(Scheme):
                B, Jdet, r = compute_B_and_jacobian(points[j], points[k], coords)
                Stress = E @ B @ Def
                StressesList[i, index, 0] = Stress[0]
                StressesList[i, index, 1] = Stress[1]
                StressesList[i, index, 2] = Stress[2]
                StressesList[i, index, 3] = Stress[3]
                index += 1

        # avg gauss stress in element
        sx_avg = StressesList[i, :, 0].mean()
        sz_avg = StressesList[i, :, 1].mean()
        txz_avg = StressesList[i, :, 2].mean()
        hoopStr_avg = StressesList[i, :, 3].mean()
        StressAvgs = np.array([sx_avg, sz_avg, txz_avg, hoopStr_avg])
        Sigma1, Sigma2, MaxShear, VonMises, hoopstress = StressFunc(StressAvgs)
        ElemStressAvg[i, 0] = Sigma1
        ElemStressAvg[i, 1] = Sigma2
        ElemStressAvg[i, 2] = MaxShear
        ElemStressAvg[i, 3] = VonMises
        ElemStressAvg[i, 4] = hoopstress

    return ElemStressAvg, StressesList

def plot_stress_field(nodes, elements, stress_values, label, cmap='jet'):
    fig, ax = plt.subplots(figsize=(6,6))
    
    polys = [nodes[elements[e]] for e in range(len(elements))]
    
    pc = PolyCollection(polys, array=stress_values, cmap=cmap, edgecolors='k', linewidths=0.5)
    ax.add_collection(pc)
    ax.autoscale()
    ax.set_aspect('equal')
    
    cbar = plt.colorbar(pc, ax=ax)
    cbar.set_label(label)
    ax.set_title(f'{label} — element averaged')
    plt.show()

def plot_fbd(nodes, elements, knownDisplacements, forces, ax=None, arrow_scale=None):   #function not needed since not plotting displacements on mesh such as in assignments
    """
    Free body diagram: mesh outline + force arrows + support symbols.
    forces: flat (DoF,) or (DoF,1) array, same layout as Fmod/force_scheme.
    knownDisplacements: flat (DoF,) bool array, same layout.
    """
    forces = np.asarray(forces).reshape(-1)
    n_nodes = len(nodes)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    plot_mesh(nodes, elements, ax, color='gray', style='-', label='mesh')

    # --- force arrows -----------------------------------------------
    # scale arrow length relative to the biggest applied force so the
    # picture stays legible regardless of the actual force magnitudes
    fx = forces[0::2]
    fy = forces[1::2]
    fmag = np.sqrt(fx**2 + fy**2)
    max_f = fmag.max() if fmag.max() > 0 else 1.0

    if arrow_scale is None:
        # aim for the longest arrow being ~15% of the model's bounding box
        span = max(np.ptp(nodes[:,0]), np.ptp(nodes[:,1]))
        arrow_scale = 0.15 * span / max_f

    for n in range(n_nodes):
        if fmag[n] == 0:
            continue
        x, y = nodes[n]
        dx, dy = fx[n]*arrow_scale, fy[n]*arrow_scale
        ax.annotate('', xy=(x+dx, y+dy), xytext=(x, y),
                    arrowprops=dict(arrowstyle='-|>', color='blue', lw=2))
        ax.text(x+dx, y+dy, f'  ({fx[n]:.0f}, {fy[n]:.0f})',
                color='blue', fontsize=8, va='center')

    # --- support symbols ----------------------------------------------
    for n in range(n_nodes):
        x, y = nodes[n]
        ux_fixed = knownDisplacements[2*n]
        uy_fixed = knownDisplacements[2*n+1]
        s = 0.03 * max(np.ptp(nodes[:,0]), np.ptp(nodes[:,1]))

        if ux_fixed and uy_fixed:
            # pin: solid triangle pointing up into the node
            tri = plt.Polygon([[x, y], [x-s, y-1.6*s], [x+s, y-1.6*s]],
                               closed=True, facecolor='none', edgecolor='black', lw=1.5)
            ax.add_patch(tri)
        elif ux_fixed:
            # roller restraining x only: vertical bar (slider is vertical)
            ax.plot([x, x], [y-1.6*s, y+1.6*s], color='black', lw=2)
            ax.plot(x, y, 'o', color='black', markersize=4)
        elif uy_fixed:
            # roller restraining y only: horizontal bar
            ax.plot([x-1.6*s, x+1.6*s], [y, y], color='black', lw=2)
            ax.plot(x, y, 'o', color='black', markersize=4)

    ax.set_aspect('equal')
    ax.set_xlabel('r')
    ax.set_ylabel('z')
    ax.set_title('Free Body Diagram: applied forces + supports')
    ax.grid(False)
    return ax

def plot_mesh_and_force(nodes, elements, q_function, numEle_radius,
                         time=0.0, arrow_scale=None, ax=None,
                         title="Mesh and applied z=0 traction"):
    """
    Plots the axisymmetric bar mesh (r vs z) with the applied input
    traction q(r, time) shown as arrows at the z=0 face.

    Assumes: force acts only at z=0, only in the z-direction (axial),
    magnitude/sign given by q_function(r, time).

    nodes         : (n_nodes, 2) array, columns = [r, z]
    elements      : (n_elem, 4) int array of node ids, CCW
    q_function    : callable q_function(r, t) -> traction value
    numEle_radius : number of elements across the radius (face node count = numEle_radius+1)
    time          : time at which to evaluate/plot the load
    arrow_scale   : manual scale for arrow length (z-units per unit traction);
                    if None, auto-scaled to ~15% of bar_length
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    for elem in elements:
        corner_r = nodes[elem, 0]
        corner_z = nodes[elem, 1]
        poly = np.column_stack([corner_z, corner_r])
        patch = patches.Polygon(poly, closed=True, fill=False,
                                 edgecolor='steelblue', linewidth=0.4)
        ax.add_patch(patch)

    ax.set_xlim(nodes[:, 1].min() - 1, nodes[:, 1].max() * 1.05)
    ax.set_ylim(-nodes[:, 0].max() * 0.15, nodes[:, 0].max() * 1.15)

    face_node_ids = np.arange(numEle_radius + 1)
    r_face = nodes[face_node_ids, 0]

    q_vals = np.array([q_function(r, time) for r in r_face])

    bar_length = nodes[:, 1].max()
    if arrow_scale is None:
        max_abs_q = np.max(np.abs(q_vals)) if np.max(np.abs(q_vals)) > 0 else 1.0
        arrow_scale = (0.15 * bar_length) / max_abs_q

    for r_i, q_i in zip(r_face, q_vals):
        dz = q_i * arrow_scale
        if abs(dz) < 1e-12:
            continue
        color = 'crimson' if q_i > 0 else 'darkgreen'
        ax.annotate('', xy=(dz, r_i), xytext=(0, r_i),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

    ax.set_xlabel('z (axial position)')
    ax.set_ylabel('r (radial position)')
    ax.set_title(f'{title}  (t = {time:g})')
    ax.set_aspect('equal', adjustable='box')
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.plot([], [], color='crimson', label='q > 0 (+z)')
    ax.plot([], [], color='darkgreen', label='q < 0 (−z)')
    ax.legend(loc='upper right', fontsize=8)

    return ax

def find_nearest_element(nodes, elements, target_r, target_z):
    centroids = np.array([nodes[elements[e]].mean(axis=0) for e in range(len(elements))])
    dists = np.sum((centroids - np.array([target_r, target_z]))**2, axis=1)
    return int(np.argmin(dists))

def get_strain_stress_history(displacement_history, elements, nodes, properties,
                                target_r, target_z, Xi=0.0, Eta=0.0):
    """B is computed ONCE since the mesh doesn't move -- cheap even for many timesteps."""
    elem_id = find_nearest_element(nodes, elements, target_r, target_z)
    coords = get_element_coords(nodes, elements, elem_id)
    B, Jdet, r_actual = compute_B_and_jacobian(Xi, Eta, coords)
    E = get_element_D_matrix(properties, elem_id)
    dof_idx = get_element_dof_indices(elements, elem_id)

    n_steps = len(displacement_history)
    strain_history = np.zeros((n_steps, 4))
    stress_history = np.zeros((n_steps, 4))
    for s, d in enumerate(displacement_history):
        Def = d[dof_idx, 0]
        strain = B @ Def
        stress_history[s] = E @ strain
        strain_history[s] = strain
    return strain_history, stress_history, elem_id, r_actual

def plot_strain_history_stacked(displacement_history, elements, nodes, properties,
                                  stations, Tstep, t_max_us=None, component=1,
                                  offset=None, ylabel='Axial strain'):
    n_steps = len(displacement_history) if t_max_us is None else min(
        int((t_max_us*2e-5)/Tstep), len(displacement_history))
    time_arr = np.arange(n_steps) * Tstep
    hist = displacement_history[:n_steps]

    traces = [get_strain_stress_history(hist, elements, nodes, properties, r_t, z_t)[0][:, component]
              for (r_t, z_t) in stations]

    if offset is None:
        offset = 1.5 * max(np.abs(t).max() for t in traces)

    fig, ax = plt.subplots(figsize=(9,5))
    for i, ((r_t,z_t), trace) in enumerate(zip(stations, traces)):
        ax.plot(time_arr*1e6, trace - i*offset, lw=1.0, label=f'z={z_t:.0f}mm, r={r_t:.1f}mm')
    ax.set_xlabel('Time (µs)')
    ax.set_ylabel(f'{ylabel} (offset per trace)')
    ax.legend(loc='upper right', fontsize=8)
    return fig, ax
#Variable force-time needs recalculation of force at every time step. This force is the equvialent nodal load not just applied force.
def get_force_vec(time,nodes,numEle_radius,get_load_matrix,q_function):
    forces_leadedge = np.zeros((numEle_radius+1,1))
    for m in range(numEle_radius):
        load_matrix = get_load_matrix(nodes[m,0], nodes[m+1,0])
        q1 = q_function(nodes[m,0], time)
        q2 = q_function(nodes[m+1,0], time)
        q = np.array([[q1],[q2]])
        f_comb = load_matrix @ q
        forces_leadedge[m]   += f_comb[0]
        forces_leadedge[m+1] += f_comb[1]

    forces = np.zeros((len(nodes),2))
    for g in range(numEle_radius+1):
        forces[g,1] = forces_leadedge[g,0]

    return forces.flatten().reshape(-1, 1)


def run_explicit_solver(MassMatrix, InverseMassMatrix, GSM, Tstep, Tend,
                        get_force_vec_fn, DoF, fixed_dofs=None):
    if fixed_dofs is None:
        free = np.arange(DoF)
        Minv_ff = InverseMassMatrix
    else:
        free = np.setdiff1d(np.arange(DoF), fixed_dofs)
        # inverse of the REDUCED mass matrix, not rows/cols of the full inverse
        Minv_ff = np.linalg.inv(MassMatrix[np.ix_(free, free)])

    #STEP 0(BOOTSTRAP) - GET d_-1. can then loop through all time steps.
    d0 = np.zeros((DoF,1))
    displacement_history = [d0]
    F0 = get_force_vec_fn(0)
    dn_prev = np.zeros((DoF,1))
    dn_prev[free] = ((Tstep**2)/2) * (Minv_ff @ F0[free])
    dn = d0

    # energy balance initialization
    nSteps = int(Tend/Tstep)
    KE = np.zeros(nSteps)
    SE = np.zeros(nSteps)
    Wext = np.zeros(nSteps)
    Wrect = np.zeros(nSteps)
    E_exact = np.zeros(nSteps)
    W = 0.0

    # momentum balance
    ez = np.zeros(DoF); ez[1::2] = 1          # axial DOFs only (r,z,r,z indexing)
    m_axial = ez @ MassMatrix
    P = np.zeros(nSteps)
    impulse =  np.zeros(nSteps)
    Imp_step = 0.0

    #LOOP
    for i in range(nSteps):
        force_scheme = get_force_vec_fn(i*Tstep)
        F_int = GSM @ dn
        rhs = force_scheme - F_int

        # update free DOFs only; fixed DOFs stay at zero
        dn_next = np.zeros((DoF,1))
        dn_next[free] = (Tstep**2 * (Minv_ff @ rhs[free])
                         + 2*dn[free] - dn_prev[free])

        velocity = (dn_next-dn_prev)/(2*Tstep)
        KE[i] = 0.5 * (velocity.T @ MassMatrix @ velocity).item()
        SE[i] = 0.5 * (dn.T @ F_int).item()
        v_half = (dn_next - dn) / Tstep                       # v at n+1/2
        E_exact[i] = (0.5 * (v_half.T @ MassMatrix @ v_half).item()
                      + 0.5 * (dn.T @ GSM @ dn_next).item())
        Pwr = (force_scheme.T @ velocity).item()  # power = force*velocity
        W += Pwr * Tstep
        Wrect[i] = W                       # work = power * time
        Wext[i] = W - 0.5 * Pwr * Tstep    # trapezoid: work up to t_i

        # momentum
        P[i] = (m_axial @ v_half).item()
        Imp_step += (ez @ force_scheme).item() * Tstep
        impulse[i] = Imp_step

        displacement_history.append(dn_next)
        dn, dn_prev = dn_next, dn
    return displacement_history, KE, SE, Wext, E_exact, Wrect, P, impulse

def plot_gws_input(q_function, t_max_us, ax=None, r=0.0):
    """Plot the raw input traction q(r=0, t) over time, for direct comparison
    against your strain-vs-time output at a downstream station."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(9,4))
    t_arr_us = np.linspace(0, t_max_us, 2000)
    q_vals = np.array([q_function(r, t*1e-6) for t in t_arr_us])
    ax.plot(t_arr_us, q_vals)
    ax.set_xlabel('Time (us)')
    ax.set_ylabel('Input traction q(0,t)')
    ax.set_title('GWS input pulse')
    return ax, t_arr_us, q_vals

def get_dt_crit(K, M, fixed_dofs=None):
    if fixed_dofs is not None:
        free = np.setdiff1d(np.arange(K.shape[0]), fixed_dofs)
        K = K[np.ix_(free, free)]
        M = M[np.ix_(free, free)]
    w2 = eigh(K, M, eigvals_only=True)
    return 2/np.sqrt(w2.max())

def check_mass_conservation(Minv, density, Volume):
    total_mass_fem = np.sum(1/np.diag(Minv)) / 2
    mass_expected = density * Volume
    return total_mass_fem, mass_expected

def analytical_z0_loads(nodes, numEle_radius, h, sigma0):
    """Consistent nodal loads at z=0 face, summed across shared edges."""
    F = np.zeros(numEle_radius+1)
    for m in range(numEle_radius):
        r1, r2 = nodes[m,0], nodes[m+1,0]
        h_e = r2 - r1
        F[m]   += sigma0*np.pi*h_e*(r1 + h_e/3)     # from analytical calculation: F_i = ∫ N_i · σ · 2πr dr  
        F[m+1] += sigma0*np.pi*h_e*(r1 + 2*h_e/3)   # the += handles elements that share nodes as contribution from both elements are added to node force
    return F
'''N₁ = 1 - ξ
N₂ = ξ
r = r₁ + h_e·ξ
F₁ = 2πσ h_e ∫₀¹ (1-ξ)(r₁ + h_e ξ) dξ = 2πσ h_e (r₁/2 + h_e/6) = πσ h_e (r₁ + h_e/3)
F₂ = 2πσ h_e ∫₀¹ ξ(r₁ + h_e ξ) dξ = 2πσ h_e (r₁/2 + h_e/3) = πσ h_e (r₁ + 2h_e/3)'''
def get_axis_dofs(nodes):
    """u_r DOFs of nodes on the axis (r = 0)."""
    return np.array([2*i for i in range(len(nodes))
                     if np.isclose(nodes[i, 0], 0.0)])
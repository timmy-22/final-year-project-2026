import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fea_core import (
    create_mesh, gaussPointScheme, get_load_matrix, get_force_vec, combined_mass_matrices,
    build_mass_matrices, run_explicit_solver, get_dt_crit, get_axis_dofs,
    build_GSM, get_strain_stress_history,find_nearest_element
)
import time as timer
import numpy as np
import matplotlib.pyplot as plt
import fea_core
import sympy as sp
import scipy
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh
from pathlib import Path
from matplotlib.ticker import NullFormatter
#----------------------------------------------------------------------
# CONFIGURATION / INPUTS
#----------------------------------------------------------------------
Scheme = 3
bar_radius, bar_length = 2, 50  # [mm]        
station = (bar_radius, 20.0)    # (r, z) to analyse [mm]
z_gauge, gauge_len = 20.0, 1.0
density = 7.850e-9              # tonne/mm^3
material = [2.00e5, 0.3, 1]
c = np.sqrt(material[0] / density)
# Tend = bar_length / c + 0.2e-5        # makes sure reflected wave doesnt distort results
Tend = 0.95 * (2 * bar_length - station[1])/c
Tcrit_ratio = 0.90   # Since Tcrit changes each step, recalculate

# fs = 200e4
beta = 0.5
numEle_radius = 2

# z_mesh = [int(np.floor(bar_length/2)), bar_length, 2*bar_length, 3*bar_length,4*bar_length,5*bar_length,6*bar_length,7*bar_length,8*bar_length,9*bar_length,10*bar_length]
# r_mesh = [int(np.floor(numEle_radius/2)),numEle_radius,2*numEle_radius,3*numEle_radius,4*numEle_radius,5*numEle_radius,6*numEle_radius,7*numEle_radius,8*numEle_radius,9*numEle_radius,10*numEle_radius]
z_mesh = [40, 80, 160, 320, 640]
r_mesh = [4, 4, 4, 4, 4]
nr_fixed = 4
f0, cycles, A = 500e3, 4, 100         # GWS: centre freq [Hz], cycles, amplitude [MPa]
#----------------------------------------------------------------------
# def q_function(r_position, time):
#     A = 100
#     f = 200e3                      # Hz
#     T = 1 / f                      # one period
#     return A * np.sin(2*np.pi*f*time) * (time <= T)   # single period forcing wave

def q_function(r_position, time):
    t0 = cycles / f0                  # centre of the window
    tau = cycles / (2 * f0)           # Gaussian width
    return A * np.sin(2*np.pi*f0*(time - t0)) * np.exp(-((time - t0)/tau)**2)
#----------------------------------------------------------------------
points, weight = gaussPointScheme(Scheme)
T = 1/2e6
t0 = 3e-6
Stress_cons_max =[]
# Stress_comb_max=[]
Stress_lump_max=[]
h_list =[]
results = {"Lumped": [], "Consistent": [], "Combined": []}   # (h, t, eps_g)

for numEle_length in z_mesh:
    numEle_radius = nr_fixed
    h = bar_length / numEle_length
    nodes, elements = create_mesh(bar_radius, bar_length, numEle_radius, numEle_length)
    properties = np.tile(np.array(material), (len(elements), 1))
    DoF = 2 * len(nodes)

    GSM = build_GSM(nodes, elements, properties, DoF, Scheme, points, weight)
    M_lumped, M_consistent, *_ = build_mass_matrices(nodes, elements, properties, density, DoF, Scheme, points, weight)
    del _
    K = csr_matrix(GSM)

    # surface nodes sorted by z, and the two gauge-end positions
    surf = np.where(np.isclose(nodes[:, 0], bar_radius))[0]
    surf = surf[np.argsort(nodes[surf, 1])]
    zs = nodes[surf, 1]
    ends = []
    for z in (z_gauge - gauge_len/2, z_gauge + gauge_len/2):
        k = np.searchsorted(zs, z) - 1
        w = (z - zs[k]) / (zs[k+1] - zs[k])
        ends.append((2*surf[k] + 1, 2*surf[k+1] + 1, w))   # u_z DOFs and weight

    def force_fn(t):
        return get_force_vec(t, nodes, numEle_radius, get_load_matrix, q_function)

    for name in results:
        if name == "Lumped":
            M = M_lumped
            Minv = diags(1.0 / M_lumped.diagonal(), format='csr')
            M_eig = diags(M_lumped.diagonal())
        elif name == "Consistent":
            M = M_consistent
            Minv = np.linalg.inv(M)
            M_eig = csr_matrix(M)
        else:
            M = combined_mass_matrices(M_lumped, M_consistent, beta)
            Minv = np.linalg.inv(M)
            M_eig = csr_matrix(M)

        fixed_dofs = get_axis_dofs(nodes)
        dt_crit = get_dt_crit(GSM, M,fixed_dofs)
        dt = Tcrit_ratio * dt_crit

        hist, *_ = run_explicit_solver(M, Minv, GSM, dt, Tend, force_fn, DoF, fixed_dofs)
        hist = np.asarray(hist).squeeze()      # (steps, DoF, 1) -> (steps, DoF)

        u = [(1 - w)*hist[:, a] + w*hist[:, b] for a, b, w in ends]   # u_z at gauge ends
        eps_g = (u[1] - u[0]) / gauge_len
        t = np.arange(hist.shape[0]) * dt
        results[name].append((h, t, eps_g))

        print(f"{name:10s} nz={numEle_length:4d} h={h:.4f} dt={dt:.3e} "
              f"peak eps={np.abs(eps_g).max():.3e}")
        del hist, Minv, M, M_eig
# np.save("convergence_results.npy", results, allow_pickle=True)
#-----------------------------------------------------------------------
#----------------------------PLOTS--------------------------------------
#-----------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm",
    "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "pdf.fonttype": 42,
})

lam = c / f0                         # wavelength [mm] (c in mm/s)
styles = {"Lumped": "^-", "Consistent": "o-", "Combined": "s-"}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.3, 2.6))

# (a) gauge strain for each mesh, combined mass
for h, t, eps in results["Combined"]:
    ax1.plot(t*1e6, eps*1e6, lw=0.9, label=rf"$\lambda/h = {lam/h:.0f}$")
ax1.set_xlabel(r"Time [$\mu$s]")
ax1.set_ylabel(r"Gauge strain $\varepsilon_g$ [$\mu\varepsilon$]")
ax1.legend(frameon=False)
ax1.text(0.02, 0.96, "(a)", transform=ax1.transAxes, va="top")

# (b) relative L2 error vs finest mesh
for name, runs in results.items():
    _, t_ref, eps_ref = runs[-1]
    hs, errs = [], []
    for h, t, eps in runs[:-1]:
        diff = np.interp(t_ref, t, eps) - eps_ref
        errs.append(np.linalg.norm(diff) / np.linalg.norm(eps_ref))
        hs.append(h)
    ax2.loglog(lam/np.array(hs), errs, styles[name], ms=4, lw=0.9, label=name)

# slope -2 guide line
x = lam / np.array([results["Lumped"][0][0], results["Lumped"][-2][0]])
ax2.loglog(x, 0.5*errs[0]*(x/x[0])**-2, "k--", lw=0.8, label=r"slope $-2$")

ax2.set_xlabel(r"Elements per wavelength, $\lambda/h$")
ax2.set_ylabel(r"Relative $L_2$ error")
ax2.legend(frameon=False)
ax2.grid(True, which="both", alpha=0.3, lw=0.4)
ax2.set_xticks([8, 16, 32, 64])
ax2.set_xticklabels(["8", "16", "32", "64"])
ax2.xaxis.set_minor_formatter(NullFormatter())
ax2.text(0.97, 0.96, "(b)", transform=ax2.transAxes,
         ha="right", va="top")

fig.tight_layout()
out = Path(fea_core.__file__).resolve().parent / "script_plots"
out.mkdir(exist_ok=True)       
out.mkdir(exist_ok=True)
fig.savefig(out / "mesh_convergence.pdf")
fig.savefig(out / "mesh_convergence.png", dpi=200)
plt.show()
import os, sys
import numpy as np
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)                        # finds PC_vibration_modes
sys.path.append(os.path.dirname(script_dir))          # finds fea_core, one folder up

import fea_core
from fea_core import (
    create_mesh, gaussPointScheme, build_GSM, build_mass_matrices, get_dt_crit,
    run_explicit_solver, get_force_vec, get_load_matrix, get_strain_stress_history,
    get_axis_dofs
)
from PC_vibration_modes import find_roots, pc_solution


#----------------------------------------------------------------------
# CONFIGURATION / INPUTS
#----------------------------------------------------------------------
Scheme = 3
bar_radius, bar_length = 1, 100         # [mm]
numEle_radius, numEle_length = 10, 300
density = 7.830e-9                      # tonne/mm^3
material = [2.07e5, 0.26, 1]
E, nu = material[0], material[1]
c0 = np.sqrt(E / density)               # [mm/s] (not divided by 1000, mode_table needs mm/s)

station_z = 60.0                        # [mm] snapped to an element centre below


# GWS input: Gaussian window x carrier
amplitude = 200                         # [MPa]
f0 = 1.0e6                              # carrier frequency [Hz]
tau = 1.5e-6                            # Gaussian half-width [s]; spectrum 1/e half-width = 1/(pi*tau)
t0 = 3 * tau                            # pulse centre in time, so it starts near zero
input_sign = -1                         # set to -1 as FEA comes out upside down relative to PC
f_max = 1.8e6                           # PC prediction ignores content above this [Hz]

def q_func(r_position, time):
    return amplitude * np.exp(-((time - t0) / tau)**2) * np.cos(2*np.pi*f0*(time - t0))
fea_core.q_function = q_func

def mode_table(nu, a, c0, mode=1, aL_max=0.8, n=300, x_max=None):
    x_hi = x_max or (1 + nu)
    f, xi = ([0.0], [0.0]) if mode == 1 else ([], [])
    for r in np.linspace(0.005, aL_max, n):
        roots = find_roots(lambda x: pc_solution(x, 2*np.pi*r, nu),
                           0.3, x_hi, n_scan=4000)
        if len(roots) >= mode:
            cp = c0*np.sqrt(roots[mode-1]/(1 + nu))
            f.append(cp*r/a); xi.append(2*np.pi*r/a)
    return np.array(f), np.array(xi)          # Hz, rad/mm

def propagate(sig, dt, dz, f_tab, xi_tab, f_max, pad_time=30e-6):
    n = len(sig)
    nfft = 2**int(np.ceil(np.log2(n + pad_time/dt)))   # padding avoids wrap-around
    S = np.fft.rfft(sig, nfft)
    f = np.fft.rfftfreq(nfft, dt)
    xi = np.interp(f, f_tab, xi_tab)
    H = np.exp(-1j*xi*dz)
    H[f > f_max] = 0
    H[f < f_tab[0]] = 0                                # below cutoff, nothing propagates
    return np.fft.irfft(S*H, nfft)[:n]

#----------------------------------------------------------------------
# FEA RUN (lumped mass, same setup as the convergence script)
#----------------------------------------------------------------------
points, weight = gaussPointScheme(Scheme)
h = bar_length / numEle_length
z_st = (int(round(station_z / h - 0.5)) + 0.5) * h      # strain is read at an element centre
Tend = t0 - 2*tau + (2*bar_length - z_st) / c0
nodes, elements = create_mesh(bar_radius, bar_length, numEle_radius, numEle_length)
properties = np.tile(np.array(material), (len(elements), 1))
DoF = 2 * len(nodes)

GSM = build_GSM(nodes, elements, properties, DoF, Scheme, points, weight)
M_lumped, *_ = build_mass_matrices(nodes, elements, properties, density, DoF, Scheme, points, weight)
Minv = np.linalg.inv(M_lumped)
fixed_dofs = get_axis_dofs(nodes)
Tstep = 0.90 * get_dt_crit(GSM, M_lumped,fixed_dofs)

def force_fn(t):
    return get_force_vec(t, nodes, numEle_radius, get_load_matrix, q_func)

history, *_ = run_explicit_solver(M_lumped,Minv, GSM, Tstep, Tend, force_fn, DoF, fixed_dofs)
strain_fea = get_strain_stress_history(history, elements, nodes, properties, bar_radius, z_st)[0][:, 1]
time = np.arange(len(strain_fea)) * Tstep

#----------------------------------------------------------------------
# UNDISPERSED INPUT AND PC MODE 1 PREDICTION
#----------------------------------------------------------------------
strain_in = input_sign * np.array([q_func(0.0, t) for t in time]) / E   # 1D strain of the loading pulse

f_tab1, xi_tab1 = mode_table(nu, bar_radius, c0, mode=1)
strain_pc1 = propagate(strain_in, Tstep, z_st, f_tab1, xi_tab1, f_max)

# baseline: same pulse delayed by z/C0, no dispersion
strain_nodisp = np.interp(time - z_st / c0, time, strain_in, left=0.0)

#----------------------------------------------------------------------
# COMPARISON NUMBERS
#----------------------------------------------------------------------
peak = np.max(np.abs(strain_fea))
rms_pc1 = np.sqrt(np.mean((strain_fea - strain_pc1)**2)) / peak
rms_nodisp = np.sqrt(np.mean((strain_fea - strain_nodisp)**2)) / peak
print(f"station z = {z_st:.2f} mm, Tstep = {Tstep:.3e} s")
print(f"RMS error / FEA peak:  PC mode 1 = {rms_pc1:.3f}   no dispersion = {rms_nodisp:.3f}")
print(f"pulse centre reaches far end and returns to station at about "
      f"{(t0 + (2*bar_length - z_st)/c0)*1e6:.1f} us (Tend = {Tend*1e6:.1f} us)")

from pathlib import Path

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm",
    "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "pdf.fonttype": 42,
})
out = Path(fea_core.__file__).resolve().parent / "script_plots"
out.mkdir(exist_ok=True)       

#----------------------------------------------------------------------
# PLOT 1: stacked traces like Govender et al.
#----------------------------------------------------------------------
traces = [
    ("Undispersed loading pulse", strain_in),
    ("FEA", strain_fea),
    ("Pochhammer--Chree mode 1 prediction", strain_pc1),
]
step = 3 * np.max(np.abs(strain_in)) * 1e6
fig1, ax = plt.subplots(figsize=(6.3, 4.0))
for i, (label, sig) in enumerate(traces):
    offset = -i * step
    ax.plot(time * 1e6, sig * 1e6 + offset, 'k', lw=0.8)
    ax.text(0.5, offset + 0.5 * step, label, fontsize=8)
ax.set_xlim(0, Tend * 1e6)
ax.set_ylim(-(len(traces) - 0.5) * step, 0.9 * step)
ax.set_yticks([])                                  # offsets make tick values meaningless
ax.set_xlabel(r"Time [$\mu$s]")
ax.set_ylabel(r"Strain (traces offset)")
fig1.tight_layout()
fig1.savefig(out / "pc_strain_stacked.pdf", bbox_inches="tight")
fig1.savefig(out / "pc_strain_stacked.png", dpi=200, bbox_inches="tight")

#----------------------------------------------------------------------
# PLOT 2: overlay for the detailed comparison
#----------------------------------------------------------------------
fig2, ax = plt.subplots(figsize=(6.3, 2.8))
ax.plot(time * 1e6, strain_fea * 1e6, lw=1.0, label="FEA")
ax.plot(time * 1e6, strain_pc1 * 1e6, "--", lw=1.0, label="PC mode 1")
ax.plot(time * 1e6, strain_nodisp * 1e6, ":", lw=1.0,
        label=r"No dispersion ($c_0$)")
ax.set_xlabel(r"Time [$\mu$s]")
ax.set_ylabel(r"Strain [$\mu\varepsilon$]")
ax.grid(alpha=0.3)
ax.legend(frameon=False, loc="upper left")
fig2.tight_layout()
fig2.savefig(out / "pc_strain_overlay.pdf", bbox_inches="tight")
fig2.savefig(out / "pc_strain_overlay.png", dpi=200, bbox_inches="tight")

print("Figures saved to:", out)
plt.show()
#----------------------------------------------------------------------
# VALIDATION METRICS
#----------------------------------------------------------------------
dt = time[1] - time[0]

# Energy-like measure (integral of strain squared over the window)
E_in  = np.trapezoid(strain_in**2,  time)
E_fea = np.trapezoid(strain_fea**2, time)
E_pc  = np.trapezoid(strain_pc1**2, time)
print(f"E_fea/E_in = {E_fea/E_in:.3f}   E_pc/E_in = {E_pc/E_in:.3f}   E_fea/E_pc = {E_fea/E_pc:.3f}")

# Peak amplitude ratio
print(f"peak ratio FEA/PC = {np.max(np.abs(strain_fea))/np.max(np.abs(strain_pc1)):.3f}")

# Time offset via cross-correlation (positive = FEA arrives later than PC)
xc = np.correlate(strain_fea, strain_pc1, mode="full")
lag = (np.argmax(xc) - (len(time) - 1)) * dt
print(f"FEA lags PC by {lag*1e6:.3f} us")

# Normalised RMS error, before and after matching amplitude
nrmse = np.sqrt(np.mean((strain_fea - strain_pc1)**2)) / np.max(np.abs(strain_pc1))
scale = np.dot(strain_fea, strain_pc1) / np.dot(strain_pc1, strain_pc1)
nrmse_scaled = np.sqrt(np.mean((strain_fea - scale*strain_pc1)**2)) / np.max(np.abs(strain_fea))
print(f"NRMSE = {nrmse:.3f}   after amplitude scaling = {nrmse_scaled:.3f}   scale = {scale:.3f}")
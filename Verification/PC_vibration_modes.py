import numpy as np
from scipy.special import jv, iv
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import importlib, fea_core
importlib.reload(fea_core)
from fea_core import (
    create_mesh, gaussPointScheme, partialOfShapeFn, partial, Shape,
    get_element_coords, get_element_dof_indices, get_element_D_matrix,
    compute_B_and_jacobian, StressFunc, get_load_matrix, q_function,
    get_force_vec, build_mass_matrices, combined_mass_matrices,
    plot_mesh, plot_stress_field, plot_mesh_and_force, compute_stresses, solve_global_eqn, 
    build_GSM, find_nearest_element, get_strain_stress_history,run_explicit_solver,
    plot_strain_history_stacked, plot_gws_input, check_mass_conservation, get_axis_dofs
)
import matplotlib.pyplot as plt
import sympy as sp
from pathlib import Path
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
#----------------------------------------------------------------------------------------
def J0(z2): #bessel functions
    s = np.sqrt(abs(z2))
    return jv(0, s) if z2 >= 0 else iv(0, s)

def J1_over(z2):
    if abs(z2) < 1e-12: #makes sure that theres no 0/0 blowup and returns exact value
        return 0.5
    s = np.sqrt(abs(z2))
    return jv(1, s)/s if z2 > 0 else iv(1, s)/s #returns J1(s)/s which is building block of frequency equation

def phi(z2):
    return J0(z2) / J1_over(z2)

def pc_solution(x, gamma_a, nu):    #global formula from Gov et al. to get roots
    beta = (1 - 2*nu) / (1 - nu)    #gamma_a isnt same as gamma, but is gamma*a as firstly a/lambda is known/set and secondly gamma*a is how it appears in Phi eqn
    phi_h = phi(gamma_a**2 * (beta*x - 1))   # (ha)^2
    phi_k = phi(gamma_a**2 * (2*x - 1))      # (ka)^2
    return (x - 1)**2 * phi_h - (beta*x -1) * (x - phi_k)

def bisect(func, lo, hi, tol=1e-12, max_iterations=200):    #bissection function to get accurate roots once sign change is found
    f_lo, f_hi = func(lo), func(hi)
    if f_lo * f_hi > 0:
        raise ValueError("No sign change in bracket")
    for _ in range(max_iterations):
        mid = 0.5 * (lo + hi)
        f_mid = func(mid)
        if f_lo * f_mid <= 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)

def find_roots(func, x_min, x_max, n_scan=10000, resid_tol=1e-6):   #create fine grid in x and looks for sign changes as this is where roots are 0
    xs = np.linspace(x_min, x_max, n_scan)
    fs = np.array([func(x) for x in xs])
    roots = []
    for i in range(n_scan - 1):
        if fs[i] * fs[i+1] < 0:
            r = bisect(func, xs[i], xs[i+1])    #bissection method to refine each root
            if abs(func(r)) < resid_tol:      # Since J1 = 0 where function flips, but doesnt pass through zero. This checks for genuine root, not a pole as function has to pass through zero
                roots.append(r)
    return roots
# this is used for plotting figure 3 only and not vibrational PC modes
def mode1_curve(nu, a, c0, aL_max=0.5, n=200):  #sweeps values of a/lambda
    """Mode 1 phase velocity vs frequency. a in mm, c0 in mm/s | f in Hz, Cp in mm/s."""
    xs = np.linspace(0.3, 1 + nu, 3000)
    f, cp = [], []
    for r in np.linspace(0.005, aL_max, n):
        func = lambda x: pc_solution(x, 2*np.pi*r, nu)
        roots = find_roots(func, 0.3, 1 + nu, n_scan=3000)  # min and max locks mode 1 vibration range since mode 1 window is max at 1+v
        if roots:
            c = c0*np.sqrt(roots[0]/(1 + nu))
            cp.append(c); f.append(c*r/a)       # f = Cp/Lambda, Lambda = a/r
    return np.array(f), np.array(cp)

def get_dt_crit(K, M, fixed_dofs=None):     #since only largest eigenvalue is needed ---eigsh is sparse solver so saves valuable memory
    if fixed_dofs is not None:
        free = np.setdiff1d(np.arange(K.shape[0]), fixed_dofs)
        K = K[np.ix_(free, free)]
        M = M[np.ix_(free, free)]
    lam_max = eigsh(csr_matrix(K), k=1, M=csr_matrix(M), which="LM",
                    return_eigenvectors=False, tol=1e-6)[0]
    return 2/np.sqrt(lam_max)

#----------------------------------INPUT CONFIGURATION-------------------------------------
nu = 0.26    # matches gov et al. parameters
density = 7.830e-9
E = 2.07e5
a = 10
wavelength = 100
C0 = np.sqrt(E/density)/1000    # in [m/s]
x_max = (1 + nu)* 2**2  # has the form (1+nu)*(Cp/C0)^2. This plots modes up to phase velocity being double wave speed. change Cp/C0 value to include more or less modes
f_max = 500     # Frequency limit in KHz for mode plot 
n_modes = 4     # how many modes are plotted
# gamma_a = 2*np.pi*0.1
# func = lambda x: pc_solution(x, gamma_a, nu)
# roots = find_roots(func, 0.3, 1 + nu)           # limits that contain mode 1
# x1 = roots[0]                                   # mode 1 = lowest genuine root
# Cp_over_C0 = np.sqrt(x1 / (1 + nu))

a_over_L = np.linspace(0.005, 2.0, 100) 
modes = {} 
for aL_ratio in a_over_L:           #similar code to mode1_curve except x_max isnt forcing only mode 1 to be found- the higher x_max is the more modes will be plotted
    gamma_a = 2*np.pi*aL_ratio
    func = lambda x: pc_solution(x, gamma_a, nu)
    roots = find_roots(func, 0.3, x_max)   # sorted ascending
    for n, x in enumerate(roots, start=1):
        Cp = C0*np.sqrt(x/(1+nu))
        f = Cp/(a/aL_ratio)                        # Lambda = a/aL
        modes.setdefault(n, []).append((f, Cp))

#----------------------------PLOT GOVENDER ET AL. 3.1---------------------------
plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm",
    "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "pdf.fonttype": 42,
})

# reference speeds [m/s]
cs = C0 / np.sqrt(2*(1 + nu))                    # shear wave speed
cR = cs * (0.862 + 1.14*nu) / (1 + nu)           # Rayleigh speed (Viktorov approx.)

fig, ax = plt.subplots(figsize=(4.5, 3.4))
for n in range(1, n_modes + 1):
    if n in modes:
        pts = np.array(modes[n])
        ax.plot(pts[:, 0], pts[:, 1]/1000, lw=1.2, label=f"Mode {n}")

# ax.axhline(C0/1000, color="k", ls="--", lw=0.8)
# ax.axhline(cR/1000, color="k", ls=":", lw=0.8)
ax.text(f_max*0.98, C0/1000 + 0.1, r"$c_0$", ha="right", va="bottom")
ax.text(f_max*0.98, cR/1000 - 0.1, r"$c_R$", ha="right", va="top")

ax.set_xlabel("Frequency [kHz]")
ax.set_ylabel(r"Phase velocity $c_p$ [km/s]")
ax.set_xlim(0, 500)
ax.set_ylim(2, 9)
ax.grid(alpha=0.3)
ax.legend(frameon=False, loc="upper left")
fig.tight_layout()

out = Path(fea_core.__file__).resolve().parent / "script_plots"
out.mkdir(exist_ok=True)                   # save next to this script
fig.savefig(out / "pc_dispersion.pdf", bbox_inches="tight")
fig.savefig(out / "pc_dispersion.png", dpi=200, bbox_inches="tight")
print("Figures saved to:", out)


#-------------------------------MODE VERIFICATION FROM MY MODEL WITH GOVENDERS VALUES----------------------------
# ---------------- CONFIG ----------------
Scheme = 3
bar_radius, bar_length = 1, 100
numEle_radius = 10
numEle_length_list = [75, 150, 300]
density = 7.830e-9
material = [2.07e5, 0.26, 1]
nu = material[1]
c0 = np.sqrt(material[0]/density)            # [mm/s]

z1_nom, z2_nom = 20.0, 50.0                  # station positions [mm]
T_pulse = 1e-6                             # pulse duration [s]
thr = 0.02                                   # keep frequencies with |U1| > thr * peak
Tend = (2*bar_length - z2_nom)/c0            # far-end reflection reaches z2 at about this time

def q_func(r_position, time):                # smooth single pulse, content from DC to ~2/T
    if time <= T_pulse:
        return 200*0.5*(1 - np.cos(2*np.pi*time/T_pulse))
    return 0.0
fea_core.q_function = q_func

# ---------------- TWO-STATION EXTRACTION ----------------
def two_station_cp(s1, s2, dt, dz, thr=0.05, pad=8):
    n = len(s1)
    taper = np.ones(n); nt = int(0.1*n)      # smooth cut-off before the reflection arrives
    taper[-nt:] = 0.5*(1 + np.cos(np.pi*np.arange(nt)/nt))
    nfft = 2**int(np.ceil(np.log2(n*pad)))
    U1 = np.fft.rfft(s1*taper, nfft); U2 = np.fft.rfft(s2*taper, nfft)
    f = np.fft.rfftfreq(nfft, dt)
    valid = np.abs(U1[1:]) > thr*np.abs(U1).max()
    k = len(U1) if valid.all() else np.argmax(~valid) + 1   # first bin below threshold
    band = slice(1, k)
    phase = np.unwrap(np.angle(U2[band]*np.conj(U1[band])))  # = -2*pi*f*dz/Cp
    return f[band], -2*np.pi*f[band]*dz/phase

# ---------------- MESH LOOP ----------------
points, weight = gaussPointScheme(Scheme)
results = {}
for N in numEle_length_list:
    h = bar_length/N
    nodes, elements = create_mesh(bar_radius, bar_length, numEle_radius, N)
    DoF = 2 * len(nodes)
    properties = np.tile(np.array(material), (len(elements), 1))
    DoF = 2*len(nodes)

    GSM = build_GSM(nodes, elements, properties, DoF, Scheme, points, weight)
    M_lumped, *_ = build_mass_matrices(nodes, elements, properties, density, DoF, Scheme, points, weight)
    Minv = np.linalg.inv(M_lumped)
    fixed_dofs = get_axis_dofs(nodes)
    Tstep = 0.90*get_dt_crit(GSM, M_lumped, fixed_dofs)

    force_fn = lambda t: get_force_vec(t, nodes, numEle_radius, get_load_matrix, q_func)
    history, *_ = run_explicit_solver(M_lumped, Minv, GSM, Tstep, Tend, force_fn, DoF, fixed_dofs)

    # stations at element centres, so the nearest-element lookup is unambiguous and dz is exact
    i1 = int(round(z1_nom/h - 0.5)); i2 = int(round(z2_nom/h - 0.5))
    z1, z2 = (i1 + 0.5)*h, (i2 + 0.5)*h
    dz = z2 - z1
    s1 = get_strain_stress_history(history, elements, nodes, properties, bar_radius, z1)[0][:, 1]
    s2 = get_strain_stress_history(history, elements, nodes, properties, bar_radius, z2)[0][:, 1]

    n_win = min(int(0.98*(2*bar_length - z2)/c0/Tstep), len(s1))   # stop before the reflection
    f, cp = two_station_cp(s1[:n_win], s2[:n_win], Tstep, dz, thr)
    results[N] = (f, cp, h)
    print(f"N={N}: z1={z1:.2f}, z2={z2:.2f}, dz={dz:.2f} mm, window={n_win*Tstep*1e6:.1f} us, "
          f"Cp/C0 at lowest bins = {cp[:3]/c0}")
    del GSM, M_lumped, Minv, history, s1, s2
    import gc; gc.collect()
#---------------------------Plotting---------------------------
T_pulse = 1.0e-6

a_paper = 10.0
scale = bar_radius / a_paper                       # f_paper-equivalent = f * scale
cs = c0 / np.sqrt(2*(1 + nu))
f_cut2 = 3.8317*cs / (2*np.pi*bar_radius)          # mode 2 cutoff [Hz], ~1.9 MHz here

f_pc, cp_pc = mode1_curve(nu, bar_radius, c0, aL_max=0.8)

cg = cp_pc / (1 - (f_pc/cp_pc)*np.gradient(cp_pc, f_pc))   # group velocity, mm/s
t_arrive = z2 / cg                                          # s, z2 in mm
t_window = n_win * Tstep
f_edge = f_pc[np.argmax(t_arrive > t_window)] * scale       # paper-equivalent Hz
print(f_edge/1e3)
print(t_window*1e6, t_arrive.max()*1e6, z2, n_win)

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm",
    "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "pdf.fonttype": 42,
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.8))

# (a) phase velocity vs frequency
for N, (f, cp, h) in results.items():
    ok = f < 0.95*f_cut2                           # single-mode range only
    ax1.plot(f[ok][::8]*scale/1e3, cp[ok][::8]/1e6, '+', ms=4,
             label=f'FEA, {N} elements')
ax1.plot(f_pc*scale/1e3, cp_pc/1e6, 'k-', lw=1.0, label='PC mode 1')
ax1.set_xlabel('Frequency [kHz]')
ax1.set_ylabel(r'Phase velocity $c_p$ [km/s]')
ax1.set_xlim(0, 200); ax1.set_ylim(2, 6)
ax1.grid(alpha=0.3)
ax1.legend(frameon=False, loc="lower left")
ax1.text(0.97, 0.96, "(a)", transform=ax1.transAxes, ha="right", va="top")

# (b) error against PC vs elements per wavelength
for N, (f, cp, h) in results.items():
    ok = f < 0.95*f_cut2
    cp_ref = np.interp(f[ok], f_pc, cp_pc)
    ax2.semilogx(cp[ok]/(f[ok]*h), 100*(cp[ok] - cp_ref)/cp_ref, '.',
                 ms=3, label=f'{N} elements')
ax2.set_xlabel('Elements per wavelength')
ax2.set_ylabel(r'Error vs PC [\%]')
ax2.grid(True, which="both", alpha=0.3)
ax2.legend(frameon=False, loc="lower right")      # after plotting
ax2.text(0.03, 0.96, "(b)", transform=ax2.transAxes, va="top")

fig.tight_layout()

out = Path(fea_core.__file__).resolve().parent / "script_plots"
out.mkdir(exist_ok=True)                     # save next to this script
fig.savefig(out / "phase_velocity.pdf", bbox_inches="tight")
fig.savefig(out / "phase_velocity.png", dpi=200, bbox_inches="tight")
print("Figures saved to:", out)
plt.show()
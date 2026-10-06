#!/usr/bin/env python3
"""Scan the Moller relic line on the Kling mass range and cache it as a CSV.

Three '#' comment lines carry what produced the file, then a MAp,eps header.
np.genfromtxt takes the first line as the names even when it is a comment, so
read it back with

    np.genfromtxt("data/moller_relic_line.csv", delimiter=",", names=True,
                  skip_header=3)
"""

# The physics below is a copy of the definitions in Relic_Abundance.ipynb that
# the Gondolo-Gelmini relic line depends on, and nothing else: the closed
# forms, the velocity expansion and the Dirac case are left in the notebook.
# The notebook is still the reference implementation; keep the two
# recognisably the same when either changes.

import argparse
import csv
import os
import time

import numpy as np
from scipy import interpolate
from scipy.integrate import simpson, solve_ivp
from scipy.special import kve

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


### Constants, as in the notebook.

alpha = 1/137

Mpl = 2.435323e18      # reduced Planck mass in GeV

OmegaDM = 0.265        # without the reduced Hubble constant factor

H0 = 1.438e-42         # in GeV
T0 = 2.35e-13          # in GeV

mleptons = np.array([5.11e-4, 0.106, 1.777])  # Masses of leptons in GeV


### Hadronic R ratio, R = sigma(e+e- -> hadrons)/sigma(e+e- -> mu+mu-), from
### the PDG compilation in data/rratio.dat (1400 points, 0.3-188.7 GeV).

def read_rratio(path=os.path.join(DATA, "rratio.dat")):
    """Read the PDG R-ratio table -> (E_cm, R, dR)."""
    E, R, dR = [], [], []
    with open(path) as f:
        for line in f:
            if line.startswith("*") or not line.strip():
                continue
            try:
                ecm = float(line[0:10])
            except ValueError:
                continue                                  # continuation line
            if ecm <= 0.0:
                continue
            E.append(ecm)
            R.append(float(line[35:45]))
            dR.append(0.5*(abs(float(line[47:57])) + abs(float(line[58:68]))))
    return np.array(E), np.array(R), np.array(dR)

ecm_rratio, Rdata, dRdata = read_rratio()

# 70 energies are quoted by more than one experiment, but interp1d needs a
# strictly increasing x, so repeats are collapsed into a single inverse-
# variance weighted mean (the stat errors are all non-zero).
weights_R = 1.0/dRdata**2
ecm_R, index_R = np.unique(ecm_rratio, return_inverse=True)
R_R = (np.bincount(index_R, weights=weights_R*Rdata)
       / np.bincount(index_R, weights=weights_R))

# Linear in both E and R:
R_interp = interpolate.interp1d(ecm_R, R_R)

def R(sqrts):
    """Hadronic R ratio at sqrt(s), interpolated from the PDG table."""
    sqrts = np.asarray(sqrts, dtype=float)
    inside = sqrts >= ecm_R[0]
    return np.where(inside, R_interp(np.clip(sqrts, ecm_R[0], ecm_R[-1])), 0.0)


### Effective dof from data/effective_dof.csv.

# The QCD transition temperature is not settled, so the table lists the
# 150-214 MeV window once per candidate, tagged in the last column by
# which choice(s) the row belongs to ('All' outside that window).
qcd_transition = "214"          # "214", "170" or "150" MeV

dof_table = np.genfromtxt(os.path.join(DATA, "effective_dof.csv"), delimiter=',',
                          names=True, dtype=None, encoding='utf-8')

keep_dof = np.array([m == "All" or qcd_transition in m
                     for m in dof_table["transition_temperature_model"]])

T_dof = dof_table["kBT_eV"][keep_dof]*1e-9        # eV -> GeV
gstar_e = dof_table["g_star_e"][keep_dof]
gstar_s = dof_table["g_star_s"][keep_dof]

# Sort on (T, g*_e), not T alone:
order = np.lexsort((gstar_e, T_dof))
T_dof, gstar_e, gstar_s = T_dof[order], gstar_e[order], gstar_s[order]

# Interpolate against log T:
logT_dof = np.log(T_dof)

# interp1d needs a strictly increasing x, so push each tie up by one float
# step.
for i in range(1, len(logT_dof)):
    if logT_dof[i] <= logT_dof[i-1]:
        logT_dof[i] = np.nextafter(logT_dof[i-1], np.inf)

gstar_e_interp = interpolate.interp1d(logT_dof, gstar_e, bounds_error=False,
                                      fill_value=(gstar_e[0], gstar_e[-1]))

gstar_s_interp = interpolate.interp1d(logT_dof, gstar_s, bounds_error=False,
                                      fill_value=(gstar_s[0], gstar_s[-1]))

def gstarfun(T):
    """g*_e at temperature T in GeV, interpolated from the tabulated values."""
    return gstar_e_interp(np.log(np.asarray(T, dtype=float)))

def gstarsfun(T):
    """g*_s at temperature T in GeV, interpolated from the tabulated values."""
    return gstar_s_interp(np.log(np.asarray(T, dtype=float)))

gstarsT0 = gstarsfun(T0)


### The Boltzmann equation.

def H(T): # Assuming radiation domination
    return np.sqrt(np.pi**2/90 * gstarfun(T))*T**2/Mpl

def Yeq(x, g, mchi):
    return 45/(2*np.pi**2)* g/gstarsfun(mchi/x) * (x/(2*np.pi))**(3/2) * np.exp(-x)

def dlogglogT(T, h=0.05):
    """d ln g*_s / d ln T, by central difference on the interpolated table."""
    T = np.asarray(T, dtype=float)
    lT = np.log(T)
    return (np.log(gstarsfun(np.exp(lT + h)))
            - np.log(gstarsfun(np.exp(lT - h))))/(2*h)

def dsdx(x, mchi):
    """ds/dx at T = mchi/x, with s = (2 pi^2/45) g*_s(T) T^3."""
    T = mchi/x
    return -(2*np.pi**2/45)*mchi**3/x**4*gstarsfun(T)*(3 + dlogglogT(T))

def dYdxcore(x, Y, g, mchi, sigmav):
    """Shared right hand side, Gelmini form:"""
    return dsdx(x, mchi)/(2*3*H(mchi/x))*sigmav*(Y**2 - Yeq(x, g, mchi)**2)

def omega(mchi, Yinf):
    return mchi*Yinf*(2*np.pi**2/45)*gstarsT0*T0**3/(3*Mpl**2*H0**2)


### Gondolo-Gelmini thermal averaging.

def sigmatilde(s, mf, Q2Nc=1.0):
    """Q_f^2 N_c^f (s - 4mf^2)^{1/2} ( s - (s - 4mf^2)/3 ), zero below threshold."""
    s = np.asarray(s, dtype=float)
    opench = s > 4*mf**2
    beta2 = np.where(opench, s - 4*mf**2, 0.0)
    return Q2Nc*np.where(opench, np.sqrt(beta2)*(s - beta2/3), 0.0)

def sigma_ann(s, vare, ma, alpha, alphaD, mchi):
    """chi chi -> f fbar annihilation cross section at Mandelstam s."""
    s = np.asarray(s, dtype=float)
    opench = s > 4*mchi**2
    beta_chi = np.where(opench, np.sqrt(np.where(opench, s - 4*mchi**2, 0.0)), 0.0)

    lept = np.zeros_like(s)
    for ml in mleptons:
        lept = lept + sigmatilde(s, ml)
    hadr = R(np.sqrt(s))*sigmatilde(s, mleptons[1])

    return np.where(opench,
                    2*np.pi*vare**2*alpha*alphaD*beta_chi
                    / (s*(s - ma**2)**2)*(lept + hadr),
                    0.0)

def sigma_ann_e(s, vare, ma, alpha, alphaD, mchi):
    """chi chi -> e+ e- and nothing else."""
    s = np.asarray(s, dtype=float)
    opench = s > 4*mchi**2
    beta_chi = np.where(opench, np.sqrt(np.where(opench, s - 4*mchi**2, 0.0)), 0.0)
    return np.where(opench,
                    2*np.pi*vare**2*alpha*alphaD*beta_chi
                    /(s*(s - ma**2)**2)*sigmatilde(s, mleptons[0]),
                    0.0)

def sigma_ann_lep(s, vare, ma, alpha, alphaD, mchi):
    """chi chi -> l+ l- over all three charged leptons, no hadrons."""
    s = np.asarray(s, dtype=float)
    opench = s > 4*mchi**2
    beta_chi = np.where(opench, np.sqrt(np.where(opench, s - 4*mchi**2, 0.0)), 0.0)
    lept = np.zeros_like(s)
    for ml in mleptons:
        lept = lept + sigmatilde(s, ml)
    return np.where(opench,
                    2*np.pi*vare**2*alpha*alphaD*beta_chi
                    /(s*(s - ma**2)**2)*lept,
                    0.0)

# How far past threshold to integrate, in units of T.
GG_UCUT = 60.0

# sqrt(s) window over which R is resonance dominated and the grid has to
# stay fine.
GG_RES_LO, GG_RES_HI = 0.28, 1.15
GG_NODES_SMOOTH = 601      # no resonance in the window: the integrand is just u^(3/2) e^-u
GG_DW_RES = 5e-4           # target sqrt(s) spacing in GeV inside the window (phi is 4 MeV)
GG_NODES_MAX = 24001

def gg_grid(mchi, T, ucut=GG_UCUT):
    """Integration nodes in u = (sqrt(s) - 2 mchi)/T, uniform in sqrt(s)."""
    wlo, whi = 2*mchi, 2*mchi + ucut*T
    if whi < GG_RES_LO or wlo > GG_RES_HI:
        nodes = GG_NODES_SMOOTH
    else:
        nodes = int(np.clip((whi - wlo)/GG_DW_RES, GG_NODES_SMOOTH, GG_NODES_MAX))
    nodes += 1 - nodes % 2          # simpson wants an odd node count
    return np.linspace(0.0, ucut, nodes)

def sigmav_moller_unit(x, ma, mchi, ucut=GG_UCUT,
                       sigma_fun=sigma_ann, smooth=False):
    """<sigma v_Moller>(x) with eps^2 alpha alphaD divided out."""
    x = float(x)
    T = mchi/x
    # smooth=True skips the resonance refinement:
    u = (np.linspace(0.0, ucut, GG_NODES_SMOOTH) if smooth
         else gg_grid(mchi, T, ucut))
    w = 2*mchi + u*T
    s = w*w

    sig = sigma_fun(s, 1.0, ma, 1.0, 1.0, mchi)
    bessel = kve(1, 2*x + u)/kve(2, x)**2*np.exp(-u)
    return simpson(2*sig*(s - 4*mchi**2)*w*w*bessel, x=w)/(8*mchi**4*T)

# One interpolant per (ma, mchi).
GG_XLO, GG_XHI, GG_XPTS = 2.0, 3e3, 193
_gg_cache = {}

def sigmav_unit_table(ma, mchi, sigma_fun=sigma_ann, smooth=False):
    """log-log cubic interpolant of sigmav_moller_unit in x, memoised."""
    key = (float(ma), float(mchi), sigma_fun.__name__)
    tab = _gg_cache.get(key)
    if tab is None:
        xg = np.logspace(np.log10(GG_XLO), np.log10(GG_XHI), GG_XPTS)
        Fg = np.array([sigmav_moller_unit(_x, ma, mchi, sigma_fun=sigma_fun,
                                          smooth=smooth) for _x in xg])
        tab = interpolate.interp1d(np.log(xg), np.log(Fg), kind="cubic",
                                   bounds_error=False,
                                   fill_value=(np.log(Fg[0]), np.log(Fg[-1])))
        _gg_cache[key] = tab
    return tab

def sigmav_moller_fast(x, vare, ma, alpha, alphaD, mchi,
                       sigma_fun=sigma_ann, smooth=False):
    """<sigma v_Moller>, reading the memoised table."""
    return (vare**2*alpha*alphaD
            * np.exp(sigmav_unit_table(ma, mchi, sigma_fun, smooth)(np.log(x))))

def dYdxggf(x, Y, g, vare, ma, alpha, alphaD, mchi):
    """dYdxcore with the tabulated thermal average."""
    return dYdxcore(x, Y, g, mchi,
                    sigmav_moller_fast(x, vare, ma, alpha, alphaD, mchi))

def omega_of_gg(ma, vare, alphaD, mchi_over_ma=0.6,
                xi=1e1, xfin=1e3, rtol=1e-6, atol=1e-25):
    """Relic Omega with the full thermal average."""
    mchi = mchi_over_ma*ma
    rhs = lambda x, Y: dYdxggf(x, Y, 2, vare, ma, alpha, alphaD, mchi)
    sol = solve_ivp(rhs, (xi, xfin), [Yeq(xi, 2, mchi)],
                    method="Radau", rtol=rtol, atol=atol)
    if not sol.success:
        return np.nan
    return omega(mchi, sol.y[0][-1])

def omega_of_gg_e(ma, vare, alphaD, mchi_over_ma=0.6,
                  xi=1e1, xfin=1e3, rtol=1e-6, atol=1e-25):
    """Relic Omega with the full thermal average, electron channel only."""
    mchi = mchi_over_ma*ma
    rhs = lambda x, Y: dYdxcore(x, Y, 2, mchi,
                                sigmav_moller_fast(x, vare, ma, alpha, alphaD, mchi,
                                                   sigma_fun=sigma_ann_e, smooth=True))
    sol = solve_ivp(rhs, (xi, xfin), [Yeq(xi, 2, mchi)], method="Radau",
                    rtol=rtol, atol=atol)
    if not sol.success:
        return np.nan
    return omega(mchi, sol.y[0][-1])

def omega_of_gg_lep(ma, vare, alphaD, mchi_over_ma=0.6,
                    xi=1e1, xfin=1e3, rtol=1e-6, atol=1e-25):
    """Relic Omega with the full thermal average, all leptons, no hadrons."""
    mchi = mchi_over_ma*ma
    rhs = lambda x, Y: dYdxcore(x, Y, 2, mchi,
                                sigmav_moller_fast(x, vare, ma, alpha, alphaD, mchi,
                                                   sigma_fun=sigma_ann_lep, smooth=True))
    sol = solve_ivp(rhs, (xi, xfin), [Yeq(xi, 2, mchi)], method="Radau",
                    rtol=rtol, atol=atol)
    if not sol.success:
        return np.nan
    return omega(mchi, sol.y[0][-1])


### Following the relic line in mass.

def solve_window(omfun, ma, v_lo, v_hi, target, tol=0.01, itmax=30):
    """Find the crossing Omega(vare) = target inside [v_lo, v_hi]."""
    om_lo = omfun(ma, v_lo)
    if np.isfinite(om_lo) and om_lo < target:
        return np.nan, "below"       # needs a smaller vare than the window holds
    om_hi = omfun(ma, v_hi)
    if np.isfinite(om_hi) and om_hi > target:
        return np.nan, "above"       # needs a larger vare than the window holds
    if not (np.isfinite(om_lo) and np.isfinite(om_hi)):
        return np.nan, None

    lo, hi = np.log(v_lo), np.log(v_hi)
    f_lo, f_hi = np.log(om_lo/target), np.log(om_hi/target)   # f_lo > 0 > f_hi
    held = 0

    for _ in range(itmax):
        # Where the chord through (lo, f_lo) and (hi, f_hi) hits zero:
        probe = (lo*f_hi - hi*f_lo)/(f_hi - f_lo)
        om_p = omfun(ma, float(np.exp(probe)))
        if not np.isfinite(om_p) or om_p <= 0:
            return np.nan, None
        f_p = np.log(om_p/target)
        if abs(f_p) < tol:
            return float(np.exp(probe)), "inside"

        if f_p > 0:                  # still too much DM, so vare must rise
            # Illinois tweak:
            if held > 0:
                f_hi *= 0.5
            lo, f_lo, held = probe, f_p, 1
        else:
            if held < 0:
                f_lo *= 0.5
            hi, f_hi, held = probe, f_p, -1

    # Tolerance never met:
    return float(np.exp((lo*f_hi - hi*f_lo)/(f_hi - f_lo))), "inside"

# log Omega vs log vare is close to a straight line of slope -2, but not
# exactly, because the freeze-out x drifts with the coupling.
LOGSLOPE = -1.8

def predict_eps(i, ma_grid, eps_found):
    """Extrapolate eps for mass i from the last two solved masses."""
    solved = [(m, e) for m, e in zip(ma_grid[:i], eps_found[:i]) if np.isfinite(e)]
    if not solved:
        return None
    if len(solved) == 1:
        return solved[-1][1]
    (m1, e1), (m2, e2) = solved[-2], solved[-1]
    slope = np.log(e2/e1)/np.log(m2/m1)
    return float(e2*(ma_grid[i]/m2)**slope)

def refine_eps(omfun, ma, guess, v_lo, v_hi, target, tol=0.01, itmax=6):
    """Walk from a nearby guess to the crossing Omega = target."""
    le = np.log(guess)
    for _ in range(itmax):
        om = omfun(ma, float(np.exp(le)))
        if not np.isfinite(om) or om <= 0:
            return np.nan
        r = np.log(om/target)
        if abs(r) < tol:
            return float(np.exp(le))
        le += r/(-LOGSLOPE)
        if not (np.log(v_lo) <= le <= np.log(v_hi)):
            return np.nan
    return float(np.exp(le))

def scan_relic_line(ma_grid, vare_lo, vare_hi, alphaD, mchi_over_ma=0.6,
                    target=None, tol=0.01, solver=None, omega_fun=None):
    """eps(ma) reproducing the relic abundance, at fixed alphaD and mass ratio."""
    # The notebook defaults to its velocity expansion omega_of, which is not
    # carried over here; the Gondolo-Gelmini Omega is the only one this needs.
    if omega_fun is None:
        omega_fun = omega_of_gg
    if target is None:
        target = OmegaDM
    ma_grid = np.asarray(ma_grid, dtype=float)
    stats = {"nsolve": 0}

    def omfun(ma, vare):
        stats["nsolve"] += 1
        return omega_fun(ma, vare, alphaD, mchi_over_ma, **(solver or {}))

    eps = np.full(ma_grid.shape, np.nan)
    stop_note = None
    i_stop = len(ma_grid)

    for i, ma_i in enumerate(ma_grid):
        guess = predict_eps(i, ma_grid, eps)

        found = np.nan
        if guess is not None and vare_lo <= guess <= vare_hi:
            found = refine_eps(omfun, ma_i, guess, vare_lo, vare_hi, target, tol)

        if not np.isfinite(found):
            found, side = solve_window(omfun, ma_i, vare_lo, vare_hi, target, tol)
            if side == "above":
                stop_note = (f"crossing lies above the window at ma = {ma_i:.4g} GeV"
                             + (f" (extrapolated eps = {guess:.3e} > {vare_hi:g})"
                                if guess is not None else ""))
                i_stop = i
                break
            # 'below' or None:

        eps[i] = found

    nfound = int(np.isfinite(eps).sum())
    info = {"nsolve": stats["nsolve"], "nfound": nfound,
            "stop_note": stop_note, "i_stop": i_stop,
            "alphaD": alphaD, "mchi_over_ma": mchi_over_ma, "target": target,
            "model": omega_fun.__name__}
    return eps, info


OMEGA_FUNS = {"all": omega_of_gg, "leptons": omega_of_gg_lep,
              "electrons": omega_of_gg_e}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    # Defaults are the span of the Kling file,
    # data/Kling/scalar_DM_Oh2_intermediate_eps_vs_mAprime.txt, which has 106
    # points over this range; the point of the script is to sample it finer.
    p.add_argument("--min-mass", type=float, default=0.0017)
    p.add_argument("--max-mass", type=float, default=1.66)
    p.add_argument("--points", type=int, default=300)
    p.add_argument("--eps-lo", type=float, default=1e-7)
    p.add_argument("--eps-hi", type=float, default=5e-2)
    p.add_argument("--alpha-d", type=float, default=0.1)
    p.add_argument("--mchi-over-ma", type=float, default=0.6)
    p.add_argument("--tol", type=float, default=1e-3)
    p.add_argument("--channels", default="all", choices=sorted(OMEGA_FUNS),
                   help="which annihilation channels are open")
    p.add_argument("--output", default="data/moller_relic_line.csv")
    args = p.parse_args()

    ma_grid = np.logspace(np.log10(args.min_mass), np.log10(args.max_mass),
                          args.points)

    t0 = time.time()
    eps, info = scan_relic_line(ma_grid, args.eps_lo, args.eps_hi,
                                alphaD=args.alpha_d,
                                mchi_over_ma=args.mchi_over_ma,
                                tol=args.tol,
                                omega_fun=OMEGA_FUNS[args.channels])
    elapsed = time.time() - t0

    out = os.path.join(HERE, args.output)
    os.makedirs(os.path.dirname(out), exist_ok=True)

    # Nothing in a bare two-column file records what produced it, the same
    # trap the micrOMEGAs CSVs have; the header comments carry it.
    with open(out, "w", newline="") as fh:
        fh.write(f"# moller_scan.py, channels = {args.channels},"
                 f" omega_fun = {info['model']}\n")
        fh.write(f"# alphaD = {info['alphaD']:g},"
                 f" mchi_over_ma = {info['mchi_over_ma']:g},"
                 f" Omega_target = {info['target']:g}\n")
        fh.write(f"# {info['nfound']} of {len(ma_grid)} masses solved in"
                 f" {info['nsolve']} ODE integrations, {elapsed:.0f} s\n")
        w = csv.writer(fh)
        w.writerow(["MAp", "eps"])
        for ma_i, eps_i in zip(ma_grid, eps):
            w.writerow([f"{ma_i:.10g}", f"{eps_i:.10g}"])

    print(f"wrote {out}: {info['nfound']}/{len(ma_grid)} points,"
          f" {elapsed:.0f} s")
    if info["stop_note"] is not None:
        print(f"stopped early at index {info['i_stop']}: {info['stop_note']}")


if __name__ == "__main__":
    main()

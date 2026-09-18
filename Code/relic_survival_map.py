#!/usr/bin/env python3
r"""For each mass ratio R, which A' masses does direct detection still allow?

The relic abundance fixes eps once (MAp, alpha_D, R) are chosen, so every point
considered here already has Omega h^2 = 0.1204 and the only question left is
whether direct detection excludes it. The scan therefore runs in two stages:

  1. sweep R at fixed alpha_D and solve the relic line eps(MAp) at each one,
     comparing it against the combined 2026 direct detection limit;
  2. read off, for each R, the mass intervals where the relic line sits below
     that limit.

Both stages are done twice, once with the PandaX-4T S2 CEvNS recast folded into
the limit and once without it. The recast is a reinterpretation of that data
under a theory assumption rather than a limit the collaboration set, so the two
answers are reported side by side rather than one replacing the other.

Run it from Code/:

    python3 relic_survival_map.py --recompute --min-r 0.501 --max-r 0.65 --points-r 40

The scan takes a few minutes on eight cores and is cached in
data/relic_survival_map.npz; without --recompute the cache is reused and only
the interval analysis and the CSVs are redone.

This script writes CSVs and prints; it draws nothing. Every figure in this
repository is produced in a notebook, reading data/relic_survival_*.csv.

Only direct detection is applied. The accelerator, beam dump and supernova
regions are not, so a point called allowed here may still be excluded by them.

One consequence worth stating plainly: **the allowed mass ranges do not depend
on alpha_D.** Both the relic eps and the direct detection limit scale as
alpha_D^(-1/2) -- the first because <sigma v> ~ eps^2 alpha alpha_D, the second
because sigma_SI ~ eps^2 alpha alpha_D -- so the ratio between them is
independent of it, measured constant to four significant figures. alpha_D sets
the eps values that are reported, not which masses survive.
"""

# The BLAS thread limits have to be set before numpy is imported, here or
# anywhere downstream: each worker is a separate process, and without this the
# W of them each open a thread pool and oversubscribe the machine. The linear
# algebra is a 1x1 Radau solve, so nothing is lost by pinning it to one thread.
import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
import math
import multiprocessing
import sys
import warnings

import numpy as np

from moller_scan import load_notebook_defs

HERE = os.path.dirname(os.path.abspath(__file__))
DD_NOTEBOOK = os.path.join(HERE, "Direct_detection.ipynb")
CACHE = os.path.join("data", "relic_survival_map.npz")

# Cell 38 is the last one the exclusion test needs; the cells past it only draw
# the ladder figure.
DD_LAST_CELL = 38

# The reduced Hubble constant the notebooks' own H0 corresponds to, used to
# report the Omega h^2 that micrOMEGAs also reports.
H0_TO_H = 2.1331969e-42

# Below this the A' propagator pole sits above the s = 4 mchi^2 threshold and
# sigma_ann carries no width, so the Gondolo-Gelmini integrand is infinite and
# solve_ivp raises rather than returning nan.
R_RESONANCE = 0.5

# The two limits reported, in the order their columns and blocks appear.
CASES = ("nocenuns", "cenuns")


def load_dd_defs():
    """Execute the direct detection notebook's definition cells and return their namespace."""
    with open(DD_NOTEBOOK) as fh:
        nb = json.load(fh)

    ns = {"__name__": "ddnotebook"}
    for cell in nb["cells"][:DD_LAST_CELL + 1]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        # The plot cells define nothing the exclusion test needs, and the
        # figures they write are not this script's to rewrite.
        if "plt.subplots(" in src or "plt.plot(" in src:
            continue
        exec(compile(src, "<ddnotebook>", "exec"), ns)
    return ns


# The inputs the combined limit is built from, as (mass table, sigma table,
# target cross section): the six experiment electron envelope and the two
# Migdal searches. They are combined in eps and not in sigma, because sigma_e
# and sigma_SI are not the same quantity and cannot be minimised together.
DD_PIECES = (("mass_data_best2026", "sigmaedataGeV_best2026", "sig_si_theo_elec"),
             ("mass_data_XENON1T_migdal", "sigmasidataGeV_XENON1T_migdal",
              "sig_si_theo"),
             ("mass_data_ds50_migdal", "sigmasidataGeV_ds50_migdal",
              "sig_si_theo"))

# The PandaX-4T S2 CEvNS recast, folded in only for the second case. It covers
# mchi = 0.020 to 0.894 GeV, so it can only tighten the limit inside that band.
DD_CENUNS = ("mass_data_cenuns", "sigmaedataGeV_cenuns", "sig_si_theo_elec")


def dd_eps_limit(dd_ns, ma_grid, alpha_d, r, with_cenuns=False):
    """The combined 2026 limit on eps along a mass grid, inf where uncovered.

    This repeats best2026_at's combination rather than calling it, because
    best2026_at can only answer on the notebook's own ma_dd grid, which starts
    at MAp = 0.01 GeV. The experiments reach an order of magnitude lower --
    SENSEI tabulates down to mchi = 0.53 MeV -- so going through best2026_at
    would report a floor that belongs to a grid rather than to any measurement.
    The pieces and the eps combination are the notebook's, via its own
    eps_dd_limit; agreement with best2026_at on the range they share is 1%, the
    residue of the extra interpolation through ma_dd that this path avoids.
    """
    ma_grid = np.asarray(ma_grid, dtype=float)
    mchi = r*ma_grid
    tables = DD_PIECES + (DD_CENUNS,) if with_cenuns else DD_PIECES
    pieces = []
    for mass_key, sigma_key, theo_key in tables:
        # eps_dd_limit refuses to extrapolate past a tabulated mass range, and
        # that refusal is carried through rather than papered over: a mass no
        # experiment reaches is untested, which is not the same as allowed.
        m_v, e_v = dd_ns["eps_dd_limit"](ma_grid, mchi,
                                         dd_ns[mass_key], dd_ns[sigma_key],
                                         sigma_theo=dd_ns[theo_key],
                                         alphaD=alpha_d)
        out = np.full(ma_grid.shape, np.inf)
        if m_v.size:
            inside = (ma_grid >= m_v.min()) & (ma_grid <= m_v.max())
            out[inside] = np.exp(np.interp(np.log(ma_grid[inside]),
                                           np.log(m_v), np.log(e_v)))
        pieces.append(out)
    return np.min(pieces, axis=0)


# The relic namespace is loaded once per worker process, never once per point.
_NS = None
_JOB = None


def _init_worker(job):
    """Pool initializer: load the notebooks' definitions into this process."""
    global _NS, _JOB
    # The notebooks read data/effective_dof.csv, data/rratio.dat and the direct
    # detection tables by relative path, so the worker has to stand in Code/.
    os.chdir(HERE)
    _NS = load_notebook_defs()
    _JOB = dict(job)
    _JOB["dd_ns"] = load_dd_defs()


def _safe_omega(ns):
    """omega_of_gg wrapped so a failed point stays a failed point.

    Where no annihilation channel is open -- mchi below the electron mass --
    the thermal average underflows and Radau raises ValueError rather than
    returning nan, which without this would take the whole scan down with it.
    scan_relic_line reads omega_fun.__name__ for its info dict, so the name is
    carried across.
    """
    omega_of_gg = ns["omega_of_gg"]

    def omega(ma, vare, alphaD, mchi_over_ma, **solver):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with np.errstate(all="ignore"):
                    return omega_of_gg(ma, vare, alphaD, mchi_over_ma, **solver)
        except Exception:
            return np.nan

    omega.__name__ = "omega_of_gg"
    return omega


def _one_r(r):
    """Relic eps(MAp) at one mass ratio, and both DD limits along it."""
    ns = _NS
    ma_grid = _JOB["ma_grid"]
    alpha_d = _JOB["alpha_d"]
    # scan_relic_line follows the crossing from its solved neighbours instead of
    # re-bracketing every mass, which is ~2 Boltzmann solves per mass rather
    # than ~7. It is the notebook's own function, kept as the reference.
    eps, _info = ns["scan_relic_line"](ma_grid, _JOB["eps_lo"], _JOB["eps_hi"],
                                       alpha_d, r,
                                       target=ns["OmegaDM"], tol=_JOB["tol"],
                                       omega_fun=_safe_omega(ns))
    dd_ns = _JOB["dd_ns"]
    # Both limits are pure interpolation, so the pair costs one relic line
    # rather than two.
    return (eps,
            dd_eps_limit(dd_ns, ma_grid, alpha_d, r, with_cenuns=False),
            dd_eps_limit(dd_ns, ma_grid, alpha_d, r, with_cenuns=True))


def compute(r_grid, job, workers):
    """Scan the relic line at every R and cache the result."""
    if workers > 1:
        with multiprocessing.Pool(workers, initializer=_init_worker,
                                  initargs=(job,)) as pool:
            out = pool.map(_one_r, list(r_grid))
    else:
        _init_worker(job)
        out = [_one_r(r) for r in r_grid]

    eps = np.array([o[0] for o in out])
    lim = np.array([o[1] for o in out])
    lim_cenuns = np.array([o[2] for o in out])
    np.savez(CACHE, ma=job["ma_grid"], r=r_grid, eps=eps, lim=lim,
             lim_cenuns=lim_cenuns, alpha_d=job["alpha_d"])
    return eps, lim, lim_cenuns


def intervals(mask, ma_grid):
    """Contiguous runs of an allowed mask, as (ma_lo, ma_hi, n) triples."""
    idx = np.where(mask)[0]
    if idx.size == 0:
        return []
    # The allowed set is genuinely not always one interval: the strongest part
    # of the limit can carve a hole out of the middle of it, and reporting only
    # min and max would quietly include that hole.
    runs = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
    return [(ma_grid[g[0]], ma_grid[g[-1]], len(g)) for g in runs]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--min-r", type=float, default=0.501,
                   help="Lower end of the Mchi/MAp range, above 0.5")
    p.add_argument("--max-r", type=float, default=0.65,
                   help="Upper end of the Mchi/MAp range")
    p.add_argument("--points-r", type=int, default=40,
                   help="Number of mass ratios in the range, linear in R")
    p.add_argument("--alpha-d", type=float, default=0.5,
                   help="Dark coupling (default 0.5); the allowed ranges do not"
                        " depend on it, only the eps values reported do")
    p.add_argument("--min-mass", type=float, default=0.0007,
                   help="Lower end of the MAp range, GeV; the default reaches"
                        " below the mchi = 0.53 MeV floor of the electron"
                        " tables, where direct detection stops saying anything")
    p.add_argument("--max-mass", type=float, default=2.0,
                   help="Upper end of the MAp range, GeV")
    p.add_argument("--points-mass", type=int, default=90,
                   help="Number of masses in the logarithmic grid")
    p.add_argument("--min-eps", type=float, default=1e-9,
                   help="Lower bracket for the relic eps search")
    p.add_argument("--max-eps", type=float, default=1e-1,
                   help="Upper bracket for the relic eps search")
    p.add_argument("--tol", type=float, default=0.01,
                   help="Tolerance on |log(Omega/target)| when solving eps")
    p.add_argument("--recompute", action="store_true",
                   help="Re-run the relic scan instead of reading the cache")
    p.add_argument("--workers", type=int, default=None,
                   help="Worker processes for the scan (default: all cores)")
    p.add_argument("--output", type=str, default=None,
                   help="Points CSV (default data/relic_survival_points.csv)")
    p.add_argument("--ranges-output", type=str, default=None,
                   help="Intervals CSV (default data/relic_survival_ranges.csv)")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    if args.alpha_d <= 0:
        p.error("--alpha-d must be positive")
    if args.min_r <= 0 or args.max_r < args.min_r:
        p.error("need 0 < --min-r <= --max-r")
    if args.min_r < R_RESONANCE:
        p.error("need --min-r >= 0.5; below it the A' propagator pole is on shell"
                " and sigma_ann has no width, so the thermal average diverges")
    if args.min_mass <= 0 or args.max_mass <= args.min_mass:
        p.error("need 0 < --min-mass < --max-mass")
    if args.min_eps <= 0 or args.max_eps <= args.min_eps:
        p.error("need 0 < --min-eps < --max-eps")
    if args.points_r < 1 or args.points_mass < 1:
        p.error("--points-r and --points-mass must be at least 1")

    verbose = not args.quiet
    os.chdir(HERE)
    out_csv = args.output or os.path.join("data", "relic_survival_points.csv")
    ranges_csv = args.ranges_output or os.path.join("data",
                                                    "relic_survival_ranges.csv")

    # R is gridded linearly: the structure lives in a narrow window just above
    # 0.5, where a logarithmic grid would spend its resolution elsewhere.
    r_grid = np.linspace(args.min_r, args.max_r, args.points_r)
    ma_grid = np.logspace(math.log10(args.min_mass), math.log10(args.max_mass),
                          args.points_mass)
    job = {"ma_grid": ma_grid, "alpha_d": args.alpha_d, "tol": args.tol,
           "eps_lo": args.min_eps, "eps_hi": args.max_eps}
    workers = max(1, min(args.workers or os.cpu_count() or 1, args.points_r))

    if verbose:
        print(f"alpha_D = {args.alpha_d:g}, R in [{args.min_r:g},"
              f" {args.max_r:g}] over {args.points_r} points")
        print(f"MAp in [{args.min_mass:g}, {args.max_mass:g}] GeV over"
              f" {args.points_mass} points; direct detection only")

    if args.recompute or not os.path.exists(CACHE):
        if verbose:
            print(f"solving the relic line at {len(r_grid)} mass ratios"
                  f" with {workers} worker(s)")
        eps, lim, lim_cenuns = compute(r_grid, job, workers)
    else:
        d = np.load(CACHE)
        if ("lim_cenuns" not in d
                or d["eps"].shape != (len(r_grid), len(ma_grid))
                or not np.allclose(d["ma"], ma_grid)
                or not np.allclose(d["r"], r_grid)
                or float(d.get("alpha_d", -1)) != args.alpha_d):
            p.error(f"{CACHE} was written for different grids or alpha_D, or"
                    " predates the CEvNS column; re-run with --recompute")
        eps, lim, lim_cenuns = d["eps"], d["lim"], d["lim_cenuns"]
        if verbose:
            print(f"read {CACHE}")

    ns = load_notebook_defs()
    h = ns["H0"]/H0_TO_H
    omh2 = ns["OmegaDM"]*h*h

    limits = {"nocenuns": lim, "cenuns": lim_cenuns}
    covered = {c: np.isfinite(limits[c]) for c in CASES}
    ratio = {c: np.where(covered[c], eps/limits[c], np.nan) for c in CASES}
    ok = {c: np.isfinite(eps) & covered[c] & (ratio[c] < 1.0) for c in CASES}

    with open(out_csv, "w") as fh:
        fh.write(f"# relic_survival_map.py, alphaD={args.alpha_d:g},"
                 " direct detection only (best2026)\n")
        fh.write(f"# every row has Omega h^2 = {omh2:.4f} by construction;"
                 " dd_covered=0 means no experiment reaches that mchi\n")
        fh.write("# the _cenuns columns fold in the PandaX-4T S2 CEvNS recast;"
                 " the allowed set is alphaD independent\n")
        fh.write("R,MAp,eps,eps_ddlimit,ratio,dd_covered,allowed,"
                 "eps_ddlimit_cenuns,ratio_cenuns,dd_covered_cenuns,"
                 "allowed_cenuns\n")
        for i in range(len(r_grid)):
            for j in range(len(ma_grid)):
                fh.write("%.10e,%.10e,%.10e,%.10e,%.10e,%d,%d,"
                         "%.10e,%.10e,%d,%d\n"
                         % (r_grid[i], ma_grid[j], eps[i, j],
                            lim[i, j], ratio["nocenuns"][i, j],
                            int(covered["nocenuns"][i, j]),
                            int(ok["nocenuns"][i, j]),
                            lim_cenuns[i, j], ratio["cenuns"][i, j],
                            int(covered["cenuns"][i, j]),
                            int(ok["cenuns"][i, j])))

    nrows = {c: 0 for c in CASES}
    with open(ranges_csv, "w") as fh:
        fh.write(f"# relic_survival_map.py, alphaD={args.alpha_d:g},"
                 " direct detection only (best2026)\n")
        fh.write(f"# allowed MAp intervals per R, Omega h^2 = {omh2:.4f};"
                 " case=cenuns folds in the PandaX-4T S2 CEvNS recast\n")
        fh.write("# interval counts from 0 within each (R, case); the set can"
                 " be disjoint where the limit is strongest\n")
        fh.write("R,case,interval,ma_lo,ma_hi,n_masses\n")
        for case in CASES:
            for i, r in enumerate(r_grid):
                for k, (lo, hi, n) in enumerate(intervals(ok[case][i], ma_grid)):
                    fh.write("%.10e,%s,%d,%.10e,%.10e,%d\n"
                             % (r, case, k, lo, hi, n))
                    nrows[case] += 1

    if verbose:
        nsolved = int(np.isfinite(eps).sum())
        print(f"\nrelic line solved at {nsolved} of {eps.size} (R, MAp) cells")
        for case in CASES:
            label = "without CEvNS" if case == "nocenuns" else "with CEvNS"
            nr = int(ok[case].any(axis=1).sum())
            print(f"\n{label}: {int(ok[case].sum())} cells allowed,"
                  f" {nr} of {len(r_grid)} mass ratios,"
                  f" {nrows[case]} interval(s)")
            if not ok[case].any():
                continue
            print("  %-9s %-6s %s" % ("R", "n", "MAp intervals [GeV]"))
            for i, r in enumerate(r_grid):
                segs = intervals(ok[case][i], ma_grid)
                if segs:
                    print("  %-9.4f %-6d %s"
                          % (r, int(ok[case][i].sum()),
                             ", ".join("%.4g-%.4g" % (lo, hi)
                                       for lo, hi, _ in segs)))
            # An interval that runs to the first or last mass the limit can be
            # evaluated at has not been closed by a measurement, it has been
            # closed by where the scan or the tables stop.
            edge = []
            for i in range(len(r_grid)):
                c = np.where(covered[case][i])[0]
                if c.size and (ok[case][i, c[0]] or ok[case][i, c[-1]]):
                    edge.append(i)
            if edge:
                print("  ! the allowed set runs to the edge of direct detection"
                      f" coverage at {len(edge)} of {len(r_grid)} mass ratios;"
                      " its end there is a limit of the tables or of"
                      " --min-mass/--max-mass, not a measured boundary")

        lost = int((ok["nocenuns"] & ~ok["cenuns"]).sum())
        print(f"\nthe CEvNS recast removes {lost} otherwise allowed cell(s)")
        print(f"\nwrote {out_csv} and {ranges_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
r"""Collect N random (MAp, eps, alpha_D, R) points that no bound rules out.

Points are drawn log-uniformly inside the given limits, tested against the
accelerator and direct detection bounds, and the survivors get their relic
abundance from the Moller (Gondolo-Gelmini) thermal average. Run it from Code/:

    python3 surviving_points_scan.py --points 400 --min-mass 0.01 --max-mass 10 --min-eps 1e-6 --max-eps 1e-2 --min-alpha-d 0.01 --max-alpha-d 1.0 --min-r 0.51 --max-r 2.0 --output data/surviving_points.csv

That takes about 15 s on 8 workers, roughly a fifth of the candidates surviving.
--min-r may not go below 0.5, and --rejected-output writes the rejected
candidates with the bound that caught each one.

Three '#' comment lines carry what produced the file, then the header, so read
it back with

    np.genfromtxt("data/surviving_points.csv", delimiter=",", names=True,
                  skip_header=3)

Exclusion does not depend on Omega, so the cheap geometric tests decide
acceptance and only survivors pay for a Boltzmann solve.
"""

# The BLAS thread limits have to be set before numpy is imported, here or
# anywhere downstream: each worker is a separate process, and without this the
# W of them each open a thread pool and oversubscribe the machine. The linear
# algebra is a 1x1 Radau solve, so nothing is lost by pinning it to one thread.
import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import json
import math
import multiprocessing
import sys
import time
import warnings

import numpy as np
from matplotlib.path import Path as MplPath

from moller_scan import load_notebook_defs

HERE = os.path.dirname(os.path.abspath(__file__))
DD_NOTEBOOK = os.path.join(HERE, "Direct_detection.ipynb")

# Cell 38 defines best2026_at, the last thing the exclusion test needs; the
# cells past it only draw the ladder figure.
DD_LAST_CELL = 38

# The reduced Hubble constant the notebooks' own H0 corresponds to, used to
# turn Omega into the Omega h^2 that micrOMEGAs reports. Derived rather than
# typed so the two cannot drift apart.
H0_TO_H = 2.1331969e-42

# Below this the A' propagator pole sits above the s = 4 mchi^2 threshold and
# sigma_ann carries no width, so the Gondolo-Gelmini integrand is infinite and
# solve_ivp raises rather than returning nan.
R_RESONANCE = 0.5

# The memoised thermal-average table is keyed on (ma, mchi), which is distinct
# for every sampled point, so it never hits and only grows.
GG_CACHE_LIMIT = 200


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


class Bounds:
    """The exclusion test the notebook never wrote down, over its own limits."""

    def __init__(self, ns, use_polygons=True):
        self.best2026_at = ns["best2026_at"]
        self.polygons = {}
        if use_polygons:
            for name, arr in ns["bounds_lines"].items():
                # The notebook fills these on log-log axes, so the region a
                # reader sees is the polygon with edges straight in log space,
                # not in linear space. Testing in linear coordinates would
                # bulge every edge the wrong way.
                self.polygons[name] = MplPath(np.log(arr[:, :2]))

    def polygon_hit(self, ma, eps):
        """Name of the first accelerator or supernova region containing the point."""
        pt = (math.log(ma), math.log(eps))
        for name, path in self.polygons.items():
            if path.contains_point(pt):
                return name
        return None

    def eps_limit(self, ma, alpha_d, r, with_cenuns=False):
        """Combined direct detection limit on eps at one point, inf where uncovered."""
        m_lim, e_lim = self.best2026_at(alpha_d, r, with_cenuns=with_cenuns)
        if len(m_lim) == 0 or ma < m_lim[0] or ma > m_lim[-1]:
            return math.inf     # outside every tabulated mass range, so untested
        return math.exp(np.interp(math.log(ma), np.log(m_lim), np.log(e_lim)))

    def classify(self, ma, eps, alpha_d, r, cenuns_hard=False):
        """(reason or None, cenuns_only) for one candidate point."""
        # The polygons are tested first: they cost less than the direct
        # detection interpolation and do not depend on alpha_D or R, so a point
        # they catch never pays for best2026_at. A point both would exclude is
        # therefore attributed to the polygon.
        hit = self.polygon_hit(ma, eps)
        if hit is not None:
            return hit, False

        if eps > self.eps_limit(ma, alpha_d, r):
            return "dd", False

        # The CEvNS recast is a reinterpretation of PandaX-4T S2 under a theory
        # assumption rather than a limit the collaboration set, so by default it
        # flags a point instead of removing it.
        cenuns_only = eps > self.eps_limit(ma, alpha_d, r, with_cenuns=True)
        if cenuns_only and cenuns_hard:
            return "cenuns", True
        return None, cenuns_only


# The relic namespace is loaded once per worker process, never once per point.
_NS = None
_CFG = None
_nsolved = 0


def _init_worker(cfg=None):
    """Pool initializer: load the relic notebook's definitions into this process."""
    global _NS, _CFG
    # The notebook reads data/effective_dof.csv and data/rratio.dat by relative
    # path, so the worker has to stand in Code/ before loading.
    os.chdir(HERE)
    _NS = load_notebook_defs()
    _CFG = cfg


def _relic(point):
    """Omega at one (ma, eps, alpha_D, R) point, nan if the solve fails."""
    global _nsolved
    ma, eps, alpha_d, r = point
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with np.errstate(all="ignore"):
                om = float(_NS["omega_of_gg"](ma, eps, alpha_d, r))
    except Exception:
        # A bad point is a failed point and not a failed scan.
        om = math.nan

    _bump_cache()
    return om


def _solve_eps(point):
    """(eps, Omega) reproducing the relic abundance at one (ma, alpha_D, R), nan if none."""
    ma, alpha_d, r = point
    eps_lo, eps_hi, target, tol = _CFG

    # Every Omega the root finder asks for is kept, so the abundance actually
    # achieved at the returned eps can be reported rather than assumed equal to
    # the target. solve_window converges to within tol in log Omega, not to it.
    seen = {}

    def omfun(m, vare):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with np.errstate(all="ignore"):
                    om = float(_NS["omega_of_gg"](m, vare, alpha_d, r))
        except Exception:
            om = math.nan
        seen[vare] = om
        return om

    try:
        eps, _side = _NS["solve_window"](omfun, ma, eps_lo, eps_hi, target, tol)
    except Exception:
        eps = math.nan
    _bump_cache()

    if not (isinstance(eps, float) and math.isfinite(eps)):
        return math.nan, math.nan
    # The tolerance branch returns the probe it has just evaluated, so the
    # lookup normally hits; only the exhausted branch extrapolates past it.
    return eps, seen.get(eps, omfun(ma, eps))


def _bump_cache():
    """Drop the memoised thermal averages periodically, since none of them recur."""
    global _nsolved
    _nsolved += 1
    if _nsolved % GG_CACHE_LIMIT == 0:
        _NS["_gg_cache"].clear()


def loguniform(rng, lo, hi, n):
    """n samples drawn uniformly in log10 between lo and hi."""
    return 10.0**rng.uniform(math.log10(lo), math.log10(hi), n)


def _write_rows(fh, survivors, flags, omegas, h):
    """Stream one batch of solved points out, returning (rows, flagged, failed)."""
    nrow = nflag = nfail = 0
    # omegas may still be an unconsumed pool.imap, so the rows are written as
    # they arrive rather than gathered first.
    for (ma_i, eps_i, aD_i, r_i), flag, om in zip(survivors, flags, omegas):
        if not math.isfinite(om):
            nfail += 1
        fh.write(f"{ma_i:.10e},{r_i:.10e},{aD_i:.10e},{eps_i:.10e},"
                 f"{om*h*h:.10e},{int(flag)}\n")
        nrow += 1
        nflag += int(flag)
    return nrow, nflag, nfail


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--points", type=int, required=True,
                   help="Number of surviving points to collect")
    p.add_argument("--min-mass", type=float, required=True,
                   help="Lower end of the MAp range, GeV")
    p.add_argument("--max-mass", type=float, required=True,
                   help="Upper end of the MAp range, GeV")
    p.add_argument("--min-eps", type=float, required=True,
                   help="Lower end of the kinetic mixing range")
    p.add_argument("--max-eps", type=float, required=True,
                   help="Upper end of the kinetic mixing range")
    p.add_argument("--min-alpha-d", type=float, required=True,
                   help="Lower end of the dark coupling range")
    p.add_argument("--max-alpha-d", type=float, required=True,
                   help="Upper end of the dark coupling range")
    p.add_argument("--min-r", type=float, required=True,
                   help="Lower end of the Mchi/MAp range, at least 0.5")
    p.add_argument("--max-r", type=float, required=True,
                   help="Upper end of the Mchi/MAp range")
    p.add_argument("--solve-eps", action="store_true",
                   help="Solve eps for the observed relic abundance instead of sampling"
                        " it, so every point lies on the relic surface; --min-eps and"
                        " --max-eps then bracket the search")
    p.add_argument("--relic-tol", type=float, default=0.01,
                   help="Tolerance on |log(Omega/target)| when solving eps (default 0.01)")
    p.add_argument("--with-cenuns", action="store_true",
                   help="Treat the PandaX-4T S2 CEvNS recast as a hard cut too")
    p.add_argument("--no-polygons", action="store_true",
                   help="Skip the accelerator and supernova regions, direct detection only")
    p.add_argument("--seed", type=int, default=0,
                   help="Seed for the sampler; a run is reproducible from it")
    p.add_argument("--workers", type=int, default=None,
                   help="Worker processes for the relic solves (default: all cores)")
    p.add_argument("--max-attempts", type=int, default=None,
                   help="Give up after this many candidates (default: 1000 per point)")
    p.add_argument("--output", type=str, default=None,
                   help="Output CSV (default data/surviving_points.csv)")
    p.add_argument("--rejected-output", type=str, default=None,
                   help="Optional second CSV holding the rejected candidates and why")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    if args.points < 1:
        p.error("--points must be at least 1")
    if args.min_mass <= 0 or args.max_mass <= args.min_mass:
        p.error("need 0 < --min-mass < --max-mass")
    if args.min_eps <= 0 or args.max_eps <= args.min_eps:
        p.error("need 0 < --min-eps < --max-eps")
    if args.min_alpha_d <= 0 or args.max_alpha_d <= args.min_alpha_d:
        p.error("need 0 < --min-alpha-d < --max-alpha-d")
    if args.min_r <= 0 or args.max_r <= args.min_r:
        p.error("need 0 < --min-r < --max-r")
    if args.min_r < R_RESONANCE:
        p.error("need --min-r >= 0.5; below it the A' propagator pole is on shell"
                " and sigma_ann has no width, so the thermal average diverges")

    verbose = not args.quiet
    os.chdir(HERE)

    out_path = args.output or os.path.join("data", "surviving_points.csv")
    max_attempts = args.max_attempts or 1000*args.points
    workers = args.workers or os.cpu_count() or 1
    workers = max(1, min(workers, args.points))

    if verbose:
        print("loading the notebooks")
    ns = load_notebook_defs()
    dd = Bounds(load_dd_defs(), use_polygons=not args.no_polygons)
    h = ns["H0"]/H0_TO_H

    if verbose:
        print(f"MAp in [{args.min_mass:g}, {args.max_mass:g}] GeV,"
              f" eps in [{args.min_eps:g}, {args.max_eps:g}]")
        print(f"alpha_D in [{args.min_alpha_d:g}, {args.max_alpha_d:g}],"
              f" Mchi/MAp in [{args.min_r:g}, {args.max_r:g}]")
        print(f"collecting {args.points} surviving points with {workers} worker(s),"
              f" seed {args.seed}")

    # The root finder works in Omega, as the notebook does; only the output is
    # converted to the Omega h^2 that micrOMEGAs reports.
    cfg = (args.min_eps, args.max_eps, ns["OmegaDM"], args.relic_tol)

    rng = np.random.default_rng(args.seed)
    pool = None
    if workers > 1:
        pool = multiprocessing.Pool(workers, initializer=_init_worker,
                                    initargs=(cfg,))
    else:
        _init_worker(cfg)

    fh = open(out_path, "w", buffering=1)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    bounds_note = ("best2026" if args.no_polygons else "best2026+kling16")
    if args.with_cenuns:
        bounds_note += "+cenuns"
    eps_note = ("solved" if args.solve_eps else "sampled")
    fh.write(f"# surviving_points_scan.py, {stamp}\n")
    fh.write(f"# N={args.points}, seed={args.seed}, channels=all, eps={eps_note},"
             f" bounds={bounds_note}, rescaling=none, h={h:.6f},"
             f" Omega_target={ns['OmegaDM']*h*h:.5f}\n")
    fh.write(f"# MAp=[{args.min_mass:g},{args.max_mass:g}],"
             f" eps=[{args.min_eps:g},{args.max_eps:g}],"
             f" alphaD=[{args.min_alpha_d:g},{args.max_alpha_d:g}],"
             f" R=[{args.min_r:g},{args.max_r:g}]\n")
    fh.write("MAp,R,alphaD,eps,omegah2,cenuns_excluded\n")

    rej_fh = None
    if args.rejected_output:
        rej_fh = open(args.rejected_output, "w", buffering=1)
        rej_fh.write(f"# surviving_points_scan.py rejected candidates, {stamp}\n")
        rej_fh.write(f"# N={args.points}, seed={args.seed}, bounds={bounds_note}\n")
        rej_fh.write("#\n")
        rej_fh.write("MAp,R,alphaD,eps,reason\n")

    kept = 0
    nflagged = 0
    nfailed = 0
    drawn = 0
    rejections = {}
    t0 = time.time()

    try:
        while kept < args.points and drawn < max_attempts:
            # Size the batch from the acceptance measured so far, with a little
            # overshoot so the run does not trail off in batches of one.
            rate = (kept/drawn) if drawn and kept else 0.25
            size = int(math.ceil((args.points - kept)/max(rate, 1e-3)*1.2))
            size = int(np.clip(size, 64, 20000))
            size = min(size, max_attempts - drawn)

            # Every coordinate is drawn in the parent, in a fixed order, so the
            # sample does not depend on how the work is later distributed.
            ma_c = loguniform(rng, args.min_mass, args.max_mass, size)
            eps_c = loguniform(rng, args.min_eps, args.max_eps, size)
            aD_c = loguniform(rng, args.min_alpha_d, args.max_alpha_d, size)
            r_c = loguniform(rng, args.min_r, args.max_r, size)
            drawn += size

            if args.solve_eps:
                # eps is not known until the abundance is solved for, so here the
                # expensive step comes first and the bounds filter its output.
                # The sampled eps column is discarded; the draw still happens so
                # that a given seed places the other three coordinates
                # identically in both modes.
                seeds = [(float(ma_c[i]), float(aD_c[i]), float(r_c[i]))
                         for i in range(size)]
                if pool is None:
                    solved = [_solve_eps(pt) for pt in seeds]
                else:
                    solved = list(pool.imap(_solve_eps, seeds, chunksize=4))

                survivors, flags, omegas = [], [], []
                for (ma_i, aD_i, r_i), (eps_i, om_i) in zip(seeds, solved):
                    if not math.isfinite(eps_i):
                        rejections["no relic eps"] = rejections.get("no relic eps", 0) + 1
                        continue
                    reason, cenuns_only = dd.classify(ma_i, eps_i, aD_i, r_i,
                                                      cenuns_hard=args.with_cenuns)
                    if reason is not None:
                        rejections[reason] = rejections.get(reason, 0) + 1
                        if rej_fh is not None:
                            rej_fh.write(f"{ma_i:.10e},{r_i:.10e},{aD_i:.10e},"
                                         f"{eps_i:.10e},{reason}\n")
                        continue
                    survivors.append((ma_i, eps_i, aD_i, r_i))
                    flags.append(cenuns_only)
                    omegas.append(om_i)
                    if kept + len(survivors) >= args.points:
                        break
                nrow, nflag, nfail = _write_rows(fh, survivors, flags, omegas, h)
                kept += nrow
                nflagged += nflag
                nfailed += nfail
                if verbose:
                    print(f"  {kept}/{args.points} kept, {drawn} drawn,"
                          f" accept {100.0*kept/drawn:.1f}%")
                continue

            survivors, flags = [], []
            for i in range(size):
                reason, cenuns_only = dd.classify(
                    float(ma_c[i]), float(eps_c[i]), float(aD_c[i]), float(r_c[i]),
                    cenuns_hard=args.with_cenuns)
                if reason is not None:
                    rejections[reason] = rejections.get(reason, 0) + 1
                    if rej_fh is not None:
                        rej_fh.write(f"{ma_c[i]:.10e},{r_c[i]:.10e},{aD_c[i]:.10e},"
                                     f"{eps_c[i]:.10e},{reason}\n")
                    continue
                survivors.append((float(ma_c[i]), float(eps_c[i]),
                                  float(aD_c[i]), float(r_c[i])))
                flags.append(cenuns_only)
                # Surplus is dropped before the relic solve, never after, so no
                # Boltzmann integration is wasted on a point that is not written.
                if kept + len(survivors) >= args.points:
                    break

            if not survivors:
                continue

            if pool is None:
                omegas = (_relic(pt) for pt in survivors)
            else:
                omegas = pool.imap(_relic, survivors, chunksize=8)

            nrow, nflag, nfail = _write_rows(fh, survivors, flags, omegas, h)
            kept += nrow
            nflagged += nflag
            nfailed += nfail

            if verbose:
                print(f"  {kept}/{args.points} kept, {drawn} drawn,"
                      f" accept {100.0*kept/drawn:.1f}%")
    except KeyboardInterrupt:
        if pool is not None:
            pool.terminate()
            pool = None
        print("\ninterrupted; the partial file is still valid", file=sys.stderr)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
        fh.close()
        if rej_fh is not None:
            rej_fh.close()

    elapsed = time.time() - t0
    if verbose:
        print(f"\nkept {kept} of {drawn} candidates"
              f" ({100.0*kept/max(drawn, 1):.2f}% accepted) in {elapsed:.1f} s"
              f" ({elapsed/max(kept, 1):.2f} s per surviving point)")
        if nflagged:
            print(f"  {nflagged} of them are excluded by the CEvNS recast alone")
        if nfailed:
            print(f"! {nfailed} relic solves failed and were written as nan")
        if rejections:
            print("rejected by:")
            for name, count in sorted(rejections.items(), key=lambda kv: -kv[1]):
                print(f"  {name:<12} {count:>7}  ({100.0*count/max(drawn, 1):.2f}%)")
        print(f"wrote {out_path}")

    if kept < args.points:
        worst = max(rejections.items(), key=lambda kv: kv[1])[0] if rejections else "none"
        print(f"! only {kept} of {args.points} points found in {drawn} candidates;"
              f" {worst} rejected the most", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

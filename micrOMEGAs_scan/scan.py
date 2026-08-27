'''
Traces the relic line eps(MAp) with micrOMEGAs.

alpha_D and Mchi/MAp are fixed in config.py; only MAp and eps are scanned.
The mass grid is logarithmic and its size is an input.  The precision in eps
is not: it follows from --tol, which is a tolerance on Omega h^2, since that
is the quantity being solved for.

Example:

python3 scan.py --min-mass 1e-2 --max-mass 3.0 --points-mass 100 \
                --min-eps 1e-6 --max-eps 1e-2

The structure is the one from Code/Relic_Abundance.ipynb.  Rather than
evaluating a full grid, the crossing is found once and then followed: each
mass predicts where its neighbour's crossing is and walks the short distance
to it, so a solved mass costs one or two micrOMEGAs calls instead of the four
or so a cold bracketed search needs.  Anything that fails there falls back to
the cold search, which also reports which side of the eps window the crossing
lies on -- and that is what stops the scan instead of letting it search on
every remaining mass.
'''

import argparse
import math
import sys

import numpy as np

import config
from micromegas import MockOracle, Oracle
from rootfind import predict_eps, refine_eps, solve_window


class Row:
    '''One solved mass.'''

    __slots__ = ('i', 'ma', 'eps', 'om', 'xf', 'method', 'ncalls')

    def __init__(self, i, ma, eps, om, xf, method, ncalls):
        self.i, self.ma, self.eps = i, ma, eps
        self.om, self.xf = om, xf
        self.method, self.ncalls = method, ncalls


def seed_search(omfun, ma_grid, e_lo, e_hi, target, tol, verbose=True):
    '''Find the first mass whose crossing lies inside the eps window.

    The ends of the mass range are tried first and then the search moves
    inward from the light end, so a line that only enters the window at one
    extreme is found in two evaluations of the window rather than after
    marching through half the grid.

    eps(MAp) rises, so the two ends also settle the question on their own in
    the hopeless cases: a crossing above the window at the lightest mass is
    above it everywhere, and one below the window at the heaviest mass is
    below it everywhere.  Either one ends the scan immediately.

    Returns (index, eps, note).  index is None if no seed was found, and note
    says why.
    '''
    n = len(ma_grid)
    order = [0] + ([n - 1] if n > 1 else []) + [i for i in range(1, n - 1)]

    for i in order:
        ma = ma_grid[i]
        eps, side = solve_window(omfun, ma, e_lo, e_hi, target, tol)
        if verbose:
            where = f'{side}' if side else 'oracle failed'
            print(f'  seed probe at MAp = {ma:.6g} GeV: {where}'
                  + (f', eps = {eps:.6g}' if side == 'inside' else ''))
        if side == 'inside' and math.isfinite(eps):
            return i, eps, None
        if i == 0 and side == 'above':
            return None, math.nan, (
                f'at the lightest mass ({ma:.6g} GeV) the crossing already lies '
                f'above eps = {e_hi:g}; since eps(MAp) rises, no mass in the '
                f'range is reachable')
        if i == n - 1 and side == 'below':
            return None, math.nan, (
                f'at the heaviest mass ({ma:.6g} GeV) the crossing still lies '
                f'below eps = {e_lo:g}; since eps(MAp) rises, no mass in the '
                f'range is reachable')

    return None, math.nan, (f'no mass in the grid has its crossing inside '
                            f'[{e_lo:g}, {e_hi:g}]')


def walk(omfun, lookup, ma_grid, start, step, solved, e_lo, e_hi, target, tol,
         emit, verbose=True, early_stop=True):
    '''Follow the line from index start in direction step (+1 or -1).

    solved is the shared list of (MAp, eps) found so far, which the prediction
    reads and every solved mass appends to, so the two directions help each
    other: the downward walk starts out extrapolating through the seed and its
    upward neighbour.

    Stopping is asymmetric because the failure that ends a direction is.  Going
    up, the crossing eventually needs a larger eps than the window holds
    ('above') and nothing heavier can be solved; going down it eventually needs
    a smaller one ('below').  The opposite report means the line has stepped
    out the far side, which is not a reason to stop, so the walk carries on.

    That reasoning is monotonicity in eps(MAp), which a resonance in the
    annihilation cross section can break: the line can leave the window and
    come back.  early_stop=False keeps searching every remaining mass instead,
    at roughly four evaluations per unsolvable one.
    '''
    fatal = 'above' if step > 0 else 'below'
    n = len(ma_grid)
    i = start + step
    stop_note = None
    rows = []

    while 0 <= i < n:
        ma = ma_grid[i]
        before = counter(omfun)
        guess = predict_eps(ma, solved)

        eps, method = math.nan, None
        if guess is not None and e_lo <= guess <= e_hi:
            eps = refine_eps(omfun, ma, guess, e_lo, e_hi, target, tol)
            method = 'refine'

        if not math.isfinite(eps):
            eps, side = solve_window(omfun, ma, e_lo, e_hi, target, tol)
            method = 'window'
            if side == fatal and early_stop:
                extra = (f' (predicted eps = {guess:.3e})'
                         if guess is not None else '')
                stop_note = (f'crossing lies {side} the eps window at '
                             f'MAp = {ma:.6g} GeV{extra}')
                break
            # The far side, or an oracle failure: this mass is not solvable
            # but the next one may be, so keep going.

        used = counter(omfun) - before
        if math.isfinite(eps):
            om, xf = lookup(ma, eps)
            row = Row(i, ma, eps, om, xf, method, used)
            rows.append(row)
            solved.append((ma, eps))
            emit(row)
            if verbose:
                print(f'  MAp = {ma:.6g} GeV   eps = {eps:.6g}   '
                      f'Omega h^2 = {om:.5g}   xf = {xf:.4g}   '
                      f'[{method}, {used} call{"s" if used != 1 else ""}]')
        elif verbose:
            print(f'  MAp = {ma:.6g} GeV   no crossing in the window '
                  f'[{used} calls]')
        i += step

    unscanned = (n - i) if step > 0 else (i + 1)
    return rows, stop_note, max(unscanned, 0)


def counter(omfun):
    '''Evaluations charged so far, for the per-mass cost column.'''
    return getattr(omfun, 'nlookups', 0)


def main(argv=None):
    p = argparse.ArgumentParser(
        description='Trace eps(MAp) at fixed relic abundance with micrOMEGAs.')
    p.add_argument('--min-mass', type=float, required=True,
                   help="Lower end of the MAp range, GeV")
    p.add_argument('--max-mass', type=float, required=True,
                   help="Upper end of the MAp range, GeV")
    p.add_argument('--points-mass', type=int, required=True,
                   help='Number of masses in the logarithmic grid')
    p.add_argument('--min-eps', type=float, required=True,
                   help='Lower end of the eps window searched at each mass')
    p.add_argument('--max-eps', type=float, required=True,
                   help='Upper end of the eps window searched at each mass')
    p.add_argument('--target', type=float, default=config.OMEGA_TARGET,
                   help='Omega h^2 to solve for (default from config.py)')
    p.add_argument('--tol', type=float, default=0.01,
                   help='Tolerance on |log(Omega/target)|; sets the eps precision')
    p.add_argument('--alpha-d', type=float, default=None,
                   help='Override config.ALPHA_D for this run')
    p.add_argument('--mchi-over-map', type=float, default=None,
                   help='Override config.MCHI_OVER_MAP for this run')
    p.add_argument('--output', type=str, default=None,
                   help='Output CSV (default data/relic_line.csv)')
    p.add_argument('--cache', type=str, default=None,
                   help='Evaluation cache CSV (default data/cache.csv)')
    p.add_argument('--no-cache', action='store_true',
                   help='Do not read or write the evaluation cache')
    p.add_argument('--keep-cache', action='store_true',
                   help='Keep the cache file after a run that completes; by '
                        'default it is deleted, since it exists to resume an '
                        'interrupted scan and cannot tell that the model has '
                        'changed under it')
    p.add_argument('--timeout', type=float, default=None,
                   help='Seconds allowed per micrOMEGAs call')
    p.add_argument('--no-early-stop', action='store_true',
                   help='Keep searching after the line leaves the eps window. '
                        'Needed only if a resonance makes eps(MAp) non-monotonic, '
                        'and it costs a full window search per unsolvable mass')
    p.add_argument('--no-sort', action='store_true',
                   help='Leave the output in walk order instead of sorting by mass')
    p.add_argument('--mock', action='store_true',
                   help='Use the analytic mock oracle instead of micrOMEGAs')
    p.add_argument('--quiet', action='store_true')
    args = p.parse_args(argv)

    if args.min_mass <= 0 or args.max_mass <= args.min_mass:
        p.error('need 0 < --min-mass < --max-mass')
    if args.min_eps <= 0 or args.max_eps <= args.min_eps:
        p.error('need 0 < --min-eps < --max-eps')
    if args.points_mass < 1:
        p.error('--points-mass must be at least 1')

    verbose = not args.quiet
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = args.output or (config.DATA_DIR / 'relic_line.csv')
    cache_path = None if (args.no_cache or args.mock) else (
        args.cache or (config.DATA_DIR / 'cache.csv'))

    if args.mock:
        oracle = MockOracle(target=args.target)
    else:
        oracle = Oracle(alpha_d=args.alpha_d, mchi_over_map=args.mchi_over_map,
                        timeout=args.timeout, cache_path=cache_path)

    ma_grid = np.logspace(math.log10(args.min_mass), math.log10(args.max_mass),
                          args.points_mass)

    alpha_d = getattr(oracle, 'alpha_d',
                      args.alpha_d if args.alpha_d is not None else config.ALPHA_D)
    ratio = getattr(oracle, 'mchi_over_map',
                    args.mchi_over_map if args.mchi_over_map is not None
                    else config.MCHI_OVER_MAP)
    if verbose:
        print(f'alpha_D = {alpha_d:g}, Mchi/MAp = {ratio:g}, '
              f'Omega h^2 target = {args.target:g}, tol = {args.tol:g}')
        print(f'MAp in [{args.min_mass:g}, {args.max_mass:g}] GeV, '
              f'{args.points_mass} points; '
              f'eps in [{args.min_eps:g}, {args.max_eps:g}]')
        if args.mock:
            print('using the MOCK oracle: results are not physics')
        print('\nseeding:')

    fh = open(out_path, 'w', buffering=1)
    fh.write('MAp,eps,omegah2,xf,method,ncalls\n')

    def emit(row):
        fh.write(f'{row.ma:.10e},{row.eps:.10e},{row.om:.10e},{row.xf:.10e},'
                 f'{row.method},{row.ncalls}\n')

    before = counter(oracle)
    i_seed, eps_seed, note = seed_search(oracle, ma_grid, args.min_eps,
                                         args.max_eps, args.target, args.tol,
                                         verbose)
    if i_seed is None:
        fh.close()
        print(f'\nno starting point found: {note}')
        print(f'{oracle.nlookups} evaluations '
              f'({getattr(oracle, "ncalls", 0)} micrOMEGAs calls)')
        # The cache is kept here on purpose: no seed usually means the eps
        # window was wrong, and the retry that widens it re-uses these points.
        if cache_path is not None:
            print(f'cache kept at {cache_path} (the run did not complete)')
        oracle.close()
        return 1

    ma_seed = ma_grid[i_seed]
    om_seed, xf_seed = oracle.lookup(ma_seed, eps_seed)
    seed_row = Row(i_seed, ma_seed, eps_seed, om_seed, xf_seed, 'seed',
                   counter(oracle) - before)
    emit(seed_row)
    solved = [(ma_seed, eps_seed)]
    rows = [seed_row]
    if verbose:
        print(f'  seed at index {i_seed}: MAp = {ma_seed:.6g} GeV, '
              f'eps = {eps_seed:.6g}, Omega h^2 = {om_seed:.5g}, '
              f'xf = {xf_seed:.4g}\n')

    notes, unscanned = {}, {}
    for label, step in (('up', +1), ('down', -1)):
        if not (0 <= i_seed + step < len(ma_grid)):
            continue
        if verbose:
            print(f'walking {label} in mass from index {i_seed}:')
        got, stop_note, left = walk(oracle, oracle.lookup, ma_grid, i_seed, step,
                                    solved, args.min_eps, args.max_eps,
                                    args.target, args.tol, emit, verbose,
                                    early_stop=not args.no_early_stop)
        rows.extend(got)
        notes[label] = stop_note
        unscanned[label] = left
        if verbose:
            if stop_note:
                print(f'  stopped: {stop_note}; {left} masses left unscanned')
            print()

    fh.close()

    if not args.no_sort:
        rows.sort(key=lambda r: r.ma)
        with open(out_path, 'w') as fh2:
            fh2.write('MAp,eps,omegah2,xf,method,ncalls\n')
            for r in rows:
                fh2.write(f'{r.ma:.10e},{r.eps:.10e},{r.om:.10e},{r.xf:.10e},'
                          f'{r.method},{r.ncalls}\n')

    nfound = len(rows)
    nlook = oracle.nlookups
    ncalls = getattr(oracle, 'ncalls', nlook)
    print(f'solved {nfound} of {len(ma_grid)} masses in {nlook} evaluations '
          f'({nlook/max(nfound, 1):.1f} per solved mass)')
    if ncalls != nlook:
        print(f'  {ncalls} micrOMEGAs calls, {nlook - ncalls} served from cache')
    if getattr(oracle, 'nfailed', 0):
        print(f'  {oracle.nfailed} evaluations failed')
    for label in ('up', 'down'):
        if notes.get(label):
            print(f'  {label}: {notes[label]}; '
                  f'{unscanned[label]} masses left unscanned')
    if nfound:
        worst = max(abs(r.om/args.target - 1.0) for r in rows)
        print(f'  MAp from {rows[0].ma:.6g} to {rows[-1].ma:.6g} GeV, '
              f'eps from {min(r.eps for r in rows):.4g} to '
              f'{max(r.eps for r in rows):.4g}')
        print(f'  worst |Omega/target - 1| over the returned points: {worst:.3g}')
    print(f'written to {out_path}')

    # The scan finished, so there is nothing left to resume and the cache is
    # only a way to get stale results later: its keys carry alpha_D and the
    # mass ratio, but nothing carries the model, so a recompile or a swap in
    # work/models leaves every entry wrong with every key still matching.
    # An interrupted run never reaches here and keeps its cache.
    if args.keep_cache:
        if cache_path is not None:
            print(f'cache kept at {cache_path} (--keep-cache); it will be '
                  f'reused by the next run, which is only safe while the '
                  f'model is unchanged')
        oracle.close()
    else:
        removed = oracle.discard_cache()
        if removed is not None:
            print(f'cache {removed} removed (the run completed)')
    return 0


if __name__ == '__main__':
    sys.exit(main())

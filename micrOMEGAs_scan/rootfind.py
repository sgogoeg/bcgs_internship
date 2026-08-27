'''
Finding eps such that Omega h^2(MAp, eps) = target.

Three pieces, none of which know what produces Omega:

  solve_window   cold search of the whole eps window, and the report of which
                 side of it the crossing lies on
  predict_eps    where the crossing probably is at the next mass, from the
                 masses already solved
  refine_eps     the walk from that prediction to the crossing

The cold search costs about four evaluations and the warm walk one or two,
which is the whole point of the arrangement: micrOMEGAs is spawned per
evaluation, so the difference is most of the runtime of a scan.
'''

import math

import numpy as np

# ---------------------------------------------------------------------------
# d log Omega / d log eps.  Omega ~ 1/<sigma v> and <sigma v> ~ eps^2 gives -2;
# the freeze-out x drifts with the coupling, which softens it, and -1.8 is what
# that came to for the scalar model in the notebook.
#
# It is used only as the first step of a walk that has no measurement of its
# own yet.  After one step refine_eps measures the slope by secant and uses
# that, because -1.8 is a property of one model in one regime: a different
# power of the coupling, a resonance in the annihilation cross section, or a
# threshold all move it, and near m_A' = 2 m_chi it moves along the line as
# well.
SEED_SLOPE = -1.8

# Guards on the measured slope.  Omega falls with eps, so the slope is
# negative; a pair of nearly equal Omegas would otherwise produce a slope near
# zero and a step of thousands of decades.
SLOPE_MIN, SLOPE_MAX = 0.3, 8.0     # bounds on |slope|
MAX_STEP = math.log(100.0)          # at most two decades of eps per step


def _clamp_slope(slope):
    '''A usable slope, or None if the measurement is unphysical.'''
    if slope is None or not math.isfinite(slope) or slope >= 0.0:
        return None
    return -min(max(abs(slope), SLOPE_MIN), SLOPE_MAX)


def solve_window(omfun, ma, e_lo, e_hi, target, tol=0.01, itmax=30):
    '''Find the crossing Omega(eps) = target inside [e_lo, e_hi].

    Omega falls monotonically with eps, so the two ends of the window either
    bracket the target or nothing inside it does.  Take the bottom and the top
    first: Omega(e_lo) is the most DM the window can give and Omega(e_hi) the
    least, so those same two values also say which side of the window the
    crossing lies on.

    Then, instead of splitting the bracket down the middle, probe where the
    straight line through its two ends crosses the target.  log Omega is very
    nearly linear in log eps, so that probe lands almost on the root: about
    four evaluations in total, against the nine or so halvings a factor-200
    window would otherwise need.

    Returns (eps, side), side being 'inside', 'below', 'above', or None if the
    oracle failed.
    '''
    om_lo = omfun(ma, e_lo)
    if math.isfinite(om_lo) and om_lo < target:
        return math.nan, 'below'      # needs a smaller eps than the window holds
    om_hi = omfun(ma, e_hi)
    if math.isfinite(om_hi) and om_hi > target:
        return math.nan, 'above'      # needs a larger eps than the window holds
    if not (math.isfinite(om_lo) and math.isfinite(om_hi)
            and om_lo > 0 and om_hi > 0):
        return math.nan, None

    lo, hi = math.log(e_lo), math.log(e_hi)
    f_lo, f_hi = math.log(om_lo/target), math.log(om_hi/target)   # f_lo > 0 > f_hi
    held = 0

    for _ in range(itmax):
        # Where the chord through (lo, f_lo) and (hi, f_hi) hits zero:
        probe = (lo*f_hi - hi*f_lo)/(f_hi - f_lo)
        om_p = omfun(ma, math.exp(probe))
        if not math.isfinite(om_p) or om_p <= 0:
            return math.nan, None
        f_p = math.log(om_p/target)
        if abs(f_p) < tol:
            return math.exp(probe), 'inside'

        if f_p > 0:                   # still too much DM, so eps must rise
            # Illinois tweak: if the same end keeps being replaced the other
            # one goes stale and the probe crawls, so halve its weight.
            if held > 0:
                f_hi *= 0.5
            lo, f_lo, held = probe, f_p, 1
        else:
            if held < 0:
                f_lo *= 0.5
            hi, f_hi, held = probe, f_p, -1

    # Tolerance never met: interpolate across what is left of the bracket,
    # which costs nothing since both ends are already known.
    return math.exp((lo*f_hi - hi*f_lo)/(f_hi - f_lo)), 'inside'


def predict_eps(ma, solved):
    '''Extrapolate eps at mass ma from the masses already solved.

    solved is [(MAp, eps), ...] for the points found so far, in any order.
    The two entries closest to ma in log mass set a local power law, which is
    continued to ma.  Re-centring on the nearest solved eps alone would
    systematically undershoot, since eps(MAp) is rising; continuing the power
    law lands much closer and saves an evaluation or two per mass.

    Taking the nearest two rather than the last two is what lets the walk run
    downwards as well as upwards: the seed and its first upward neighbour
    extrapolate backwards just as well as forwards.

    Returned UNCLIPPED on purpose: an out-of-window prediction is the signal
    that the relic line may have left the scan range.
    '''
    usable = [(m, e) for m, e in solved
              if math.isfinite(e) and e > 0 and m > 0]
    if not usable:
        return None
    if len(usable) == 1:
        return usable[0][1]

    lma = math.log(ma)
    usable.sort(key=lambda me: abs(math.log(me[0]) - lma))
    (m1, e1), (m2, e2) = usable[0], usable[1]
    if m1 == m2:
        return e1
    slope = math.log(e2/e1)/math.log(m2/m1)
    if not math.isfinite(slope):
        return e1
    return float(e1*(ma/m1)**slope)


def refine_eps(omfun, ma, guess, e_lo, e_hi, target, tol=0.01, itmax=8,
               seed_slope=SEED_SLOPE):
    '''Walk from a nearby guess to the crossing Omega = target.

    One evaluation per step.  The step is a Newton step in log-log,

        d log eps = -log(Omega/target) / (d log Omega / d log eps),

    and the slope in the denominator is measured by secant from the previous
    two evaluations, so nothing here assumes the -2 power counting holds.
    Only the very first step, which has no pair to measure from, uses
    seed_slope.

    Returns nan if the walk leaves [e_lo, e_hi], if the oracle fails, or if
    itmax runs out with the tolerance still unmet.  Returning nan rather than
    the last iterate matters: the caller reacts by searching the whole window,
    whereas a returned iterate would be recorded as a solved point and would
    then poison every extrapolation after it.
    '''
    if not (math.isfinite(guess) and guess > 0):
        return math.nan
    llo, lhi = math.log(e_lo), math.log(e_hi)
    le = math.log(guess)
    prev = None                       # (log eps, log Omega) of the last step

    for _ in range(itmax):
        om = omfun(ma, math.exp(le))
        if not math.isfinite(om) or om <= 0:
            return math.nan
        lom = math.log(om)
        r = lom - math.log(target)
        if abs(r) < tol:
            return math.exp(le)

        slope = None
        if prev is not None and prev[0] != le:
            slope = _clamp_slope((lom - prev[1])/(le - prev[0]))
        if slope is None:
            slope = seed_slope

        prev = (le, lom)
        step = -r/slope
        le += math.copysign(min(abs(step), MAX_STEP), step)
        if not (llo <= le <= lhi):
            return math.nan
    return math.nan


def measure_slope(omfun, ma, eps, target, frac=0.05):
    '''d log Omega / d log eps at a solved point, by finite difference.

    Not used by the scan; it is here for checking, at a given mass, how far
    the local slope really is from the -1.8 that seeds the first step.
    '''
    e1, e2 = eps*(1.0 - frac), eps*(1.0 + frac)
    om1, om2 = omfun(ma, e1), omfun(ma, e2)
    if not (math.isfinite(om1) and math.isfinite(om2) and om1 > 0 and om2 > 0):
        return math.nan
    return math.log(om2/om1)/math.log(e2/e1)

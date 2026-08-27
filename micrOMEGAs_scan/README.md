# micrOMEGAs relic-line scan

Traces the line `eps(MAp)` on which micrOMEGAs returns the observed relic
abundance, at fixed `alpha_D` and `Mchi/MAp`.

This is the logic of `Code/Relic_Abundance.ipynb` with micrOMEGAs in place of
the Boltzmann solver, and it differs in kind from `micrOMEGAs_scan/`: that one
evaluates every point of a grid blindly, which parallelises perfectly; this one
finds the crossing once and then follows it, so each mass costs one or two
micrOMEGAs calls instead of the four a cold search needs — but the points are
sequential by construction.

## Setup

Edit `config.py`: `MICROMEGAS_PATH`, `EXECUTABLE`, `PAR_FILE_NAME`, the
`PAR_NAMES` mapping, and the fixed `ALPHA_D` and `MCHI_OVER_MAP`. Nothing
else in the package hardcodes a path or a parameter name.

Per call the scan writes

    Mchi  <MCHI_OVER_MAP * MAp>
    MAp   <MAp>
    epsD  <eps>
    gD    <sqrt(4 pi ALPHA_D)>

into `PAR_FILE_NAME` inside `MICROMEGAS_PATH`, runs `./scan_omega <par>` there,
and reads `xf` and `Omega h^2` from the last line of stdout that is two
numbers. Set `GD` in `config.py` if you would rather fix `gD` than `alpha_D`.

## Running

    python3 scan.py --min-mass 1e-2 --max-mass 3.0 --points-mass 100 \
                    --min-eps 1e-6 --max-eps 1e-2

`--points-mass` sets the logarithmic mass grid. The precision in `eps` is not
an input: it follows from `--tol`, a tolerance on `|log(Omega/target)|`, which
is the quantity actually being solved. A `--tol` of 0.01 puts `eps` within
about `0.01/1.8`, roughly half a percent.

`--target` overrides `config.OMEGA_TARGET` (0.12) for one run;
`--alpha-d` and `--mchi-over-map` likewise.

`--mock` runs the whole machinery against an analytic surrogate, with no
micrOMEGAs involved. Useful for checking the arguments and the output format
before committing to a long run.

## Output

`data/relic_line.csv`, sorted by mass:

    MAp,eps,omegah2,xf,method,ncalls

`omegah2` is the value micrOMEGAs actually returned at the `eps` on that row,
not the target — so the column is the verification. `method` is `seed`,
`refine` (warm start) or `window` (cold search), and `ncalls` is what that mass
cost. Rows are written as they are solved, so a long run can be watched.

Every evaluation is also appended to `data/cache.csv`. The scan is
deterministic, so re-running one that was interrupted replays the earlier work
without spawning a process. `--no-cache` turns that off.

## How it works

`rootfind.py`, none of which knows micrOMEGAs exists:

- **`solve_window`** — the cold search. `Omega` falls monotonically with `eps`,
  so the two ends of the window either bracket the crossing or nothing inside
  does; the two end evaluations therefore also say which side of the window the
  crossing lies on. The bracket is then closed by log-log regula falsi with the
  Illinois tweak rather than bisection: about four evaluations against the nine
  or so halvings a wide window would need.
- **`predict_eps`** — the two solved masses nearest in log mass set a local
  power law, continued to the next mass. Returned unclipped: a prediction
  outside the window is the signal that the line has left the range.
- **`refine_eps`** — a Newton walk in log-log from that prediction, one
  evaluation per step. **The slope is measured by secant from the previous two
  evaluations**, so nothing assumes the `Omega ~ eps^-2` power counting; the
  `-1.8` in `SEED_SLOPE` is only the first step of a walk that has no
  measurement yet. That matters wherever the counting fails — a different power
  of the coupling, a threshold, or the resonance at `MAp = 2 Mchi`, where the
  slope moves along the line as well as with the model.

`scan.py` puts these together:

1. **Seed.** Cold-search the lightest mass, then the heaviest, then inward from
   the light end until one crossing lands inside the window. Because `eps(MAp)`
   rises, the two ends also settle the hopeless cases outright: above the
   window at the lightest mass means above it everywhere, below it at the
   heaviest means below it everywhere. Either ends the scan in two evaluations.
2. **Walk, both directions.** From the seed, upward and downward in mass. The
   prediction reads the shared list of solved points, so the downward walk
   starts out extrapolating through the seed and its upward neighbour rather
   than from nothing. Any failure — a prediction outside the window, a Newton
   walk that wandered off — falls back to the cold search at that mass.
3. **Stop.** Going up, a crossing reported `above` the window means nothing
   heavier is reachable; going down, `below` means nothing lighter is. The
   opposite report is the line stepping out the far side and is not a reason to
   stop. Both are monotonicity arguments, which a resonance can break —
   `--no-early-stop` keeps searching every remaining mass instead, at about
   four evaluations per unsolvable one.

A micrOMEGAs call that fails, times out, or returns unparseable output becomes
a failed point, not a failed scan: that mass is skipped, reported, and the walk
carries on across the gap. A **negative** `Omega h^2` counts as a failure too:
`darkOmega` does return one rather than an error, and the root finders need
`Omega > 0` to work in logs.

## The evaluation cache

Every micrOMEGAs evaluation is cached in `data/cache.csv`, so a scan that is
interrupted and restarted with the same grid replays its earlier work without
spawning a process. **A run that completes deletes the cache**: at that point
there is nothing left to resume, and all a surviving cache can do is go stale.
A run that is interrupted, or that ends without finding a seed, keeps it --
those are the cases where the next run has something to pick up. `--keep-cache`
keeps it regardless.

A cached point is keyed on
**alpha_D, Mchi/MAp, MAp and eps** -- all four. The first two matter as much as
the last two: they are fixed within a run but not between runs, and a key
without them hands back the previous alpha_D's Omega after `config.py` is
edited. That failure is silent in the worst way, because a cache hit spawns no
process, so it does not even rewrite the `.par` file: the visible symptom is a
`scan.par` whose `gD` never changes, while the numbers coming out are for the
old alpha_D.

Cache files written before this was fixed cannot be read -- nothing in them
records which alpha_D they hold -- and are rejected with an error naming the
file rather than loaded. Delete them, or point `--cache` elsewhere. Rebuilding
a cache costs one scan.

Note that the cache cannot see the *model*. Recompiling micrOMEGAs, or swapping
the model in `work/models/`, invalidates every cached point without changing
any key, and no check can catch it. That is the reason a completed run deletes
the cache rather than leaving it: the window in which a stale cache can be read
back is now only as long as the gap between an interrupted scan and its resume.
If you use `--keep-cache`, delete the file yourself when the model changes.

## Notes from the first real run

Against `micromegas_7.1.4/DarkPhotonComplexScalar`, `alpha_D = 0.5`,
`Mchi/MAp = 0.6`, over `MAp` in [0.05, 100] GeV with 50 points and `eps` in
[1e-6, 1e-1]: 50 masses solved in 83 micrOMEGAs calls, under a second in total.
Re-running the executable on every returned `eps` reproduces the recorded
`Omega` exactly, with a worst deviation from 0.12 of 0.8%.

Three things that turned up and are worth knowing:

- **Below about 0.03 GeV micrOMEGAs returns a negative `Omega h^2`.** Those
  masses cannot be scanned; keep `--min-mass` above it.
- **`eps(MAp)` is not monotonic** in this model — it turns over between roughly
  0.15 and 0.28 GeV, in the hadronic region. The early stop assumes
  monotonicity, so a scan whose eps window is narrow enough for the line to
  leave and re-enter it there needs `--no-early-stop`. The window above is wide
  enough that it never came up.
- **`scan_omega.cpp` prints `Omega` with `%.2e`**, three significant figures,
  so `eps` cannot be pinned down better than about 0.2% no matter what `--tol`
  says. `--tol 0.005` reaches that floor; anything smaller costs a few more
  calls and buys nothing. Widen the `printf` if you need more.

The seed in that run is worth a look as well: the cold search at the lightest
mass failed, because `Omega` at the top of the eps window came back negative,
so the seed was taken at the heavy end and the scan walked *down* the whole
grid. The lightest mass was then solved anyway, by the warm walk, which never
had to evaluate the eps that broke the cold search.

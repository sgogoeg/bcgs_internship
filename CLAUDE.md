# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Physics research code for a dark photon dark matter model (complex scalar DM
coupled through a kinetically mixed dark photon `A'`). The work computes the
relic abundance and direct detection limits, and compares hand-rolled Boltzmann
solutions against micrOMEGAs.

Scanned parameters are the dark photon mass `MAp` and the kinetic mixing `eps`.
`alpha_D` and the mass ratio `Mchi/MAp` are held fixed within any one scan.

## The pipeline

The repository is one chain, and the pieces only make sense together. Nothing
automates the chain end to end — each stage is run by hand.

1. **`FeynRules/<variant>/DP_Complex_Scalar.fr`** — the model. Three variants
   exist and they differ *only* in which fermions the dark photon current
   couples to: `Full` (charged leptons + up/down quarks), `Only leptons`
   (quarks dropped), `Only electrons` (electron alone). Comparing these three
   is the point of the exercise, not an accident.
2. **`FeynRules/<variant>/ScalarSingletDM-CH/*.mdl`** — the CalcHEP model files
   FeynRules writes out. Switching variant means copying these five files into
   the micrOMEGAs model directory's `work/models/` and rebuilding.
3. **micrOMEGAs** (external, *not* in this repo) — see paths below. Builds
   `scan_omega`, which reads a `.par` file and prints one line, `xf Omega_h^2`.
4. **`micrOMEGAs_scan/`** — traces the relic line `eps(MAp)` by calling
   `scan_omega` repeatedly. Writes a CSV.
5. **`Code/*.ipynb`** — the analysis notebooks, which read those CSVs from
   `Code/data/` and plot them against the notebooks' own Boltzmann solutions.

`Code/data/micromegas_relic_line{,_onlye,_onlyl}.csv` are the stage-4 outputs
for the `Full`, `Only electrons` and `Only leptons` variants respectively.
Nothing inside a CSV records which variant produced it — see *Validating a
micrOMEGAs scan* for the checks that recover that, and run them on every new
file.

## External dependencies (not in the repo)

- micrOMEGAs 7.1.4, model dir `~/heptools/micromegas_7.1.4/DarkPhotonComplexScalar`
- Path, executable name and `.par` filename are all set in
  `micrOMEGAs_scan/config.py`. **That file is hand-edited by the user** —
  treat `MICROMEGAS_PATH`, `EXECUTABLE` and `PAR_FILE_NAME` as theirs to set,
  and prefer CLI overrides (`--alpha-d`, `--mchi-over-map`) over editing the
  physics constants in it.

## Commands

Notebooks run on the repo-root `venv/` (Python 3.12; numpy, scipy, matplotlib,
ipykernel). It supplies a *kernel* — no jupyterlab or notebook server is
installed — so point the IDE at `venv/bin/python`.

`venv/bin/pip` carries a shebang from an earlier location of the venv and fails;
use `venv/bin/python -m pip` instead.

### Running a notebook headlessly

Do this whenever you add or edit cells; it is the only check that they actually
run. nbconvert and nbformat are not installed, so extract with plain json:

```bash
cd Code && ../venv/bin/python -c "
import json; nb=json.load(open('Relic_Abundance.ipynb'))
open('/tmp/nb.py','w').write(\"import matplotlib; matplotlib.use('Agg')\n\"+
  ''.join(''.join(c['source'])+'\n' for c in nb['cells'] if c['cell_type']=='code'))"
../venv/bin/python /tmp/nb.py
```

A full run of `Relic_Abundance.ipynb` is ~6.5 minutes, dominated by the
parameter scans.

It rewrites `Code/figures/*.pdf`, which is also the strongest available test:
**to prove a refactor was presentation-only, render `figures/*.pdf` to PNG
before and after and compare byte-for-byte.** If you only meant to check that
the cells run, `git checkout -- Code/figures/` afterwards.

### Running a scan

From `micrOMEGAs_scan/`:

```bash
python3 scan.py --min-mass 0.035 --max-mass 3.0 --points-mass 300 \
                --min-eps 1e-6 --max-eps 5e-3 --alpha-d 0.1 --tol 0.005 \
                --output ../Code/data/micromegas_relic_line.csv
```

Exercise the scanner without micrOMEGAs (analytic surrogate, no processes
spawned) — the fastest way to check changes to the root-finding logic:

```bash
python3 scan.py --mock --min-mass 1e-2 --max-mass 3.0 --points-mass 20 \
                --min-eps 1e-6 --max-eps 1e-2
```

### Rebuilding micrOMEGAs after swapping the model variant

```bash
cd ~/heptools/micromegas_7.1.4/DarkPhotonComplexScalar
cp "<repo>/FeynRules/<variant>/ScalarSingletDM-CH/"*.mdl work/models/
make main=scan_omega.cpp
```

There is no test suite, linter, or CI. Verification is by re-running the
executable on the returned points, by the figure diff above, and by the
consistency checks in *Validating a micrOMEGAs scan*.

## `micrOMEGAs_scan/` architecture

Deliberately layered, so the root finding never learns that micrOMEGAs exists:

- **`micromegas.py`** — the oracle. `Oracle(ma, eps) -> Omega`; writes the
  `.par`, spawns the executable, parses stdout, caches. `MockOracle` is a drop-in
  analytic stand-in. Any micrOMEGAs failure returns `nan` (a failed *point*,
  never a failed scan).
- **`rootfind.py`** — `solve_window` (cold log-log regula falsi with the Illinois
  tweak), `predict_eps` (local power-law extrapolation from solved neighbours),
  `refine_eps` (warm Newton walk). These see only a two-argument function.
- **`scan.py`** — the driver: seed search, then a bidirectional walk.
- **`config.py`** — every path and parameter name.

The strategy, ported from `Code/Relic_Abundance.ipynb` (`scan_relic_line`): find
the crossing once, then *follow* it. Each mass extrapolates from its solved
neighbours and walks a short distance, costing ~1–2 micrOMEGAs calls instead of
the ~4 a cold bracketed search needs. Failures fall back to the cold search. The
seed is sought at the light end, then the heavy end, then inward; the walk then
runs both directions from wherever the seed landed.

If you change the scanner, the notebook function it mirrors is the reference
implementation — keep them recognisably the same.

## The physics in `Code/Relic_Abundance.ipynb`

### Three treatments of `<sigma v>`

The notebook computes the same relic line three ways, in increasing fidelity.
They share everything except the cross section, which is the point: `dYdxcore`
is the single right-hand side and each treatment supplies its own `<sigma v>`.

1. **Closed forms** (`eps_approx0..4`) — algebraic, instant, on the full `ma`
   grid.
2. **Boltzmann + velocity expansion** (`omega_of`) — `sigmavrel`, the `6/x`
   thermal average.
3. **Boltzmann + Gondolo-Gelmini** (`omega_of_gg`) — the exact
   `<sigma v_Moller>` integral over `s`.

**The expansion is not a small correction away from the exact average.** At the
freeze-out `x ~ 20` it overestimates `<sigma v>` by about 1.8x, which moves
`eps` by ~25%. The two agree only as `1 - 15.5/x`, measured out to `x = 3000`;
that residual is a genuine `O(v^4)/O(v^2) ~ 10/x` term with an `O(1)`
coefficient, not a bug. Any comparison that mixes the two treatments is
comparing this factor, not physics.

### The non-self-conjugate factor of 2

`chi` is a complex scalar (and `psi` a Dirac fermion), so annihilation is
`chi chibar`, never `chi chi`. With `Y` counting **both** species (`gchi = 2`
for the scalar, `GDIRAC = 4` for the Dirac case) the collision term carries a
`1/2`. Four places must move together, or the line shifts by `sqrt(2)`:

- `dYdxcore` — the `1/2` (written as `/(2*3*H)`).
- `sigmavrelic` — a compensating `2`, since `sigma_0^req` doubles.
- `xfreeze` — a `1/2` inside the log, because the rate that depletes a `chi`
  is `n_chibar <sigma v> = n_eq <sigma v>/2`.
- `omega()` — **no** extra factor, because `Y` is already the total.

The equivalent convention is `Y = n_chi/s` with `g = 1` (scalar) or `2`
(Dirac), no `1/2`, and `Omega = 2 m Y s0/rho_c`. Verified numerically: the two
give `Omega = 0.033077` identically, and the mixed version gives 0.518 of that.

**Measured consequence.** With the `1/2` in, the notebook's lines sit
`sqrt(2)` above the Kling curve; with it out they land on it (expansion
Boltzmann / Kling = 0.950). That is unresolved: either Kling used the
self-conjugate convention, or something else absorbs the 2.

### Constants, all verified numerically against Kolb & Turner

- `Mpl = mpl/sqrt(8pi)` exactly; `H` coefficient `sqrt(pi^2/90)/Mpl` reproduces
  KT's `1.66/mpl` to 1e-4.
- `T0` -> 2.727 K, `H0` -> `h = 0.674`, `OmegaDM*h^2 = 0.1204`, `g*s(T0) = 3.910`.
- `3.79 = 1/0.264` where `0.264 = (2pi^2/45)/1.66`; `0.038 = 0.145 * 0.264`
  (KT's rounding, 0.65% low, worth `dx_f = 0.007`).
- `Yeq` prefactor `45/(2pi^2)(1/2pi)^{3/2} = 0.144748`, identical to KT's
  `45/(2pi^4)sqrt(pi/8)`.
- **`sigmavrelic` needs `x_f^(n+1)`, not `x_f`** — `xf**2` for the p-wave
  scalar. This was wrong once and hid because the denominator was also off by
  `x_f`; the two cancelled to within 30%.
- `Kl` and `sigmatilde` must agree: the mass correction is `(1 + r^2/2)`, the
  standard vector-current factor `(2/3)(s + 2 m_f^2)`, **not** `(1 + r^2/6)`.

### The closed forms, and how good each is

Against the Boltzmann line each belongs to, at `alphaD = 0.1`,
`mchi = 0.6 ma`:

| form | `x_f` from | accuracy |
|---|---|---|
| `eps_approx0/1` | fixed target | 0.64-0.77 — the `6e-9` target is s-wave; p-wave needs ~1.5e-8 |
| `eps_approx2` | `x_f = 20` fixed | 1.05-1.31, drifts because true `x_f` runs 15.3 -> 20.1 |
| `eps_approx3` | KT (5.48), coupled to `eps` | **1.03**, flat — the best one |
| `eps_approx4` | `Gamma = H` | 1.19 vs the expansion, 0.96 vs Moller |

`eps_approx3` requires solving `eps` and `x_f` **together** (`relic_line_point`),
since `x_f` depends on the model `sigma_0` and `sigma_0^req` depends on `x_f`.
The fixed point contracts by `~1.8/x_f ~ 0.1` per step: a dozen iterations,
vectorised over the mass grid.

`eps_approx4`'s agreement with the exact average is a **cancellation**, not
insight: `Gamma = H` overestimates `x_f` by ~15% while the expansion
overestimates `<sigma v>`. Making it self-consistent (`eps_approx5`, Moller on
both sides) moves it to 1.27 — worse. Only `eps_approx3` agrees for the right
reason.

### Gondolo-Gelmini implementation notes

- Integrate in `w = sqrt(s)` (`ds = 2w dw`), so the threshold sits at the fixed
  point `w = 2 mchi` and the decay is a plain `e^-u`, `u = (w - 2mchi)/T`.
- Use **exponentially scaled Bessels**: `K1(z) = kve(1,z)e^-z`,
  `K2(x) = kve(2,x)e^-x`, so the ratio carries `exp(2x - z) = e^-u <= 1`.
  Bare `kv` overflows past `x ~ 700`.
- Three speedups make it scan-able, none of which touches the integrand:
  `eps^2 alpha alphaD` factors straight out (`sigmav_moller_unit`); the grid
  refines only when the thermal window overlaps `sqrt(s)` in [0.28, 1.15] GeV
  (`gg_grid`, `smooth=True` skips it); and a memoised log-log cubic table in
  `x` per `(ma, mchi, sigma_fun)` (`sigmav_unit_table`). Result: ~0.15 s per
  `omega_of_gg` call, *faster* than the expansion path.
- `sigma_fun=` and `smooth=` are threaded through so channel variants share the
  same integral: `sigma_ann` (all), `sigma_ann_lep` (no hadrons),
  `sigma_ann_e` (electrons only), `sigma_ann_dirac`.
- Accuracy: quadrature within 1.7e-4 of a 64001-node reference, table
  interpolation median ~1e-5.

## `Code/Relic_Convergence.ipynb`

Five code cells, no markdown, plots only. It re-executes the parent notebook's
code cells **up to `scan_relic_line`** (so the Moller machinery never loads),
then redefines every level of approximation in the **self-conjugate** convention
and shows them converging on the Kling line. Medians: 0.63, 0.67, 1.06, 0.99,
1.14, 0.95 (Boltzmann). The convergence is real but not monotonic — level 5 is
`Gamma = H`, a cruder freeze-out condition than KT's.

Note it defines the levels explicitly rather than importing them from the parent
notebook: deriving them from live state gave a wrong answer once, because the
parent's `sigmav_target` had been edited underneath it.

## Validating a micrOMEGAs scan

Columns are `MAp, eps, omegah2, xf, method, ncalls`. Nothing in a CSV records
which model variant, `alpha_D` or target produced it, so identity has to be
recovered by comparison.

**Run these two checks on every new file. They have caught three separate
problems already.**

1. **Sub-threshold identity.** Below `m_A' = 2 m_mu/1.2 = 0.176 GeV` only
   `e+e-` is open, so *every* channel variant must agree there. A constant
   offset is an overall coupling, not a channel effect: a ratio of 0.448
   revealed a scan run at `alphaD = 0.5` (`1/0.448^2 = 4.98`). After the fix it
   reads 0.998 +- 0.002.
2. **Ordering.** `eps(all channels) <= eps(leptons) <= eps(e only)`, since
   dropping channels lowers `<sigma v>` and raises `eps`. The buggy file had
   the electron line *below* the full one, which is impossible.

Compare only on the mass range the files genuinely share — `np.interp` clamps
at the ends, so extrapolating one file past its last point manufactures a
disagreement that is not there. The all-channel file currently starts at
0.0325 GeV and the two variant files at 0.1 GeV.

Also check `md5sum` — the "all leptons" file was once a byte-identical copy of
the electron one.

**Current state, measured 08-29** (all three files, both checks passing):

| comparison | below 0.176 GeV | above the 2-pion threshold |
|---|---|---|
| electrons/leptons | 1.0000 | 1.416 |
| leptons/all | 1.0009 | 1.411 |
| electrons/all | 1.0009 | 1.997 |

Ordering holds at 100% of points; all three md5sums differ.

**Agreement with the notebook** (python Moller / micrOMEGAs, 0.10-1.62 GeV):

| channels | median | scatter in log |
|---|---|---|
| `e+e-` only | 1.025 | 0.011 |
| all leptons | 1.027 | 0.012 |
| all channels | 1.048 | 0.227 |

The scatter is the result: **20x smaller once hadrons are removed**. The two
implementations of the thermal average agree to ~2.5%; the whole all-channel
disagreement is the hadronic `R(sqrt(s))` treatment. Independently, the size of
the muon channel opening agrees to 0.4% (micrOMEGAs 0.7048, python 0.7075).

*Re-derived 08-29 against the regenerated all-channel file: unchanged.*
`all channels` is 1.0443 / 0.2263, `all leptons` 1.0272 / 0.0116, `e+e- only`
1.0254 / 0.0113. For contrast the velocity expansion against the same
micrOMEGAs line is 0.8694 / 0.2256 — the ~15% offset is the expansion, the
0.23 scatter is the hadrons, and they are independent.

## Things that will bite you

These are all established by measurement against the real code, not guesses.

- **`scan_omega` prints Omega with `%.2e`** (3 significant figures), so `eps`
  cannot be resolved better than ~0.2% no matter what `--tol` says. `--tol 0.005`
  reaches that floor. Widen the `printf` in `scan_omega.cpp` if you need more.
- **micrOMEGAs returns *negative* `Omega h^2` below ~0.03 GeV** for this model,
  and just above that Omega stops falling monotonically with `eps`, so no
  crossing can be bracketed. Keep `--min-mass` at 0.035 or above. The notebooks'
  own Boltzmann solutions go lower; that is why the curves have different
  starting masses.
- **`eps(MAp)` is not monotonic** — it turns over at the muon and 2-pion
  thresholds (`MAp` ≈ 0.176 and 0.233 GeV; the first is the same threshold the
  sub-threshold check above relies on). The early stop assumes monotonicity;
  `--no-early-stop` exists for windows narrow enough that the line leaves and
  re-enters.
- **The cache cannot see the model.** Its key covers `alpha_D`, `Mchi/MAp`,
  `MAp`, `eps` — but recompiling micrOMEGAs or swapping `work/models/`
  invalidates every entry while every key still matches, and nothing can detect
  it. A completed run therefore *deletes* its cache; only an interrupted or
  seedless run keeps one. If results look wrong after a model swap, suspect a
  surviving cache first — the `alphaD = 0.5` file that check 1 caught was
  exactly this.
- **Unit mismatch between the notebooks and micrOMEGAs.** The notebooks work in
  `Omega` (`OmegaDM = 0.265`); micrOMEGAs returns `Omega h^2`. The equivalent
  target is `OmegaDM * h**2` with `h = H0/2.1331969e-42 ≈ 0.6741`, i.e. 0.12042 —
  derived from the notebook's own `H0` so the two cannot drift apart. Do not
  compare a scan run at 0.12 against notebook curves without noting it.

## Conventions

- **Notebook output is plots only.** The notebooks were deliberately sanitized;
  when adding cells, do not add `print`. The one exception is
  `scan_relic_line`'s `if verbose:` block, which is opt-in and defaults off.
- Docstrings are a single summary line; comment blocks are one complete
  sentence.
- Notebook plots: log-log, `figsize=(10,10)`, major+minor grids, a rounded text
  box carrying the fixed parameters (`alpha_D`, `Mchi/MAp`, `Omega_DM`) read back
  out of the scan `info` dicts rather than typed in, a legend lifted above the
  axes, and `plt.savefig("figures/<name>.pdf", bbox_inches='tight')`. Match this
  when adding cells.
- Comments in this codebase explain *why* a choice was made and what was
  measured, not what the line does. Several carry numbers that were established
  empirically (e.g. the `-1.8` seed slope). Keep that register.

## Open issues

- The `sqrt(2)` offset against Kling recorded under *The non-self-conjugate
  factor of 2* is also unresolved.

*Resolved 08-29:* the two Dirac problems are fixed. The markdown now carries
`(s - 4 m_psi^2)^{-1/2}` (it read `^{+1/2}`, which is dimensionally `E^0` and
pure p-wave), and both the markdown `<sigma v>` and `sigmavreldirac` now read
`16 pi` rather than `8 pi`. Verified: the Moller average built from
`sigma_ann_dirac` now converges on `sigmavreldirac` as `1 - 10.4/x`, a clean
`O(v^2)` residual with a constant coefficient, exactly as the scalar converges
as `1 - 15.5/x`. Before the fix it sat at a flat 2.000.

*Resolved 08-29:* the note that `micromegas_relic_line.csv` had been re-run
without hadrons no longer holds. The file was regenerated on 08-27 16:13 (397
points, 0.0325-3 GeV) and now shows `leptons/all = 1.411` above the 2-pion
threshold, i.e. hadrons are present, with both consistency checks passing.

## Where things live in `Relic_Abundance.ipynb`

Cell indices drift as cells are added; grep for the `def` if one is off.

| what | cell |
|---|---|
| `sigmavrel`, `Kl`, `R`, `read_rratio` | 6 |
| `sigmavrelic`, `xfreeze`, `relic_line_point`, `sigmav_target`, `xfreeze_gamma` | 19 |
| `H`, `s`, `Yeq`, `dsdx`, `dYdxcore`, `dYdx` | 25 |
| `omega_of`, `solve_window`, `scan_relic_line` | 28 |
| `sigmatilde`, `sigma_ann`, `sigmav_moller`, `dYdxgg` | 33 |
| the fast path: `gg_grid`, `sigmav_unit_table`, `sigmav_moller_fast`, `omega_of_gg{,_e,_lep}`, `sigma_ann_{e,lep}` | 34 |
| Dirac: `sigmavreldirac`, `sigma_ann_dirac`, `GDIRAC`, `omega_of_gg_dirac` | 63 |

Scan results, all `eps_*` with a matching `info_*`/`scan_info_*`:

| variable | grid | treatment |
|---|---|---|
| `eps_relic1` | `ma_scan1` | expansion — the section that introduces it |
| `eps_exp_m` | `ma_moller` | expansion, for the Moller comparison |
| `eps_gg`, `eps_gg_e`, `eps_gg_lep`, `eps_gg_dirac` | `ma_moller` | Moller, by channel |
| `eps_relic1m`, `eps_relic2..6` | `ma_scan1..6` | Moller, the parameter scans |
| `eps_relic_dirac` | `ma_scan_dirac` | Dirac, expansion |

**`eps_relic1m` exists because `eps_relic1` cannot be switched.** The parameter
scans all run on `omega_of_gg`, but `ma_scan1` is defined back in the
velocity-expansion section, *before* `omega_of_gg` exists, so passing
`omega_fun=omega_of_gg` there is a `NameError`. The `alpha_D = 0.1` baseline is
recomputed in the parameter section instead, and the plot points at `1m`.

`ma_moller` is the comparison grid: ~50 masses, deliberately lopsided, 14 points
below 0.4 GeV and 34 between 0.4 and 1.2 where `sqrt(s) = 1.2 m_A'` sweeps the
rho, omega and phi. That is the only place the two averagings can be told apart.

## Why the expansion fails where it does

The disagreement is not spread over the mass range; it sits entirely in
0.45-0.9 GeV and **oscillates in sign** — the expansion runs ~20% high on the
rising flank of the rho, then ~40% low at the peak.

The cause is that `RKmu` samples `R` at exactly `sqrt(s) = 2 mchi`. At freeze
out `<v_rel^2> = 6/x ~ 0.3`, so `sqrt(s) = 2 mchi (1 + v^2/8)` is spread by
3-4%, i.e. +-25-30 MeV at `sqrt(s) ~ 0.8` GeV — wide compared with the omega
(8.5 MeV) and phi (4.2 MeV) widths. The Gondolo-Gelmini integral smears them;
a point sample does not. Overshooting the flanks and undershooting the peak is
the signature of a sharp resonance sampled pointwise instead of convolved.

Outside that window the two agree to a few percent, which is why removing the
hadrons collapses the scatter by 20x.

## The Kling benchmark

`Code/data/Kling/scalar_DM_Oh2_intermediate_eps_vs_mAprime.txt` — two columns,
`m_A'` in GeV and `eps`, 106 points over 0.0017-1.66 GeV.

**It uses `m_chi = 0.6 m_A'`, the same ratio as this work.** That is not stated
in the file; it was recovered from the resonance positions. Taking
`d ln eps / d ln m_A'` and locating the spikes gives `m_A' = 0.628-0.692` and
`0.841-0.865`, i.e. `sqrt(s) = 1.2 m_A' = 0.753-0.830` and `1.009-1.038` —
the rho/omega (0.775-0.783) and the phi (1.019). At `m_chi = m_A'/3` those
features would land at completely different masses. So the curve is directly
comparable and any offset is normalisation, not benchmark.

## The scalar/Dirac ratio: expect `sqrt(2 x_f)`, not `sqrt(x_f)`

Two different numbers are both correct, and mistaking one for the other looks
like a bug:

- **Closed forms** `eps_approx_1` vs `eps_approx_1_dirac` give **4.47-5.07**,
  and should be compared with `sqrt(x_f) ~ 4.45`. Those two use the *same* fixed
  `<sigma v>` target for both models, so the ratio is just
  `<sv>_Dirac/<sv>_scalar(x_f) = x_f`.
- **The relic lines** (`eps_gg` vs `eps_gg_dirac`) give **5.65-7.34, median
  6.55**, and should be compared with `sqrt(2 x_f) ~ 6.3`. The extra 2 is the
  `(n+1)` in `sigma_0^req ~ (n+1) x_f^(n+1)`: p-wave needs `2 x_f^2`, s-wave
  `x_f`, while both models share the same `sigma_0^model`. The residual above
  6.3 is the two freeze-out points differing and the exact average suppressing
  the p-wave more than the s-wave.

The in-notebook check prints the first pair against `sqrt(x_f)`. Do not apply
that yardstick to the second pair.

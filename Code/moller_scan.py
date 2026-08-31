#!/usr/bin/env python3
"""Scan the Moller relic line on the Kling mass range and cache it as a CSV.

Three '#' comment lines carry what produced the file, then a MAp,eps header.
np.genfromtxt takes the first line as the names even when it is a comment, so
read it back with

    np.genfromtxt("data/moller_relic_line.csv", delimiter=",", names=True,
                  skip_header=3)
"""

# The notebook is the reference implementation of every piece of physics here,
# so rather than duplicate it this script executes its code cells and calls
# scan_relic_line itself. Plotting cells and the cells that run their own
# scans are skipped: they cost minutes and define nothing needed here.

import argparse
import csv
import json
import os
import time

import numpy as np

NOTEBOOK = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "Relic_Abundance.ipynb")

# Everything through the fast Gondolo-Gelmini path; cell 35 onward only
# consumes it.
LAST_CELL = 34


def load_notebook_defs():
    """Execute the notebook's definition cells and return their namespace."""
    with open(NOTEBOOK) as fh:
        nb = json.load(fh)

    ns = {"__name__": "notebook"}
    for cell in nb["cells"][:LAST_CELL + 1]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        # Plot cells define nothing, and the in-notebook scans would re-run
        # the very computation this script exists to avoid.
        if ("plt.subplots(" in src or "plt.plot(" in src
                or "scan_relic_line(ma_scan" in src):
            continue
        exec(compile(src, "<notebook>", "exec"), ns)
    return ns


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
    p.add_argument("--channels", default="all",
                   choices=["all", "leptons", "electrons", "dirac"],
                   help="which annihilation channels are open")
    p.add_argument("--output", default="data/moller_relic_line.csv")
    args = p.parse_args()

    # The notebook reads data/ by relative path.
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    ns = load_notebook_defs()

    omega_fun = {"all": "omega_of_gg", "leptons": "omega_of_gg_lep",
                 "electrons": "omega_of_gg_e"}[args.channels] \
        if args.channels != "dirac" else None
    if omega_fun is None:
        raise SystemExit("the Dirac path lives in a later cell; not supported here")

    ma_grid = np.logspace(np.log10(args.min_mass), np.log10(args.max_mass),
                          args.points)

    t0 = time.time()
    eps, info = ns["scan_relic_line"](ma_grid, args.eps_lo, args.eps_hi,
                                      alphaD=args.alpha_d,
                                      mchi_over_ma=args.mchi_over_ma,
                                      tol=args.tol,
                                      omega_fun=ns[omega_fun])
    elapsed = time.time() - t0

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.output)
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

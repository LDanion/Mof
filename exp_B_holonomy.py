#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

# Tout lancer + convergence
python exp_B_holonomy.py --loop all --convergence --outdir out_expB

# Juste un cercle
python exp_B_holonomy.py --loop circle --R 1.0 --a 1.0 --N 4000 --outdir out_expB_circle

EXP B — Temporal holonomy from a non-integrable one-form (MOF)

We compute the line integral ∮_γ τ over closed loops γ in R^2,
for a non-exact 1-form: τ = a(-y dx + x dy), with dτ = 2a dx∧dy ≠ 0.

Key outputs:
- Numerical ∮ τ for different loops (circle, square, figure-eight)
- Theoretical value 2a * oriented_area(loop) where applicable
- Convergence study vs discretization N
- Figures and a CSV summary

Usage examples:
  python exp_B_holonomy.py --loop circle --R 1.0 --a 1.0 --N 2000
  python exp_B_holonomy.py --loop all --outdir out_expB
  python exp_B_holonomy.py --loop square --a 0.7 --N 400 --convergence

Dependencies: numpy, matplotlib
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# One-form (MOF) definition
# -----------------------------

@dataclass(frozen=True)
class MetronomicOneForm:
    """
    τ = a(-y dx + x dy)
    For a param curve (x(t), y(t)) sampled as points, we compute:
      ∮ τ = Σ a(-y_i Δx_i + x_i Δy_i)
    using a consistent discretization.
    """
    a: float = 1.0

    def integral_closed(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Discrete line integral over a closed curve.
        The curve is assumed closed; if not, we will close it internally.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        if x.ndim != 1 or y.ndim != 1 or x.size != y.size:
            raise ValueError("x and y must be 1D arrays of same length.")

        # Close the loop if necessary
        if not (np.isclose(x[0], x[-1]) and np.isclose(y[0], y[-1])):
            x = np.concatenate([x, x[:1]])
            y = np.concatenate([y, y[:1]])

        dx = np.diff(x)
        dy = np.diff(y)

        # Use left-point evaluation (consistent Riemann sum)
        xL = x[:-1]
        yL = y[:-1]

        return float(np.sum(self.a * (-yL * dx + xL * dy)))


def oriented_area_polygon(x: np.ndarray, y: np.ndarray) -> float:
    """
    Oriented area using the shoelace formula for a closed polygonal chain
    given by points (x_i, y_i). If not closed, closed internally.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if not (np.isclose(x[0], x[-1]) and np.isclose(y[0], y[-1])):
        x = np.concatenate([x, x[:1]])
        y = np.concatenate([y, y[:1]])

    return 0.5 * float(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))


# -----------------------------
# Loop generators
# -----------------------------

def loop_circle(R: float, N: int) -> Tuple[np.ndarray, np.ndarray, str]:
    t = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False)
    x = R * np.cos(t)
    y = R * np.sin(t)
    return x, y, f"circle(R={R})"

def loop_square(R: float, N: int) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Square centered at origin, vertices (±R, ±R).
    N points distributed along edges.
    """
    if N < 4:
        raise ValueError("N must be >= 4 for square.")
    # Points per edge (roughly)
    n = max(1, N // 4)
    # Build edges
    x1 = np.linspace(-R, R, n, endpoint=False); y1 = -R * np.ones_like(x1)
    y2 = np.linspace(-R, R, n, endpoint=False); x2 = R * np.ones_like(y2)
    x3 = np.linspace(R, -R, n, endpoint=False); y3 = R * np.ones_like(x3)
    y4 = np.linspace(R, -R, n, endpoint=False); x4 = -R * np.ones_like(y4)

    x = np.concatenate([x1, x2, x3, x4])
    y = np.concatenate([y1, y2, y3, y4])

    # If length differs from N, pad by repeating last point(s) (harmless for integral)
    if x.size < N:
        pad = N - x.size
        x = np.concatenate([x, np.full(pad, x[-1])])
        y = np.concatenate([y, np.full(pad, y[-1])])
    else:
        x = x[:N]
        y = y[:N]

    return x, y, f"square(R={R})"

def loop_figure_eight(R: float, N: int) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Lemniscate-like loop (Gerono):
      x = R cos t
      y = R sin t cos t
    This has two lobes with opposite orientation; total oriented area is 0,
    so ∮ τ should be ~ 0 for τ = a(-y dx + x dy).
    """
    t = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False)
    x = R * np.cos(t)
    y = R * np.sin(t) * np.cos(t)
    return x, y, f"figure_eight(R={R})"


LOOPS = {
    "circle": loop_circle,
    "square": loop_square,
    "figure8": loop_figure_eight,
}


# -----------------------------
# Plotting helpers
# -----------------------------

def ensure_outdir(outdir: str) -> str:
    os.makedirs(outdir, exist_ok=True)
    return outdir

def save_path_plot(x: np.ndarray, y: np.ndarray, title: str, outpath: str) -> None:
    plt.figure()
    plt.plot(x, y)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()

def save_convergence_plot(Ns: np.ndarray, errs: np.ndarray, title: str, outpath: str) -> None:
    plt.figure()
    plt.loglog(Ns, errs, marker="o")
    plt.xlabel("N (number of points)")
    plt.ylabel("abs(error)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


# -----------------------------
# Experiment runner
# -----------------------------

def run_one(loop_name: str, R: float, a: float, N: int) -> Dict[str, float]:
    if loop_name not in LOOPS:
        raise ValueError(f"Unknown loop '{loop_name}'. Choose among {list(LOOPS.keys())} or 'all'.")

    x, y, label = LOOPS[loop_name](R=R, N=N)
    mof = MetronomicOneForm(a=a)

    I = mof.integral_closed(x, y)
    A = oriented_area_polygon(x, y)
    I_th = 2.0 * a * A

    return {
        "loop": loop_name,
        "R": float(R),
        "a": float(a),
        "N": int(N),
        "integral_num": float(I),
        "area_oriented": float(A),
        "integral_theory": float(I_th),
        "abs_error": float(abs(I - I_th)),
    }


def convergence(loop_name: str, R: float, a: float, Nmin: int, Nmax: int, nsteps: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convergence study: compute abs error vs N on a log-spaced grid.
    """
    Ns = np.unique(np.round(np.logspace(np.log10(Nmin), np.log10(Nmax), nsteps)).astype(int))
    errs = []
    for N in Ns:
        res = run_one(loop_name=loop_name, R=R, a=a, N=int(N))
        errs.append(res["abs_error"])
    return Ns.astype(int), np.array(errs, dtype=float)


def write_csv(rows: List[Dict[str, float]], outpath: str) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(outpath, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    p = argparse.ArgumentParser(description="EXP B — Holonomy ∮τ for a non-integrable one-form τ = a(-y dx + x dy).")
    p.add_argument("--loop", type=str, default="all", choices=["all"] + sorted(list(LOOPS.keys())),
                   help="Which loop to compute (default: all).")
    p.add_argument("--R", type=float, default=1.0, help="Loop scale parameter (default: 1.0).")
    p.add_argument("--a", type=float, default=1.0, help="One-form amplitude a (default: 1.0).")
    p.add_argument("--N", type=int, default=2000, help="Number of discretization points (default: 2000).")
    p.add_argument("--outdir", type=str, default="out_expB", help="Output directory (default: out_expB).")
    p.add_argument("--convergence", action="store_true", help="Also run convergence study.")
    p.add_argument("--Nmin", type=int, default=50, help="Min N for convergence (default: 50).")
    p.add_argument("--Nmax", type=int, default=20000, help="Max N for convergence (default: 20000).")
    p.add_argument("--nsteps", type=int, default=10, help="Number of N samples in convergence (default: 10).")
    args = p.parse_args()

    outdir = ensure_outdir(args.outdir)

    loops_to_run = sorted(list(LOOPS.keys())) if args.loop == "all" else [args.loop]
    summary_rows: List[Dict[str, float]] = []

    # Run main computations + path plots
    for loop_name in loops_to_run:
        x, y, label = LOOPS[loop_name](R=args.R, N=args.N)
        res = run_one(loop_name=loop_name, R=args.R, a=args.a, N=args.N)
        summary_rows.append(res)

        # Save path figure
        fig_path = os.path.join(outdir, f"path_{loop_name}_N{args.N}_R{args.R}_a{args.a}.png")
        save_path_plot(x, y,
                       title=f"{label} | ∮τ (num)={res['integral_num']:.6g}, 2aA (th)={res['integral_theory']:.6g}",
                       outpath=fig_path)

        # Convergence study if requested
        if args.convergence:
            Ns, errs = convergence(loop_name=loop_name, R=args.R, a=args.a,
                                   Nmin=args.Nmin, Nmax=args.Nmax, nsteps=args.nsteps)
            fig_conv = os.path.join(outdir, f"conv_{loop_name}_R{args.R}_a{args.a}.png")
            save_convergence_plot(Ns, errs,
                                  title=f"Convergence | {label} | error = |∮τ - 2aA|",
                                  outpath=fig_conv)

            # Save convergence data
            conv_csv = os.path.join(outdir, f"conv_{loop_name}_R{args.R}_a{args.a}.csv")
            with open(conv_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["N", "abs_error"])
                for n, e in zip(Ns, errs):
                    w.writerow([int(n), float(e)])

    # Save summary CSV
    summary_csv = os.path.join(outdir, f"summary_R{args.R}_a{args.a}_N{args.N}.csv")
    write_csv(summary_rows, summary_csv)

    # Print summary nicely
    print("\n=== EXP B — Holonomy summary ===")
    print(f"τ = a(-y dx + x dy), a={args.a}, R={args.R}, N={args.N}")
    for r in summary_rows:
        print(f"- {r['loop']:8s}  ∮τ(num)={r['integral_num']:+.8e}   2aA(th)={r['integral_theory']:+.8e}   err={r['abs_error']:.3e}")
    print(f"\nSaved outputs in: {os.path.abspath(outdir)}")
    print(f"Summary CSV: {os.path.abspath(summary_csv)}\n")


if __name__ == "__main__":
    main()

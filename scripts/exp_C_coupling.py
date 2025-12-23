#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

python exp_C_coupling.py --sweep --klist "0,0.01,0.05,0.1,0.2,0.5,1.0" --outdir out_expC_sweep

EXP C — Interaction → effective integrability (dispersion reduction)

Two coupled oscillators with a metronomic one-form (MOF) defining effective time:
  dT_i/dt = 1 + eps * a(x_i, y_i),  a>=0

We do NOT modify the ODE by MOF; MOF is an effective-time reading along trajectories.

Dynamics (coupled harmonic oscillators with optional detuning):
  x1' = y1
  y1' = -w1^2 x1 - k (x1 - x2)
  x2' = y2
  y2' = -w2^2 x2 - k (x2 - x1)

Measure:
- Define events using a Poincaré section: upward crossings of x=0 with y>0.
- Record event indices n and corresponding effective times T1_eff(n), T2_eff(n)
- Compute ΔT_eff(n) = T1_eff(n) - T2_eff(n)
- Compute dispersion std(ΔT_eff) after discarding transient events.

Expected outcome:
- As coupling k increases, dispersion std(ΔT_eff) decreases (effective integrability / coherence).

Outputs:
- For each k: plots + CSV
- If sweeping k: summary CSV + plot std vs k

Dependencies: numpy, matplotlib
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from typing import Tuple, Dict, List

import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# RK4 integrator
# -----------------------------

def rk4_integrate(rhs, z0: np.ndarray, t: np.ndarray, rhs_args: tuple) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    z0 = np.asarray(z0, dtype=float)

    dt = t[1] - t[0]
    if not np.allclose(np.diff(t), dt):
        raise ValueError("t grid must be evenly spaced for RK4.")

    Z = np.zeros((t.size, z0.size), dtype=float)
    Z[0] = z0

    for i in range(t.size - 1):
        ti = t[i]
        zi = Z[i]

        k1 = rhs(ti, zi, *rhs_args)
        k2 = rhs(ti + 0.5*dt, zi + 0.5*dt*k1, *rhs_args)
        k3 = rhs(ti + 0.5*dt, zi + 0.5*dt*k2, *rhs_args)
        k4 = rhs(ti + dt, zi + dt*k3, *rhs_args)

        Z[i+1] = zi + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

    return Z


# -----------------------------
# Coupled oscillators
# -----------------------------

def rhs_coupled_osc(t: float, z: np.ndarray, w1: float, w2: float, k: float) -> np.ndarray:
    x1, y1, x2, y2 = z
    dx1 = y1
    dy1 = -(w1**2)*x1 - k*(x1 - x2)
    dx2 = y2
    dy2 = -(w2**2)*x2 - k*(x2 - x1)
    return np.array([dx1, dy1, dx2, dy2], dtype=float)


# -----------------------------
# MOF tempo
# -----------------------------

@dataclass(frozen=True)
class MOFTempo:
    eps: float = 0.3
    kind: str = "radius"  # "radius" or "energy"

    def a(self, x: np.ndarray, y: np.ndarray, omega: float = 1.0) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if self.kind == "radius":
            return x**2 + y**2
        elif self.kind == "energy":
            return 0.5*(y**2 + (omega**2)*x**2)
        else:
            raise ValueError(f"Unknown a(x,y) kind: {self.kind}")

    def tempo(self, x: np.ndarray, y: np.ndarray, omega: float = 1.0) -> np.ndarray:
        return 1.0 + self.eps * self.a(x, y, omega=omega)

    def teff(self, t: np.ndarray, x: np.ndarray, y: np.ndarray, omega: float = 1.0) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        dt = t[1] - t[0]
        if not np.allclose(np.diff(t), dt):
            raise ValueError("t must be evenly spaced for T_eff computation.")
        tempo = self.tempo(x, y, omega=omega)
        T = np.zeros_like(t)
        T[1:] = np.cumsum(0.5*(tempo[1:] + tempo[:-1]) * dt)
        return T


# -----------------------------
# Event detection (Poincaré section)
# -----------------------------

def upward_zero_crossings(t: np.ndarray, x: np.ndarray, y: np.ndarray, Teff: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect events where x crosses 0 upward (x_i <= 0 < x_{i+1}) and y_i > 0.
    We linearly interpolate event time and Teff at the crossing.
    Returns: t_events, Teff_events
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    Teff = np.asarray(Teff, dtype=float)

    idx = np.where((x[:-1] <= 0.0) & (x[1:] > 0.0) & (y[:-1] > 0.0))[0]
    if idx.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    t_ev = []
    T_ev = []
    for i in idx:
        # linear interpolation fraction for x=0
        denom = (x[i+1] - x[i])
        if denom == 0.0:
            continue
        alpha = (0.0 - x[i]) / denom
        alpha = float(np.clip(alpha, 0.0, 1.0))

        t0 = t[i] + alpha * (t[i+1] - t[i])
        T0 = Teff[i] + alpha * (Teff[i+1] - Teff[i])

        t_ev.append(t0)
        T_ev.append(T0)

    return np.array(t_ev, dtype=float), np.array(T_ev, dtype=float)


# -----------------------------
# Plotting
# -----------------------------

def ensure_outdir(outdir: str) -> str:
    os.makedirs(outdir, exist_ok=True)
    return outdir

def save_plot_xy(x: np.ndarray, y: np.ndarray, title: str, outpath: str) -> None:
    plt.figure()
    plt.plot(x, y)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()

def save_plot_timeseries(t: np.ndarray, a: np.ndarray, b: np.ndarray, title: str, ylabel: str, outpath: str) -> None:
    plt.figure()
    plt.plot(t, a, label="osc1")
    plt.plot(t, b, label="osc2")
    plt.xlabel("t")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()

def save_plot_delta(n: np.ndarray, dT: np.ndarray, title: str, outpath: str) -> None:
    plt.figure()
    plt.plot(n, dT, marker="o")
    plt.xlabel("event index n")
    plt.ylabel("ΔT_eff(n) = T1_eff - T2_eff")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()

def save_plot_std_vs_k(ks: np.ndarray, stds: np.ndarray, title: str, outpath: str) -> None:
    plt.figure()
    plt.plot(ks, stds, marker="o")
    plt.xlabel("coupling k")
    plt.ylabel("std(ΔT_eff) after transient")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


# -----------------------------
# CSV helpers
# -----------------------------

def write_csv(path: str, header: List[str], rows: np.ndarray) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([float(v) for v in r])


# -----------------------------
# Run a single k
# -----------------------------

def run_one_k(args, outdir: str, k: float) -> Dict[str, float]:
    t = np.linspace(0.0, args.tmax, args.nsteps, endpoint=True)

    w1 = args.w1
    w2 = args.w2

    z0 = np.array([args.x10, args.y10, args.x20, args.y20], dtype=float)
    Z = rk4_integrate(rhs_coupled_osc, z0=z0, t=t, rhs_args=(w1, w2, k))

    x1, y1, x2, y2 = Z[:, 0], Z[:, 1], Z[:, 2], Z[:, 3]

    mof = MOFTempo(eps=args.eps, kind=args.akind)
    T1 = mof.teff(t, x1, y1, omega=w1)
    T2 = mof.teff(t, x2, y2, omega=w2)

    # Events
    t1_ev, T1_ev = upward_zero_crossings(t, x1, y1, T1)
    t2_ev, T2_ev = upward_zero_crossings(t, x2, y2, T2)

    ne = int(min(t1_ev.size, t2_ev.size))
    if ne == 0:
        raise RuntimeError("No events detected. Increase tmax or adjust initial conditions.")

    # Align by event index (simple & robust for near-periodic regimes)
    T1_ev = T1_ev[:ne]
    T2_ev = T2_ev[:ne]
    dT = T1_ev - T2_ev

    # Discard transient events
    n0 = min(args.ntrans, ne)
    dT_ss = dT[n0:] if ne > n0 else np.array([], dtype=float)

    std_dt = float(np.std(dT_ss)) if dT_ss.size > 1 else float("nan")
    mean_dt = float(np.mean(dT_ss)) if dT_ss.size > 0 else float("nan")

    # Outputs
    kout = ensure_outdir(os.path.join(outdir, f"k_{k:.6g}".replace(".", "p")))
    # Phase plots
    save_plot_xy(x1, y1, f"Osc1 phase | w1={w1} | k={k}", os.path.join(kout, "phase_osc1.png"))
    save_plot_xy(x2, y2, f"Osc2 phase | w2={w2} | k={k}", os.path.join(kout, "phase_osc2.png"))

    # Time series x(t)
    save_plot_timeseries(t, x1, x2, f"x(t) | coupled oscillators | k={k}", "x", os.path.join(kout, "x_timeseries.png"))

    # ΔT_eff across events
    n = np.arange(ne, dtype=int)
    save_plot_delta(n, dT, f"ΔT_eff vs event index | k={k} | discard first {n0}", os.path.join(kout, "deltaTeff_events.png"))

    # CSV exports (dense timeseries + event series)
    write_csv(os.path.join(kout, "timeseries.csv"),
              ["t", "x1", "y1", "T1_eff", "x2", "y2", "T2_eff"],
              np.column_stack([t, x1, y1, T1, x2, y2, T2]))

    write_csv(os.path.join(kout, "events.csv"),
              ["n", "T1_eff_event", "T2_eff_event", "delta_Teff"],
              np.column_stack([n, T1_ev, T2_ev, dT]))

    return {
        "k": float(k),
        "w1": float(w1),
        "w2": float(w2),
        "eps": float(args.eps),
        "akind": str(args.akind),
        "n_events": float(ne),
        "n_transient_discarded": float(n0),
        "std_delta_Teff": std_dt,
        "mean_delta_Teff": mean_dt,
        "outdir_k": kout,
    }


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="EXP C — Coupling reduces MOF-induced timing dispersion (effective integrability).")

    # Simulation grid
    p.add_argument("--tmax", type=float, default=200.0, help="Integration horizon.")
    p.add_argument("--nsteps", type=int, default=20001, help="Number of time steps (evenly spaced).")

    # Oscillator params
    p.add_argument("--w1", type=float, default=1.0, help="Oscillator 1 frequency w1.")
    p.add_argument("--w2", type=float, default=1.05, help="Oscillator 2 frequency w2 (detuning helps show effect).")

    # Coupling
    p.add_argument("--k", type=float, default=0.0, help="Coupling strength k (single run if --sweep not set).")
    p.add_argument("--sweep", action="store_true", help="Sweep k over a list (see --klist).")
    p.add_argument("--klist", type=str, default="0,0.01,0.05,0.1,0.2,0.5,1.0",
                   help="Comma-separated list of k values for sweep.")

    # Initial conditions
    p.add_argument("--x10", type=float, default=1.0, help="Initial x1.")
    p.add_argument("--y10", type=float, default=0.0, help="Initial y1.")
    p.add_argument("--x20", type=float, default=0.2, help="Initial x2.")
    p.add_argument("--y20", type=float, default=0.0, help="Initial y2.")

    # MOF
    p.add_argument("--eps", type=float, default=0.3, help="MOF strength eps (tempo = 1 + eps*a).")
    p.add_argument("--akind", choices=["radius", "energy"], default="radius", help="a(x,y) choice for tempo modulation.")

    # Event/transient handling
    p.add_argument("--ntrans", type=int, default=10, help="Number of initial events to discard as transient.")

    # Output
    p.add_argument("--outdir", type=str, default="out_expC", help="Output directory.")

    #args = p.parse ( )  # <-- safeguard against accidental typo? No.
    # Fix: correct call:
    # args = p.parse_args()
    args = p.parse_args()

    outdir = ensure_outdir(args.outdir)

    summaries: List[Dict[str, float]] = []

    if args.sweep:
        ks = [float(x.strip()) for x in args.klist.split(",") if x.strip() != ""]
    else:
        ks = [float(args.k)]

    for k in ks:
        res = run_one_k(args, outdir, k)
        summaries.append(res)

    # Write global summary
    sum_csv = os.path.join(outdir, "summary.csv")
    # Union of keys (robust)
    all_keys = sorted(set().union(*[set(s.keys()) for s in summaries]))
    with open(sum_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=all_keys)
        w.writeheader()
        for s in summaries:
            w.writerow(s)

    # If sweeping, produce std vs k plot
    if args.sweep:
        ks_arr = np.array([s["k"] for s in summaries], dtype=float)
        std_arr = np.array([s["std_delta_Teff"] for s in summaries], dtype=float)
        save_plot_std_vs_k(ks_arr, std_arr,
                           title=f"Dispersion reduction vs coupling | eps={args.eps}, a={args.akind}, w1={args.w1}, w2={args.w2}",
                           outpath=os.path.join(outdir, "std_deltaTeff_vs_k.png"))

    print("\n=== EXP C — Summary ===")
    for s in summaries:
        print(f"- k={s['k']:.6g}  std(ΔT_eff)={s['std_delta_Teff']:.6g}  mean(ΔT_eff)={s['mean_delta_Teff']:.6g}  events={int(s['n_events'])}")
    print(f"\nSaved outputs in: {os.path.abspath(outdir)}")
    print(f"Summary CSV: {os.path.abspath(sum_csv)}\n")


if __name__ == "__main__":
    main()

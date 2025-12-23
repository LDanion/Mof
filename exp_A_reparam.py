#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

# Les deux systèmes (par défaut)
python exp_A_reparam.py --system both --outdir out_expA --eps 0.3 --akind radius

# Oscillateur seul, ω=2
python exp_A_reparam.py --system osc --omega 2.0 --tmax 40 --eps 0.2 --outdir out_expA_osc

# Van der Pol seul, μ=3
python exp_A_reparam.py --system vdp --mu 3.0 --tmax 80 --eps 0.15 --outdir out_expA_vdp

EXP A — Effective time reparametrization induced by a metronomic one-form (MOF)

Goal:
- Show that the MOF changes the *tempo* (effective time) along a trajectory,
  without changing the spatial/phase trajectory itself.

We integrate standard ODEs in physical time t:
  - Harmonic oscillator: x' = y, y' = -ω^2 x
  - Van der Pol:         x' = y, y' = μ(1-x^2)y - x

Then define an effective time via a (simple) MOF along trajectories:
  dT_eff = (1 + ε * a(x,y)) dt  with a(x,y) >= 0
  => T_eff(t) = ∫ (1 + ε a(x(t),y(t))) dt

This is the minimal, reproducible demonstration of "metronomic modulation"
as a time reading / reparametrization, not a new force field.

Outputs:
- Phase portrait (x vs y) for each system
- Time series: x(t) and x(T_eff) (via interpolation)
- Tempo curve: dT_eff/dt = 1 + ε a(x(t),y(t))
- CSV data exports

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
# Systems
# -----------------------------

def rhs_osc(t: float, z: np.ndarray, omega: float) -> np.ndarray:
    x, y = z
    return np.array([y, -(omega**2) * x], dtype=float)

def rhs_vdp(t: float, z: np.ndarray, mu: float) -> np.ndarray:
    x, y = z
    return np.array([y, mu * (1.0 - x**2) * y - x], dtype=float)


# -----------------------------
# RK4 integrator (deterministic, no SciPy required)
# -----------------------------

def rk4_integrate(rhs, z0: np.ndarray, t: np.ndarray, rhs_args: tuple) -> np.ndarray:
    """
    Classic fixed-step RK4 integration.
    t must be evenly spaced.
    """
    t = np.asarray(t, dtype=float)
    z0 = np.asarray(z0, dtype=float)

    dt = t[1] - t[0]
    if not np.allclose(np.diff(t), dt):
        raise ValueError("t grid must be evenly spaced for this RK4 integrator.")

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
# MOF definition along trajectory
# -----------------------------

@dataclass(frozen=True)
class MOFTempo:
    """
    Defines dT_eff/dt = 1 + eps * a(x,y), with a(x,y) >= 0.
    This is a minimal "metronomic modulation" of effective time.
    """
    eps: float = 0.5
    kind: str = "radius"   # "radius" or "energy"

    def a(self, x: np.ndarray, y: np.ndarray, omega: float = 1.0) -> np.ndarray:
        """
        Non-negative modulation function.
        - radius: a = x^2 + y^2
        - energy (oscillator): a = 0.5*(y^2 + ω^2 x^2)  (still fine for VdP too)
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        if self.kind == "radius":
            return x**2 + y**2
        elif self.kind == "energy":
            return 0.5*(y**2 + (omega**2)*x**2)
        else:
            raise ValueError(f"Unknown MOF a(x,y) kind: {self.kind}")

    def tempo(self, x: np.ndarray, y: np.ndarray, omega: float = 1.0) -> np.ndarray:
        return 1.0 + self.eps * self.a(x, y, omega=omega)

    def teff(self, t: np.ndarray, x: np.ndarray, y: np.ndarray, omega: float = 1.0) -> np.ndarray:
        """
        Compute T_eff(t) by cumulative trapezoidal integration of tempo(t).
        """
        t = np.asarray(t, dtype=float)
        dt = t[1] - t[0]
        if not np.allclose(np.diff(t), dt):
            raise ValueError("t must be evenly spaced to compute T_eff reliably here.")

        tempo = self.tempo(x, y, omega=omega)
        # cumulative trapezoid: T[0]=0
        T = np.zeros_like(t)
        T[1:] = np.cumsum(0.5*(tempo[1:] + tempo[:-1]) * dt)
        return T


# -----------------------------
# Helpers
# -----------------------------

def ensure_outdir(outdir: str) -> str:
    os.makedirs(outdir, exist_ok=True)
    return outdir

def write_csv(path: str, header: List[str], rows: np.ndarray) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([float(v) for v in r])

def interp_x_of_T(T: np.ndarray, x: np.ndarray, Tgrid: np.ndarray) -> np.ndarray:
    """
    Interpolate x(T_eff) on a uniform grid of T values.
    Assumes T is increasing (tempo > 0).
    """
    return np.interp(Tgrid, T, x)

def plot_phase(x: np.ndarray, y: np.ndarray, title: str, outpath: str) -> None:
    plt.figure()
    plt.plot(x, y)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()

def plot_time_series(t: np.ndarray, x: np.ndarray, title: str, outpath: str) -> None:
    plt.figure()
    plt.plot(t, x)
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()

def plot_tempo(t: np.ndarray, tempo: np.ndarray, title: str, outpath: str) -> None:
    plt.figure()
    plt.plot(t, tempo)
    plt.xlabel("t")
    plt.ylabel("dT_eff/dt")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()

def plot_teff(t: np.ndarray, T: np.ndarray, title: str, outpath: str) -> None:
    plt.figure()
    plt.plot(t, T)
    plt.xlabel("t")
    plt.ylabel("T_eff")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()

def plot_x_vs_T(Tgrid: np.ndarray, xT: np.ndarray, title: str, outpath: str) -> None:
    plt.figure()
    plt.plot(Tgrid, xT)
    plt.xlabel("T_eff")
    plt.ylabel("x")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


# -----------------------------
# Experiment
# -----------------------------

def run_system_osc(args, outdir: str) -> Dict[str, float]:
    omega = args.omega
    t = np.linspace(0.0, args.tmax, args.nsteps, endpoint=True)
    z0 = np.array([args.x0, args.y0], dtype=float)

    Z = rk4_integrate(rhs_osc, z0=z0, t=t, rhs_args=(omega,))
    x, y = Z[:, 0], Z[:, 1]

    mof = MOFTempo(eps=args.eps, kind=args.akind)
    T = mof.teff(t, x, y, omega=omega)
    tempo = mof.tempo(x, y, omega=omega)

    # Uniform T grid (same number of samples)
    Tgrid = np.linspace(T[0], T[-1], t.size)
    xT = interp_x_of_T(T, x, Tgrid)

    # Plots
    plot_phase(x, y, f"OSC phase | ω={omega}", os.path.join(outdir, "osc_phase.png"))
    plot_time_series(t, x, "OSC x(t)", os.path.join(outdir, "osc_x_of_t.png"))
    plot_teff(t, T, "OSC T_eff(t)", os.path.join(outdir, "osc_Teff_of_t.png"))
    plot_tempo(t, tempo, f"OSC tempo dT_eff/dt (eps={args.eps}, a={args.akind})", os.path.join(outdir, "osc_tempo.png"))
    plot_x_vs_T(Tgrid, xT, "OSC x(T_eff) (interpolated)", os.path.join(outdir, "osc_x_of_Teff.png"))

    # CSV exports
    write_csv(os.path.join(outdir, "osc_timeseries.csv"),
              ["t", "x", "y", "tempo", "T_eff"],
              np.column_stack([t, x, y, tempo, T]))

    write_csv(os.path.join(outdir, "osc_x_of_Teff.csv"),
              ["T_eff_uniform", "x_interp"],
              np.column_stack([Tgrid, xT]))

    return {
        "system": "oscillator",
        "omega": float(omega),
        "eps": float(args.eps),
        "Teff_end": float(T[-1]),
        "t_end": float(t[-1]),
        "tempo_min": float(np.min(tempo)),
        "tempo_max": float(np.max(tempo)),
    }


def run_system_vdp(args, outdir: str) -> Dict[str, float]:
    mu = args.mu
    t = np.linspace(0.0, args.tmax, args.nsteps, endpoint=True)
    z0 = np.array([args.x0, args.y0], dtype=float)

    Z = rk4_integrate(rhs_vdp, z0=z0, t=t, rhs_args=(mu,))
    x, y = Z[:, 0], Z[:, 1]

    # For VdP, omega isn't a system parameter, but MOF energy-kind can still use omega=1
    omega_for_mof = 1.0

    mof = MOFTempo(eps=args.eps, kind=args.akind)
    T = mof.teff(t, x, y, omega=omega_for_mof)
    tempo = mof.tempo(x, y, omega=omega_for_mof)

    Tgrid = np.linspace(T[0], T[-1], t.size)
    xT = interp_x_of_T(T, x, Tgrid)

    plot_phase(x, y, f"Van der Pol phase | μ={mu}", os.path.join(outdir, "vdp_phase.png"))
    plot_time_series(t, x, "VdP x(t)", os.path.join(outdir, "vdp_x_of_t.png"))
    plot_teff(t, T, "VdP T_eff(t)", os.path.join(outdir, "vdp_Teff_of_t.png"))
    plot_tempo(t, tempo, f"VdP tempo dT_eff/dt (eps={args.eps}, a={args.akind})", os.path.join(outdir, "vdp_tempo.png"))
    plot_x_vs_T(Tgrid, xT, "VdP x(T_eff) (interpolated)", os.path.join(outdir, "vdp_x_of_Teff.png"))

    write_csv(os.path.join(outdir, "vdp_timeseries.csv"),
              ["t", "x", "y", "tempo", "T_eff"],
              np.column_stack([t, x, y, tempo, T]))

    write_csv(os.path.join(outdir, "vdp_x_of_Teff.csv"),
              ["T_eff_uniform", "x_interp"],
              np.column_stack([Tgrid, xT]))

    return {
        "system": "van_der_pol",
        "mu": float(mu),
        "eps": float(args.eps),
        "Teff_end": float(T[-1]),
        "t_end": float(t[-1]),
        "tempo_min": float(np.min(tempo)),
        "tempo_max": float(np.max(tempo)),
    }


def write_summary(outdir: str, summaries: List[Dict[str, float]]) -> None:
    if not summaries:
        return
    keys = list(summaries[0].keys())
    path = os.path.join(outdir, "summary.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for s in summaries:
            w.writerow(s)


def main() -> None:
    p = argparse.ArgumentParser(description="EXP A — Effective time reparametrization with MOF on oscillator + Van der Pol.")
    p.add_argument("--system", choices=["osc", "vdp", "both"], default="both",
                   help="Which system to run (default: both).")
    p.add_argument("--outdir", type=str, default="out_expA", help="Output directory.")
    p.add_argument("--tmax", type=float, default=60.0, help="Integration time horizon.")
    p.add_argument("--nsteps", type=int, default=6001, help="Number of time samples (evenly spaced).")
    p.add_argument("--x0", type=float, default=1.0, help="Initial x.")
    p.add_argument("--y0", type=float, default=0.0, help="Initial y.")

    # Oscillator params
    p.add_argument("--omega", type=float, default=1.0, help="Oscillator frequency ω.")

    # VdP params
    p.add_argument("--mu", type=float, default=2.0, help="Van der Pol parameter μ.")

    # MOF params
    p.add_argument("--eps", type=float, default=0.3, help="MOF strength ε (ensure 1+ε a(x,y) stays positive).")
    p.add_argument("--akind", choices=["radius", "energy"], default="radius",
                   help="Modulation function a(x,y): radius=x^2+y^2 or energy=0.5*(y^2+ω^2 x^2).")

    args = p.parse_args()
    outdir = ensure_outdir(args.outdir)

    summaries: List[Dict[str, float]] = []

    if args.system in ("osc", "both"):
        sub = ensure_outdir(os.path.join(outdir, "oscillator"))
        summaries.append(run_system_osc(args, sub))

    if args.system in ("vdp", "both"):
        sub = ensure_outdir(os.path.join(outdir, "van_der_pol"))
        summaries.append(run_system_vdp(args, sub))

    write_summary(outdir, summaries)

    print("\n=== EXP A — Effective time reparametrization summary ===")
    for s in summaries:
        name = s["system"]
        print(f"- {name:12s}  Teff_end={s['Teff_end']:.6g} (for t_end={s['t_end']:.6g})"
              f"  tempo∈[{s['tempo_min']:.6g},{s['tempo_max']:.6g}]  eps={s['eps']}")
    print(f"\nSaved outputs in: {os.path.abspath(outdir)}")
    print("Each system folder contains phase plots, x(t), T_eff(t), tempo(t), and x(T_eff), plus CSV exports.\n")


if __name__ == "__main__":
    main()

# MOF — Metronomic One-Form  
**Effective-Time Reparametrization in Dynamical Systems**

## Overview

MOF (Metronomic One-Form) is a minimal numerical framework for exploring **effective-time reparametrization** in dynamical systems.

The project introduces an *effective time* variable (T_eff), defined through a local, state-dependent temporal modulation, without modifying the underlying equations of motion. The goal is to study how temporal structure, holonomy, and interaction constraints affect the accumulation and dispersion of effective time along trajectories.

No new force, field, or degree of freedom is introduced. The framework operates strictly at the level of **time parametrization** and **geometric consistency**.

---

## Scientific Motivation

In many nonlinear systems, dynamics are invariant under time reparametrization, yet observable quantities (such as phase alignment or event timing) may depend on how time is accumulated along trajectories.

MOF investigates three core questions:

1. Can effective-time accumulation be defined consistently using geometric principles?
2. Does effective-time reparametrization alone alter intrinsic dynamics?
3. Can interaction between subsystems reduce temporal dispersion without enforcing strict integrability?

These questions are addressed through controlled numerical experiments.

---

## Experiments

The repository currently includes three main experiments:

### **Experiment A — Geometric Consistency**
- Validation of effective-time holonomy using Stokes’ theorem
- Path-dependent accumulation on closed loops
- Benchmark geometries (circle, square, figure-eight)

### **Experiment B — Neutral Reparametrization**
- Single oscillator under effective-time reparametrization
- Demonstrates that reparametrization alone preserves phase-space dynamics
- Applied to harmonic and Van der Pol oscillators

### **Experiment C — Interaction-Induced Temporal Coherence**
- Two weakly coupled oscillators
- Measurement of dispersion of effective-time increments
- Shows strong reduction of temporal dispersion as coupling increases
<img width="1184" height="1900" alt="image" src="https://github.com/user-attachments/assets/5935f889-2e6d-4949-bea3-eb955f86d854" />

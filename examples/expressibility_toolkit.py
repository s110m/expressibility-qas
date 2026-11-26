#!/usr/bin/env python3
"""
expressibility_toolkit.py
==========================

A single-file, drop-in toolkit for measuring the **expressibility** of a
parameterized quantum circuit (PQC), following:

    Sim, Johnson & Aspuru-Guzik (2019), "Expressibility and Entangling
    Capability of Parameterized Quantum Circuits for Hybrid Quantum-Classical
    Algorithms", arXiv:1905.10876.

--------------------------------------------------------------------------
THE IDEA
--------------------------------------------------------------------------
A PQC family {U(theta)} is "expressible" if, as you sweep theta uniformly,
the states U(theta)|0> cover the Hilbert space as evenly as a truly
Haar-random unitary would. The paper turns this into a measurable quantity:

  1. Draw two independent parameter samples x, y and build the states
         |psi_x> = U(x)|0>,   |psi_y> = U(y)|0>
  2. Compute the fidelity   f = |<psi_y|psi_x>|^2   in [0, 1]
  3. Repeat many times to build an empirical distribution P_PQC(f)
  4. Compare it to the analytic fidelity distribution of Haar-random states
     on the same Hilbert space dimension N = 2^n_qubits:
         P_Haar(f) = (N - 1) * (1 - f)^(N - 2)
  5. Expressibility is reported as the (small) KL divergence
         D_KL( P_PQC(f) || P_Haar(f) )
     A circuit that is "maximally expressible" has D_KL -> 0. Larger D_KL
     means the circuit's reachable states are concentrated in a smaller
     region of Hilbert space than a Haar-random ensemble would be.

--------------------------------------------------------------------------
WHAT THIS FILE GIVES YOU
--------------------------------------------------------------------------
- `run_expressibility(...)`  the one function you actually call.
- Two ways to sample fidelities:
    * "exact"  mode: statevector overlap (fast, noiseless, needs a simulator
      that can build a Statevector -- no hardware/shots required).
    * "shots"  mode: a SWAP test evaluated with shots on any Qiskit
      backend/AerSimulator (use this if you want to study the effect of
      finite sampling or run on real/noisy backends).
- Three ways to describe "your circuit" to the toolkit -- pick whichever
  matches how your ansatz is already defined (see "HOW TO USE" below):
    * `sampler_from_ansatz(...)`   you already have a `QuantumCircuit`
      with free `Parameter`s (e.g. built with Qiskit's `TwoLocal`,
      `EfficientSU2`, or your own custom ansatz).
    * `sampler_from_builder(...)`  you have a function
      `build(qc, theta, ...)` that appends gates in-place given a flat
      parameter vector (a common pattern for hand-rolled circuit template
      libraries).
    * a plain `Callable[[], QuantumCircuit]` you write yourself -- the
      toolkit doesn't care how you generate the circuit, only that calling
      it twice gives two independent samples from your parameter
      distribution.
- Plotting (`plot_empirical_vs_haar`) and a small `ExpressibilityResult`
  dataclass bundling everything (fidelities, histograms, KL, figure).

--------------------------------------------------------------------------
DEPENDENCIES
--------------------------------------------------------------------------
Required:   qiskit>=1.0, numpy, scipy, matplotlib
Optional:   qiskit-aer   (only needed for mode="shots")

    pip install qiskit numpy scipy matplotlib
    pip install qiskit-aer   # optional, for shot-based mode

--------------------------------------------------------------------------
HOW TO USE THIS IN ANOTHER PROJECT
--------------------------------------------------------------------------
Copy this single file into your project (no other files from this repo
are required) and do one of:

    # (A) You already have a parameterized QuantumCircuit ansatz
    from expressibility_toolkit import sampler_from_ansatz, run_expressibility
    from qiskit.circuit.library import EfficientSU2

    ansatz = EfficientSU2(num_qubits=4, reps=2)
    sampler = sampler_from_ansatz(ansatz)
    result = run_expressibility(sampler, n_qubits=4, n_pairs=1000, mode="exact")
    print("KL divergence to Haar:", result.KL)   # lower = more expressible

    # (B) You have a "build-in-place" style ansatz function
    from expressibility_toolkit import sampler_from_builder, run_expressibility

    def my_ansatz(qc, theta):
        for i, t in enumerate(theta):
            qc.ry(t, i % qc.num_qubits)
        for i in range(qc.num_qubits - 1):
            qc.cx(i, i + 1)

    sampler = sampler_from_builder(my_ansatz, n_qubits=4, n_params=8)
    result = run_expressibility(sampler, n_qubits=4, n_pairs=1000, mode="exact")

    # (C) Full manual control -- any callable that returns a fresh circuit
    def my_sampler():
        # ... build and return a QuantumCircuit with randomized parameters ...
        ...
    result = run_expressibility(my_sampler, n_qubits=4)

Then inspect `result`:
    result.KL       -> scalar expressibility score (nats); compare circuits
                       by ranking this ascending (lower = more expressible).
    result.fidelity -> raw fidelity samples, for your own statistics/plots.
    result.figure   -> matplotlib Figure (empirical histogram vs Haar curve).

Run this file directly (`python expressibility_toolkit.py`) to see a
self-contained demo comparing a few toy 1-qubit circuits against Haar.
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import rel_entr

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector, state_fidelity, random_unitary
from qiskit.circuit.library import UnitaryGate

try:
    from qiskit_aer import AerSimulator
    _AER_AVAILABLE = True
except Exception:
    AerSimulator = None
    _AER_AVAILABLE = False


# ==========================================================================
# Haar reference distribution, histograms, divergence
# ==========================================================================

def haar_bin_masses(bins: np.ndarray, n_qubits: int) -> np.ndarray:
    """
    Probability mass the Haar-random fidelity distribution assigns to each
    histogram bin, obtained by integrating the analytic PDF
        P_Haar(f) = (N-1) (1-f)^(N-2),   N = 2**n_qubits
    over each bin [l, u]:
        mass = (1-l)^(N-1) - (1-u)^(N-1)
    """
    N = 2 ** n_qubits
    l, u = bins[:-1], bins[1:]
    P = (1.0 - l) ** (N - 1) - (1.0 - u) ** (N - 1)
    P = P / P.sum()  # renormalize for numerical safety
    return P


def empirical_hist(samples: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """Normalize `samples` into a probability mass per histogram bin."""
    weights = np.ones_like(samples, dtype=float) / max(1, len(samples))
    P, _ = np.histogram(samples, bins=bins, weights=weights, range=(0.0, 1.0))
    return P


def kl_divergence(P: np.ndarray, Q: np.ndarray, eps: float = 1e-12) -> float:
    """KL(P || Q) in nats, with clipping so empty bins don't blow up."""
    P = np.clip(P, eps, 1.0); P = P / P.sum()
    Q = np.clip(Q, eps, 1.0); Q = Q / Q.sum()
    return float(np.sum(rel_entr(P, Q)))


def plot_empirical_vs_haar(
    fidelity: np.ndarray,
    bins: np.ndarray,
    n_qubits: int,
    title: Optional[str] = None,
    show: bool = True,
    save_path: Optional[str] = None,
):
    """Plot the empirical fidelity histogram against the Haar curve."""
    P_emp = empirical_hist(fidelity, bins)
    P_haar = haar_bin_masses(bins, n_qubits)
    KL = kl_divergence(P_emp, P_haar)

    l, u = bins[:-1], bins[1:]
    bins_x = 0.5 * (l + u)
    weights = np.ones_like(fidelity) / max(1, len(fidelity))

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.hist(fidelity, bins=bins, weights=weights, alpha=0.6, label="Empirical")
    ax.plot(bins_x, P_haar, linewidth=2, label=f"Haar (n={n_qubits})")
    ttl = title or "Expressibility"
    ax.set_title(f"{ttl} — KL(P||Haar)={KL:.6f} nats")
    ax.set_xlabel(r"fidelity  $f=|\langle \psi_y | \psi_x \rangle|^2$")
    ax.set_ylabel("probability")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.2)

    if save_path is not None:
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    return fig, KL


# ==========================================================================
# Fidelity sampling
# ==========================================================================

def fidelity_exact(circuit_sampler: Callable[[], QuantumCircuit], n_pairs: int) -> np.ndarray:
    """
    Exact fidelity samples via statevector overlap (no shots, no noise).
    Calls `circuit_sampler()` twice per pair to get two independent draws.
    """
    fvals: List[float] = []
    for _ in range(n_pairs):
        psi_x = Statevector.from_instruction(circuit_sampler())
        psi_y = Statevector.from_instruction(circuit_sampler())
        fvals.append(state_fidelity(psi_x, psi_y))
    return np.asarray(fvals, dtype=float)


def swap_test_fidelity(
    Ux: QuantumCircuit,
    Uy: QuantumCircuit,
    shots: int = 20000,
    backend: Optional[Any] = None,
) -> float:
    """
    Estimate |<psi_y|psi_x>|^2 with a destructive SWAP test evaluated on
    `shots` measurement outcomes. Requires qiskit-aer (or another backend
    passed in explicitly).
    """
    if backend is None and not _AER_AVAILABLE:
        raise RuntimeError(
            "qiskit-aer is required for shot-based SWAP test. "
            "Install with: pip install qiskit-aer"
        )
    n = Ux.num_qubits
    assert Uy.num_qubits == n, "Ux and Uy must act on the same number of qubits."

    qc = QuantumCircuit(1 + 2 * n, 1)
    a, x0, y0 = 0, 1, 1 + n

    qc.h(a)
    qc.compose(Ux, qubits=range(x0, x0 + n), inplace=True)
    qc.compose(Uy, qubits=range(y0, y0 + n), inplace=True)
    for i in range(n):
        qc.cswap(a, x0 + i, y0 + i)
    qc.h(a)
    qc.measure(a, 0)

    backend = backend or AerSimulator()
    qc_t = transpile(qc, backend)
    result = backend.run(qc_t, shots=shots).result()
    counts = result.get_counts()
    p0 = counts.get("0", 0) / shots
    f_est = 2 * p0 - 1  # E[Z] = 2*p0 - 1 = |<psi_y|psi_x>|^2
    return max(0.0, min(1.0, f_est))


def fidelity_shots(
    circuit_sampler: Callable[[], QuantumCircuit],
    n_pairs: int,
    shots: int = 20000,
    backend: Optional[Any] = None,
) -> np.ndarray:
    """Shot-based fidelity samples via repeated SWAP tests."""
    fvals: List[float] = []
    for _ in range(n_pairs):
        Ux = circuit_sampler()
        Uy = circuit_sampler()
        fvals.append(swap_test_fidelity(Ux, Uy, shots=shots, backend=backend))
    return np.asarray(fvals, dtype=float)


# ==========================================================================
# Main API
# ==========================================================================

@dataclass
class ExpressibilityResult:
    fidelity: np.ndarray   # raw f = |<psi_y|psi_x>|^2 samples
    bins: np.ndarray       # histogram bin edges on [0, 1]
    P_emp: np.ndarray      # empirical probability mass per bin
    P_haar: np.ndarray     # Haar-random probability mass per bin
    KL: float              # KL(P_emp || P_haar) in nats -- the expressibility score
    figure: Any = None     # matplotlib Figure, if plotted


def run_expressibility(
    circuit_sampler: Callable[[], QuantumCircuit],
    n_qubits: int,
    n_pairs: int = 400,
    mode: str = "exact",              # "exact" or "shots"
    shots: int = 20000,
    nbins: int = 75,
    seed: Optional[int] = None,
    backend: Optional[Any] = None,
    title: Optional[str] = None,
    show: bool = True,
    save_path: Optional[str] = None,
) -> ExpressibilityResult:
    """
    Compute the expressibility of a PQC described by `circuit_sampler`.

    Args:
        circuit_sampler: () -> QuantumCircuit(n_qubits). Each call must
            return one independently-sampled circuit (i.e. it should
            internally randomize its own parameters). Called twice per pair.
        n_qubits: number of qubits the returned circuits act on.
        n_pairs: number of (x, y) pairs to draw. More pairs => smoother
            histogram and a more stable KL estimate (a few hundred to a
            couple thousand is typical).
        mode: "exact" for noiseless statevector fidelity, or "shots" for a
            SWAP test with finite sampling (needs qiskit-aer or a backend).
        shots: shots per SWAP test, only used when mode == "shots".
        nbins: number of equal-width histogram bins on [0, 1].
        seed: optional RNG seed (seeds numpy + stdlib random) for
            reproducibility of parameter draws.
        backend: optional backend/AerSimulator for shot-based mode.
        title: plot title.
        show: whether to display the plot interactively.
        save_path: if given, save the figure as a PNG here.

    Returns:
        ExpressibilityResult(fidelity, bins, P_emp, P_haar, KL, figure)
        -- `KL` is the number you report/compare: lower means the circuit's
        reachable states are statistically closer to Haar-random, i.e. more
        expressible.
    """
    if seed is not None:
        import random
        np.random.seed(seed)
        random.seed(seed)

    if mode.lower() == "exact":
        fidelity = fidelity_exact(circuit_sampler, n_pairs)
    elif mode.lower() == "shots":
        fidelity = fidelity_shots(circuit_sampler, n_pairs, shots=shots, backend=backend)
    else:
        raise ValueError("mode must be 'exact' or 'shots'")

    bins = np.linspace(0.0, 1.0, nbins + 1)
    P_emp = empirical_hist(fidelity, bins)
    P_haar = haar_bin_masses(bins, n_qubits)
    KL = kl_divergence(P_emp, P_haar)

    fig, _ = plot_empirical_vs_haar(fidelity, bins, n_qubits, title=title, show=show, save_path=save_path)

    return ExpressibilityResult(fidelity=fidelity, bins=bins, P_emp=P_emp, P_haar=P_haar, KL=KL, figure=fig)


# ==========================================================================
# Sampler adapters -- turn "your circuit" into circuit_sampler()
# ==========================================================================

def sampler_from_ansatz(
    ansatz: QuantumCircuit,
    low: float = 0.0,
    high: float = 2 * math.pi,
    rng: Optional[np.random.Generator] = None,
) -> Callable[[], QuantumCircuit]:
    """
    Adapt an existing parameterized `QuantumCircuit` (e.g. built with
    Qiskit's `TwoLocal`/`EfficientSU2`, or any circuit you already have with
    free `Parameter`s) into a `circuit_sampler` by binding every free
    parameter to an i.i.d. uniform value in [low, high) on each call.
    """
    rng = rng or np.random.default_rng()
    params = list(ansatz.parameters)
    if not params:
        raise ValueError("ansatz has no free Parameters to randomize.")

    def _make() -> QuantumCircuit:
        values = rng.uniform(low, high, size=len(params))
        return ansatz.assign_parameters(dict(zip(params, values)), inplace=False)

    return _make


def sampler_from_builder(
    builder: Callable[..., None],
    n_qubits: int,
    n_params: int,
    low: float = 0.0,
    high: float = 2 * math.pi,
    rng: Optional[np.random.Generator] = None,
    extra_args: Sequence[Any] = (),
) -> Callable[[], QuantumCircuit]:
    """
    Adapt a "build gates in-place" style ansatz function into a
    `circuit_sampler`. `builder` is called as:

        builder(qc, theta, *extra_args)

    where `qc` is a fresh `QuantumCircuit(n_qubits)` and `theta` is a numpy
    array of `n_params` i.i.d. uniform values in [low, high). This matches
    the common pattern used by hand-rolled circuit template libraries
    (e.g. `circuit_k(qc, theta, L, repeat)` style functions).
    """
    rng = rng or np.random.default_rng()

    def _make() -> QuantumCircuit:
        qc = QuantumCircuit(n_qubits)
        theta = rng.uniform(low, high, size=n_params)
        builder(qc, theta, *extra_args)
        return qc

    return _make


# ==========================================================================
# Toy example samplers (useful as sanity checks / teaching examples)
# ==========================================================================

def sampler_idle(n_qubits: int) -> Callable[[], QuantumCircuit]:
    """Identity circuit -- always |0...0>. Expect large KL (not expressible)."""
    def _make() -> QuantumCircuit:
        return QuantumCircuit(n_qubits)
    return _make


def sampler_circuit_A() -> Callable[[], QuantumCircuit]:
    """1-qubit: H - Rz(t1) - Rz(t2) - H. Two Rz's collapse to one free angle."""
    rng = np.random.default_rng()
    def _make() -> QuantumCircuit:
        t1, t2 = rng.uniform(0, 2 * math.pi, size=2)
        qc = QuantumCircuit(1)
        qc.h(0); qc.rz(t1, 0); qc.rz(t2, 0); qc.h(0)
        return qc
    return _make


def sampler_circuit_B() -> Callable[[], QuantumCircuit]:
    """1-qubit: H - Rz(t) - Rx(p) - H. Genuinely 2 free axes -> more expressible."""
    rng = np.random.default_rng()
    def _make() -> QuantumCircuit:
        t, p = rng.uniform(0, 2 * math.pi, size=2)
        qc = QuantumCircuit(1)
        qc.h(0); qc.rz(t, 0); qc.rx(p, 0); qc.h(0)
        return qc
    return _make


def sampler_random_unitary(n_qubits: int) -> Callable[[], QuantumCircuit]:
    """Haar-random unitary on n qubits -- the reference case, KL should be ~0."""
    def _make() -> QuantumCircuit:
        qc = QuantumCircuit(n_qubits)
        U = random_unitary(2 ** n_qubits)
        qc.append(UnitaryGate(U), range(n_qubits))
        return qc
    return _make


# ==========================================================================
# __main__ demo
# ==========================================================================

def _maybe_use_agg_backend():
    """Use a non-interactive matplotlib backend if no display is available."""
    import matplotlib
    if not os.environ.get("DISPLAY", "") and sys.platform != "win32":
        matplotlib.use("Agg")


def main():
    _maybe_use_agg_backend()

    outdir = os.path.abspath("./expressibility_outputs")
    os.makedirs(outdir, exist_ok=True)
    print("Running expressibility demos...\n")

    demos = [
        ("Idle (1q)", sampler_idle(1), 1),
        ("Circuit A: H-Rz-Rz-H (1q)", sampler_circuit_A(), 1),
        ("Circuit B: H-Rz-Rx-H (1q)", sampler_circuit_B(), 1),
        ("Haar-random unitary (1q)", sampler_random_unitary(1), 1),
    ]

    for i, (name, sampler, nq) in enumerate(demos, start=1):
        print(f"[{i}] {name}, exact")
        fname = name.split(":")[0].split("(")[0].strip().lower().replace(" ", "_")
        res = run_expressibility(
            circuit_sampler=sampler,
            n_qubits=nq,
            n_pairs=400,
            mode="exact",
            nbins=75,
            title=f"{name} — Exact",
            show=False,
            save_path=os.path.join(outdir, f"{fname}_exact.png"),
        )
        print(f"    KL(P||Haar) = {res.KL:.6f} nats")

    if _AER_AVAILABLE:
        print("[5] Circuit B (1q), shots (SWAP test)")
        res = run_expressibility(
            circuit_sampler=sampler_circuit_B(),
            n_qubits=1,
            n_pairs=200,
            mode="shots",
            shots=20000,
            nbins=75,
            title="Circuit B (1q) — Shots (SWAP test)",
            show=False,
            save_path=os.path.join(outdir, "circuitB_1q_shots.png"),
        )
        print(f"    KL(P||Haar) = {res.KL:.6f} nats")
    else:
        print("[5] Skipping shot-based demo (qiskit-aer not installed).")

    print("\nDone. Figures saved under:", outdir)


if __name__ == "__main__":
    main()

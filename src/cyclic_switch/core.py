"""Exact model of a quantum switch acting on Pauli channels in a superposition of orders.

Conventions
-----------
* ``N`` single-qubit Pauli channels ``E_i(rho) = sum_k w[i, k] sigma_k rho sigma_k`` with
  ``sigma = (I, X, Y, Z)``.  Complete depolarisation is ``w[i] = (1/4, 1/4, 1/4, 1/4)``.
* An *order set* is a list of permutations; permutation ``c`` lists the channel indices in
  the order in which they act on the target qubit.  The control register has one basis state
  ``|c>`` per permutation.
* The switch acts as ``F_k = sum_c |c><c| (x) W[k, c]`` on a (control (x) target) input, where
  ``W[k, c]`` is the product of the Kraus operators of tuple ``k`` in the order ``c``.
* A *code* is an encoder isometry ``V`` from one logical qubit into control (x) target and a
  decoding channel ``R`` from control (x) target back to one qubit.  Its quality is the
  entanglement fidelity ``Fe``; the average (Haar) state fidelity is ``(2 Fe + 1) / 3`` and the
  best measure-and-prepare strategy reaches ``Fe = 1/2`` (average fidelity ``2/3``).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

PAULIS = np.array(
    [
        [[1, 0], [0, 1]],
        [[0, 1], [1, 0]],
        [[0, -1j], [1j, 0]],
        [[1, 0], [0, -1]],
    ],
    dtype=complex,
)

CLASSICAL_FE = 0.5  # entanglement-fidelity limit of measure-and-prepare for one qubit
CLASSICAL_AVG_FIDELITY = 2.0 / 3.0


def fe_to_avg_fidelity(fe: float) -> float:
    """Convert entanglement fidelity to average state fidelity for a qubit."""
    return (2.0 * fe + 1.0) / 3.0


def cyclic_orders(n: int) -> list[tuple[int, ...]]:
    """The ``n`` cyclic shifts of ``(0, ..., n-1)``."""
    return [tuple((c + j) % n for j in range(n)) for c in range(n)]


def all_orders(n: int) -> list[tuple[int, ...]]:
    """All ``n!`` permutations of ``(0, ..., n-1)``."""
    return list(itertools.permutations(range(n)))


def pauli_weights(n: int, p: float = 1.0, bias: float = 0.0) -> np.ndarray:
    """Pauli weights of ``n`` identical depolarising channels of strength ``p``.

    ``E(rho) = (1 - p) rho + p I/2``: identity weight ``1 - 3p/4``, others ``p/4``.
    ``bias`` moves a fraction of the identity weight to the Z error (a miscalibrated,
    dephasing-biased channel) while keeping the weights normalised.
    """
    w = np.array([1 - 0.75 * p, p / 4, p / 4, p / 4], dtype=float)
    shift = bias * w[0]
    w[0] -= shift
    w[3] += shift
    return np.tile(w, (n, 1))


@dataclass(frozen=True)
class SwitchTensors:
    """Pre-computed Kraus data of a switch.

    ``ops[k, c]`` is the 2x2 operator applied for Kraus tuple ``k`` when the order is ``c``;
    ``wk[k]`` is the (order independent) probability weight of tuple ``k``.
    """

    ops: np.ndarray  # (K, C, 2, 2)
    wk: np.ndarray  # (K,)
    orders: tuple[tuple[int, ...], ...]

    @property
    def n_orders(self) -> int:
        return len(self.orders)


def build_switch(weights: np.ndarray, orders: Sequence[Sequence[int]]) -> SwitchTensors:
    """Build the Kraus tensors of the switch for the given channels and order set."""
    weights = np.asarray(weights, dtype=float)
    n = weights.shape[0]
    tuples = list(itertools.product(range(4), repeat=n))
    ops = np.zeros((len(tuples), len(orders), 2, 2), dtype=complex)
    wk = np.zeros(len(tuples))
    for ik, k in enumerate(tuples):
        wk[ik] = np.prod([weights[i, k[i]] for i in range(n)])
        for ic, order in enumerate(orders):
            m = np.eye(2, dtype=complex)
            for ch in order:  # first listed channel acts first
                m = PAULIS[k[ch]] @ m
            ops[ik, ic] = m
    return SwitchTensors(ops=ops, wk=wk, orders=tuple(tuple(o) for o in orders))


def fourier_basis(c: int) -> np.ndarray:
    """Columns are the discrete-Fourier measurement vectors of the control register."""
    j = np.arange(c)
    return np.exp(2j * np.pi * np.outer(j, j) / c) / np.sqrt(c)


def fixed_protocol(t: SwitchTensors) -> float:
    """Entanglement fidelity of the fixed protocol (no encoder or decoder learning).

    The logical qubit is sent through the target only, the control starts in the equal
    superposition, is read out in the Fourier basis, and the best Pauli correction is applied
    for each readout outcome.
    """
    c = t.n_orders
    a = np.ones(c, dtype=complex) / np.sqrt(c)
    v = fourier_basis(c)
    cond = np.einsum("cm,c,kcab->kmab", v.conj(), a, t.ops)  # (K, outcome, 2, 2)
    ov = np.einsum("gba,kmba->kmg", PAULIS.conj(), cond)
    score = np.einsum("k,kmg->mg", t.wk, np.abs(ov) ** 2) / 4.0
    return float(score.max(axis=1).sum())


def direct_fe(p: float, n: int = 1) -> float:
    """Entanglement fidelity of ``n`` definite-order depolarising channels of strength ``p``."""
    return 1.0 - 0.75 * (1.0 - (1.0 - p) ** n)


# ---------------------------------------------------------------------------------------
# Encoder / decoder design (PyTorch)
# ---------------------------------------------------------------------------------------
def _contributions(
    ops: torch.Tensor, v: torch.Tensor, r: torch.Tensor
) -> torch.Tensor:
    """Per-order trace terms ``Tr(R_j F_k V)`` split by control branch, shape (J, K, C)."""
    c = ops.shape[1]
    vc = v.reshape(c, 2, 2)  # (c, target_out, logical)
    fv = torch.einsum("kcab,cbd->kcad", ops, vc)  # (K, C, 2, 2)
    rr = r.reshape(r.shape[0] // 2, 2, c, 2)  # (J, logical_out, c, target)
    return torch.einsum("jacb,kcba->jkc", rr, fv)


def _fe_from_contrib(contrib: torch.Tensor, wk: torch.Tensor, lam: float) -> torch.Tensor:
    coh = contrib.sum(-1).abs() ** 2
    inc = (contrib.abs() ** 2).sum(-1)
    return (wk[None] * ((1 - lam) * coh + lam * inc)).sum() / 4.0


def _fe_replacement(
    ops: torch.Tensor, v: torch.Tensor, r: torch.Tensor, wk: torch.Tensor
) -> torch.Tensor:
    """Fidelity when the control register is replaced by the maximally mixed state.

    The replacement channel has Kraus operators ``|c'><c| / sqrt(C)`` on the control.
    """
    c = ops.shape[1]
    fv = torch.einsum("kcab,cbd->kcad", ops, v.reshape(c, 2, 2))
    rr = r.reshape(r.shape[0] // 2, 2, c, 2)
    term = torch.einsum("jxqb,kcbx->jkqc", rr, fv)
    return (wk[None, :, None, None] * term.abs() ** 2).sum() / (4.0 * c)


def _objective(
    ops: torch.Tensor, v: torch.Tensor, r: torch.Tensor, wk: torch.Tensor, lam: float, noise: str
) -> torch.Tensor:
    contrib = _contributions(ops, v, r)
    if noise == "dephasing":
        return _fe_from_contrib(contrib, wk, lam)
    if noise == "replacement":
        coh = (wk[None] * contrib.sum(-1).abs() ** 2).sum() / 4.0
        return (1 - lam) * coh + lam * _fe_replacement(ops, v, r, wk)
    raise ValueError(f"unknown noise model {noise!r}")


def _isometry(m: torch.Tensor) -> torch.Tensor:
    q, _ = torch.linalg.qr(m)
    return q


def _randc(shape: tuple[int, ...], gen: torch.Generator) -> torch.Tensor:
    m = torch.randn(*shape, 2, dtype=torch.float64, generator=gen)
    return torch.complex(m[..., 0], m[..., 1])


def fixed_encoder(n_orders: int) -> np.ndarray:
    """Target-only encoder with the control in the equal superposition, shape (2C, 2)."""
    a = np.ones(n_orders) / np.sqrt(n_orders)
    return np.kron(a[:, None], np.eye(2)).astype(complex)


def evaluate_code(
    t: SwitchTensors, v: np.ndarray, r: np.ndarray, lam: float = 0.0, noise: str = "dephasing"
) -> float:
    """Entanglement fidelity of a given code under control noise of strength ``lam``.

    ``noise`` is ``"dephasing"`` (order-basis dephasing) or ``"replacement"`` (control
    replaced by the maximally mixed state with probability ``lam``).

    ``v`` has shape (2C, 2) and ``r`` has shape (J, 2, 2C) (Kraus operators of the decoder).
    """
    ops = torch.tensor(t.ops, dtype=torch.complex128)
    wk = torch.tensor(t.wk, dtype=torch.float64)
    rt = torch.tensor(r).reshape(r.shape[0] * 2, -1)
    return float(_objective(ops, torch.tensor(v), rt, wk, lam, noise))


def design_code(
    t: SwitchTensors,
    lam: float = 0.0,
    seed: int = 0,
    steps: int = 400,
    lr: float = 0.05,
    learn_encoder: bool = True,
    n_kraus: int | None = None,
    noise: str = "dephasing",
) -> dict[str, object]:
    """Gradient-ascent search for the encoder and decoder that maximise ``Fe``.

    With ``learn_encoder=False`` the encoder is the target-only fixed encoder and only the
    decoder is optimised.  ``lam`` is the control-noise probability assumed by the optimiser
    (``noise`` selects the model, see ``evaluate_code``).  Returns the best ``Fe`` seen, the encoder ``v`` (2C, 2) and decoder ``r``
    (J, 2, 2C).
    """
    c = t.n_orders
    d = 2 * c
    j = n_kraus or d
    gen = torch.Generator().manual_seed(seed)
    ops = torch.tensor(t.ops, dtype=torch.complex128)
    wk = torch.tensor(t.wk, dtype=torch.float64)
    mv = _randc((d, 2), gen).requires_grad_(learn_encoder)
    mr = _randc((2 * j, d), gen).requires_grad_(True)
    v_fixed = torch.tensor(fixed_encoder(c))
    params = [mr] + ([mv] if learn_encoder else [])
    opt = torch.optim.Adam(params, lr=lr)
    best_fe, best_v, best_r = -1.0, None, None
    for _ in range(steps):
        opt.zero_grad()
        v = _isometry(mv) if learn_encoder else v_fixed
        r = _isometry(mr)
        fe = _objective(ops, v, r, wk, lam, noise)
        (-fe).backward()
        opt.step()
        if fe.item() > best_fe:
            best_fe = fe.item()
            best_v, best_r = v.detach().clone(), r.detach().clone()
    return {
        "fe": float(best_fe),
        "v": best_v.numpy(),
        "r": best_r.reshape(j, 2, d).numpy(),
    }


def hoeffding_shots(avg_fidelity: float, alpha: float = 0.05) -> float:
    """Shots after which a Hoeffding lower bound on the success rate exceeds 2/3.

    Each shot prepares one of the six Pauli eigenstates, applies the code and measures in the
    preparation basis; the success probability averaged over the six states equals the
    average fidelity.  Returns ``inf`` when the fidelity does not exceed ``2/3``.
    """
    margin = avg_fidelity - CLASSICAL_AVG_FIDELITY
    if margin <= 1e-6:
        return float("inf")
    return float(np.log(1.0 / alpha) / (2.0 * margin**2))


def certification_rate(
    avg_fidelity: float, shots: int, trials: int, rng: np.random.Generator, alpha: float = 0.05
) -> float:
    """Monte Carlo probability that the Hoeffding test certifies fidelity above ``2/3``."""
    if shots > 10**8:
        return float("nan")  # infeasible budget; not simulated
    succ = rng.binomial(shots, min(max(avg_fidelity, 0.0), 1.0), size=trials) / shots
    bound = succ - np.sqrt(np.log(1.0 / alpha) / (2.0 * shots))
    return float(np.mean(bound > CLASSICAL_AVG_FIDELITY))

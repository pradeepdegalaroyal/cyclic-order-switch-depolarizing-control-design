# Method

All quantities are computed exactly (density-operator level, no sampling) unless stated. Abbreviations are spelled out at first use.

## Model

* **Channels.** `N` single-qubit Pauli channels `E_i(rho) = sum_k w_ik sigma_k rho sigma_k` with Pauli operators `sigma = (I, X, Y, Z)`. A completely depolarising channel has `w_ik = 1/4`. A depolarising channel of strength `p` has identity weight `1 - 3p/4` and weight `p/4` on each of X, Y and Z.
* **Quantum switch with an order set.** The control register has one basis state `|c>` per order `c` in an order set. For Kraus tuple `k` the switch applies `F_k = sum_c |c><c| (x) W_kc`, where `W_kc` is the product of the Pauli operators of tuple `k` taken in order `c`. The order sets are the `N` cyclic shifts and, as an ablation, all `N!` permutations. A definite order is the special case of a single order.
* **Code.** One logical qubit is encoded by an isometry `V` into control (x) target, passes through the switch and is decoded by a channel `R` (Kraus operators `R_j`, at most `2C` of them, `C` = number of orders) back to one qubit. Quality is the entanglement fidelity `Fe = (1/4) sum_{j,k} w_k |Tr(R_j F_k V)|^2`. The average state fidelity is `(2 Fe + 1)/3`.
* **Classical limit.** The best measure-and-prepare strategy has `Fe = 1/2` (average fidelity `2/3`). A definite order of `N` completely depolarising channels gives `Fe = 1/4`.

## Methods compared

| Label | Description | Role |
|---|---|---|
| Classical limit | measure-and-prepare, `Fe = 1/2` | classical baseline |
| Definite order | the same `N` channels in series, no coherent control, `Fe = 1 - (3/4)(1 - (1-p)^N)` | classical-order baseline |
| Fixed protocol | logical qubit in the target only, control in the equal superposition, Fourier-basis readout of the control, best Pauli correction per outcome | standard quantum-switch baseline |
| Learned decoder only | same target-only encoder, decoder over control (x) target found by gradient ascent | ablation: decoding |
| Learned encoder + decoder | encoder isometry `V` and decoder isometry (Stinespring form of `R`) both parametrised through the QR (orthogonal-triangular) factorisation of complex matrices and optimised with Adam (adaptive moment estimation) in PyTorch | proposed design |

## Metrics

Entanglement fidelity `Fe` (and the average fidelity), the margin over the classical limit, the control-noise probability `lambda*` at which `Fe` falls to `1/2`, the cyclic-versus-all-orders gain, and the number of shots a Hoeffding lower bound needs to certify an average fidelity above `2/3` at 95 % confidence.

## Ablation

(a) What is learned: nothing (fixed protocol), decoder only, encoder and decoder. (b) Order set: `N` cyclic orders versus all `N!` orders for `N = 3, 4`.

## Noise and robustness sweeps

* **Control dephasing.** After the switch the control register is dephased in the order basis: `rho -> (1-lambda) rho + lambda sum_c P_c rho P_c`, `lambda` in {0, 0.15, 0.3, 0.5, 0.75, 1}, for `N = 2` and `N = 4`. A *noise-oblivious* code (best code at `lambda = 0`) is compared with a code *re-designed* for each `lambda`.
* **Control replacement.** A harsher control error: with probability `lambda` the control register is replaced by the maximally mixed state (Kraus operators `|c'><c| / sqrt(C)`), same grid, same oblivious versus re-designed comparison.
* **Channel strength.** Depolarising strength `p` in {0.4, 0.55, 0.7, 0.85, 1}, learned switch code versus definite order and a single channel use.
* **Finite shots.** Each shot prepares one of the six Pauli eigenstates, applies the code and measures in the preparation basis; the success probability averaged over the six states equals the average fidelity. Certification uses a one-sided Hoeffding bound; a Monte Carlo check with 2000 trials is run at the predicted shot count and at twice that count.

## Seeds

Five random restarts per configuration (three for the 24-order case and for decoder-only learning and for the noise sweeps), reported as mean ± standard deviation, with the best restart used as the code in the oblivious-code comparison and in the Pareto plot. Fixed numbers (fixed protocol, definite order) are analytic and have no seed.

## Scale of the study

Simulation on a laptop, `N <= 5` channels, control dimension up to 24. No hardware is involved; control dephasing, control replacement and channel strength are the only imperfections modelled.

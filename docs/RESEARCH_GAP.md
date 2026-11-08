# Research gap

## State of the art (from the supplied literature)

Noise sets the capacity of a communication link, and a fully depolarising channel carries no information when it is used in a definite order. Wang et al. (2026, digital object identifier (DOI) 10.1038/s42005-026-02698-1) show experimentally on a programmable silicon photonic chip that a superposition of cyclic orders of four completely depolarising channels restores a measured output fidelity of 0.712 ± 0.013, above the classical threshold of 2/3. Ren et al. (2026, doi:10.1038/s41377-025-02173-6) note that most entanglement-assisted communication with dense coding is still proof-of-principle and report about three times the classical channel capacity over 20 km of fibre. Breuer et al. (2016, doi:10.1103/RevModPhys.88.021002) and Tanimura (2020, doi:10.1063/5.0011599) set out how memory and strong system-environment coupling break the memoryless (Markovian) picture and why numerically exact treatments are needed. On the classical and learning side, recent receivers adapt to imperfect channel knowledge (Kumar et al., 2026, doi:10.1186/s13634-026-01327-x; Pan et al., 2026, doi:10.1007/s44163-026-02499-z) and a variational quantum circuit has been used as a learned decoder (Devipriya, 2026, doi:10.1016/j.aeue.2026.156615). Hybrid qubits with Hamiltonian protection take a different route to noise robustness (Pita-Vidal et al., 2027, doi:10.1016/j.physrep.2026.10.002).

## Related work table

| Paper | Year | Role | What it does | What it does NOT cover |
|---|---|---|---|---|
| Wang et al., doi:10.1038/s42005-026-02698-1 | 2026 | adjacent | Superposition of cyclic orders of four completely depolarising channels, fidelity 0.712 ± 0.013 above 2/3 | Dependence on the number of channels and orders, optimisation of the code, robustness sweeps, shot cost of the certificate |
| Ren et al., doi:10.1038/s41377-025-02173-6 | 2026 | adjacent | Continuous-variable entanglement-assisted communication over 20 km fibre, about 3x classical capacity | Qubit channels with no capacity at all; learned encoders |
| Breuer et al., doi:10.1103/RevModPhys.88.021002 | 2016 | landmark | Defines and classifies memory effects in open quantum dynamics | Causal-order superposition as a resource; code design |
| Tanimura, doi:10.1063/5.0011599 | 2020 | review | Hierarchical equations of motion for numerically exact non-perturbative dynamics | Information transmission figures of merit; learned control |
| Pita-Vidal et al., doi:10.1016/j.physrep.2026.10.002 | 2027 | recent | Hamiltonian-protected hybrid semiconductor-superconductor qubits for noise robustness | Protection by coherent control of operation order |
| Devipriya, doi:10.1016/j.aeue.2026.156615 | 2026 | adjacent | Variational quantum circuit as reliability scorer in a wireless receiver | Equal-parameter classical baseline; a fundamental fidelity limit to compare with |
| Kumar et al., doi:10.1186/s13634-026-01327-x | 2026 | adjacent | Recurrent-network detector robust to 10 % and 20 % channel-estimate mismatch | Quantum links; re-design versus reuse of a nominal design under mismatch |
| Pan et al., doi:10.1007/s44163-026-02499-z | 2026 | adjacent | Channel-state-conditioned denoising for time-varying channels | Quantum channels; an exact optimum to benchmark against |
| Shao et al., doi:10.1007/978-981-92-1678-9_21 | 2027 | adjacent | Shows that a spectrum-efficiency gain cannot exceed the Shannon limit once normalised properly | Quantum threshold tests; finite-shot certification |
| Zhang et al., doi:10.1186/s13634-026-01314-2 | 2026 | adjacent | Stepwise optimisation of antenna and access allocation in satellite-terrestrial networks | Quantum channels; control-register resources |

## The gap

The recent switch experiment (Wang et al., 2026) establishes that a single operating point beats the 2/3 threshold. The landmark and review papers on open systems (Breuer et al., 2016; Tanimura, 2020) say that real noise has structure and that exact treatments matter, and the learning-based receivers (Kumar et al., 2026; Pan et al., 2026; Devipriya, 2026) adapt a decoder to the channel. Nobody in this set asks, for the *switch of depolarising channels*, (i) which encoding and decoding reach the best fidelity as the number of channels and the set of orders grow, (ii) whether using the control register as a data carrier is necessary, (iii) how much control noise and channel weakening the advantage tolerates and whether re-designing the code for the noise helps, and (iv) how many shots are needed to certify the margin.

## Falsifiable hypotheses and metric

* **H1.** A gradient-designed encoder and decoder that use the control register reach an entanglement fidelity `Fe > 1/2` (average fidelity above 2/3) for every `N >= 2` with cyclic orders, while the fixed protocol that encodes in the target only stays below `1/2`. Falsified if the learned `Fe <= 1/2` or the fixed protocol exceeds `1/2`.
* **H2.** Using all `N!` orders gives a higher best `Fe` than the `N` cyclic orders at `N = 3` and `4`. Falsified if the gain is within restart spread.
* **H3.** Re-designing the code for the control noise (order-basis dephasing, or replacement of the control by the maximally mixed state) raises the tolerance `lambda*` (where `Fe = 1/2`) above that of the noise-oblivious code. Falsified if the two curves coincide within spread.

Success metric: entanglement fidelity `Fe` (mean ± standard deviation over restarts) against the classical limit `1/2`, plus the shots needed to certify it.

## Novelty in one sentence

This study treats the control register of the order-superposition switch as a learned part of the code and maps, in exact simulation, how the fidelity margin over the classical threshold scales with the number of channels and orders, with control dephasing and channel strength, and with the shot budget of the certificate.

## Scale and honesty

Everything is exact simulation of an idealised switch on a laptop (up to five channels and 24 orders); no hardware data is used and the numbers are not a reproduction of the cited experiment, which may use a different encoding and fidelity definition.

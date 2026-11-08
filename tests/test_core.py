import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cyclic_switch import (  # noqa: E402
    all_orders,
    build_switch,
    certification_rate,
    cyclic_orders,
    design_code,
    direct_fe,
    evaluate_code,
    fe_to_avg_fidelity,
    fixed_protocol,
    hoeffding_shots,
    pauli_weights,
)


def test_orders():
    assert cyclic_orders(3) == [(0, 1, 2), (1, 2, 0), (2, 0, 1)]
    assert len(all_orders(4)) == 24


def test_weights_normalised_and_operators_unitary():
    t = build_switch(pauli_weights(3, 0.7), cyclic_orders(3))
    assert np.isclose(t.wk.sum(), 1.0)
    for k in range(0, len(t.wk), 7):
        for c in range(3):
            m = t.ops[k, c]
            assert np.allclose(m.conj().T @ m, np.eye(2))


def test_single_depolarising_channel_limit():
    t = build_switch(pauli_weights(1, 1.0), cyclic_orders(1))
    assert np.isclose(fixed_protocol(t), 0.25)
    assert np.isclose(direct_fe(1.0, 1), 0.25)
    assert np.isclose(direct_fe(0.0, 3), 1.0)


def test_fixed_protocol_below_classical_limit():
    f2 = fixed_protocol(build_switch(pauli_weights(2, 1.0), cyclic_orders(2)))
    assert np.isclose(f2, 0.375)
    assert f2 < 0.5
    assert np.isclose(fe_to_avg_fidelity(0.5), 2 / 3)


def test_learned_code_beats_classical_limit_n2():
    t = build_switch(pauli_weights(2, 1.0), cyclic_orders(2))
    r = design_code(t, seed=0, steps=300)
    assert r["fe"] > 0.55
    assert np.isclose(evaluate_code(t, r["v"], r["r"], 0.0), r["fe"], atol=1e-6)


def test_full_dephasing_hurts():
    t = build_switch(pauli_weights(2, 1.0), cyclic_orders(2))
    r = design_code(t, seed=0, steps=300)
    assert evaluate_code(t, r["v"], r["r"], 1.0) < evaluate_code(t, r["v"], r["r"], 0.0)


def test_certification_helpers():
    assert hoeffding_shots(0.6) == float("inf")
    n = hoeffding_shots(0.75)
    rate = certification_rate(0.75, int(np.ceil(2 * n)), 300, np.random.default_rng(0))
    assert rate > 0.8

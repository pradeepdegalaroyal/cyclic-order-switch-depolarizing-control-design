"""Learned encoder-decoder design for the quantum switch of depolarising channels."""
from .core import (
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

__all__ = [
    "all_orders",
    "build_switch",
    "certification_rate",
    "cyclic_orders",
    "design_code",
    "direct_fe",
    "evaluate_code",
    "fe_to_avg_fidelity",
    "fixed_protocol",
    "hoeffding_shots",
    "pauli_weights",
]
__version__ = "0.1.0"

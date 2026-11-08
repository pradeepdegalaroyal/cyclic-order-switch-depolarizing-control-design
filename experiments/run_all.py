"""Reproduce every result and figure of the study.

Usage::

    python experiments/run_all.py            # full study (about 8-10 minutes on a laptop)
    python experiments/run_all.py --quick    # smoke run (under one minute)

Raw numbers go to ``results/`` and figures to ``figures/``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from cyclic_switch.core import (  # noqa: E402
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

torch.set_num_threads(6)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def experiment_scaling(cfg: dict) -> tuple[pd.DataFrame, dict]:
    """E1: fixed protocol vs decoder-only learning vs full encoder+decoder learning."""
    rows = []
    best_codes: dict[str, dict] = {}
    for n, kind in cfg["configs"]:
        orders = cyclic_orders(n) if kind == "cyclic" else all_orders(n)
        t = build_switch(pauli_weights(n, 1.0), orders)
        name = f"{kind}-N{n}"
        fixed = fixed_protocol(t)
        rows.append(dict(config=name, n_channels=n, order_set=kind, n_orders=len(orders),
                         method="fixed protocol", seed=0, fe=fixed))
        log(f"{name}: fixed protocol Fe = {fixed:.4f}")
        if kind == "cyclic" and n <= cfg["decoder_only_max_n"]:
            for s in range(cfg["seeds_dec"]):
                r = design_code(t, seed=s, learn_encoder=False, steps=cfg["steps_dec"], lr=0.03)
                rows.append(dict(config=name, n_channels=n, order_set=kind, n_orders=len(orders),
                                 method="learned decoder only", seed=s, fe=r["fe"]))
        seeds = cfg["seeds_full_big"] if len(orders) > 6 else cfg["seeds"]
        best = None
        for s in range(seeds):
            r = design_code(t, seed=s, steps=cfg["steps"])
            rows.append(dict(config=name, n_channels=n, order_set=kind, n_orders=len(orders),
                             method="learned encoder + decoder", seed=s, fe=r["fe"]))
            if best is None or r["fe"] > best["fe"]:
                best = r
        best_codes[name] = best
        log(f"{name}: learned best Fe = {best['fe']:.4f}")
    return pd.DataFrame(rows), best_codes


def experiment_dephasing(cfg: dict, best_codes: dict) -> pd.DataFrame:
    """E2a: control dephasing; noise-oblivious code vs code re-designed for the noise."""
    rows = []
    for n in cfg["noise_n"]:
        t = build_switch(pauli_weights(n, 1.0), cyclic_orders(n))
        code0 = best_codes[f"cyclic-N{n}"]
        for lam in cfg["lams"]:
            obl = evaluate_code(t, code0["v"], code0["r"], lam)
            rows.append(dict(n_channels=n, lam=lam, method="oblivious code", seed=0, fe=obl))
            for s in range(cfg["seeds_noise"]):
                r = design_code(t, lam=lam, seed=s, steps=cfg["steps_noise"])
                rows.append(dict(n_channels=n, lam=lam, method="re-designed code", seed=s, fe=r["fe"]))
            log(f"dephasing N={n} lam={lam}: oblivious {obl:.4f}")
    return pd.DataFrame(rows)


def experiment_replacement(cfg: dict) -> pd.DataFrame:
    """E2c: harsher control noise, the control is replaced by the maximally mixed state."""
    rows = []
    for n in cfg["noise_n"]:
        t = build_switch(pauli_weights(n, 1.0), cyclic_orders(n))
        code0 = max((design_code(t, seed=s, steps=cfg["steps_noise"]) for s in range(cfg["seeds_noise"])),
                    key=lambda r: r["fe"])
        for lam in cfg["lams"]:
            obl = evaluate_code(t, code0["v"], code0["r"], lam, noise="replacement")
            rows.append(dict(n_channels=n, lam=lam, method="oblivious code", seed=0, fe=obl))
            for s in range(cfg["seeds_noise"]):
                r = design_code(t, lam=lam, seed=s, steps=cfg["steps_noise"], noise="replacement")
                rows.append(dict(n_channels=n, lam=lam, method="re-designed code", seed=s, fe=r["fe"]))
            log(f"replacement N={n} lam={lam}: oblivious {obl:.4f}")
    return pd.DataFrame(rows)


def experiment_strength(cfg: dict) -> pd.DataFrame:
    """E2b: channel strength p below 1 (partially depolarising channels)."""
    rows = []
    for n in cfg["noise_n"]:
        for p in cfg["ps"]:
            t = build_switch(pauli_weights(n, p), cyclic_orders(n))
            rows.append(dict(n_channels=n, p=p, method="definite order", seed=0, fe=direct_fe(p, n)))
            rows.append(dict(n_channels=n, p=p, method="single channel", seed=0, fe=direct_fe(p, 1)))
            rows.append(dict(n_channels=n, p=p, method="fixed protocol", seed=0, fe=fixed_protocol(t)))
            for s in range(cfg["seeds_noise"]):
                r = design_code(t, seed=s, steps=cfg["steps_noise"])
                rows.append(dict(n_channels=n, p=p, method="learned code", seed=s, fe=r["fe"]))
            log(f"strength N={n} p={p} done")
    return pd.DataFrame(rows)


def experiment_certification(noise_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """E3: shots needed to certify average fidelity above 2/3 for the re-designed code."""
    rng = np.random.default_rng(2026)
    rows = []
    sub = noise_df[noise_df.method == "re-designed code"]
    for (noise, n, lam), g in sub.groupby(["noise", "n_channels", "lam"]):
        fe = g.fe.max()
        favg = fe_to_avg_fidelity(fe)
        n_star = hoeffding_shots(favg)
        row = dict(noise=noise, n_channels=n, lam=lam, fe_best=fe, avg_fidelity=favg,
                   hoeffding_shots=n_star)
        for mult in (1, 2):
            if np.isfinite(n_star):
                shots = int(np.ceil(mult * n_star))
                row[f"cert_rate_{mult}x"] = certification_rate(favg, shots, cfg["mc_trials"], rng)
            else:
                row[f"cert_rate_{mult}x"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smoke run under one minute")
    ap.add_argument("--reuse", action="store_true",
                    help="reuse existing optimisation CSVs, recompute certification and figures")
    args = ap.parse_args()
    if args.quick:
        cfg = dict(configs=[(2, "cyclic"), (3, "cyclic")], decoder_only_max_n=2, seeds_dec=1,
                   steps_dec=200, seeds=1, seeds_full_big=1, steps=150, noise_n=[2],
                   lams=[0.0, 0.5], ps=[0.7, 1.0], seeds_noise=1, steps_noise=100, mc_trials=200)
    else:
        cfg = dict(configs=[(2, "cyclic"), (3, "cyclic"), (4, "cyclic"), (5, "cyclic"),
                            (3, "all"), (4, "all")],
                   decoder_only_max_n=4, seeds_dec=3, steps_dec=1500, seeds=5, seeds_full_big=3,
                   steps=600, noise_n=[2, 4], lams=[0.0, 0.15, 0.3, 0.5, 0.75, 1.0],
                   ps=[0.4, 0.55, 0.7, 0.85, 1.0], seeds_noise=3, steps_noise=400, mc_trials=2000)
    out = ROOT / ("results_quick" if args.quick else "results")
    fig_dir = ROOT / ("figures_quick" if args.quick else "figures")
    out.mkdir(exist_ok=True)
    fig_dir.mkdir(exist_ok=True)
    t0 = time.time()

    if args.reuse:
        scaling = pd.read_csv(out / "scaling.csv")
        deph = pd.read_csv(out / "noise_dephasing.csv")
        strength = pd.read_csv(out / "noise_strength.csv")
        if (out / "noise_replacement.csv").exists():
            repl = pd.read_csv(out / "noise_replacement.csv")
        else:
            repl = experiment_replacement(cfg)
            repl.to_csv(out / "noise_replacement.csv", index=False)
    else:
        scaling, best_codes = experiment_scaling(cfg)
        scaling.to_csv(out / "scaling.csv", index=False)
        deph = experiment_dephasing(cfg, best_codes)
        deph.to_csv(out / "noise_dephasing.csv", index=False)
        strength = experiment_strength(cfg)
        strength.to_csv(out / "noise_strength.csv", index=False)
        repl = experiment_replacement(cfg)
        repl.to_csv(out / "noise_replacement.csv", index=False)
    both = pd.concat([deph.assign(noise="dephasing"), repl.assign(noise="replacement")])
    cert = experiment_certification(both, cfg)
    cert.to_csv(out / "certification.csv", index=False)

    summary = {
        "classical_limit_fe": 0.5,
        "classical_limit_avg_fidelity": 2 / 3,
        "runtime_seconds": round(time.time() - t0, 1),
        "torch_threads": 6,
        "quick": bool(args.quick),
        "scaling": scaling.groupby(["config", "method"]).fe.agg(["mean", "std", "max", "count"])
        .reset_index().to_dict(orient="records"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    import make_figures

    make_figures.make_all(out, fig_dir)
    log(f"done in {time.time() - t0:.0f} s; results in {out}, figures in {fig_dir}")


if __name__ == "__main__":
    main()

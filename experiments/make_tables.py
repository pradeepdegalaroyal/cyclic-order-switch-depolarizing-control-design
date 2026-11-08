"""Print the markdown tables used in the README from the files in ``results/``."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

R = Path(__file__).resolve().parents[1] / "results"


def ms(g: pd.Series) -> str:
    sd = g.std(ddof=1) if len(g) > 1 else float("nan")
    return f"{g.mean():.4f}" if pd.isna(sd) else f"{g.mean():.4f} ± {sd:.4f}"


def scaling_table() -> str:
    s = pd.read_csv(R / "scaling.csv")
    lines = ["| Configuration | Orders | Fixed protocol | Learned decoder only | Learned encoder + decoder (mean ± std) | Best restart | Best average fidelity | Restarts |",
             "|---|---|---|---|---|---|---|---|"]
    for c in dict.fromkeys(s.config):
        g = s[s.config == c]
        fixed = g[g.method == "fixed protocol"].fe.iloc[0]
        dec = g[g.method == "learned decoder only"].fe
        full = g[g.method == "learned encoder + decoder"].fe
        n_ord = int(g.n_orders.iloc[0])
        lines.append(f"| {c} | {n_ord} | {fixed:.4f} | {ms(dec) if len(dec) else '–'} | {ms(full)} | "
                     f"{full.max():.4f} | {(2 * full.max() + 1) / 3:.4f} | {len(full)} |")
    return "\n".join(lines)


def noise_table(name: str) -> str:
    d = pd.read_csv(R / name)
    lines = ["| N | noise probability | Oblivious code | Re-designed code (mean ± std) |", "|---|---|---|---|"]
    for (n, lam), g in d.groupby(["n_channels", "lam"]):
        obl = g[g.method == "oblivious code"].fe.iloc[0]
        red = g[g.method == "re-designed code"].fe
        lines.append(f"| {n} | {lam} | {obl:.4f} | {ms(red)} |")
    return "\n".join(lines)


def strength_table() -> str:
    d = pd.read_csv(R / "noise_strength.csv")
    lines = ["| N | p | Learned switch code (mean ± std) | Fixed protocol | Definite order | One channel use |", "|---|---|---|---|---|---|"]
    for (n, p), g in d.groupby(["n_channels", "p"]):
        val = lambda m: g[g.method == m].fe
        lines.append(f"| {n} | {p} | {ms(val('learned code'))} | {val('fixed protocol').iloc[0]:.4f} | "
                     f"{val('definite order').iloc[0]:.4f} | {val('single channel').iloc[0]:.4f} |")
    return "\n".join(lines)


def cert_table() -> str:
    c = pd.read_csv(R / "certification.csv")
    lines = ["| Noise model | N | noise probability | Best Fe | Average fidelity | Hoeffding shots | Certified at 1x shots | Certified at 2x shots |",
             "|---|---|---|---|---|---|---|---|"]
    for _, r in c.iterrows():
        shots = "no margin" if r.hoeffding_shots == float("inf") else f"{r.hoeffding_shots:,.0f}"
        c1 = "–" if pd.isna(r.cert_rate_1x) else f"{r.cert_rate_1x:.3f}"
        c2 = "–" if pd.isna(r.cert_rate_2x) else f"{r.cert_rate_2x:.3f}"
        lines.append(f"| {r.noise} | {r.n_channels} | {r.lam} | {r.fe_best:.4f} | {r.avg_fidelity:.4f} | {shots} | {c1} | {c2} |")
    return "\n".join(lines)


if __name__ == "__main__":
    out = ["## Scaling", scaling_table(), "", "## Dephasing", noise_table("noise_dephasing.csv"), "",
           "## Replacement", noise_table("noise_replacement.csv"), "", "## Strength", strength_table(), "",
           "## Certification", cert_table()]
    text = "\n".join(out)
    (R / "tables.md").write_text(text, encoding="utf-8")
    print(text)

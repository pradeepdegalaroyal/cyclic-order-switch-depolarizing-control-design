"""Figures for the study (Okabe-Ito palette, 200 dpi)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OI = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
      "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7"}
DPI = 200
CLASSICAL = 0.5

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def _label(config: str) -> str:
    kind, n = config.split("-N")
    return f"N={n}\n{'cyclic' if kind == 'cyclic' else 'all orders'}"


def fig_scaling_bars(scaling: pd.DataFrame, path: Path) -> None:
    methods = [("fixed protocol", OI["sky"], "//"), ("learned decoder only", OI["orange"], "xx"),
               ("learned encoder + decoder", OI["green"], "..")]
    configs = list(dict.fromkeys(scaling.config))
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    w = 0.26
    for i, (m, col, hatch) in enumerate(methods):
        means, stds, xs = [], [], []
        for j, c in enumerate(configs):
            g = scaling[(scaling.config == c) & (scaling.method == m)].fe
            if len(g):
                xs.append(j + (i - 1) * w)
                means.append(g.mean())
                stds.append(g.std(ddof=1) if len(g) > 1 else 0.0)
        ax.bar(xs, means, w, yerr=stds, color=col, edgecolor="black", hatch=hatch, capsize=3, label=m)
    ax.axhline(CLASSICAL, color=OI["vermillion"], ls="--", lw=1.5, label="measure-and-prepare limit (0.5)")
    ax.axhline(0.25, color="black", ls=":", lw=1.2, label="definite order (0.25)")
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels([_label(c) for c in configs])
    ax.set_ylabel("entanglement fidelity Fe (dimensionless)")
    ax.set_xlabel("switch configuration (completely depolarising channels)")
    ax.set_title("Only an encoder that uses the control register clears the classical limit")
    ax.set_ylim(0, 0.85)
    ax.legend(fontsize=8, loc="upper left", ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def fig_restarts(scaling: pd.DataFrame, path: Path) -> None:
    sub = scaling[scaling.method == "learned encoder + decoder"]
    configs = list(dict.fromkeys(sub.config))
    data = [sub[sub.config == c].fe.values for c in configs]
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    bp = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", lw=1.5))
    for b in bp["boxes"]:
        b.set(facecolor="#d9eef9", edgecolor="black")
    rng = np.random.default_rng(0)
    for i, d in enumerate(data, start=1):
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(d)), d, s=28, color=OI["blue"], edgecolor="black",
                   zorder=3, label="individual restart" if i == 1 else None)
    ax.axhline(CLASSICAL, color=OI["vermillion"], ls="--", lw=1.5, label="classical limit (0.5)")
    ax.set_xticks(range(1, len(configs) + 1))
    ax.set_xticklabels([_label(c) for c in configs])
    ax.set_ylabel("Fe after training (dimensionless)")
    ax.set_xlabel("switch configuration")
    ax.set_title("Spread of the learned encoder-decoder over random restarts")
    ax.set_ylim(0.47, 0.74)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def fig_pareto(scaling: pd.DataFrame, path: Path) -> None:
    sub = scaling[scaling.method == "learned encoder + decoder"]
    pts = sub.groupby("config").agg(fe=("fe", "max"), c=("n_orders", "first")).reset_index()
    pts = pts.sort_values("c")
    front, best = [], -1.0
    for _, r in pts.sort_values(["c", "fe"], ascending=[True, False]).iterrows():
        if r.fe > best + 1e-9:
            front.append(r.config)
            best = r.fe
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for i, (_, r) in enumerate(pts.iterrows()):
        on = r.config in front
        ax.scatter(r.c, r.fe, s=90 if on else 55, color=OI["green"] if on else OI["sky"],
                   edgecolor="black", marker="D" if on else "o", zorder=3)
        ax.annotate(_label(r.config).replace("\n", ", "), (r.c, r.fe), textcoords="offset points",
                    xytext=(-20 if i % 2 == 0 else -20, -16 if i % 2 == 0 else 12), fontsize=8)
    fp = pts[pts.config.isin(front)]
    ax.step(fp.c, fp.fe, where="post", color=OI["green"], lw=1.5, label="Pareto front")
    ax.axhline(CLASSICAL, color=OI["vermillion"], ls="--", lw=1.2, label="classical limit (0.5)")
    ax.set_xscale("log")
    ax.set_xlabel("control register dimension = number of orders (log scale)")
    ax.set_ylabel("best learned Fe (dimensionless)")
    ax.set_title("Quality versus control-register cost")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def _band(ax, df, x, color, label, ls="-"):
    g = df.groupby(x).fe.agg(["mean", "std"]).fillna(0.0)
    ax.plot(g.index, g["mean"], color=color, ls=ls, marker="o", ms=4, label=label)
    ax.fill_between(g.index, g["mean"] - g["std"], g["mean"] + g["std"], color=color, alpha=0.2)


def _dephasing_panel(ax, df, title, xlabel):
    cols = {2: OI["blue"], 4: OI["vermillion"]}
    for n in sorted(df.n_channels.unique()):
        d = df[df.n_channels == n]
        _band(ax, d[d.method == "re-designed code"], "lam", cols.get(n, "k"), f"N={n}, re-designed for the noise")
        _band(ax, d[d.method == "oblivious code"], "lam", cols.get(n, "k"), f"N={n}, designed at noise 0", ls="--")
    ax.axhline(CLASSICAL, color="black", ls=":", lw=1.3, label="classical limit (0.5)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("entanglement fidelity Fe (dimensionless)")
    ax.set_title(title)
    ax.legend(fontsize=7, frameon=False)


def fig_noise(deph: pd.DataFrame, repl: pd.DataFrame, strength: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    _dephasing_panel(axes[0], deph, "(a) Control dephasing in the order basis",
                     "dephasing probability (dimensionless)")
    _dephasing_panel(axes[1], repl, "(b) Control replaced by maximally mixed state",
                     "replacement probability (dimensionless)")
    ax = axes[2]
    cols = {2: OI["blue"], 4: OI["vermillion"]}
    for n in sorted(strength.n_channels.unique()):
        s = strength[strength.n_channels == n]
        _band(ax, s[s.method == "learned code"], "p", cols.get(n, "k"), f"N={n}, learned switch code")
        _band(ax, s[s.method == "definite order"], "p", cols.get(n, "k"), f"N={n}, definite order", ls="--")
    s1 = strength[strength.method == "single channel"].groupby("p").fe.mean()
    ax.plot(s1.index, s1.values, color=OI["green"], ls="-.", marker="s", ms=4, label="one channel use")
    ax.axhline(CLASSICAL, color="black", ls=":", lw=1.3)
    ax.set_xlabel("depolarising strength p of every channel (dimensionless)")
    ax.set_ylabel("entanglement fidelity Fe (dimensionless)")
    ax.set_title("(c) Partially depolarising channels")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def fig_certification(cert: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    titles = {"dephasing": "(a) Order-basis dephasing", "replacement": "(b) Control replacement"}
    for ax, noise in zip(axes, ["dephasing", "replacement"]):
        c = cert[cert.noise == noise]
        ns = sorted(c.n_channels.unique())
        lams = sorted(c.lam.unique())
        w = 0.38
        for i, n in enumerate(ns):
            d = c[c.n_channels == n].set_index("lam").reindex(lams)
            vals = d.hoeffding_shots.replace(np.inf, np.nan).values
            xs = np.arange(len(lams)) + (i - 0.5) * w
            ax.bar(xs, np.nan_to_num(vals, nan=0.0), w, color=[OI["blue"], OI["vermillion"]][i % 2],
                   edgecolor="black", hatch=["//", "xx"][i % 2], label=f"N={n}")
            for x, v in zip(xs, vals):
                if np.isnan(v):
                    ax.text(x, 260, "no margin", rotation=90, ha="center", va="bottom", fontsize=7)
        ax.set_yscale("log")
        ax.set_xticks(range(len(lams)))
        ax.set_xticklabels([str(l) for l in lams])
        ax.set_xlabel("control noise probability (dimensionless)")
        ax.set_title(titles[noise])
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("Hoeffding shots to certify F > 2/3 (count)")
    fig.suptitle("Shot cost of certifying the margin over the classical threshold (95 % confidence)")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def make_all(results: Path, fig_dir: Path) -> None:
    scaling = pd.read_csv(results / "scaling.csv")
    deph = pd.read_csv(results / "noise_dephasing.csv")
    strength = pd.read_csv(results / "noise_strength.csv")
    repl = pd.read_csv(results / "noise_replacement.csv")
    cert = pd.read_csv(results / "certification.csv")
    fig_scaling_bars(scaling, fig_dir / "fig1_scaling_bars.png")
    fig_restarts(scaling, fig_dir / "fig2_restart_spread.png")
    fig_pareto(scaling, fig_dir / "fig3_pareto_cost_quality.png")
    fig_noise(deph, repl, strength, fig_dir / "fig4_noise_sweeps.png")
    fig_certification(cert, fig_dir / "fig5_certification_shots.png")

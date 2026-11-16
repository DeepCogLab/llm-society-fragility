"""Publication figures (PDF + PNG) from the unified agent-round table and analysis CSVs."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "agent_rounds_v3.parquet"
RES = ROOT / "results" / "tables"
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
SHORT = {"Qwen2.5-7B-Instruct": "Qwen2.5-7B", "Qwen3-14B": "Qwen3-14B", "Qwen3-30B-A3B-GPTQ-Int4": "Qwen3-30B-A3B",
         "Llama-3.1-8B-Instruct": "Llama-3.1-8B", "GLM-4-9B-0414": "GLM-4-9B"}
ORDER = ["Qwen2.5-7B", "Qwen3-14B", "Qwen3-30B-A3B", "Llama-3.1-8B", "GLM-4-9B"]
COL = {"Qwen2.5-7B": "#2a6fdb", "Qwen3-14B": "#d1495b", "Qwen3-30B-A3B": "#edae49", "Llama-3.1-8B": "#00798c", "GLM-4-9B": "#7b4b94"}
SCH = {"initial_only": ("#c0392b", "evidence at init only (ρ=0)"), "persistent": ("#1f618d", "evidence re-accessible (ρ=1)"),
       "no_relay": ("#7f8c8d", "ρ=0, relay of non-seeds blocked")}
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans",
                     "axes.titlesize": 9, "legend.frameon": False, "figure.dpi": 150})


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def boot(v, n=2000, seed=0):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    if not len(v):
        return np.nan, np.nan, np.nan
    b = np.random.default_rng(seed).choice(v, (n, len(v))).mean(1)
    return v.mean(), np.quantile(b, .025), np.quantile(b, .975)


def load():
    df = pd.read_parquet(DATA)
    df["M"] = df.model.map(SHORT)
    return df


def deltas(df, schedule, arm, source=None):
    x = df[(df.seed == 0) & (df.schedule == schedule)]
    if source:
        x = x[x.source.isin(source)]
    e = x.groupby(["M", "replicate", "topology", "arm", "round"]).wrong.mean().unstack("arm")
    if arm not in e or "control" not in e:
        return None
    return (e[arm] - e["control"]).groupby(["M", "replicate", "round"]).mean().unstack("round")


def fig_memory(df):
    models = [m for m in ORDER if m in set(df.M)]
    fig, axes = plt.subplots(1, len(models), figsize=(2.3 * len(models), 2.3), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, models):
        ax.axvspan(1.5, 3.5, color="#f4d03f", alpha=.25, lw=0)
        for sch, arm, key in [("initial_only", "treated", "initial_only"), ("persistent", "treated", "persistent"), ("initial_only", "no_relay_treated", "no_relay")]:
            d = deltas(df[df.M == m], sch, arm, source=["v3", "v4"])
            if d is None or m not in d.index.get_level_values(0):
                continue
            g = d.loc[m]
            est = [boot(g[t].values, seed=t) for t in range(9)]
            mu, lo, hi = zip(*est)
            c, lab = SCH[key]
            ax.plot(range(9), np.array(mu) * 100, color=c, lw=1.6, label=lab, ls="--" if key == "no_relay" else "-")
            ax.fill_between(range(9), np.array(lo) * 100, np.array(hi) * 100, color=c, alpha=.15, lw=0)
        ax.axhline(0, color="k", lw=.5)
        ax.set_title(m)
        ax.set_xlabel("round")
        ax.set_xticks(range(0, 9, 2))
    axes[0].set_ylabel("extra non-seed error (pp)\ntreated − control")
    axes[0].text(2.5, axes[0].get_ylim()[1] * .92, "perturbation", ha="center", fontsize=7, color="#9a7d0a")
    axes[-1].legend(loc="center left", bbox_to_anchor=(1, .5))
    save(fig, "fig1_perturbation_memory")


def fig_individual_vs_network(df):
    prof = []
    for f in sorted((ROOT / "data" / "profiles_v3").glob("*.csv")) + sorted((ROOT / "data" / "profiles_v4").glob("*.csv")):
        d = pd.read_csv(f)
        d["M"] = SHORT.get(f.stem, f.stem)
        prof.append(d)
    prof = pd.concat(prof)
    prof = prof[prof.stage.isin(["validation", "profile"])]
    models = [m for m in ORDER if m in set(df.M)]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3))
    s = prof[(prof.support >= 0) & (prof.kind == "report") & (prof.interface == "natural") & (prof.history == "aligned") & (prof.q == .7)]
    for m in models:
        g = s[s.M == m].groupby("support").changed_private.mean()
        if len(g):
            axes[0].plot(g.index, g.values * 100, "o-", color=COL[m], ms=3, label=m)
    axes[0].plot([0, 1, 2, 3, 4], [0, 0, 50, 100, 100], color="k", ls=":", lw=1, label="Bayes (independent peers)")
    axes[0].set_title("A  isolation: private evidence shown")
    axes[0].set_xlabel("# of 4 peers opposing own evidence")
    axes[0].set_ylabel("switch away from own evidence (%)")
    df = df.sort_values(["model", "schedule", "block", "arm", "node", "round"])
    df["prev"] = df.groupby(["model", "schedule", "block", "arm", "node"]).wrong.shift(1)
    for ax, sch, title in [(axes[1], "persistent", "B  network, evidence re-accessible"), (axes[2], "initial_only", "C  network, evidence at init only")]:
        x = df[(df.arm == "control") & (df.seed == 0) & (df["round"] >= 1) & (df.schedule == sch) & (df.n_inbox == 4)]
        for m in models:
            g = x[x.M == m]
            if not len(g):
                continue
            # probability of disagreeing with own previous public choice as a function of # neighbours disagreeing with it
            opp = np.where(g.prev == 1, 4 - g.n_wrong_inbox, g.n_wrong_inbox)
            switched = (g.wrong != g.prev).astype(float)
            r = pd.Series(switched.values).groupby(opp).mean()
            ax.plot(r.index, r.values * 100, "o-", color=COL[m], ms=3)
        ax.plot([0, 1, 2, 3, 4], [0, 0, 50, 100, 100], color="k", ls=":", lw=1)
        ax.set_title(title)
        ax.set_xlabel("# of 4 neighbours opposing own last choice")
    axes[1].set_ylabel("switch own choice (%)")
    axes[0].legend(fontsize=6.5, loc="upper left")
    fig.tight_layout()
    save(fig, "fig2_individual_vs_network_rules")


def fig_anchors(df, pred_path):
    models = [m for m in ORDER if m in set(df.M)]
    x = df[(df.seed == 0) & (df["round"] >= 6) & df.schedule.isin(["initial_only", "anchor25", "anchor50", "persistent"])]
    e = x.groupby(["M", "anchor_fraction", "replicate", "topology", "arm"]).wrong.mean().unstack("arm")
    e["d"] = e.treated - e.control
    t = e.groupby(["M", "anchor_fraction", "replicate"])[["d", "control", "treated"]].mean()
    pred = pd.read_csv(pred_path) if Path(pred_path).exists() else None
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))
    for m in models:
        if m not in t.index.get_level_values(0):
            continue
        g = t.loc[m]
        fr = sorted(set(g.index.get_level_values(0)))
        for ax, col in [(axes[0], "d"), (axes[1], "control")]:
            est = [boot(g.loc[f][col].values, seed=int(f * 100)) for f in fr]
            mu, lo, hi = map(np.array, zip(*est))
            ax.errorbar(np.array(fr) + (ORDER.index(m) - 2) * .012, mu * 100, yerr=[(mu - lo) * 100, (hi - mu) * 100], fmt="o", color=COL[m], ms=3.5, capsize=1.5, lw=1, label=m)
        if pred is not None:
            pm = {v: k for k, v in SHORT.items()}.get(m)
            p = pred[(pred.model == pm) & (pred["round"] >= 6)]
            if len(p):
                p = p.groupby(["anchor_fraction", "block", "arm"]).pred_nonseed_error.mean().unstack("arm")
                p["d"] = p.treated - p.control
                pp = p.groupby("anchor_fraction")[["d", "control"]].mean()
                axes[0].plot(pp.index, pp.d * 100, color=COL[m], lw=1, alpha=.6)
                axes[1].plot(pp.index, pp.control * 100, color=COL[m], lw=1, alpha=.6)
        mu_d = g.groupby(level=0).d.mean() * 100
        mu_c = g.groupby(level=0).control.mean() * 100
        axes[2].plot(mu_c.values, mu_d.values, "-", color=COL[m], lw=1)
        axes[2].scatter(mu_c.values, mu_d.values, c=[COL[m]], s=[10 + 40 * f for f in mu_c.index], zorder=3)
    axes[0].set_title("A  perturbation memory (tail)")
    axes[0].set_ylabel("extra non-seed error (pp)")
    axes[1].set_title("B  normal-operation error (control)")
    axes[1].set_ylabel("non-seed error (%)")
    for ax in axes[:2]:
        ax.set_xlabel("anchor fraction ρ (agents re-accessing evidence)")
        ax.set_xticks([0, .25, .5, .75, 1])
    axes[2].set_title("C  learning–resilience trade-off")
    axes[2].set_xlabel("normal error (%)")
    axes[2].set_ylabel("perturbation memory (pp)")
    axes[0].legend(fontsize=6.5)
    axes[0].text(.98, .98, "lines: ABM prediction\nfrozen before runs", transform=axes[0].transAxes, ha="right", va="top", fontsize=6.5, color="#555")
    fig.tight_layout()
    save(fig, "fig3_anchor_dose_response")


def fig_phase(phase_csv, params_csv):
    if not Path(phase_csv).exists():
        return
    r = pd.read_csv(phase_csv)
    p = pd.read_csv(params_csv)
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6))
    for ax, col, title, cmap in [(axes[0], "delta_tail", "A  perturbation memory (pp)", "Reds"), (axes[1], "control_tail", "B  normal-operation error (%)", "Blues")]:
        piv = r[r.n == 96].groupby(["w_over_a", "rho"])[col].mean().unstack("rho") * 100
        im = ax.imshow(piv.values, origin="lower", aspect="auto", cmap=cmap, extent=[-.05, 1.05, -.5, len(piv) - .5])
        ax.set_yticks(range(len(piv)))
        ax.set_yticklabels([f"{v:g}" for v in piv.index])
        ax.set_xlabel("anchor fraction ρ")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=.046)
        for _, row in p.iterrows():
            yv = np.interp(row.w_over_a, piv.index.values, np.arange(len(piv)))
            ax.scatter([0, 1], [yv, yv], marker="D", s=18, color=COL.get(row.model, "k"), edgecolor="k", lw=.5, zorder=4)
    axes[0].set_ylabel("private-evidence weight w/a")
    fig.tight_layout()
    save(fig, "fig4_phase_diagram")


def main():
    df = load()
    fig_memory(df)
    fig_individual_vs_network(df)
    fig_anchors(df, ROOT / "results" / "abm_frozen" / "abm_predictions_frozen_pre_v4.csv")
    fig_phase(RES / "phase_diagram_sim.csv", RES / "parametric_rules.csv")
    print("figures written to", FIG)


if __name__ == "__main__":
    main()

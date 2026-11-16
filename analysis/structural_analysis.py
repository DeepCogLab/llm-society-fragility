"""Structural analyses of v3 (+v4 when present) agent-round data.

Inference unit: task replicate (WS and ER of the same task are averaged first).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "agent_rounds_v3.parquet"
OUT = ROOT / "results" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
SHORT = {"Qwen2.5-7B-Instruct": "Qwen2.5-7B", "Qwen3-14B": "Qwen3-14B", "Qwen3-30B-A3B-GPTQ-Int4": "Qwen3-30B-A3B",
         "Llama-3.1-8B-Instruct": "Llama-3.1-8B", "GLM-4-9B-0414": "GLM-4-9B", "Mistral-7B-Instruct-v0.3": "Mistral-7B"}
LLR = np.log(.7 / .3)


def boot(values, n=4000, seed=0):
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    b = rng.choice(v, (n, len(v))).mean(1)
    return v.mean(), np.quantile(b, .025), np.quantile(b, .975)


def load():
    df = pd.read_parquet(DATA)
    df["M"] = df.model.map(SHORT)
    df = df.sort_values(["model", "schedule", "block", "arm", "node", "round"])
    df["prev"] = df.groupby(["model", "schedule", "block", "arm", "node"]).wrong.shift(1)
    return df


def perturbation_memory(df):
    """d_t = treated - control nonseed error, per task; retention = mean d(6..8) / mean d(2..3)."""
    ns = df[(df.seed == 0) & df.arm.isin(["control", "treated", "no_relay_treated"])]
    e = ns.groupby(["M", "schedule", "replicate", "topology", "arm", "round"]).wrong.mean().unstack("arm")
    rows = []
    for arm in ["treated", "no_relay_treated"]:
        if arm not in e:
            continue
        d = (e[arm] - e["control"]).groupby(["M", "schedule", "replicate", "round"]).mean().unstack("round")
        for (m, s), g in d.groupby(level=[0, 1]):
            if g.isna().all().all():
                continue
            for t in range(9):
                est, lo, hi = boot(g[t].values, seed=t)
                rows.append({"model": m, "schedule": s, "arm": arm, "round": t, "delta": est, "lo": lo, "hi": hi, "n_tasks": g[t].notna().sum()})
            peak = g[[2, 3]].mean(1)
            tail = g[[6, 7, 8]].mean(1)
            est, lo, hi = boot(tail.values, seed=99)
            rows.append({"model": m, "schedule": s, "arm": arm, "round": "tail6-8", "delta": est, "lo": lo, "hi": hi, "n_tasks": len(tail)})
            # retention: ratio of means with bootstrap over tasks
            rng = np.random.default_rng(7)
            idx = rng.integers(0, len(g), (4000, len(g)))
            pk, tl = peak.values, tail.values
            ratios = tl[idx].mean(1) / np.where(pk[idx].mean(1) == 0, np.nan, pk[idx].mean(1))
            rows.append({"model": m, "schedule": s, "arm": arm, "round": "retention", "delta": tl.mean() / pk.mean() if pk.mean() else np.nan,
                         "lo": np.nanquantile(ratios, .025), "hi": np.nanquantile(ratios, .975), "n_tasks": len(g)})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "perturbation_memory.csv", index=False)
    return out


def bayes_choice(row):
    """Naive Bayes: peers' public choices as independent q-signals, private signal if visible.
    Returns P(choose the wrong option) from a rational agent ignoring dependence (ties -> keep previous)."""
    vis = row.anchor == 1
    llr_wrong = (2 * row.n_wrong_inbox - row.n_inbox) * LLR
    if vis:
        llr_wrong += (-1 if row.private_correct else 1) * LLR
    if llr_wrong > 0:
        return 1.0
    if llr_wrong < 0:
        return 0.0
    return float(row.prev) if not np.isnan(row.prev) else .5


def rules(df):
    """Effective decision rule per model x schedule from CONTROL arms (no injected messages)."""
    d = df[(df.arm == "control") & (df.seed == 0) & (df["round"] >= 1)].copy()
    d["bayes"] = [bayes_choice(r) for r in d.itertuples()]
    d["ctx"] = list(zip(d.anchor, np.where(d.anchor == 1, d.private_correct, -1), d.prev, d.n_inbox, d.n_wrong_inbox))
    rows = []
    for (m, s), g in d.groupby(["M", "schedule"]):
        agree = (g.wrong == (g.bayes > .5)).where(g.bayes != .5)
        # social weight: P(adopt majority | private visible and majority opposes private)
        vis = g[(g.anchor == 1) & (g.n_inbox == 4)]
        opp = vis[vis.n_wrong_inbox.where(vis.private_correct == 1, 4 - vis.n_wrong_inbox) >= 3]
        follow_majority = np.where(opp.private_correct == 1, opp.wrong, 1 - opp.wrong).mean() if len(opp) else np.nan
        inv = g[(g.n_inbox == 4) & (g.n_wrong_inbox == 2)]
        inertia = (inv.wrong == inv.prev).mean() if len(inv) else np.nan
        maj = g[(g.n_inbox == 4) & (g.n_wrong_inbox != 2)]
        maj_follow = (maj.wrong == (maj.n_wrong_inbox > 2)).mean()
        rows.append({"model": m, "schedule": s, "n": len(g), "bayes_agreement": agree.mean(),
                     "follow_peer_majority_against_visible_private(3-4 of 4)": follow_majority,
                     "follow_4peer_majority(k!=2)": maj_follow, "inertia_at_tie(k=2)": inertia,
                     "error_rate": g.wrong.mean()})
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "effective_rules.csv", index=False)
    # Between-model divergence of effective rules within schedule (weighted by pooled context frequency)
    div = []
    for s, g in d.groupby("schedule"):
        tab = g.groupby(["ctx", "M"]).wrong.mean().unstack("M")
        w = g.groupby("ctx").size()
        tab = tab.loc[w.index]
        models = [c for c in tab.columns]
        for i, a in enumerate(models):
            for b in models[i + 1:]:
                ok = tab[a].notna() & tab[b].notna()
                div.append({"schedule": s, "a": a, "b": b,
                            "weighted_mean_abs_diff": float(np.average((tab[a] - tab[b]).abs()[ok], weights=w[ok]))})
    pd.DataFrame(div).to_csv(OUT / "rule_divergence.csv", index=False)
    return r, pd.DataFrame(div)


def lockin(df):
    """Local self-stability of extra errors at the end; orphaned errors (no audit mention)."""
    ns = df[(df.seed == 0)]
    key = ["M", "schedule", "block", "node", "round"]
    c = ns[ns.arm == "control"].set_index(key)
    rows = []
    for arm in ["treated"]:
        t = ns[ns.arm == arm].set_index(key)
        j = t.join(c[["wrong"]], rsuffix="_ctrl", how="inner").reset_index()
        late = j[j["round"] == 8]
        extra = late[(late.wrong == 1) & (late.wrong_ctrl == 0)]
        for (m, s), g in extra.groupby(["M", "schedule"]):
            ever = ns[(ns.arm == arm) & (ns.M == m) & (ns.schedule == s)].groupby(["block", "node"]).n_injected_inbox.sum()
            direct = [ever.get((b, n), 0) > 0 for b, n in zip(g.block, g.node)]
            rows.append({"model": m, "schedule": s, "extra_wrong_agents_t8": len(g),
                         "share_inbox_majority_or_tie_wrong": float(((g.n_wrong_inbox * 2) >= g.n_inbox).mean()),
                         "share_never_directly_exposed": 1 - float(np.mean(direct)),
                         "share_message_mentions_audit": float(g.msg_audit.mean()),
                         "mean_confidence": float(g.confidence.mean())})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "lockin_orphaned_errors.csv", index=False)
    return out


def main():
    df = load()
    pd.set_option("display.width", 250)
    pm = perturbation_memory(df)
    print(pm[pm["round"].isin(["tail6-8", "retention"])].round(3).to_string())
    r, div = rules(df)
    print(r.round(3).to_string())
    print(div.groupby("schedule").weighted_mean_abs_diff.describe().round(3))
    print(lockin(df).round(3).to_string())


if __name__ == "__main__":
    main()

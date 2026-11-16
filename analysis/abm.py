"""Individual-rule ABM bridge: fit per-model decision rules from UNPERTURBED (control)
network logs, then simulate perturbed societies on the exact same graphs/exposures/
seeds/anchors. Used (1) to test whether the macro perturbation memory follows from
individual rules + network structure, and (2) to freeze predictions for the v4 anchor
sweep before those LLM runs exist.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiment_code"))
from society.core import Spec, graph_for, neighbors_for, selected_nodes, task_for  # noqa: E402
from society.structural_v4 import anchors_for  # noqa: E402

DATA = ROOT / "data" / "agent_rounds_v3.parquet"
OUT = ROOT / "results" / "abm_rerun"  # never overwrite results/abm_frozen


def add_prev(df):
    df = df.sort_values(["model", "schedule", "block", "arm", "node", "round"]).copy()
    df["prev"] = df.groupby(["model", "schedule", "block", "arm", "node"]).wrong.shift(1)
    return df


def fit_rules(df, model, fit_arms=("control",), schedules=("initial_only", "persistent")):
    """Nonparametric rule table with hierarchical shrinkage to coarser cells."""
    d = df[(df.model == model) & df.arm.isin(fit_arms) & df.schedule.isin(schedules)]
    t0 = d[d["round"] == 0].groupby("private_correct").wrong.mean().to_dict()
    d = d[d["round"] >= 1].copy()
    d["vis"] = (d.anchor == 1).astype(int)  # anchor flag in v3 == persistent schedule
    d["pc"] = np.where(d.vis == 1, d.private_correct, -1)
    d["frac"] = (d.n_wrong_inbox / d.n_inbox.clip(lower=1)).round(2)
    fine = d.groupby(["vis", "pc", "prev", "n_inbox", "n_wrong_inbox"]).wrong.agg(["sum", "size"])
    coarse = d.groupby(["vis", "pc", "prev", "frac"]).wrong.agg(["sum", "size"])
    coarser = d.groupby(["vis", "pc", "prev"]).wrong.agg(["sum", "size"])
    return {"t0": t0, "fine": fine["sum"].div(1).to_dict(), "fine_n": fine["size"].to_dict(),
            "coarse": coarse["sum"].to_dict(), "coarse_n": coarse["size"].to_dict(),
            "coarser": coarser["sum"].to_dict(), "coarser_n": coarser["size"].to_dict()}


def rule_p(rule, vis, pc, prev, n, k, a=2.0):
    key0 = (vis, pc if vis else -1, prev)
    p0 = (rule["coarser"].get(key0, 0) + .5) / (rule["coarser_n"].get(key0, 0) + 1)
    frac = round(k / max(n, 1), 2)
    key1 = key0 + (frac,)
    p1 = (rule["coarse"].get(key1, 0) + a * p0) / (rule["coarse_n"].get(key1, 0) + a)
    key2 = key0 + (n, k)
    return (rule["fine"].get(key2, 0) + a * p1) / (rule["fine_n"].get(key2, 0) + a)


def build_table(rule):
    """Pre-compute probabilities for all (vis, pc, prev, n, k)."""
    tab = np.zeros((2, 2, 2, 5, 5))
    for vis in (0, 1):
        for pc in (0, 1):
            for prev in (0, 1):
                for n in range(5):
                    for k in range(n + 1):
                        tab[vis, pc, prev, n, k] = rule_p(rule, vis, pc if vis else -1, float(prev), n, k)
    return tab


def simulate_block(spec, rule_tab, t0, fraction, mc, rng_seed, seed_fraction=None):
    graph, task = graph_for(spec), task_for(spec)
    from dataclasses import replace
    seeds = set(selected_nodes(replace(spec, fraction=seed_fraction) if seed_fraction else spec, graph))
    anchors = anchors_for(spec, fraction)
    truth = task["truth"]
    pc = np.array([int(x == truth) for x in task["private"]])
    vis = np.array([int(i in anchors) for i in range(spec.n)])
    nonseed = np.array([i not in seeds for i in range(spec.n)])
    nbrs = {(t, i): neighbors_for(spec, graph, i, t) for t in range(1, spec.rounds) for i in range(spec.n)}
    rng = np.random.default_rng(rng_seed)
    u = rng.random((spec.rounds, mc, spec.n))  # common random numbers across arms
    states, result = {}, {}
    for arm in ("control", "treated", "no_relay_treated"):
        s = np.zeros((spec.rounds, mc, spec.n), dtype=int)
        s[0] = (u[0] < np.where(pc == 1, t0[1], t0[0])[None, :]).astype(int)
        for t in range(1, spec.rounds):
            for i in range(spec.n):
                ns = nbrs[t, i]
                k = np.zeros(mc, dtype=int)
                for j in ns:
                    if arm == "no_relay_treated" and j not in seeds and t >= spec.onset:
                        w = states["control"][t - 1, :, j]
                    else:
                        w = s[t - 1, :, j]
                    if arm != "control" and j in seeds and spec.onset <= t < spec.onset + spec.duration:
                        w = np.ones(mc, dtype=int)
                    k = k + w
                p = rule_tab[vis[i], pc[i], s[t - 1, :, i], len(ns), k]
                s[t, :, i] = (u[t, :, i] < p).astype(int)
        states[arm] = s
        result[arm] = s[:, :, nonseed].mean(axis=(1, 2))
    return result


def specs():
    for topology in ("WS", "ER"):
        for rep in range(70000, 70024):
            yield Spec(phase="network_response_v3", topology=topology, replicate=rep, n=24, rounds=9,
                       reliability=.7, onset=2, duration=2, fraction=.125, wording=rep % 3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mc", type=int, default=100)
    p.add_argument("--fractions", default="0,0.25,0.5,0.75,1")
    p.add_argument("--tag", default="frozen_pre_v4")
    p.add_argument("--seed-fractions", default="0.125")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    df = add_prev(pd.read_parquet(DATA))
    df = df[df.source == "v3"]
    rows, rules_out = [], {}
    fractions = [float(x) for x in args.fractions.split(",")]
    for model in sorted(df.model.unique()):
        rule = fit_rules(df, model)
        tab = build_table(rule)
        rules_out[model] = {"t0": rule["t0"], "table": tab.tolist()}
        for spec in specs():
            for f in fractions:
              for sf in [float(x) for x in args.seed_fractions.split(",")]:
                res = simulate_block(spec, tab, rule["t0"], f, args.mc, abs(hash((model, spec.block_id, f, sf))) % 2**32, sf)
                for arm, curve in res.items():
                    for t, v in enumerate(curve):
                        rows.append({"model": model, "anchor_fraction": f, "seed_fraction": sf, "topology": spec.topology, "replicate": spec.replicate,
                                     "block": spec.block_id, "arm": arm, "round": t, "pred_nonseed_error": v})
        print(json.dumps({"model": model, "done": True}), flush=True)
    pd.DataFrame(rows).to_csv(OUT / f"abm_predictions_{args.tag}.csv", index=False)
    (OUT / f"abm_rules_{args.tag}.json").write_text(json.dumps({"created": time.time(), "created_local": time.ctime(),
        "fit_data": "v3 control arms only (initial_only + persistent), rounds>=1; treated/no_relay never used",
        "rules": rules_out}, indent=1))


if __name__ == "__main__":
    main()

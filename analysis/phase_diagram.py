"""Parametric bridge model and phase diagram.

Rule family (fitted per model on v3 CONTROL arms, nonseed, rounds>=1):
  logit P(wrong) = b + a*(2k-n) + h*(2*prev-1) + w*vis*(1-2*pc)
Then sweep anchor fraction rho and private weight w (relative to a) on random graphs
(WS/ER, mean degree 4, exposure 4) of size n in {24, 96}, seeds = 1/8 for 2 rounds.
"""
from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "tables"
SHORT = {"Qwen2.5-7B-Instruct": "Qwen2.5-7B", "Qwen3-14B": "Qwen3-14B", "Qwen3-30B-A3B-GPTQ-Int4": "Qwen3-30B-A3B",
         "Llama-3.1-8B-Instruct": "Llama-3.1-8B", "GLM-4-9B-0414": "GLM-4-9B"}


def fit_params(df):
    df = df.sort_values(["model", "schedule", "block", "arm", "node", "round"]).copy()
    df["prev"] = df.groupby(["model", "schedule", "block", "arm", "node"]).wrong.shift(1)
    d = df[(df.arm == "control") & (df.seed == 0) & (df["round"] >= 1) & df.schedule.isin(["initial_only", "persistent"])]
    rows = []
    for m, g in d.groupby("model"):
        X = pd.DataFrame({"const": 1.0, "social": 2 * g.n_wrong_inbox - g.n_inbox, "inertia": 2 * g.prev - 1,
                          "private": g.anchor * (1 - 2 * g.private_correct)})
        fit = sm.Logit(g.wrong.values, X.values).fit_regularized(alpha=1e-3, disp=0)
        b, a, h, w = fit.params
        t0 = df[(df.model == m) & (df["round"] == 0)].groupby("private_correct").wrong.mean()
        rows.append({"model": SHORT.get(m, m), "b": b, "a": a, "h": h, "w": w, "w_over_a": w / a,
                     "t0_wrong_given_private_correct": t0.get(1), "t0_wrong_given_private_wrong": t0.get(0)})
    return pd.DataFrame(rows)


def simulate(params, n, rho, topology, rep, mc=64, seed_frac=1 / 8, rounds=9, onset=2, dur=2, q=.7):
    rng = np.random.default_rng(hash((n, rho, topology, rep, round(params["w"], 3))) % 2**32)
    g = nx.watts_strogatz_graph(n, 4, .18, seed=rep) if topology == "WS" else nx.gnm_random_graph(n, 2 * n, seed=rep)
    nbr = [list(g.neighbors(i)) for i in range(n)]
    pc = (rng.random(n) < q).astype(int)
    order = rng.permutation(n)
    seeds = order[: max(1, round(seed_frac * n))]
    anchors = np.zeros(n, int)
    anchors[rng.permutation(n)[: round(rho * n)]] = 1
    nonseed = np.ones(n, bool)
    nonseed[seeds] = False
    exposures = [[rng.permutation(nbr[i])[:4] for i in range(n)] for _ in range(rounds)]
    u = rng.random((rounds, mc, n))
    out = {}
    for arm in ("control", "treated"):
        s = np.zeros((rounds, mc, n), int)
        s[0] = (u[0] < np.where(pc == 1, params["t0c"], params["t0w"])).astype(int)
        for t in range(1, rounds):
            k = np.zeros((mc, n))
            nn = np.zeros(n)
            prev = s[t - 1].copy()
            if arm == "treated" and onset <= t < onset + dur:
                prev[:, seeds] = 1
            for i in range(n):
                ex = exposures[t][i]
                nn[i] = len(ex)
                if len(ex):
                    k[:, i] = prev[:, ex].sum(1)
            z = params["b"] + params["a"] * (2 * k - nn) + params["h"] * (2 * s[t - 1] - 1) + params["w"] * anchors * (1 - 2 * pc)
            s[t] = (u[t] < 1 / (1 + np.exp(-z))).astype(int)
        out[arm] = s[:, :, nonseed].mean(axis=(1, 2))
    return out


def main():
    df = pd.read_parquet(ROOT / "data" / "agent_rounds_v3.parquet")
    params = fit_params(df)
    params.to_csv(OUT / "parametric_rules.csv", index=False)
    print(params.round(3).to_string())
    base = params[params.model == "Qwen2.5-7B"].iloc[0]
    rows = []
    for n in (24, 96):
        for rho in np.linspace(0, 1, 11):
            for wa in (0, .5, 1, 2, 4, 8):
                p = {"b": base.b, "a": base.a, "h": base.h, "w": wa * base.a,
                     "t0c": base.t0_wrong_given_private_correct, "t0w": base.t0_wrong_given_private_wrong}
                for topology in ("WS", "ER"):
                    for rep in range(12 if n == 24 else 6):
                        r = simulate(p, n, rho, topology, rep)
                        rows.append({"n": n, "rho": round(rho, 2), "w_over_a": wa, "topology": topology, "rep": rep,
                                     "control_tail": r["control"][-3:].mean(), "treated_tail": r["treated"][-3:].mean(),
                                     "delta_tail": r["treated"][-3:].mean() - r["control"][-3:].mean(),
                                     "delta_peak": r["treated"][2:4].mean() - r["control"][2:4].mean()})
        print("n done", n, flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "phase_diagram_sim.csv", index=False)
    piv = res.groupby(["n", "w_over_a", "rho"]).delta_tail.mean().unstack("rho")
    pd.set_option("display.width", 250)
    print((piv * 100).round(1))
    print((res.groupby(["n", "w_over_a", "rho"]).control_tail.mean().unstack("rho") * 100).round(1))


if __name__ == "__main__":
    main()

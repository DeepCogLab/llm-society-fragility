"""Cross-family ABM test (2026-11-16).

Variant `control_only`: fit the individual-rule ABM on CONTROL logs only (initial_only + persistent controls;
v3 for Qwen, v4 for GLM/Llama), simulate perturbed arms on the identical graphs/seeds. Fully prospective
with respect to treated outcomes; same code/hyperparameters as the frozen Qwen predictions (analysis/abm.py).

Variant `plus_premium`: additionally uses one individual-level number per model, the *persuasion premium*
of the injected message = P(flip | exposed to >=1 injected msg) - P(flip | same inbox size and same number of
wrong msgs, none injected), measured at the exposure step (rounds 2-4, previously-correct non-seeds, treated
initial_only arm). The tail (rounds 6-8) is then produced by the control-fitted network rule alone.
This decomposes the effect into first-contact persuasion (individual) and propagation (structural).
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
import abm  # noqa: E402
from society.core import graph_for, neighbors_for, selected_nodes, task_for  # noqa: E402
from society.structural_v4 import anchors_for  # noqa: E402
from dataclasses import replace  # noqa: E402

D = abm.add_prev(pd.read_parquet(ROOT / "data/agent_rounds_all.parquet"))
OUT = ROOT / "results/tables"
TAIL = [6, 7, 8]
COND = {"initial_only": (0.0, .125), "persistent": (1.0, .125), "init_seed6": (0.0, .25)}
MC = 200


def premium(m):
    d = D[(D.model == m) & (D.schedule == "initial_only") & (D.arm == "treated") & (D.seed == 0)
          & D["round"].isin([2, 3, 4]) & (D.prev == 0) & (D.n_inbox > 0)]
    a = d[d.n_injected_inbox >= 1].groupby(["n_inbox", "n_wrong_inbox"]).wrong.agg(["mean", "size"])
    b = d[d.n_injected_inbox == 0].groupby(["n_inbox", "n_wrong_inbox"]).wrong.agg(["mean", "size"])
    j = a.join(b, lsuffix="_i", rsuffix="_o", how="inner")
    j = j[j.size_o >= 5]
    w = j.size_i
    return float(((j.mean_i - j.mean_o) * w).sum() / w.sum()), int(w.sum())


def simulate(spec, tab, t0, fraction, rng_seed, seed_fraction, prem):
    graph, task = graph_for(spec), task_for(spec)
    seeds = set(selected_nodes(replace(spec, fraction=seed_fraction), graph))
    anchors = anchors_for(spec, fraction)
    pc = np.array([int(x == task["truth"]) for x in task["private"]])
    vis = np.array([int(i in anchors) for i in range(spec.n)])
    nonseed = np.array([i not in seeds for i in range(spec.n)])
    nbrs = {(t, i): neighbors_for(spec, graph, i, t) for t in range(1, spec.rounds) for i in range(spec.n)}
    u = np.random.default_rng(rng_seed).random((spec.rounds, MC, spec.n))
    states, out = {}, {}
    for arm in ("control", "treated", "no_relay_treated"):
        s = np.zeros((spec.rounds, MC, spec.n), dtype=int)
        s[0] = (u[0] < np.where(pc == 1, t0[1], t0[0])[None, :]).astype(int)
        for t in range(1, spec.rounds):
            window = arm != "control" and spec.onset <= t < spec.onset + spec.duration
            for i in range(spec.n):
                ns = nbrs[t, i]
                k = np.zeros(MC, dtype=int)
                inj = 0
                for j in ns:
                    w = states["control"][t - 1, :, j] if (arm == "no_relay_treated" and j not in seeds and t >= spec.onset) else s[t - 1, :, j]
                    if window and j in seeds:
                        w = np.ones(MC, dtype=int); inj += 1
                    k = k + w
                prev = s[t - 1, :, i]
                p = tab[vis[i], pc[i], prev, len(ns), k]
                if inj and prem:
                    p = np.where(prev == 0, np.clip(p + prem, 0, 1), p)
                s[t, :, i] = (u[t, :, i] < p).astype(int)
        states[arm] = s
        out[arm] = s[:, :, nonseed].mean(axis=(1, 2))
    return out


def main(models):
    rows, prem_rows = [], []
    for m in models:
        rule = abm.fit_rules(D[D.source.isin(["v3", "v4"])], m)
        tab = abm.build_table(rule)
        pr, n = premium(m)
        prem_rows.append(dict(model=m, premium=pr, n_exposures=n))
        for variant, pv in (("control_only", 0.0), ("plus_premium", pr)):
            for spec in abm.specs():
                for sched, (f, sf) in COND.items():
                    seed = int.from_bytes(f"{spec.block_id}{sched}".encode()[:8], "little") % 2**32
                    res = simulate(spec, tab, rule["t0"], f, seed, sf, pv)
                    for arm, curve in res.items():
                        rows.append(dict(model=m, variant=variant, schedule=sched, block=spec.block_id,
                                         arm=arm, terr=float(np.mean(curve[TAIL]))))
        print(m, "done", flush=True)
    P = pd.DataFrame(rows)
    key = ["model", "variant", "schedule", "block"]
    c = P[P.arm == "control"].set_index(key)["terr"]
    P = P[P.arm != "control"].copy()
    P["pred_delta"] = (P["terr"].values - c.loc[list(P[key].itertuples(index=False, name=None))].values) * 100
    S = P.groupby(["model", "variant", "schedule", "arm"]).pred_delta.mean().reset_index()
    obs = pd.read_csv(OUT / "v4_v5_tail_deltas.csv")
    obs = obs[obs.source.isin(["v3", "v4"])][["model", "schedule", "arm", "n_tasks", "delta", "lo", "hi"]]
    S = S.merge(obs, on=["model", "schedule", "arm"], how="inner")
    S["inside_ci"] = (S.pred_delta >= S.lo) & (S.pred_delta <= S.hi)
    S.to_csv(OUT / "cross_family_abm.csv", index=False)
    pd.DataFrame(prem_rows).to_csv(OUT / "persuasion_premium.csv", index=False)
    print(pd.DataFrame(prem_rows).round(3).to_string())
    print(S.round(2).to_string())


if __name__ == "__main__":
    main(sys.argv[1:] or ["GLM-4-9B-0414", "Qwen2.5-7B-Instruct", "Qwen3-14B", "Qwen3-30B-A3B-GPTQ-Int4"])

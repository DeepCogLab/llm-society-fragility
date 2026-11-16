"""Corrected re-run of analysis/cross_family_abm.py (does NOT overwrite results/tables).

Why: the abstract task (v3/v4) and the natural-language task (v5) reuse the same network block ids.
analysis/cross_family_abm.py computes `prev` (an agent's previous decision) on the pooled table, grouping by
(model, schedule, block, arm, node) without `source`, so `prev` is interleaved across the two tasks, and
`premium()` pools abstract and natural-language exposures. Here both steps are restricted to the abstract
task (source in v3, v4). Simulation code and hyper-parameters are imported unchanged.

Output: paper/scripts/out/cross_family_abm_fixed.csv, persuasion_premium_fixed.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis"))
import abm  # noqa: E402
import cross_family_abm as cf  # noqa: E402  (module import also builds the original, pooled table; we replace it)

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)
RAW = pd.read_parquet(ROOT / "data" / "agent_rounds_all.parquet")
D = abm.add_prev(RAW[RAW.source.isin(["v3", "v4"])])
cf.D = D  # premium() and friends now see abstract-task rows only
TAIL = [6, 7, 8]


def main(models):
    rows, prem_rows = [], []
    for m in models:
        rule = abm.fit_rules(D, m)
        tab = abm.build_table(rule)
        pr, n = cf.premium(m)
        prem_rows.append(dict(model=m, premium=pr, n_exposures=n))
        for variant, pv in (("control_only", 0.0), ("plus_premium", pr)):
            for spec in abm.specs():
                for sched, (f, sf) in cf.COND.items():
                    seed = int.from_bytes(f"{spec.block_id}{sched}".encode()[:8], "little") % 2**32
                    res = cf.simulate(spec, tab, rule["t0"], f, seed, sf, pv)
                    for arm, curve in res.items():
                        rows.append(dict(model=m, variant=variant, schedule=sched, block=spec.block_id, arm=arm,
                                         terr=float(np.mean(curve[TAIL])), **{f"r{t}": float(curve[t]) for t in range(9)}))
        print(m, "done", flush=True)
    P = pd.DataFrame(rows)
    key = ["model", "variant", "schedule", "block"]
    c = P[P.arm == "control"].set_index(key)
    P = P[P.arm != "control"].copy()
    idx = list(P[key].itertuples(index=False, name=None))
    P["pred_delta"] = (P["terr"].values - c.loc[idx, "terr"].values) * 100
    for t in range(9):
        P[f"d{t}"] = (P[f"r{t}"].values - c.loc[idx, f"r{t}"].values) * 100
    S = P.groupby(["model", "variant", "schedule", "arm"])[["pred_delta"] + [f"d{t}" for t in range(9)]].mean().reset_index()
    obs = pd.read_csv(ROOT / "results" / "tables" / "v4_v5_tail_deltas.csv")
    obs = obs[obs.source.isin(["v3", "v4"])][["model", "schedule", "arm", "n_tasks", "delta", "lo", "hi"]]
    S = S.merge(obs, on=["model", "schedule", "arm"], how="inner")
    S["inside_ci"] = (S.pred_delta >= S.lo) & (S.pred_delta <= S.hi)
    S.to_csv(OUT / "cross_family_abm_fixed.csv", index=False)
    pd.DataFrame(prem_rows).to_csv(OUT / "persuasion_premium_fixed.csv", index=False)
    pd.set_option("display.width", 220)
    print(pd.DataFrame(prem_rows).round(3).to_string())
    print(S[["model", "variant", "schedule", "arm", "pred_delta", "delta", "lo", "hi", "inside_ci"]].round(2).to_string())


if __name__ == "__main__":
    main(sys.argv[1:] or ["Qwen2.5-7B-Instruct", "Qwen3-14B", "Qwen3-30B-A3B-GPTQ-Int4", "GLM-4-9B-0414", "Llama-3.1-8B-Instruct"])

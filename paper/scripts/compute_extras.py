"""Extra numbers for plan A that numbers.json does not contain: tail effects split by topology.
Same conventions as compute_numbers.py (non-seed agents, treated minus control, rounds 6-8, bootstrap over tasks).
Writes paper/scripts/out/extras.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
D = pd.read_parquet(ROOT / "data" / "agent_rounds_all.parquet")
D = D[D.source != "v5"]  # abstract task only (block ids are reused by the natural-language task)
SHORT = {"Qwen2.5-7B-Instruct": "Qwen2.5-7B", "Qwen3-14B": "Qwen3-14B", "Qwen3-30B-A3B-GPTQ-Int4": "Qwen3-30B",
         "GLM-4-9B-0414": "GLM-4-9B", "Llama-3.1-8B-Instruct": "Llama-3.1-8B"}


def boot(x, n=4000, seed=0):
    x = np.asarray(x, float)
    b = np.random.default_rng(seed).choice(x, (n, len(x))).mean(1)
    return [float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


out = {}
for m, short in SHORT.items():
    for sched, arm in [("initial_only", "treated"), ("persistent", "treated"), ("initial_only", "no_relay_treated")]:
        g = D[(D.model == m) & (D.schedule == sched)]
        t, c = g[g.arm == arm], g[g.arm == "control"]
        seeds = t[t.seed == 1][["block", "node"]].drop_duplicates().assign(is_seed=1)
        lv = {}
        for name, df in (("t", t), ("c", c)):
            x = df.merge(seeds, on=["block", "node"], how="left")
            x = x[x.is_seed.isna() & x["round"].isin([6, 7, 8])]
            lv[name] = x.groupby(["topology", "replicate"]).wrong.mean()
        j = pd.concat(lv, axis=1).dropna()
        d = (j.t - j.c) * 100
        for topo in ("WS", "ER"):
            out[f"{short}|{sched}|{arm}|{topo}"] = boot(d.xs(topo, level="topology").values)

p = Path(__file__).resolve().parent / "out" / "extras.json"
p.write_text(json.dumps(out, indent=1), encoding="utf-8")
for k, v in out.items():
    print(f"{k:55s} {v[0]:6.1f} [{v[1]:5.1f},{v[2]:5.1f}]")

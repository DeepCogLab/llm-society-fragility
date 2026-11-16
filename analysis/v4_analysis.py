"""v4/v5 analysis: tail deltas (non-seed agents, rounds 6-8), paired by block, task-clustered bootstrap;
comparison with frozen ABM predictions."""
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
D = pd.read_parquet(ROOT / "data/agent_rounds_all.parquet")
OUT = ROOT / "results/tables"; OUT.mkdir(exist_ok=True, parents=True)
TAIL = [6, 7, 8]
PAIR = {"init_seed1": ("v3", "initial_only"), "init_seed6": ("v3", "initial_only"),
        "persist_seed6": ("v3", "persistent")}
ANCH = {"initial_only": 0.0, "anchor25": .25, "anchor50": .5, "persistent": 1.0, "in_history": 0.0}

def boot(x, n=4000, seed=0):
    x = np.asarray(x, float); r = np.random.default_rng(seed)
    b = r.choice(x, (n, len(x))).mean(1)
    return x.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)

def block_tail(df_t, df_c, seed_mask_from_treated=True):
    """non-seed (per treated arm seeds) tail error, treated minus control, by block"""
    seeds = df_t[df_t.seed == 1][["block", "node"]].drop_duplicates()
    seeds["is_seed"] = 1
    out = {}
    for name, df in (("t", df_t), ("c", df_c)):
        x = df[df["round"].isin(TAIL)].merge(seeds, on=["block", "node"], how="left")
        x = x[x.is_seed.isna()]
        out[name] = x.groupby(["block", "replicate"]).wrong.mean()
    j = pd.concat(out, axis=1).dropna()
    # task-level (replicate) clustering: average WS/ER blocks within replicate
    j = j.groupby(level="replicate").mean()
    return j

rows = []
for (src, m), g in D[D.source.isin(["v3", "v4", "v5"])].groupby(["source", "model"]):
    for s in g.schedule.unique():
        gs = g[g.schedule == s]
        for arm in ["treated", "no_relay_treated", "treated_authority", "treated_neutral"]:
            t = gs[gs.arm == arm]
            if t.empty: continue
            if s in PAIR:
                psrc, ps = PAIR[s]
                # Qwen controls live in v3; Llama/GLM core controls were run under v4
                c = D[(D.source.isin([psrc, "v4"])) & (D.model == m) & (D.schedule == ps) & (D.arm == "control")]
            else:
                c = gs[gs.arm == "control"]
            if c.empty: continue
            j = block_tail(t, c)
            if len(j) < 8: continue
            d = (j.t - j.c) * 100
            est, lo, hi = boot(d.values, seed=len(rows))
            rows.append(dict(source=src, model=m, schedule=s, arm=arm, n_tasks=len(j),
                             ctrl=j.c.mean() * 100, treat=j.t.mean() * 100, delta=est, lo=lo, hi=hi))
R = pd.DataFrame(rows)

# frozen predictions
P = pd.read_csv(ROOT / "results/abm_frozen/abm_predictions_frozen_pre_v4.csv")
Pd = pd.read_csv(ROOT / "results/abm_frozen/abm_predictions_dose_frozen_pre_v4.csv")
def pred(pf, m, **kw):
    q = pf[(pf.model == m) & pf["round"].isin(TAIL)]
    for k, v in kw.items(): q = q[np.isclose(q[k], v)]
    if q.empty: return np.nan
    p = q.pivot_table(index="block", columns="arm", values="pred_nonseed_error")
    return (p.treated - p.control).mean() * 100 if {"treated", "control"} <= set(p.columns) else np.nan
pr = []
for _, r in R.iterrows():
    if r.arm != "treated": pr.append(np.nan); continue
    if r.schedule in ANCH and r.schedule != "in_history":
        pr.append(pred(P, r.model, anchor_fraction=ANCH[r.schedule]))
    elif r.schedule in PAIR:
        a = 1.0 if r.schedule.startswith("persist") else 0.0
        sf = 0.25 if r.schedule.endswith("6") else 0.0417
        pr.append(pred(Pd, r.model, anchor_fraction=a, seed_fraction=sf))
    else: pr.append(np.nan)
R["frozen_pred"] = pr
R.to_csv(OUT / "v4_v5_tail_deltas.csv", index=False)
pd.set_option("display.width", 200)
print(R.round(2).to_string())

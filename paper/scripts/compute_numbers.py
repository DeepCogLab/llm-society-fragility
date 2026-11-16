"""Recompute every number quoted in the three paper drafts from data/agent_rounds_all.parquet.

Writes paper/scripts/numbers.json (machine readable) and prints a readable summary.
Conventions follow analysis/v4_analysis.py: outcome = error rate of non-seed agents,
treated minus control, paired by network block, WS/ER averaged within task, bootstrap over the 24 tasks.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
D = pd.read_parquet(ROOT / "data" / "agent_rounds_all.parquet")
OUT = Path(__file__).resolve().parent / "numbers.json"
SHORT = {"Qwen2.5-7B-Instruct": "Qwen2.5-7B", "Qwen3-14B": "Qwen3-14B", "Qwen3-30B-A3B-GPTQ-Int4": "Qwen3-30B",
         "GLM-4-9B-0414": "GLM-4-9B", "Llama-3.1-8B-Instruct": "Llama-3.1-8B"}
MODELS = list(SHORT)
TAIL = [6, 7, 8]
PAIR = {"init_seed1": "initial_only", "init_seed6": "initial_only", "persist_seed6": "persistent"}
D["grp"] = np.where(D.source == "v5", "nl", "abs")  # abstract task (v3+v4) vs natural-language task (v5)
D = D.sort_values(["grp", "model", "schedule", "block", "arm", "node", "round"])
D["prev"] = D.groupby(["grp", "model", "schedule", "block", "arm", "node"]).wrong.shift(1)


def boot(x, n=4000, seed=0):
    x = np.asarray(x, float)
    b = np.random.default_rng(seed).choice(x, (n, len(x))).mean(1)
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def arms(grp, model, schedule, arm):
    """Return (treated df, control df) following the pairing logic of v4_analysis.py."""
    g = D[(D.grp == grp) & (D.model == model)]
    t = g[(g.schedule == schedule) & (g.arm == arm)]
    cs = PAIR.get(schedule, schedule)
    c = g[(g.schedule == cs) & (g.arm == "control")]
    return t, c


def per_round(grp, model, schedule, arm):
    """24 x 9 matrix of treated-minus-control non-seed error (per task, per round), plus levels."""
    t, c = arms(grp, model, schedule, arm)
    if t.empty or c.empty:
        return None
    seeds = t[t.seed == 1][["block", "node"]].drop_duplicates().assign(is_seed=1)
    lv = {}
    for name, df in (("t", t), ("c", c)):
        x = df.merge(seeds, on=["block", "node"], how="left")
        x = x[x.is_seed.isna()]
        lv[name] = x.groupby(["block", "replicate", "round"]).wrong.mean()
    j = pd.concat(lv, axis=1).dropna()
    j = j.groupby(level=["replicate", "round"]).mean()
    return (j.t - j.c).unstack("round"), j.t.unstack("round"), j.c.unstack("round")


def summarize(grp, model, schedule, arm):
    r = per_round(grp, model, schedule, arm)
    if r is None:
        return None
    d, lt, lc = r
    out = {"n_tasks": int(len(d)), "round": {}}
    for t in range(9):
        m, lo, hi = boot(d[t].values * 100, seed=t)
        out["round"][t] = [m, lo, hi]
    tail = d[TAIL].mean(1) * 100
    out["tail"] = list(boot(tail.values, seed=99))
    out["peak3"] = list(boot(d[3].values * 100, seed=98))
    out["ctrl_tail"] = float(lc[TAIL].mean(1).mean() * 100)
    out["treat_tail"] = float(lt[TAIL].mean(1).mean() * 100)
    out["treat_round"] = [float(lt[t].mean() * 100) for t in range(9)]
    out["ctrl_round"] = [float(lc[t].mean() * 100) for t in range(9)]
    pk = d[3].values
    out["retention"] = float(d[TAIL].mean(1).mean() / pk.mean()) if pk.mean() > 0.005 else None
    return out


def collapse(grp, model, schedule, arm):
    """Share of networks whose error rate at t=8 exceeds 50% (all agents / non-seeds), treated vs control."""
    t, c = arms(grp, model, schedule, arm)
    if t.empty or c.empty:
        return None
    seeds = t[t.seed == 1][["block", "node"]].drop_duplicates().assign(is_seed=1)
    res = {}
    for name, df in (("treated", t), ("control", c)):
        x = df[df["round"] == 8].merge(seeds, on=["block", "node"], how="left")
        res[name + "_all"] = float((x.groupby("block").wrong.mean() > .5).mean())
        res[name + "_nonseed"] = float((x[x.is_seed.isna()].groupby("block").wrong.mean() > .5).mean())
        res["n_networks"] = int(x.block.nunique())
    return res


N = {"meta": {"rows": int(len(D)), "rows_by_source": D.groupby("source").size().to_dict(),
              "rows_by_model": {SHORT[k]: int(v) for k, v in D.groupby("model").size().items()},
              "blocks_shared_abs_nl": bool(set(D[D.grp == "abs"].block) & set(D[D.grp == "nl"].block))}}

# ---- 1. main effects ----
conds = [("abs", s, a) for s in ["initial_only", "persistent"] for a in ["treated", "no_relay_treated"]] + \
        [("abs", s, "treated") for s in ["anchor25", "anchor50", "in_history", "init_fullmem", "init_seed1", "init_seed6", "persist_seed6"]] + \
        [("nl", "initial_only", a) for a in ["treated", "treated_authority", "treated_neutral", "no_relay_treated"]] + \
        [("nl", "persistent", a) for a in ["treated", "no_relay_treated"]]
N["effects"], N["collapse"] = {}, {}
for m in MODELS:
    for grp, s, a in conds:
        r = summarize(grp, m, s, a)
        if r is None:
            continue
        key = f"{SHORT[m]}|{grp}|{s}|{a}"
        N["effects"][key] = r
        N["collapse"][key] = collapse(grp, m, s, a)

# ---- 2. lock-in anatomy at t=8 (abstract task, treated, rho=0 and rho=1) ----
N["lockin"] = {}
ns = D[(D.grp == "abs") & (D.seed == 0)]
for m in MODELS:
    for s in ["initial_only", "persistent"]:
        key = ["block", "node", "round"]
        t = ns[(ns.model == m) & (ns.schedule == s) & (ns.arm == "treated")]
        c = ns[(ns.model == m) & (ns.schedule == s) & (ns.arm == "control")]
        if t.empty or c.empty:
            continue
        j = t.set_index(key).join(c.set_index(key)[["wrong"]], rsuffix="_ctrl", how="inner").reset_index()
        late = j[j["round"] == 8]
        extra = late[(late.wrong == 1) & (late.wrong_ctrl == 0)]
        ever = t.groupby(["block", "node"]).n_injected_inbox.sum()
        direct = np.array([ever.get((b, n), 0) > 0 for b, n in zip(extra.block, extra.node)])
        N["lockin"][f"{SHORT[m]}|{s}"] = {
            "extra_wrong_t8": int(len(extra)),
            "share_local_majority_or_tie_wrong": float(((extra.n_wrong_inbox * 2) >= extra.n_inbox).mean()) if len(extra) else None,
            "share_never_directly_exposed": float(1 - direct.mean()) if len(extra) else None,
            "share_mentions_audit": float(extra.msg_audit.mean()) if len(extra) else None}

# ---- 3. effective rules in unperturbed networks ----
LLR = np.log(.7 / .3)
N["rules"] = {}
ctl = D[(D.grp == "abs") & (D.arm == "control") & (D.seed == 0) & (D["round"] >= 1) & D.schedule.isin(["initial_only", "persistent"])].copy()
for (m, s), g in ctl.groupby(["model", "schedule"]):
    g4 = g[g.n_inbox == 4]
    opp = np.where(g4.prev == 1, 4 - g4.n_wrong_inbox, g4.n_wrong_inbox)
    sw = pd.Series((g4.wrong != g4.prev).astype(float).values).groupby(opp).agg(["mean", "size"])
    vis = (g.anchor == 1).values
    llr = (2 * g.n_wrong_inbox - g.n_inbox).values * LLR + np.where(vis, np.where(g.private_correct == 1, -1, 1) * LLR, 0)
    bay = np.where(llr > 0, 1.0, np.where(llr < 0, 0.0, np.nan))
    ok = ~np.isnan(bay)
    maj = g4[g4.n_wrong_inbox != 2]
    tie = g4[g4.n_wrong_inbox == 2]
    v4 = g4[g4.anchor == 1]
    oppv = v4[np.where(v4.private_correct == 1, v4.n_wrong_inbox, 4 - v4.n_wrong_inbox) >= 3]
    N["rules"][f"{SHORT[m]}|{s}"] = {
        "n": int(len(g)),
        "switch_by_opposing": {int(k): [float(v["mean"]), int(v["size"])] for k, v in sw.iterrows()},
        "bayes_agreement": float((g.wrong.values[ok] == bay[ok]).mean()),
        "follow_majority_of_4": float((maj.wrong == (maj.n_wrong_inbox > 2)).mean()),
        "keep_previous_at_tie": float((tie.wrong == tie.prev).mean()),
        "follow_majority_against_visible_evidence": float(np.where(oppv.private_correct == 1, oppv.wrong, 1 - oppv.wrong).mean()) if len(oppv) else None,
        "control_error": float(g.wrong.mean())}

# ---- 4. isolated-agent profiles (Qwen only) ----
N["isolation"] = {}
for f in sorted((ROOT / "data" / "profiles_v3").glob("*.csv")):
    p = pd.read_csv(f)
    p = p[p.stage.isin(["validation", "profile"])]
    s = p[(p.support >= 0) & (p.kind == "report") & (p.interface == "natural") & (p.history == "aligned") & (p.q == .7)]
    N["isolation"][SHORT.get(f.stem, f.stem)] = {int(k): [float(v["mean"]), int(v["size"])] for k, v in s.groupby("support").changed_private.agg(["mean", "size"]).iterrows()}

# ---- 5. first-contact persuasion premium (matched on inbox size and number of wrong messages) ----
N["premium"] = {}
for grp, sched in [("abs", "initial_only"), ("abs", "persistent"), ("nl", "initial_only"), ("nl", "persistent")]:
    for m in MODELS:
        for arm in ["treated", "treated_neutral"]:
            d = D[(D.grp == grp) & (D.model == m) & (D.schedule == sched) & (D.arm == arm) & (D.seed == 0)
                  & D["round"].isin([2, 3, 4]) & (D.prev == 0) & (D.n_inbox > 0)]
            if d.empty:
                continue
            a = d[d.n_injected_inbox >= 1].groupby(["n_inbox", "n_wrong_inbox"]).wrong.agg(["mean", "size"])
            b = d[d.n_injected_inbox == 0].groupby(["n_inbox", "n_wrong_inbox"]).wrong.agg(["mean", "size"])
            j = a.join(b, lsuffix="_i", rsuffix="_o", how="inner")
            j = j[j.size_o >= 5]
            if j.empty:
                continue
            w = j.size_i
            N["premium"][f"{SHORT[m]}|{grp}|{sched}|{arm}"] = {
                "flip_injected": float((j.mean_i * w).sum() / w.sum()), "flip_ordinary": float((j.mean_o * w).sum() / w.sum()),
                "premium": float(((j.mean_i - j.mean_o) * w).sum() / w.sum()), "n_exposures": int(w.sum())}

# ---- 6. does the authority claim travel? (natural-language task, wrong non-seeds at t=8) ----
N["claim_recited"] = {}
x = D[(D.grp == "nl") & (D.seed == 0) & (D["round"] == 8) & (D.wrong == 1)]
for (m, s, a), g in x.groupby(["model", "schedule", "arm"]):
    N["claim_recited"][f"{SHORT[m]}|{s}|{a}"] = [float(g.msg_claim.mean()), int(len(g))]

# ---- 7. bridge: frozen predictions vs observed ----
T = pd.read_csv(ROOT / "results" / "tables" / "v4_v5_tail_deltas.csv")
F = T[T.source.isin(["v3", "v4"]) & T.frozen_pred.notna()].copy()
F["inside"] = (F.frozen_pred >= F.lo) & (F.frozen_pred <= F.hi)
pro = F[F.source == "v4"]
N["bridge_frozen"] = {"n": int(len(F)), "r": float(np.corrcoef(F.frozen_pred, F.delta)[0, 1]),
                      "mae": float((F.frozen_pred - F.delta).abs().mean()), "inside": int(F.inside.sum()),
                      "mean_signed_error": float((F.frozen_pred - F.delta).mean()),
                      "prospective_n": int(len(pro)), "prospective_inside": int(pro.inside.sum()),
                      "prospective_r": float(np.corrcoef(pro.frozen_pred, pro.delta)[0, 1]),
                      "prospective_mae": float((pro.frozen_pred - pro.delta).abs().mean()),
                      "misses": [f"{SHORT[r.model]}|{r.schedule}: pred {r.frozen_pred:.1f} obs {r.delta:.1f} [{r.lo:.1f},{r.hi:.1f}]" for r in F[~F.inside].itertuples()]}
C = pd.read_csv(ROOT / "results" / "tables" / "cross_family_abm.csv")
N["bridge_cross"] = {}
for v, g in C.groupby("variant"):
    N["bridge_cross"][v] = {"n": int(len(g)), "inside": int(g.inside_ci.sum()), "r": float(np.corrcoef(g.pred_delta, g.delta)[0, 1]),
                            "mae": float((g.pred_delta - g.delta).abs().mean())}
    for fam, gg in g.groupby(g.model.str.startswith("GLM")):
        N["bridge_cross"][v]["GLM" if fam else "Qwen"] = {"n": int(len(gg)), "inside": int(gg.inside_ci.sum()), "mae": float((gg.pred_delta - gg.delta).abs().mean())}

OUT.write_text(json.dumps(N, indent=1), encoding="utf-8")

# ---------------- readable summary ----------------
print("rows", N["meta"])
print("\n== tail effect (pp) [95% CI] | peak at t=3 | retention | control tail error | collapse treated/control (all agents)")
for k, v in N["effects"].items():
    c = N["collapse"][k]
    ret = "  n/a" if v["retention"] is None else f"{v['retention']:5.2f}"
    print(f"{k:55s} {v['tail'][0]:6.1f} [{v['tail'][1]:5.1f},{v['tail'][2]:5.1f}]  peak {v['peak3'][0]:5.1f}  ret {ret}  ctrl {v['ctrl_tail']:5.1f}"
          f"  collapse {c['treated_all']*100:4.0f}%/{c['control_all']*100:3.0f}% (nonseed {c['treated_nonseed']*100:4.0f}%/{c['control_nonseed']*100:3.0f}%)")
print("\n== lock-in"); [print(k, v) for k, v in N["lockin"].items()]
print("\n== rules"); [print(k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items() if a != "switch_by_opposing"},
                            {a: round(b[0], 3) for a, b in v["switch_by_opposing"].items()}) for k, v in N["rules"].items()]
print("\n== isolation"); [print(k, {a: round(b[0], 3) for a, b in v.items()}) for k, v in N["isolation"].items()]
print("\n== premium"); [print(k, {a: round(b, 3) for a, b in v.items()}) for k, v in N["premium"].items()]
print("\n== claim recited"); [print(k, round(v[0], 3), v[1]) for k, v in N["claim_recited"].items()]
print("\n== bridge frozen", json.dumps(N["bridge_frozen"], indent=1))
print("== bridge cross", json.dumps(N["bridge_cross"], indent=1))

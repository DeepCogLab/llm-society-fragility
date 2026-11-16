"""Extract a unified agent-round table from v3 and v4 network runs (read-only)."""
from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

V3 = Path("results/response_v3/networks")
V4 = Path("results/structural_v4/networks")
OUT = Path("results/structural_v4/analysis")
AUDIT = re.compile(r"audit", re.I)
FRAC = {"persistent": 1.0, "initial_only": 0.0, "init_fullmem": 0.0, "anchor25": .25, "anchor50": .5, "in_history": 1.0,
        "init_seed1": 0.0, "init_seed6": 0.0, "persist_seed6": 1.0}
V5 = Path("results/semantic_v5/networks")
CLAIM = re.compile(r"\b(lab|labs|laboratory|sequencing|genome|regulator|postmortem|commander|conclusive|conclusively|tooling)\b", re.I)


def runs():
    for p in sorted(V3.glob("*/q0.70/*/*/*.json")):
        model, _, schedule, block, arm = p.parts[-5], p.parts[-4], p.parts[-3], p.parts[-2], p.stem
        yield str(p), "v3", model, schedule, block, arm
    for p in sorted(V4.glob("*/*/*/*.json")):
        model, schedule, block, arm = p.parts[-4], p.parts[-3], p.parts[-2], p.stem
        yield str(p), "v4", model, schedule, block, arm
    for p in sorted(V5.glob("*/*/*/*.json")):
        model, schedule, block, arm = p.parts[-4], p.parts[-3], p.parts[-2], p.stem
        yield str(p), "v5", model, schedule, block, arm


def extract(item):
    path, source, model, schedule, block, arm = item
    run = json.loads(Path(path).read_bytes())
    truth = run["task"]["truth"]
    private = run["task"]["private"]
    anchors = set(run.get("anchors", range(24) if schedule == "persistent" else []))
    spec = run["spec"]
    seed_fraction = run["identity"].get("seed_fraction", spec["fraction"])
    rows = []
    for e in run["events"]:
        inbox = e["received"]
        rows.append({
            "source": source, "model": model, "schedule": schedule, "anchor_fraction": FRAC[schedule], "block": block,
            "topology": spec["topology"], "replicate": spec["replicate"], "arm": arm, "round": e["round"], "node": e["node"],
            "wrong": int(e["wrong"]), "confidence": e["confidence"], "seed": int(e["seed_node"]), "anchor": int(e["node"] in anchors),
            "private_correct": int(private[e["node"]] == truth), "dist": e["distance_from_seeds"],
            "n_inbox": len(inbox), "n_wrong_inbox": sum(m["choice"] != truth for m in inbox),
            "n_injected_inbox": sum(bool(m.get("injected")) for m in inbox),
            "n_audit_text_inbox": sum(bool(AUDIT.search(m["text"])) and not m.get("injected") for m in inbox),
            "msg_audit": int(bool(AUDIT.search(e["outgoing"]["text"]))),
            "msg_chars": len(e["outgoing"]["text"]),
            "cites_planted": int("E900" in e["outgoing"]["roots"]),
            "seed_fraction": seed_fraction,
            "msg_claim": int(bool(CLAIM.search(e["outgoing"]["text"]))),
        })
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=24)
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    items = list(runs())
    with ProcessPoolExecutor(args.workers) as ex:
        frames = [pd.DataFrame(r) for r in ex.map(extract, items, chunksize=8)]
    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(OUT / "agent_rounds.parquet", index=False)
    print(df.groupby(["source", "model", "schedule", "arm"]).block.nunique().to_string())


if __name__ == "__main__":
    main()

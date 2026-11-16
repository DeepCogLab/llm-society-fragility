"""v4: structural determinants of persistence after transient perturbation.

Manipulates WHERE evidence is stored in the society: the fraction of agents that
can re-access their original private evidence each round (anchor fraction), plus an
information-equivalent presentation control (in_history). Re-uses the exact v3
Spec / task / graph / seed / prompt construction so that results pair with v3 runs
(common random numbers). v3 files are imported, never modified.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import time
from dataclasses import asdict, replace
from pathlib import Path

import networkx as nx

from .client import LLMClient
from .core import Spec, atomic_json, digest, graph_for, graph_summary, neighbors_for, seed_for, selected_nodes, task_for
from .experiment import SYSTEM, injected_message, message_from_decision, prompt_for, summarize
from . import response_v3 as v3

PROTOCOL4 = "structural-determinants-v4"
OUTPUT = Path("results/structural_v4")
CONFIG = Path("configs/structural_v4.json")
_CFG = json.loads((Path(__file__).resolve().parents[1] / CONFIG).read_text())["schedules"]
config_memory = {k: v["memory"] for k, v in _CFG.items() if "memory" in v}
config_dose = {name: info["seed_fraction"] for name, info in json.loads((Path(__file__).resolve().parents[1] / CONFIG).read_text())["schedules"].items() if "seed_fraction" in info}


def log(event, **values):
    print(json.dumps({"event": event, "time": time.strftime("%H:%M:%S"), **values}, ensure_ascii=False), flush=True)


def model_registry():
    base = {m["id"]: m for m in json.loads(Path("configs/models.json").read_text())["models"]}
    mistral = json.loads(Path("configs/cross_family_grammar_runtime1200.json").read_text())["models"][0]
    base[mistral["id"]] = mistral
    base["GLM-4-9B-0414"] = {"id": "GLM-4-9B-0414", "family": "GLM", "url": "http://127.0.0.1:18245/v1",
                             "fingerprint": "GLM-4-9B-0414-bf16-vllm0.14.1-20261115"}
    base["Llama-3.1-8B-Instruct"]["fingerprint"] = "Llama-3.1-8B-Instruct-bf16-vllm0.14.1-20261115"
    return base


def anchors_for(spec: Spec, fraction: float) -> set[int]:
    """Nested anchor sets: the 25% set is a subset of the 50% set within a block."""
    order = list(range(spec.n))
    random.Random(seed_for(PROTOCOL4, "anchors", spec.block_id)).shuffle(order)
    return set(order[: round(fraction * spec.n)])


def node_prompt(spec, task, node, t, inbox, prior, schedule, anchors):
    if schedule == "in_history":
        prompt = v3.private_prompt(spec, task, node, t, inbox, prior, "persistent")
        if t > 0:
            line = f"Your private sensor report E{node:03d} observes {task['private'][node]}; reliability {spec.reliability:.2f}."
            if prompt.count(line) != 1:
                raise RuntimeError("Expected one private report line")
            prompt = prompt.replace(line, f"At initialization (round 0) your private sensor report E{node:03d} observed {task['private'][node]}; reliability {spec.reliability:.2f}. It is not re-sent in later rounds.")
        return prompt
    if schedule in config_memory:
        # Prompt-only override: block_id / seeds still come from the unmodified spec.
        spec = replace(spec, memory=config_memory[schedule])
    return v3.private_prompt(spec, task, node, t, inbox, prior, "persistent" if node in anchors else "initial_only")


def network_specs(config):
    n = config["network"]
    for topology in n["topologies"]:
        for rep in range(n["start"], n["start"] + n["reps"]):
            yield Spec(phase=n["phase"], topology=topology, replicate=rep, n=n["n"], rounds=n["rounds"],
                       reliability=n["q"], onset=n["onset"], duration=n["duration"], fraction=n["fraction"], wording=rep % 3)


async def run_arm(client, spec, schedule, fraction, arm, baseline, path):
    identity = {"protocol": PROTOCOL4, "model": client.model, "spec": asdict(spec), "schedule": schedule,
                "anchor_fraction": fraction, "arm": arm}
    dose = config_dose.get(schedule)
    if dose is not None:
        identity["seed_fraction"] = dose
    if schedule in config_memory:
        identity["memory"] = config_memory[schedule]
    if path.exists():
        saved = json.loads(path.read_bytes())
        if saved["identity"] != identity:
            raise RuntimeError(f"Identity mismatch {path}")
        return saved
    graph, task = graph_for(spec), task_for(spec)
    # Dose variants keep the base spec's graph, task, exposures, anchors and sampling seeds
    # (paired with the base schedule's control); only the nested seed set grows/shrinks.
    seeds = set(selected_nodes(replace(spec, fraction=dose) if dose is not None else spec, graph))
    anchors = anchors_for(spec, fraction) if schedule != "in_history" else set(range(spec.n))
    dist = nx.multi_source_dijkstra_path_length(graph, seeds)
    outgoing, histories, events = [], [[] for _ in range(spec.n)], []
    for t in range(spec.rounds):
        inboxes, prompts = [], []
        for node in range(spec.n):
            inbox = []
            if t:
                for sender in neighbors_for(spec, graph, node, t):
                    m = outgoing[t - 1][sender]
                    if arm == "no_relay_treated" and sender not in seeds and t >= spec.onset:
                        m = baseline["outgoing"][t - 1][sender]
                    if arm != "control" and sender in seeds and spec.onset <= t < spec.onset + spec.duration:
                        m = injected_message(m, task, spec)
                    inbox.append(m)
            inboxes.append(inbox)
            prompts.append(node_prompt(spec, task, node, t, inbox, histories[node], schedule, anchors))
        decisions = await asyncio.gather(*(client.complete(SYSTEM, p, seed_for(v3.PROTOCOL, "network", spec.block_id, node, t), spec.temperature)
                                           for node, p in enumerate(prompts)))
        messages = []
        for node, d in enumerate(decisions):
            message = message_from_decision(d, node, t, inboxes[node], "natural")
            messages.append(message)
            e = {"round": t, "node": node, "choice": d["choice"], "confidence": d["confidence"], "wrong": d["choice"] != task["truth"],
                 "seed_node": node in seeds, "anchor": node in anchors, "distance_from_seeds": dist.get(node),
                 "received": inboxes[node], "outgoing": message,
                 "direct_injected_exposure": any(m["injected"] for m in inboxes[node]), "call_key": d["call_key"],
                 "prompt_hash": digest(prompts[node]), "prompt_chars": len(prompts[node])}
            events.append(e)
            histories[node].append(e)
        outgoing.append(messages)
    result = {"identity": identity, "spec": asdict(spec), "model": client.model["id"], "family": client.model["family"],
              "schedule": schedule, "anchor_fraction": fraction, "anchors": sorted(anchors), "arm": arm, "mode": "natural",
              "treatment": arm != "control", "block_id": spec.block_id, "task": task, "seeds": sorted(seeds),
              "graph": graph_summary(graph), "events": events, "outgoing": outgoing}
    result["metrics"] = summarize(result)
    atomic_json(path, result)
    return result


async def run_networks(client, config, schedules, blocks_in_flight):
    sem = asyncio.Semaphore(blocks_in_flight)
    jobs = [(spec, s) for s in schedules for spec in network_specs(config)]
    done = [0]

    async def block(spec, schedule):
        info = config["schedules"][schedule]
        async with sem:
            folder = OUTPUT / "networks" / client.model["id"] / schedule / spec.block_id
            base = None
            if "control" in info["arms"]:
                base = await run_arm(client, spec, schedule, info["anchor_fraction"], "control", None, folder / "control.json")
            for arm in info["arms"]:
                if arm != "control":
                    await run_arm(client, spec, schedule, info["anchor_fraction"], arm, base, folder / f"{arm}.json")
            done[0] += 1
            log("v4_block_done", model=client.model["id"], schedule=schedule, topology=spec.topology,
                replicate=spec.replicate, done=done[0], total=len(jobs))
    await asyncio.gather(*(block(*j) for j in jobs))


async def run_profile(client, config):
    v3config = v3.load_config()
    v3config["profile"] = config["profile"]
    folder = OUTPUT / "profiles" / client.model["id"]
    cells = list(v3.profile_cells(v3config, "profile"))

    async def one(cell):
        path = folder / (digest(cell)[:24] + ".json")
        identity = {"protocol": PROTOCOL4, "model": client.model, "cell": cell}
        if path.exists():
            return
        spec, task, private, inbox, prior, prompt = v3.profile_context(cell)
        seed = seed_for(v3.PROTOCOL, "profile", cell["replicate"], cell["q"], cell["private_correct"], cell["support"], cell["history"])
        d = await client.complete(SYSTEM, prompt, seed, config["temperature"])
        atomic_json(path, {"identity": identity, **cell, "truth": task["truth"], "private": private, "decision": d,
                           "wrong": d["choice"] != task["truth"], "changed_private": d["choice"] != private,
                           "prompt_hash": digest(prompt)})
    for start in range(0, len(cells), 256):
        await asyncio.gather(*(one(c) for c in cells[start:start + 256]))
        log("v4_profile_progress", model=client.model["id"], completed=min(start + 256, len(cells)), total=len(cells))


def freeze(config):
    files = [Path("society/structural_v4.py"), CONFIG, Path("society/response_v3.py"), Path("society/experiment.py"),
             Path("society/core.py"), Path("society/client.py")]
    record = {"design": config, "system": SYSTEM, "sources": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    path = OUTPUT / "frozen_design.json"
    if path.exists():
        old = json.loads(path.read_bytes())
        if old["sources"] != record["sources"] or old["design"] != config:
            # Allow additions only by explicit versioned amendment.
            amend = OUTPUT / "amendments"
            amend.mkdir(parents=True, exist_ok=True)
            atomic_json(amend / f"{int(time.time())}.json", record)
            log("v4_source_amendment_recorded")
        return
    record["frozen"] = time.time()
    atomic_json(path, record)


async def main_async(args):
    config = json.loads(CONFIG.read_text())
    OUTPUT.mkdir(parents=True, exist_ok=True)
    freeze(config)
    model = model_registry()[args.model]
    client = LLMClient(model, OUTPUT / "cache" / (args.model + ".sqlite"), args.concurrency)
    try:
        await client.inspect()
        if args.profile:
            await run_profile(client, config)
        schedules = [s for s in args.schedules.split(",") if s]
        if schedules:
            await run_networks(client, config, schedules, args.blocks)
        log("v4_worker_done", model=args.model, actual_calls=client.actual_calls, cache_hits=client.cache_hits)
    finally:
        await client.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--schedules", default="")
    p.add_argument("--profile", action="store_true")
    p.add_argument("--concurrency", type=int, default=48)
    p.add_argument("--blocks", type=int, default=6)
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()

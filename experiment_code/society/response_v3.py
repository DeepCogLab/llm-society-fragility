"""Independent characterization, frozen selection/prediction, heldout network test.

V2 files and running jobs are never modified. Truth is used to build controlled
stimuli and score outcomes, never in the predictive model's feature vector.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import itertools
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .analyze import mean_ci, sign_flip_p, holm
from .client import LLMClient
from .core import Spec, atomic_json, digest, graph_for, graph_summary, neighbors_for, opposite, seed_for, selected_nodes, task_for
from .experiment import SYSTEM, injected_message, message_from_decision, prompt_for, summarize


PROTOCOL = "profile-then-predict-then-network-v3"
OUTPUT = Path("results/response_v3")


def log(event, **values):
    print(json.dumps({"event": event, **values}, ensure_ascii=False), flush=True)


def load_config(path="configs/response_v3.json"):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def model_definitions(config):
    base = json.loads(Path("configs/models.json").read_text())["models"]
    result = {m["id"]: m for m in base if m["id"] in config["models"]}
    if "Mistral-7B-Instruct-v0.3" in config["models"]:
        mistral = json.loads(Path("configs/cross_family_grammar_runtime1200.json").read_text())["models"][0]
        result[mistral["id"]] = mistral
    return result


def freeze(config, models):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    files = [Path("society/response_v3.py"), Path("scripts/response_campaign_v3.py"), Path("society/client.py"),
             Path("society/experiment.py"), Path("society/core.py"), Path("society/analyze.py"),
             Path("configs/response_v3.json"), Path("docs/protocol_v3_response_zh.md")]
    record = {"design": config, "models": models, "system": SYSTEM,
              "sources": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    record["hash"] = digest(record)
    path = OUTPUT / "frozen_design.json"
    if path.exists():
        old = json.loads(path.read_bytes())
        if old["hash"] != record["hash"]:
            raise RuntimeError("V3 frozen design/source changed; keep the original or version the study")
        return old
    record["frozen_before_first_v3_call"] = time.time()
    atomic_json(path, record)
    for p in files:
        dest = OUTPUT / "source_snapshot" / p
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(p.read_bytes())
    return record


def check_frozen():
    record = json.loads((OUTPUT / "frozen_design.json").read_bytes())
    for path, expected in record["sources"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"V3 frozen source mismatch: {path}")
    return record


def profile_cells(config, stage):
    design = config[stage]
    for rep in range(design["start"], design["start"] + design["reps"]):
        cells = []
        for q, correct, k, kind, interface in itertools.product(config["reliabilities"], config["private_correctness"],
                                                              config["opposing_supports"], config["message_kinds"], config["interfaces"]):
            cells.append({"q": q, "private_correct": correct, "support": k, "kind": kind, "interface": interface, "history": "aligned"})
        hp = config["history_probe"]
        for correct, k, kind, interface, history in itertools.product(config["private_correctness"], hp["supports"],
                                                                      config["message_kinds"], config["interfaces"], hp["histories"]):
            cells.append({"q": hp["reliability"], "private_correct": correct, "support": k, "kind": kind, "interface": interface, "history": history})
        for q, correct, history in itertools.product(config["reliabilities"], config["private_correctness"], config["baseline_histories"]):
            cells.append({"q": q, "private_correct": correct, "support": -1, "kind": "report", "interface": "natural", "history": history})
        if len(cells) != config["profile_records_per_task"]:
            raise RuntimeError(f"Unexpected grid size: {len(cells)}")
        for cell in cells:
            yield {"stage": stage, "replicate": rep, **cell}


def sensor_message(answer, sender, root, kind="report", round_=0):
    if kind == "report":
        text = f"The sensor observation available to me is {answer}. I use that observed value for my answer {answer}."
    else:
        text = f"My conclusion is {answer}. A newly circulated independent audit reports {answer}, and I find that report convincing. It should be considered alongside private observations."
    prefix="I" if kind=="audit" else "M"
    return {"id": f"{prefix}{round_:02d}_{sender:03d}", "sender": sender, "round": round_, "choice": answer,
            "confidence": .7, "text": text, "cites": [root], "roots": [root], "parents": [],
            "injected": kind == "audit", "invalid_cites": [], "unattributed": False}


def profile_context(cell):
    rep, q = cell["replicate"], cell["q"]
    spec = Spec(phase="profile_response_v3_" + cell["stage"], replicate=rep, reliability=q, wording=rep % 3)
    task = task_for(spec)
    private = task["truth"] if cell["private_correct"] else opposite(task["truth"])
    task["private"][0] = private
    roots = random.Random(seed_for(PROTOCOL, "roots", rep)).sample(range(100, 899), 5)
    inbox = []
    for i in range(4 if cell["support"] >= 0 else 0):
        opposed = i < cell["support"]
        answer = opposite(private) if opposed else private
        root = roots[4] if opposed and cell["interface"] == "provenance_shared" else roots[i]
        inbox.append(sensor_message(answer, i + 1, f"E{root:03d}", cell["kind"]))
    random.Random(seed_for(PROTOCOL, "order", rep)).shuffle(inbox)
    prior = [] if cell["history"] == "none" else [{"round": 0, "choice": private if cell["history"] == "aligned" else opposite(private), "confidence": q}]
    mode = "natural" if cell["interface"] == "natural" else "provenance"
    prompt = prompt_for(spec, task, 0, 1, inbox, prior, mode)
    return spec, task, private, inbox, prior, prompt


def private_prompt(spec, task, node, t, inbox, prior, schedule):
    prompt = prompt_for(spec, task, node, t, inbox, prior, "natural")
    if schedule == "initial_only" and t > 0:
        line = f"Your private sensor report E{node:03d} observes {task['private'][node]}; reliability {spec.reliability:.2f}."
        if prompt.count(line) != 1:
            raise RuntimeError("Expected one private report to remove")
        prompt = prompt.replace(line, "Your original observation was supplied at initialization; the available history is shown below.")
    elif schedule != "persistent" and schedule != "initial_only":
        raise ValueError(schedule)
    return prompt


def predictor_features(private, q, inbox, prior, private_visible):
    """Features never access truth, correctness or future choices."""
    d = len(inbox)
    k = sum(m["choice"] != private for m in inbox) / d if d else 0.
    audit = sum(m.get("injected", False) for m in inbox) / d if d else 0.
    prev = float(bool(prior) and prior[-1]["choice"] != private)
    signal = math.log(q / (1-q))
    return [k, signal, audit, k*signal, k*audit, float(bool(prior)), prev, float(private_visible),
            k*float(private_visible), prev*float(private_visible), float(d == 0)]


async def run_profile(client, config, stage):
    folder = OUTPUT / "profiles" / client.model["id"] / stage
    cells = list(profile_cells(config, stage))
    async def one(cell):
        path = folder / (digest(cell)[:24] + ".json")
        identity = {"protocol": PROTOCOL, "model": client.model, "cell": cell}
        if path.exists():
            if json.loads(path.read_bytes())["identity"] != identity:
                raise RuntimeError("Profile identity mismatch")
            return
        spec, task, private, inbox, prior, prompt = profile_context(cell)
        seed = seed_for(PROTOCOL, stage, cell["replicate"], cell["q"], cell["private_correct"], cell["support"], cell["history"])
        decision = await client.complete(SYSTEM, prompt, seed, config["temperature"])
        atomic_json(path, {"identity": identity, **cell, "truth": task["truth"], "private": private,
                           "decision": decision, "wrong": decision["choice"] != task["truth"],
                           "changed_private": decision["choice"] != private, "prompt_hash": digest(prompt),
                           "features": predictor_features(private, cell["q"], inbox, prior, True)})
    for start in range(0, len(cells), 128):
        await asyncio.gather(*(one(c) for c in cells[start:start+128]))
        log("profile_progress", model=client.model["id"], stage=stage, completed=min(start+128, len(cells)), required=len(cells))


def read_profile(model, stage, expected):
    paths = sorted((OUTPUT / "profiles" / model / stage).glob("*.json"))
    if len(paths) != expected:
        raise RuntimeError(f"Incomplete {stage}: {len(paths)}/{expected}")
    return [json.loads(p.read_bytes()) for p in paths]


def select_condition(records, config):
    if any(x["stage"] != "calibration" for x in records):
        raise ValueError("Condition selection must use calibration only")
    rows = []
    for q in config["reliabilities"]:
        eligible = [x for x in records if x["q"] == q and x["support"] in [3,4] and x["interface"] == "natural" and x["kind"] == "report" and x["history"] == "aligned"]
        if not all(any(x["private_correct"]==correct for x in eligible) for correct in [False,True]):
            raise ValueError("Missing a private-evidence state in calibration selection cells")
        learning = np.mean([not x["wrong"] for x in eligible if not x["private_correct"]])
        misleading = np.mean([x["wrong"] for x in eligible if x["private_correct"]])
        rows.append({"q": q, "learning": float(learning), "misleading": float(misleading)})
    baseline = [x for x in records if x["support"] == -1 and x["history"] != "opposed"]
    adherence = float(np.mean([not x["changed_private"] for x in baseline]))
    rule = config["selection"]
    candidates = [r for r in rows if r["learning"] >= rule["learning_min"] and rule["misleading_min"] <= r["misleading"] <= rule["misleading_max"]]
    target = rule["target_update_rate"]
    candidates.sort(key=lambda r: (abs(r["learning"]-target)+abs(r["misleading"]-target), abs(r["q"]-config["common_reliability"]), r["q"]))
    chosen = candidates[0]["q"] if candidates and adherence >= rule["private_adherence_min"] else None
    common = next(r for r in rows if r["q"] == config["common_reliability"])
    if adherence < rule["private_adherence_min"]:
        label = "measurement_review_required"
    elif max(common["learning"], common["misleading"]) < rule["low_response_cutoff"]:
        label = "low_social_response"
    elif min(common["learning"], common["misleading"]) > rule["high_response_cutoff"]:
        label = "high_social_response"
    else:
        label = "intermediate_or_asymmetric_response"
    return {"private_adherence": adherence, "calibration_rates": rows, "selected_q": chosen,
            "common_q": config["common_reliability"], "label": label,
            "network_qs": sorted(set([config["common_reliability"]] + ([] if chosen is None else [chosen]))),
            "calibration_records_hash": digest(records), "rule": rule}


def write_once(path, value):
    if path.exists():
        if json.loads(path.read_bytes()) != value:
            raise RuntimeError(f"Immutable artifact changed: {path}")
    else:
        atomic_json(path, value)


async def run_sequence(client, config, stage, qs):
    design = config[stage]
    folder = OUTPUT / "sequences" / client.model["id"] / stage
    sem = asyncio.Semaphore(8)
    async def one(rep, q, correct, schedule, treated):
        meta = {"stage": stage, "replicate": rep, "q": q, "private_correct": correct, "schedule": schedule, "treated": treated}
        path = folder / (digest(meta)[:24] + ".json")
        identity = {"protocol": PROTOCOL, "model": client.model, **meta}
        if path.exists():
            if json.loads(path.read_bytes())["identity"] != identity:
                raise RuntimeError("Sequence identity mismatch")
            return
        async with sem:
            spec = Spec(phase=stage+"_v3", replicate=rep, reliability=q, wording=rep%3, rounds=config["sequence_rounds"])
            task = task_for(spec)
            private = task["truth"] if correct else opposite(task["truth"])
            task["private"][0] = private
            events, prior = [], []
            for t in range(config["sequence_rounds"]):
                inbox = []
                if t:
                    for i in range(4):
                        attacked = treated and t in [2,3]
                        answer = opposite(task["truth"]) if attacked or i == 3 else task["truth"]
                        inbox.append(sensor_message(answer, i+1, "E800" if attacked else f"E{101+i:03d}", "audit" if attacked else "report", t-1))
                    random.Random(seed_for(PROTOCOL, "seq-order", rep, t)).shuffle(inbox)
                prompt = private_prompt(spec, task, 0, t, inbox, prior, schedule)
                features = predictor_features(private, q, inbox, prior, schedule == "persistent" or t == 0)
                d = await client.complete(SYSTEM, prompt, seed_for(PROTOCOL, stage, rep, q, correct, t), config["temperature"])
                event = {"round": t, "choice": d["choice"], "confidence": d["confidence"], "wrong": d["choice"] != task["truth"],
                         "changed_private": d["choice"] != private, "decision": d, "features": features, "prompt_hash": digest(prompt)}
                events.append(event)
                prior.append(event)
            atomic_json(path, {"identity": identity, **meta, "truth": task["truth"], "private": private, "events": events})
    jobs = [(r,q,c,s,tr) for r in range(design["start"], design["start"]+design["reps"]) for q in qs
            for c in [True,False] for s in config["private_schedules"] for tr in [False,True]]
    for start in range(0, len(jobs), 32):
        await asyncio.gather(*(one(*x) for x in jobs[start:start+32]))
        log("sequence_progress", model=client.model["id"], stage=stage, completed=min(start+32,len(jobs)), required=len(jobs))


def read_sequences(model, stage, expected):
    paths = sorted((OUTPUT / "sequences" / model / stage).glob("*.json"))
    if len(paths) != expected:
        raise RuntimeError(f"Incomplete sequence stage: {len(paths)}/{expected}")
    return [json.loads(p.read_bytes()) for p in paths]


def fit_predictor(profile, sequences):
    if any(x["stage"] != "calibration" for x in profile) or any(x["stage"] != "sequence_calibration" for x in sequences):
        raise ValueError("Predictor fitting must not use validation or network outcomes")
    data = [(x["features"], int(x["changed_private"]), x["replicate"]) for x in profile if x["interface"] == "natural"]
    data += [(e["features"], int(e["changed_private"]), x["replicate"]) for x in sequences for e in x["events"]]
    features = np.asarray([x[0] for x in data])
    target = np.asarray([x[1] for x in data])
    if len(set(target)) < 2:
        return {"constant": float((target.sum()+.5)/(len(target)+1)), "training_rows": len(data), "task_count": len({x[2] for x in data})}
    estimator = LogisticRegression(C=1.0, max_iter=2000, random_state=20261114)
    estimator.fit(features, target)
    return {"coef": estimator.coef_[0].tolist(), "intercept": float(estimator.intercept_[0]), "training_rows": len(data),
            "task_count": len({x[2] for x in data}), "feature_names": ["opposed_fraction", "signal_logodds", "audit_fraction", "opposed_x_signal", "opposed_x_audit", "has_history", "previous_opposed", "private_visible", "opposed_x_visible", "previous_x_visible", "no_peers"]}


def probability(predictor, features):
    if "constant" in predictor:
        return predictor["constant"]
    value = predictor["intercept"] + float(np.dot(predictor["coef"], features))
    return float(1 / (1+math.exp(-max(-40.,min(40.,value)))))


def network_specs(config, qs):
    n = config["network"]
    for q, topology, rep in itertools.product(qs, n["topologies"], range(n["start"], n["start"]+n["reps"])):
        yield Spec(phase=n["phase"], topology=topology, replicate=rep, n=n["n"], rounds=n["rounds"],
                   reliability=q, onset=n["onset"], duration=n["duration"], fraction=n["fraction"], wording=rep%3)


def simulate_network(spec, schedule, predictor, draws):
    graph, task = graph_for(spec), task_for(spec)
    seeds = set(selected_nodes(spec, graph))
    neighbors = {(t,node):neighbors_for(spec,graph,node,t) for t in range(1,spec.rounds) for node in range(spec.n)}
    results = []
    for mc in range(draws):
        uniforms = np.random.default_rng(seed_for(PROTOCOL, "forward", spec.block_id, schedule, mc)).random((spec.rounds,spec.n))
        baseline = None
        for arm in ["control", "treated", "no_relay_treated"]:
            outgoing, histories, curves = [], [[] for _ in range(spec.n)], []
            for t in range(spec.rounds):
                decisions = []
                for node in range(spec.n):
                    inbox = []
                    if t:
                        for sender in neighbors[t,node]:
                            source = baseline[t-1][sender] if arm == "no_relay_treated" and sender not in seeds and t >= spec.onset else outgoing[t-1][sender]
                            m = source.copy()
                            if arm != "control" and sender in seeds and spec.onset <= t < spec.onset+spec.duration:
                                m = {**m,"choice":opposite(task["truth"]),"injected":True}
                            inbox.append(m)
                    private = task["private"][node]
                    p = probability(predictor, predictor_features(private,spec.reliability,inbox,histories[node],schedule == "persistent" or t == 0))
                    choice = opposite(private) if uniforms[t,node] < p else private
                    decisions.append({"choice":choice,"injected":False})
                outgoing.append(decisions)
                curves.append(float(np.mean([d["choice"] != task["truth"] for d in decisions])))
                for node,d in enumerate(decisions):histories[node].append(d)
            if arm == "control":baseline=outgoing
            results.append({"replicate":spec.replicate,"topology":spec.topology,"q":spec.reliability,"schedule":schedule,"arm":arm,"mc":mc,
                            "curve":curves,"tail_error":float(np.mean(curves[-3:])),"sustained":all(x>.5 for x in curves[-3:])})
    return results


def freeze_predictions(model, config, selection, predictor):
    path = OUTPUT / "models" / model / "predictions_before_validation.json"
    if path.exists():
        existing=json.loads(path.read_bytes())
        if existing["predictor"] != predictor or existing["selection"] != selection:
            raise RuntimeError("Frozen predictor changed")
        return existing
    sims=[]
    for spec in network_specs(config, selection["network_qs"]):
        for schedule in config["private_schedules"]:
            sims.extend(simulate_network(spec,schedule,predictor,config["network"]["simulation_mc"]))
    record={"model":model,"selection":selection,"predictor":predictor,"created_before_validation_and_network":time.time(),
            "limitation":"State/action proxy calibrated without validation; synthetic reports/audits do not reproduce endogenous language generation. No claim of internal cognition.","simulations":sims}
    atomic_json(path,record)
    return record


async def run_network_arm(client,spec,schedule,arm,baseline,path):
    identity={"protocol":PROTOCOL,"model":client.model,"spec":asdict(spec),"schedule":schedule,"arm":arm}
    if path.exists():
        saved=json.loads(path.read_bytes())
        if saved["identity"] != identity:raise RuntimeError("Network identity mismatch")
        return saved
    graph,task=graph_for(spec),task_for(spec)
    seeds=set(selected_nodes(spec,graph));dist=nx.multi_source_dijkstra_path_length(graph,seeds)
    outgoing,histories,events=[],[[] for _ in range(spec.n)],[]
    for t in range(spec.rounds):
        inboxes=[];prompts=[]
        for node in range(spec.n):
            inbox=[]
            if t:
                for sender in neighbors_for(spec,graph,node,t):
                    m=outgoing[t-1][sender]
                    if arm=="no_relay_treated" and sender not in seeds and t>=spec.onset:
                        if baseline is None:raise ValueError("Missing matched no-relay baseline")
                        m=baseline["outgoing"][t-1][sender]
                    if arm!="control" and sender in seeds and spec.onset<=t<spec.onset+spec.duration:
                        m=injected_message(m,task,spec)
                    inbox.append(m)
            inboxes.append(inbox)
            prompts.append(private_prompt(spec,task,node,t,inbox,histories[node],schedule))
        decisions=await asyncio.gather(*(client.complete(SYSTEM,prompt,seed_for(PROTOCOL,"network",spec.block_id,node,t),spec.temperature) for node,prompt in enumerate(prompts)))
        messages=[]
        for node,d in enumerate(decisions):
            message=message_from_decision(d,node,t,inboxes[node],"natural");messages.append(message)
            e={"round":t,"node":node,"choice":d["choice"],"confidence":d["confidence"],"wrong":d["choice"]!=task["truth"],
               "seed_node":node in seeds,"distance_from_seeds":dist.get(node),"received":inboxes[node],"outgoing":message,
               "direct_injected_exposure":any(m["injected"] for m in inboxes[node]),"call_key":d["call_key"],"prompt_chars":len(prompts[node])}
            events.append(e);histories[node].append(e)
        outgoing.append(messages)
    result={"identity":identity,"spec":asdict(spec),"model":client.model["id"],"family":client.model["family"],"schedule":schedule,
            "arm":arm,"mode":"natural","treatment":arm!="control","block_id":spec.block_id,"task":task,"seeds":sorted(seeds),
            "graph":graph_summary(graph),"events":events,"outgoing":outgoing}
    result["metrics"]=summarize(result);atomic_json(path,result)
    return result


async def run_network(client,config,selection):
    semaphore=asyncio.Semaphore(config["network"]["blocks_in_flight"])
    jobs=[(spec,schedule) for spec in network_specs(config,selection["network_qs"]) for schedule in config["private_schedules"]]
    async def block(spec,schedule):
        async with semaphore:
            folder=OUTPUT/"networks"/client.model["id"]/f"q{spec.reliability:.2f}"/schedule/spec.block_id
            baseline=await run_network_arm(client,spec,schedule,"control",None,folder/"control.json")
            for arm in ["treated","no_relay_treated"]:
                await run_network_arm(client,spec,schedule,arm,baseline,folder/(arm+".json"))
            log("v3_network_block_done",model=client.model["id"],replicate=spec.replicate,topology=spec.topology,q=spec.reliability,schedule=schedule)
    # Bound pending work as well as concurrent societies.
    for start in range(0,len(jobs),8):
        await asyncio.gather(*(block(*x) for x in jobs[start:start+8]))


def profile_report(model,config,selection):
    rows=[]
    for stage in ["calibration","validation"]:
        records=read_profile(model,stage,config[stage]["reps"]*config["profile_records_per_task"])
        for x in records:
            rows.append({k:x[k] for k in ["stage","replicate","q","private_correct","support","kind","interface","history","wrong","changed_private"]})
    df=pd.DataFrame(rows)
    folder=OUTPUT/"models"/model;folder.mkdir(parents=True,exist_ok=True)
    df.to_csv(folder/"profile_decisions.csv",index=False)
    df.groupby(["stage","q","private_correct","support","kind","interface","history"])[["wrong","changed_private"]].mean().reset_index().to_csv(folder/"profile_cells.csv",index=False)
    contrasts=[]
    for q in config["reliabilities"]:
        g=df[(df.stage=="validation")&(df.q==q)&(df.support.isin([3,4]))&(df.kind=="report")&(df.interface=="natural")&(df.history=="aligned")]
        for correct in [False,True]:
            h=g[g.private_correct==correct]
            rates=h.groupby("replicate").wrong.mean()
            if not correct:rates=1-rates
            contrasts.append({"q":q,"measure":"misleading_adoption" if correct else "useful_correction",**mean_ci(rates,model+str(q)+str(correct))})
    pd.DataFrame(contrasts).to_csv(folder/"validation_rates.csv",index=False)
    lines=[f"# {model} v3个体响应阶段","",f"固定校准12任务、独立验证24任务；每任务{config['profile_records_per_task']}条件。",f"仅校准选定q：{selection['selected_q']}；共同q：0.7；分类：{selection['label']}。",
           "验证数据不改变选定条件和预测器。数值是当前协议下的条件响应，不是永久模型特征。","","|q|测量|验证比例|95%区间|","|---|---|---|---|"]
    for r in contrasts:lines.append(f"|{r['q']}|{r['measure']}|{r['estimate']:.3f}|[{r['lo']:.3f},{r['hi']:.3f}]|")
    (folder/"个体响应阶段报告.md").write_text("\n".join(lines),encoding="utf-8")
    sequence_rows=[]
    for stage in ["sequence_calibration","sequence_validation"]:
        expected=config[stage]["reps"]*len(selection["network_qs"])*8
        for run in read_sequences(model,stage,expected):
            for event in run["events"]:
                sequence_rows.append({k:run[k] for k in ["stage","replicate","q","private_correct","schedule","treated"]} |
                                     {k:event[k] for k in ["round","wrong","changed_private","confidence"]})
    sf=pd.DataFrame(sequence_rows);sf.to_csv(folder/"sequence_decisions.csv",index=False)
    paired=sf.pivot(index=["stage","replicate","q","private_correct","schedule","round"],columns="treated",values="wrong").reset_index()
    paired["delta"]=paired[True].astype(float)-paired[False].astype(float)
    sequence_summary=[]
    for key,g in paired.groupby(["stage","q","private_correct","schedule","round"]):
        sequence_summary.append(dict(zip(["stage","q","private_correct","schedule","round"],key)) |
                                {"control_wrong":g[False].astype(float).mean(),"treated_wrong":g[True].astype(float).mean(),
                                 **mean_ci(g.groupby("replicate").delta.mean(),f"sequence-v3-{model}-{key}")})
    pd.DataFrame(sequence_summary).to_csv(folder/"sequence_paired_summary.csv",index=False)


def network_report(model,config,selection,predictions):
    rows=[];observed=[];audit={"preperiod_mismatches":0,"no_relay_indirect_mismatches":0,"outside_window":0,"paired_runs":0}
    for spec in network_specs(config,selection["network_qs"]):
        for schedule in config["private_schedules"]:
            folder=OUTPUT/"networks"/model/f"q{spec.reliability:.2f}"/schedule/spec.block_id
            control=json.loads((folder/"control.json").read_bytes())
            ci={(e["round"],e["node"]):e for e in control["events"]}
            base={"replicate":spec.replicate,"topology":spec.topology,"q":spec.reliability,"schedule":schedule}
            private_error=float(np.mean([x!=control["task"]["truth"] for x in control["task"]["private"]]))
            def observed_row(run,arm):
                return {**base,"arm":arm,"tail_error":run["metrics"]["tail_error"],"final_error":run["metrics"]["final_error"],
                        "sustained":run["metrics"]["sustained_failure"],"final_correct_majority":run["metrics"]["final_error"]<.5,
                        "final_tie":run["metrics"]["final_error"]==.5,"private_only_error":private_error,"private_majority_correct":private_error<.5}
            observed.append(observed_row(control,"control"))
            for arm in ["treated","no_relay_treated"]:
                treated=json.loads((folder/(arm+".json")).read_bytes());audit["paired_runs"]+=1
                observed.append(observed_row(treated,arm))
                for e in treated["events"]:
                    c=ci[e["round"],e["node"]]
                    if e["round"]<spec.onset and e["choice"]!=c["choice"]:audit["preperiod_mismatches"]+=1
                    if e["direct_injected_exposure"] and not spec.onset<=e["round"]<spec.onset+spec.duration:audit["outside_window"]+=1
                    if arm=="no_relay_treated" and not e["seed_node"] and (e["distance_from_seeds"] is None or e["distance_from_seeds"]>=2) and e["choice"]!=c["choice"]:audit["no_relay_indirect_mismatches"]+=1
                for t in range(spec.rounds):
                    ce=[e for e in control["events"] if e["round"]==t and not e["seed_node"]]
                    te=[e for e in treated["events"] if e["round"]==t and not e["seed_node"]]
                    rows.append({"replicate":spec.replicate,"topology":spec.topology,"q":spec.reliability,"schedule":schedule,"arm":arm,"round":t,
                                 "control_error":float(np.mean([e["wrong"] for e in ce])),"treated_error":float(np.mean([e["wrong"] for e in te])),
                                 "harm":float(np.mean([e["wrong"] and not c["wrong"] for c,e in zip(ce,te)])),
                                 "help":float(np.mean([c["wrong"] and not e["wrong"] for c,e in zip(ce,te)]))})
    if any(audit[k] for k in ["preperiod_mismatches","no_relay_indirect_mismatches","outside_window"]):raise RuntimeError(f"V3 integrity check failed: {audit}")
    folder=OUTPUT/"models"/model;df=pd.DataFrame(rows);df["delta"]=df.treated_error-df.control_error
    df.to_csv(folder/"network_round_pairs.csv",index=False)
    summaries=[]
    for (q,schedule,arm),g in df[df['round']>=6].groupby(["q","schedule","arm"]):
        arr=g.groupby("replicate").delta.mean()
        summaries.append({"q":q,"schedule":schedule,"arm":arm,"control_error":g.control_error.mean(),"treated_error":g.treated_error.mean(),
                          "harm":g.harm.mean(),"help":g.help.mean(),**mean_ci(arr,f"v3-{model}-{q}-{schedule}-{arm}"),"p_signflip":sign_flip_p(arr,f"v3-{model}-{q}-{schedule}-{arm}")})
    result=pd.DataFrame(summaries);result["p_holm_within_model_all_conditions"]=holm(result.p_signflip.to_numpy());result.to_csv(folder/"network_tail_effects.csv",index=False)
    interaction=[]
    for (q,arm),g in df[df['round']>=6].groupby(["q","arm"]):
        wide=g.groupby(["replicate","schedule"]).delta.mean().unstack()
        arr=wide.initial_only-wide.persistent
        interaction.append({"q":q,"arm":arm,"contrast":"initial_only_effect_minus_persistent_effect",**mean_ci(arr,f"v3-interaction-{model}-{q}-{arm}"),"p_signflip":sign_flip_p(arr,f"v3-interaction-{model}-{q}-{arm}")})
    interaction=pd.DataFrame(interaction);interaction["p_holm_within_model"]=holm(interaction.p_signflip.to_numpy());interaction.to_csv(folder/"private_schedule_interaction.csv",index=False)
    atomic_json(folder/"network_integrity.json",audit)
    # The numerical forward prediction targets ALL-agent error, not nonseed error.
    prediction_rows=[]
    for x in predictions["simulations"]:
        prediction_rows.append({k:x[k] for k in ["replicate","topology","q","schedule","arm","tail_error"]})
    keys=["replicate","topology","q","schedule","arm"]
    forecast=pd.DataFrame(prediction_rows).groupby(keys).tail_error.mean().rename("predicted_tail_error").reset_index()
    forecast.to_csv(folder/"frozen_forward_predictions.csv",index=False)
    actual=pd.DataFrame(observed);actual.to_csv(folder/"observed_network_runs.csv",index=False)
    function_rows=[]
    for (q,schedule),g in actual.groupby(["q","schedule"]):
        c=g[g.arm=="control"].copy();c["learning_gain"]=c.private_only_error-c.tail_error
        gain=c.groupby("replicate").learning_gain.mean()
        function_rows.append({"q":q,"schedule":schedule,"measure":"normal_learning_gain_vs_private_only",**mean_ci(gain,f"v3-normal-gain-{model}-{q}-{schedule}")})
        control_sustained=c.set_index(["replicate","topology"]).sustained
        for arm in ["treated","no_relay_treated"]:
            tr=g[g.arm==arm].set_index(["replicate","topology"])
            newly=tr.sustained & ~control_sustained
            function_rows.append({"q":q,"schedule":schedule,"measure":arm+"_new_sustained_task_rate",**mean_ci(newly.groupby("replicate").max().astype(float),f"v3-sustained-{model}-{q}-{schedule}-{arm}"),
                                  "caution":"Zero-event bootstrap is not a population risk bound."})
    pd.DataFrame(function_rows).to_csv(folder/"collective_function_and_learning.csv",index=False)
    evaluated=actual.merge(forecast,on=keys,validate="one_to_one")
    if len(evaluated)!=len(actual):raise RuntimeError("Missing pre-run predictions")
    evaluated["absolute_error"]=(evaluated.tail_error-evaluated.predicted_tail_error).abs()
    evaluated["squared_error"]=(evaluated.tail_error-evaluated.predicted_tail_error)**2
    evaluated["private_only_absolute_error"]=(evaluated.tail_error-evaluated.private_only_error).abs()
    evaluated["improvement_vs_private_only"]=evaluated.private_only_absolute_error-evaluated.absolute_error
    evaluated.to_csv(folder/"forward_prediction_evaluation.csv",index=False)
    evaluations=[]
    for (q,schedule,arm),g in evaluated.groupby(["q","schedule","arm"]):
        arr=g.groupby("replicate").improvement_vs_private_only.mean()
        evaluations.append({"q":q,"schedule":schedule,"arm":arm,"mae":g.absolute_error.mean(),"rmse":float(np.sqrt(g.squared_error.mean())),
                            "private_only_mae":g.private_only_absolute_error.mean(),**mean_ci(arr,f"v3-prediction-{model}-{q}-{schedule}-{arm}")})
    pd.DataFrame(evaluations).to_csv(folder/"forward_prediction_summary.csv",index=False)
    lines=[f"# {model} v3群体阶段结果","", "按独立任务聚类；所有条件完整收齐后生成。共同条件与校准条件分列。", "",result.to_csv(index=False),"", "私人证据可用性×扰动的配对交互：",interaction.to_csv(index=False),"",
           "单Agent预测器只在校准数据拟合、在验证和网络开始前冻结；状态代理不能再现全部自然语言变化。完整原始消息与响应均保留。"]
    (folder/"群体实验阶段报告.md").write_text("\n".join(lines),encoding="utf-8")


async def worker(model_name,config):
    frozen=check_frozen();models=frozen["models"]
    model=models[model_name];folder=OUTPUT/"models"/model_name;folder.mkdir(parents=True,exist_ok=True)
    complete=folder/"complete.json"
    if complete.exists():
        if json.loads(complete.read_bytes())["design_hash"]!=frozen["hash"]:raise RuntimeError("Completion belongs to another design")
        return
    client=LLMClient(model,OUTPUT/"cache"/(model_name+".sqlite"),config["client_concurrency"])
    def state(stage,**more):atomic_json(folder/"status.json",{"model":model_name,"stage":stage,"updated":time.time(),**more})
    try:
        info=await client.inspect()
        import platform
        from importlib.metadata import version
        atomic_json(folder/"environment.json",{"model_endpoint":info,"python":platform.python_version(),
                    "packages":{name:version(name) for name in ["numpy","pandas","scikit-learn","httpx","networkx"]},"inspected":time.time()})
        state("calibration")
        await run_profile(client,config,"calibration")
        records=read_profile(model_name,"calibration",config["calibration"]["reps"]*config["profile_records_per_task"])
        selection=select_condition(records,config);write_once(folder/"selection_from_calibration.json",selection)
        state("sequence_calibration",selection=selection)
        await run_sequence(client,config,"sequence_calibration",selection["network_qs"])
        expected=config["sequence_calibration"]["reps"]*len(selection["network_qs"])*8
        sequences=read_sequences(model_name,"sequence_calibration",expected)
        predictor=fit_predictor(records,sequences)
        state("freeze_predictions",selection=selection)
        predictions=freeze_predictions(model_name,config,selection,predictor)
        state("independent_validation",selection=selection)
        await run_profile(client,config,"validation")
        await run_sequence(client,config,"sequence_validation",selection["network_qs"])
        read_sequences(model_name,"sequence_validation",config["sequence_validation"]["reps"]*len(selection["network_qs"])*8)
        profile_report(model_name,config,selection)
        if selection["label"]=="measurement_review_required":
            state("measurement_review_required",selection=selection)
            raise RuntimeError("Private-report comprehension gate failed; profile retained, no network calls launched")
        state("network_confirmation",selection=selection)
        await run_network(client,config,selection)
        network_report(model_name,config,selection,predictions)
        atomic_json(complete,{"model":model_name,"design_hash":frozen["hash"],"finished":time.time(),"selection":selection,"actual_calls_this_worker":client.actual_calls,"cache_hits_this_worker":client.cache_hits})
        state("complete",selection=selection)
    finally:
        await client.close()


def main():
    p=argparse.ArgumentParser();p.add_argument("--model",required=True);args=p.parse_args()
    asyncio.run(worker(args.model,load_config()))


if __name__=="__main__":main()

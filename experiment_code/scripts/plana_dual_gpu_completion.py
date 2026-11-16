"""Complete Plan A's fixed experiment matrix on both GPUs, without changing prompts.

User authorization, 2026-11-16: both GPUs may be used. Existing protocol, model
fingerprints, seeds, precision and sampling parameters remain unchanged. Batch
scheduling/device placement change; this is recorded in the campaign manifest.
Run from the original server project, as `python -m scripts.plana_dual_gpu_completion`.
"""
from __future__ import annotations

import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "plana_completion_20261116"
PY = "/home/vllm/miniconda3/envs/vllm/bin/python3.10"
VLLM = "/home/vllm/miniconda3/envs/vllm/bin/vllm"
Q7, Q30, Q14 = "Qwen2.5-7B-Instruct", "Qwen3-30B-A3B-GPTQ-Int4", "Qwen3-14B"
GLM, LL = "GLM-4-9B-0414", "Llama-3.1-8B-Instruct"
MODELS = [Q7, Q30, Q14, GLM, LL]
PORTS = {Q7: 18241, Q30: 18242, Q14: 18240, GLM: 18245, LL: 18243}
LOCK = threading.Lock()


def write_json(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


def log(event, **kw):
    with LOCK:
        print(json.dumps({"event": event, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **kw}), flush=True)


def get_json(url):
    with urllib.request.urlopen(url, timeout=10) as r:
        return json.load(r)


def services(port):
    found = []
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            cmd = (p / "cmdline").read_bytes().rstrip(b"\0").decode().split("\0")
            if "serve" in cmd and "--port" in cmd and cmd[cmd.index("--port") + 1] == str(port) and any("vllm" in a for a in cmd[:3]):
                found.append((int(p.name), cmd))
        except (OSError, UnicodeError, ValueError, IndexError):
            continue
    return found


def stop_service(model):
    for pid, cmd in services(PORTS[model]):
        if "--served-model-name" not in cmd or cmd[cmd.index("--served-model-name") + 1] != model:
            raise RuntimeError(f"Refusing to stop unexpected service on {PORTS[model]}")
        # Stop only after this campaign's workers have exited, and no requests remain.
        import re
        with urllib.request.urlopen(f"http://127.0.0.1:{PORTS[model]}/metrics", timeout=10) as r:
            metrics = r.read().decode()
        vals = re.findall(r'^vllm:num_requests_(?:running|waiting)\{[^\n]*\}\s+([\d.]+)', metrics, re.M)
        if not vals or any(float(v) for v in vals):
            raise RuntimeError(f"Service {model} not verifiably idle")
        os.kill(pid, signal.SIGTERM)
        for _ in range(90):
            if not services(PORTS[model]):
                break
            time.sleep(2)
        else:
            raise RuntimeError(f"Service {model} did not stop")
        log("service_stopped", model=model, pid=pid)
    time.sleep(8)


def service_command(model):
    if model in (Q7, Q30, Q14):
        plan = json.loads((ROOT / "results/environment/single_gpu_service_plan.json").read_text())
        return list(plan[model]["command"])
    path = "/mnt/models/models/" + ("Meta-Llama-3.1-8B-Instruct" if model == LL else model)
    return [PY, VLLM, "serve", path, "--served-model-name", model,
            "--host", "127.0.0.1", "--port", str(PORTS[model]), "--dtype", "bfloat16",
            "--gpu-memory-utilization", "0.90", "--max-model-len", "2048",
            "--max-num-seqs", "128", "--max-num-batched-tokens", "8192",
            "--enable-prefix-caching", "--generation-config", "vllm", "--seed", "20261110",
            "--structured-outputs-config", '{"backend": "xgrammar", "disable_any_whitespace": true}']


def start_service(model, gpu):
    if services(PORTS[model]):
        info = get_json(f"http://127.0.0.1:{PORTS[model]}/v1/models")
        if not any(x["id"] == model for x in info["data"]):
            raise RuntimeError(f"Model mismatch for {model}")
        log("service_reused", model=model, gpu=gpu)
        return
    cmd = service_command(model)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    with (OUT / f"service_{model}.log").open("ab") as f:
        proc = subprocess.Popen(cmd, cwd="/home/vllm", env=env, stdin=subprocess.DEVNULL,
                                stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
    log("service_started", model=model, gpu=gpu, pid=proc.pid, command=cmd)
    for _ in range(180):
        if proc.poll() is not None:
            raise RuntimeError(f"Service {model} exited with {proc.returncode}")
        try:
            data = get_json(f"http://127.0.0.1:{PORTS[model]}/v1/models")
            if any(x["id"] == model for x in data["data"]):
                log("service_ready", model=model, gpu=gpu)
                return
        except Exception:
            pass
        time.sleep(5)
    raise RuntimeError(f"Service {model} readiness timeout")


def jobs_for(model):
    jobs = []
    if model in (Q7, Q30):
        jobs.append(("memory", "society.structural_v4", ["--schedules", "init_fullmem"]))
    if model in (GLM, LL):
        jobs.append(("dose", "society.structural_v4", ["--schedules", "persist_seed6"]))
    jobs += [
        ("nl_initial", "society.semantic_v5", ["--schedules", "initial_only", "--arms", "control,treated,treated_neutral,no_relay_treated"]),
        ("nl_persistent", "society.semantic_v5", ["--schedules", "persistent", "--arms", "control,treated,no_relay_treated"]),
    ]
    return [(f"{model}_{name}", module, args) for name, module, args in jobs]


def run_job(model, name, module, args):
    env = dict(os.environ, REPAIR_DANGLING_CITE_MODELS=LL, PYTHONUNBUFFERED="1")
    cmd = [PY, "-u", "-m", module, "--model", model, *args, "--concurrency", "48", "--blocks", "8"]
    for attempt in range(1, 5):
        with (OUT / f"{name}.log").open("ab") as f:
            proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                    stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        log("worker_started", job=name, pid=proc.pid, attempt=attempt, command=cmd)
        code = proc.wait()
        log("worker_exit", job=name, code=code, attempt=attempt)
        if code == 0:
            return
        time.sleep(10 * attempt)
    raise RuntimeError(f"Job {name} failed four attempts; see its log")


def run_models(models, gpu):
    for model in models:
        start_service(model, gpu)
    errors = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        fs = [pool.submit(run_job, m, *job) for m in models for job in jobs_for(m)]
        for f in concurrent.futures.as_completed(fs):
            try:
                f.result()
            except Exception as exc:
                errors.append(str(exc))
                log("job_error", gpu=gpu, error=str(exc))
    if errors:
        raise RuntimeError("; ".join(errors))


def gpu_lane(gpu):
    first, second = ([Q7, Q30], GLM) if gpu == 0 else ([Q14], LL)
    run_models(first, gpu)
    for model in first:
        stop_service(model)
    run_models([second], gpu)
    log("gpu_lane_complete", gpu=gpu)


def audit():
    from society import structural_v4 as v4, semantic_v5 as v5
    specs = list(v5.specs(24))
    result = []
    for model in MODELS:
        for protocol, schedule, arm in [
            *[("structural_v4", s, a) for s, aa in (("init_fullmem", ("control", "treated")), ("persist_seed6", ("treated",))) for a in aa],
            *[("semantic_v5", s, a) for s in ("initial_only", "persistent") for a in ("control", "treated", "no_relay_treated")],
            ("semantic_v5", "initial_only", "treated_neutral"),
        ]:
            good, missing, invalid = 0, [], []
            for spec in specs:
                p = ROOT / "results" / protocol / "networks" / model / schedule / spec.block_id / (arm + ".json")
                if not p.exists():
                    missing.append(spec.block_id)
                    continue
                try:
                    d = json.loads(p.read_bytes())
                    assert d["model"] == model and d["schedule"] == schedule and d["arm"] == arm
                    assert d["block_id"] == spec.block_id
                    assert len(d["events"]) == 216
                    assert {(e["round"], e["node"]) for e in d["events"]} == {(r,n) for r in range(9) for n in range(24)}
                    assert all(e["choice"] in ("A", "B") for e in d["events"])
                    good += 1
                except Exception as exc:
                    invalid.append({"path": str(p), "error": repr(exc)})
            result.append({"model": model, "protocol": protocol, "schedule": schedule, "arm": arm,
                           "complete": good, "required": 48, "missing": missing, "invalid": invalid})
    write_json(OUT / "coverage.json", result)
    return all(r["complete"] == 48 for r in result)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "campaign.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        sources = {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in
                   ("society/client.py", "society/structural_v4.py", "society/semantic_v5.py", "configs/structural_v4.json", "configs/models.json")}
        write_json(OUT / "manifest.json", {"started": time.time(), "pid": os.getpid(), "gpus": [0, 1],
                   "authorization": "2026-11-16 user explicitly permits both GPUs at full load",
                   "sample": "24 fixed tasks x WS/ER x 24 agents x 9 rounds; no outcome-dependent stopping",
                   "note": "Original prompts, model precision, fingerprints and sampling seeds retained; GPU placement and batching change.",
                   "jobs": {m: jobs_for(m) for m in MODELS}, "source_sha256": sources})
        log("campaign_started", pid=os.getpid(), gpus=[0, 1])
        audit()
        (OUT / "coverage.before.json").write_bytes((OUT / "coverage.json").read_bytes())
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            lanes = [pool.submit(gpu_lane, gpu) for gpu in (0, 1)]
            errors = []
            for f in concurrent.futures.as_completed(lanes):
                try:
                    f.result()
                except Exception as exc:
                    errors.append(repr(exc))
                    log("lane_error", error=repr(exc))
        complete = audit()
        status = {"finished": time.time(), "coverage_complete": complete, "errors": errors}
        if complete and not errors:
            with (OUT / "extraction.log").open("ab") as f:
                code = subprocess.call([PY, "-u", "-m", "society.structural_extract", "--workers", "12"], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
            status["extraction_code"] = code
            status["parquet"] = str(ROOT / "results/structural_v4/analysis/agent_rounds.parquet")
        write_json(OUT / "completion.json", status)
        log("campaign_finished", **status)
        if not complete or errors or status.get("extraction_code") != 0:
            raise SystemExit(1)


if __name__ == "__main__":
    main()

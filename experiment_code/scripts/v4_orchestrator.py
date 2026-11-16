"""GPU0-only sequential orchestrator for the v4 structural study.

Policy (user instruction): only physical GPU0 is used; GPU1 is never touched.
Stages are resumable via a state file. Each service is stopped only when idle and
only by exact PID+cmdline match. Workers are resumable (existing runs are skipped).
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = ROOT / "results" / "environment"
OUT = ROOT / "results" / "structural_v4"
STATE = OUT / "orchestrator_state.json"
PY = "/home/vllm/miniconda3/envs/vllm/bin/python3.10"
VLLM = "/home/vllm/miniconda3/envs/vllm/bin/vllm"
PLAN = json.loads((ENV / "single_gpu_service_plan.json").read_text())


def log(event, **kw):
    print(json.dumps({"event": event, "t": time.strftime("%m-%d %H:%M:%S"), **kw}, ensure_ascii=False), flush=True)


def state():
    return json.loads(STATE.read_text()) if STATE.exists() else {"done": []}


def mark(stage):
    s = state()
    s["done"].append(stage)
    s["updated"] = time.ctime()
    STATE.write_text(json.dumps(s, indent=1))


def get(url):
    with urllib.request.urlopen(url, timeout=10) as r:
        return r.read().decode()


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def services_on_port(port):
    found = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            cmd = proc.joinpath("cmdline").read_bytes().rstrip(b"\0").decode().split("\0")
        except Exception:
            continue
        if "serve" in cmd and "--port" in cmd and cmd[cmd.index("--port") + 1] == str(port) and any("vllm" in c for c in cmd[:3]):
            found.append((int(proc.name), cmd))
    return found


def idle(port):
    try:
        text = get(f"http://127.0.0.1:{port}/metrics")
    except Exception:
        return True
    import re
    vals = re.findall(r'^vllm:num_requests_(?:running|waiting)\{[^\n]*\}\s+([\d.]+)', text, re.M)
    return all(float(x) == 0 for x in vals)


def stop_port(port):
    for pid, cmd in services_on_port(port):
        while not idle(port):
            time.sleep(20)
        os.kill(pid, signal.SIGTERM)
        deadline = time.time() + 180
        while alive(pid) and time.time() < deadline:
            time.sleep(2)
        if alive(pid):
            raise RuntimeError(f"service {pid} on {port} did not exit")
        log("service_stopped", port=port, pid=pid)
    time.sleep(10)


def start(name, command, port):
    if services_on_port(port):
        log("service_already_running", port=port)
    else:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
        logf = (ENV / f"{name}_v4_gpu0.service.log").open("ab")
        subprocess.Popen(command, cwd="/home/vllm", env=env, stdin=subprocess.DEVNULL, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
        log("service_started", name=name, port=port)
    deadline = time.time() + 900
    while time.time() < deadline:
        try:
            data = json.loads(get(f"http://127.0.0.1:{port}/v1/models"))
            if any(m["id"] == name for m in data["data"]):
                log("service_ready", name=name)
                return
        except Exception:
            pass
        time.sleep(5)
    raise RuntimeError(f"{name} not ready")


def plan_command(model, util=None):
    cmd = list(PLAN[model]["command"])
    if util is not None:
        cmd[cmd.index("--gpu-memory-utilization") + 1] = str(util)
    return cmd


def new_command(path, name, port, util):
    return [PY, VLLM, "serve", path, "--served-model-name", name, "--host", "127.0.0.1", "--port", str(port),
            "--dtype", "bfloat16", "--gpu-memory-utilization", str(util), "--max-model-len", "2048",
            "--max-num-seqs", "64", "--max-num-batched-tokens", "4096", "--enable-prefix-caching",
            "--generation-config", "vllm", "--seed", "20261110",
            "--structured-outputs-config", '{"backend": "xgrammar", "disable_any_whitespace": true}']


def running(module, model, schedules=None):
    """PIDs of live worker processes for (module, model[, schedules arg])."""
    pids = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            cmd = proc.joinpath("cmdline").read_bytes().rstrip(b"\0").decode().split("\0")
        except Exception:
            continue
        if module in cmd and "--model" in cmd and cmd[cmd.index("--model") + 1] == model:
            if schedules is None or ("--schedules" in cmd and cmd[cmd.index("--schedules") + 1] == schedules):
                pids.append(int(proc.name))
    return pids


def kill_job(pid):
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass


def workers(jobs, deadline=None):
    """jobs: list of (name, module, model, args). Adopt live identical workers, else launch; retry failures.
    At `deadline` (epoch seconds) all remaining workers are terminated (outputs are atomic and resumable)."""
    procs, tries = {}, {j[0]: 0 for j in jobs}
    spec = {j[0]: j for j in jobs}

    def sched_of(args):
        return args[args.index("--schedules") + 1] if "--schedules" in args else None

    def launch(name):
        _, module, model, args = spec[name]
        tries[name] += 1
        logf = (ROOT / "results" / "jobs" / f"{name}.log").open("ab")
        procs[name] = subprocess.Popen([sys.executable, "-u", "-m", module, "--model", model, *args],
                                       cwd=ROOT, stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        log("worker_started", job=name, attempt=tries[name])
    adopted = {}
    for name, (_, module, model, args) in spec.items():
        live = running(module, model, sched_of(args))
        if live:
            adopted[name] = live[0]
            log("worker_adopted", job=name, pid=live[0])
        else:
            launch(name)
    while procs or adopted:
        time.sleep(30)
        if deadline and time.time() > deadline:
            for name, pid in adopted.items():
                kill_job(pid)
            for name, p in procs.items():
                p.terminate()
            for name, p in procs.items():
                try:
                    p.wait(120)
                except Exception:
                    p.kill()
            log("stage_deadline_reached", unfinished=sorted(list(procs) + list(adopted)))
            return
        for name, pid in list(adopted.items()):
            if not alive(pid) or Path(f"/proc/{pid}/stat").read_text().split()[2] == "Z":
                del adopted[name]
                log("adopted_worker_gone", job=name)
                launch(name)  # resumable verification pass; exits quickly when complete
        for name, p in list(procs.items()):
            code = p.poll()
            if code is None:
                continue
            del procs[name]
            log("worker_exit", job=name, code=code)
            if code and tries[name] < 4:
                time.sleep(60)
                launch(name)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    s = state()
    if "wait_mistral" not in s["done"]:
        # Mistral v2 was intentionally paused (see results/environment/mistral_v2_paused_20261115.json).
        stop_port(18244)
        mark("wait_mistral")
    V4, V5 = "society.structural_v4", "society.semantic_v5"
    sched_qwen = ["--schedules", "anchor25,anchor50,in_history", "--concurrency", "48", "--blocks", "8"]
    dose = ["--schedules", "init_seed6,init_seed1,persist_seed6", "--concurrency", "24", "--blocks", "4"]
    sched_new = ["--profile", "--schedules", "initial_only,persistent,anchor25,anchor50,in_history", "--concurrency", "48", "--blocks", "8"]
    dose_new = ["--schedules", "init_seed6", "--concurrency", "16", "--blocks", "4"]
    q7, q30, q14, ll, glm = "Qwen2.5-7B-Instruct", "Qwen3-30B-A3B-GPTQ-Int4", "Qwen3-14B", "Llama-3.1-8B-Instruct", "GLM-4-9B-0414"
    if "qwen7_30" not in s["done"]:
        start("Qwen2.5-7B-Instruct", plan_command("Qwen2.5-7B-Instruct"), 18241)
        start("Qwen3-30B-A3B-GPTQ-Int4", plan_command("Qwen3-30B-A3B-GPTQ-Int4"), 18242)
        workers([("v4_" + q7, V4, q7, sched_qwen), ("v4_" + q30, V4, q30, sched_qwen),
                 ("v4dose_" + q7, V4, q7, dose), ("v4dose_" + q30, V4, q30, dose),
                 ("v5_Qwen3-30B", V5, q30, ["--concurrency", "24", "--blocks", "4"]),
                 ("v5_Qwen2.5-7B", V5, q7, ["--arms", "control,treated", "--concurrency", "24", "--blocks", "4"])])
        stop_port(18241)
        stop_port(18242)
        mark("qwen7_30")
    import datetime as _dt
    def at(hhmm):  # next occurrence of local clock time (2026-11-16 morning)
        h, m = map(int, hhmm.split(":"))
        now = _dt.datetime.now()
        d = now.replace(hour=h, minute=m, second=0, microsecond=0)
        if d < now:
            d += _dt.timedelta(days=1)
        return d.timestamp()
    if "qwen14" not in s["done"]:
        # 2026-11-15 23:20 user request: add memory ablation + v5 wording decomposition; finish everything by 09:00.
        start("Qwen3-14B", plan_command("Qwen3-14B"), 18240)
        workers([("v4_" + q14, V4, q14, sched_qwen), ("v4dose_" + q14, V4, q14, dose),
                 ("v4mem_" + q14, V4, q14, ["--schedules", "init_fullmem", "--concurrency", "32", "--blocks", "6"]),
                 ("v5w_" + q14, V5, q14, ["--schedules", "initial_only", "--arms", "control,treated,treated_authority,treated_neutral",
                                         "--concurrency", "32", "--blocks", "6"])],
                # v5 persistent for 14B dropped 23:25 to fit the 09:00 deadline (partial files kept, resumable)
                deadline=at("02:00"))
        stop_port(18240)
        mark("qwen14")
    if "llama_glm_core" not in s["done"]:
        up = []
        for name, path, port, util in [(ll, "/mnt/models/models/Meta-Llama-3.1-8B-Instruct", 18243, 0.42),
                                       (glm, "/mnt/models/models/GLM-4-9B-0414", 18245, 0.48)]:
            try:
                start(name, new_command(path, name, port, util), port)
                up.append(name)
            except Exception as exc:  # a failed family must not block the restore stage
                log("service_failed", name=name, error=str(exc)[:300])
                stop_port(port)
        core = ["--schedules", "initial_only,persistent", "--concurrency", "48", "--blocks", "8"]
        mem = ["--schedules", "init_fullmem", "--concurrency", "24", "--blocks", "4"]
        prof = ["--profile", "--schedules", "", "--concurrency", "16"]
        jobs = [(k + m, V4, m, a) for m in up for k, a in (("v4core_", core), ("v4mem_", mem), ("v4prof_", prof))]
        # 01:25 addition (throughput ~30k calls/h/model leaves slack before 08:15): lower-priority extras.
        extra = ["--schedules", "init_seed6,in_history", "--concurrency", "16", "--blocks", "3"]
        v5x = ["--schedules", "initial_only", "--arms", "control,treated,treated_neutral", "--concurrency", "16", "--blocks", "3"]
        jobs += [(k + m, mod, m, a) for m in up for k, mod, a in (("v4extra_", V4, extra), ("v5x_", V5, v5x))]
        if jobs:
            workers(jobs, deadline=at("08:15"))
        stop_port(18243)
        stop_port(18245)
        mark("llama_glm_core")
    if "restore" not in s["done"]:
        start("Qwen2.5-7B-Instruct", plan_command("Qwen2.5-7B-Instruct"), 18241)
        start("Qwen3-30B-A3B-GPTQ-Int4", plan_command("Qwen3-30B-A3B-GPTQ-Int4"), 18242)
        mark("restore")
    log("orchestrator_complete")


if __name__ == "__main__":
    main()

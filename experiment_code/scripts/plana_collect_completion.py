"""Wait for the remote Plan A campaign, validate its data, and build local results.

The report is built in an isolated directory. Manuscript text is not changed by
this collector, because new findings need to be read before revising claims.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "plana_completion_20261116"
REMOTE = "/home/vllm/projects/local_disruptions_20261114"
PYREMOTE = "/home/vllm/miniconda3/envs/vllm/bin/python3.10"


def log(message):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), message, flush=True)


def status(value):
    value["updated"] = time.time()
    tmp = OUT / "collector_status.tmp"
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    tmp.replace(OUT / "collector_status.json")


def remote_status():
    code = (
        "import json,os;from pathlib import Path;"
        f"p=Path({REMOTE!r})/'results/plana_completion_20261116';"
        "m=json.loads((p/'manifest.json').read_text());"
        "c=p/'completion.json';"
        "print(json.dumps({'manifest':m,'done':json.loads(c.read_text()) if c.exists() else None,"
        "'alive':Path('/proc/'+str(m['pid'])).exists()}))"
    )
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "vllm-server",
                        PYREMOTE + " -c " + shlex.quote(code)], capture_output=True, text=True, timeout=45, check=True)
    return json.loads(r.stdout)


def download(remote, local):
    subprocess.run(["scp", "-q", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                    "vllm-server:" + remote, str(local)], check=True, timeout=300)


def collect(done):
    import pandas as pd

    if not done["coverage_complete"] or done.get("errors") or done.get("extraction_code") != 0:
        raise RuntimeError("Remote campaign did not pass all completion checks: " + json.dumps(done))
    for name in ("manifest.json", "completion.json", "coverage.json", "coverage.before.json", "campaign.log"):
        download(REMOTE + "/results/plana_completion_20261116/" + name, OUT / name)
    coverage = json.loads((OUT / "coverage.json").read_text())
    assert all(x["complete"] == x["required"] == 48 and not x["invalid"] and not x["missing"] for x in coverage)
    candidate = OUT / "agent_rounds_all.parquet"
    download(done["parquet"], candidate)
    old = pd.read_parquet(ROOT / "data/agent_rounds_all.parquet")
    new = pd.read_parquet(candidate)
    key = ["source", "model", "schedule", "block", "arm", "node", "round"]
    assert not new.duplicated(key).any(), "Duplicate agent decisions in collected data"
    old_index = old.set_index(key).sort_index()
    new_index = new.set_index(key).sort_index()
    assert old_index.index.isin(new_index.index).all(), "Old observations missing in collected data"
    # The extractor reconstructs the old records identically; verify all fields.
    pd.testing.assert_frame_equal(old_index, new_index.loc[old_index.index, old_index.columns], check_dtype=False)
    project = OUT / "report_project"
    for name in ("data", "analysis", "paper/scripts", "results/tables", "results/abm_frozen"):
        (project / name).mkdir(parents=True, exist_ok=True)
    shutil.copy2(candidate, project / "data/agent_rounds_all.parquet")
    shutil.copytree(ROOT / "data/profiles_v3", project / "data/profiles_v3", dirs_exist_ok=True)
    shutil.copytree(ROOT / "results/tables", project / "results/tables", dirs_exist_ok=True)
    shutil.copytree(ROOT / "results/abm_frozen", project / "results/abm_frozen", dirs_exist_ok=True)
    shutil.copytree(ROOT / "paper/scripts", project / "paper/scripts", dirs_exist_ok=True)
    shutil.copy2(ROOT / "analysis/v4_analysis.py", project / "analysis/v4_analysis.py")
    scripts = ["analysis/v4_analysis.py", "paper/scripts/compute_numbers.py", "paper/scripts/make_tables_planA.py",
               "paper/scripts/make_figures_planA.py"]
    for script in scripts:
        log("Generating " + script)
        with (OUT / (Path(script).stem + ".log")).open("w", encoding="utf-8") as f:
            subprocess.run([sys.executable, str(project / script)], cwd=project, stdout=f, stderr=subprocess.STDOUT, check=True)
    tables = project / "paper/planA_shock_vs_persistence/tables"
    assert all(r"\na" not in p.read_text(encoding="utf-8") for p in tables.glob("*.tex")), "Table gaps remain"
    n = json.loads((project / "paper/scripts/numbers.json").read_text(encoding="utf-8"))
    rows = []
    for k, e in n["effects"].items():
        model, group, schedule, arm = k.split("|")
        rows.append({"model": model, "group": group, "schedule": schedule, "arm": arm,
                     "n_tasks": e["n_tasks"], "delta_pp": e["tail"][0], "ci_lo": e["tail"][1], "ci_hi": e["tail"][2],
                     "shock_pp": e["peak3"][0], "control_tail_pct": e["ctrl_tail"]})
    pd.DataFrame(rows).to_csv(OUT / "effects_completed.csv", index=False, encoding="utf-8-sig")
    # Only publish the unified dataset after all analysis and table checks pass.
    backup = OUT / "agent_rounds_all.before.parquet"
    if not backup.exists():
        shutil.copy2(ROOT / "data/agent_rounds_all.parquet", backup)
    tmp = ROOT / "data/agent_rounds_all.completion.tmp"
    shutil.copy2(candidate, tmp)
    tmp.replace(ROOT / "data/agent_rounds_all.parquet")
    result = {"state": "complete", "old_rows": len(old), "new_rows": len(new), "added_rows": len(new)-len(old),
              "sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(), "report_project": str(project),
              "effects_csv": str(OUT / "effects_completed.csv"), "manuscript_needs_review": True}
    status(result)
    log(json.dumps(result))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    status({"state": "waiting_remote", "pid": __import__("os").getpid()})
    consecutive_errors = 0
    while True:
        try:
            remote = remote_status()
            consecutive_errors = 0
        except (subprocess.SubprocessError, ValueError, OSError) as exc:
            consecutive_errors += 1
            log(f"Connection retry {consecutive_errors}: {exc}")
            time.sleep(60)
            continue
        if remote["done"] is not None:
            status({"state": "collecting"})
            collect(remote["done"])
            return
        if not remote["alive"]:
            raise RuntimeError("Remote campaign exited without a completion record")
        time.sleep(60)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log(repr(exc))
        status({"state": "failed", "error": repr(exc)})
        raise

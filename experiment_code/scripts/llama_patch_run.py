"""20261116 04:40 Llama completion run after the dangling-cite repair (GPU0 only).

Temporarily frees GPU0 (stops Qwen2.5-7B / Qwen3-30B services that were just restored), runs the Llama
jobs that failed on the xgrammar `"]}` artifact, then restores both services. Hard deadline 07:50.
"""
import datetime as _dt
from scripts import v4_orchestrator as o

V4, V5 = "society.structural_v4", "society.semantic_v5"


def at(hhmm):
    h, m = map(int, hhmm.split(":"))
    now = _dt.datetime.now()
    d = now.replace(hour=h, minute=m, second=0, microsecond=0)
    return (d + _dt.timedelta(days=1) if d < now else d).timestamp()

LL = "Llama-3.1-8B-Instruct"


def main():
    o.log("llama_patch_begin")
    o.stop_port(18241)
    o.stop_port(18242)
    try:
        o.start(LL, o.new_command("/mnt/models/models/Meta-Llama-3.1-8B-Instruct", LL, 18243, 0.85), 18243)
        jobs = [("v4core_" + LL, V4, LL, ["--schedules", "persistent,initial_only", "--concurrency", "64", "--blocks", "8"]),
                ("v4mem_" + LL, V4, LL, ["--schedules", "init_fullmem", "--concurrency", "32", "--blocks", "4"]),
                ("v5x_" + LL, V5, LL, ["--schedules", "initial_only", "--arms", "control,treated,treated_neutral",
                                         "--concurrency", "24", "--blocks", "3"])]
        o.workers(jobs, deadline=at("07:50"))
    except Exception as exc:
        o.log("llama_patch_error", error=str(exc)[:300])
    finally:
        o.stop_port(18243)
        o.start("Qwen2.5-7B-Instruct", o.plan_command("Qwen2.5-7B-Instruct"), 18241)
        o.start("Qwen3-30B-A3B-GPTQ-Int4", o.plan_command("Qwen3-30B-A3B-GPTQ-Int4"), 18242)
        o.log("llama_patch_complete")


if __name__ == "__main__":
    main()

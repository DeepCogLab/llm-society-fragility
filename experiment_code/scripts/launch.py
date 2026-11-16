"""Detach one managed job, retaining command, pid, logs and explicit completion status."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def launch(name, module, arguments):
    root = Path(__file__).resolve().parents[1]
    jobs = root / "results" / "jobs"
    jobs.mkdir(parents=True, exist_ok=True)
    pidfile = jobs / (name + ".pid")
    if pidfile.exists():
        pid = int(pidfile.read_text())
        try:
            os.kill(pid, 0)
            raise RuntimeError(f"Job already active: {name}, pid={pid}")
        except ProcessLookupError:
            pass
    cmd = [sys.executable, "-u", "-m", module, *arguments]
    exitfile=jobs/(name+'.exit.json')
    if exitfile.exists():
        import time
        history=jobs/'history';history.mkdir(exist_ok=True)
        exitfile.rename(history/(name+'.'+str(time.time_ns())+'.exit.json'))
    (jobs / (name + ".command.json")).write_text(json.dumps(cmd,indent=2),encoding="utf-8")
    # A wrapper process writes exit status even when the experiment fails.
    wrapper = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", name]
    with (jobs / (name + ".log")).open("ab") as log, open(os.devnull,"rb") as stdin:
        process = subprocess.Popen(wrapper,cwd=root,stdout=log,stderr=subprocess.STDOUT,stdin=stdin,start_new_session=True)
    pidfile.write_text(str(process.pid))
    print(json.dumps({"job":name,"pid":process.pid,"log":str(jobs / (name + ".log"))}))


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--worker")
    p.add_argument("name",nargs="?")
    p.add_argument("module",nargs="?")
    p.add_argument("arguments",nargs=argparse.REMAINDER)
    args=p.parse_args()
    if args.worker:
        import time
        jobs=Path(__file__).resolve().parents[1]/"results"/"jobs"
        command=json.loads((jobs/(args.worker+".command.json")).read_text())
        start=time.time()
        code=subprocess.call(command)
        (jobs/(args.worker+".exit.json")).write_text(json.dumps({"exit_code":code,"started":start,"finished":time.time()}))
        sys.exit(code)
    else:
        launch(args.name,args.module,args.arguments)

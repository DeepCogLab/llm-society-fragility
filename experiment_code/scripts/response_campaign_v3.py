"""Durable v3 orchestration; never stops or rotates an existing model service."""
from __future__ import annotations

import argparse
import json
import os
import socket
import time
import urllib.request
from pathlib import Path

from scripts.launch import launch
from society.core import atomic_json
from society.response_v3 import OUTPUT, freeze, load_config, model_definitions


ROOT=Path(__file__).resolve().parents[1]


def name_for(model):
    return "response_v3_"+{"Qwen2.5-7B-Instruct":"qwen7","Qwen3-14B":"qwen14","Qwen3-30B-A3B-GPTQ-Int4":"qwen30"}[model]


def ready(model):
    try:
        with urllib.request.urlopen(model["url"]+"/models",timeout=5) as r:
            return any(x["id"]==model["id"] for x in json.load(r)["data"])
    except Exception:return False


def managed_alive(name):
    path=ROOT/"results/jobs"/(name+".pid")
    if not path.exists():return False
    pid=int(path.read_text())
    proc=Path(f"/proc/{pid}/cmdline")
    if not proc.exists():return False
    cmd=proc.read_bytes().rstrip(b"\0").decode().split("\0")
    # Refuse to interpret a recycled PID as our active worker.
    return len(cmd)>=4 and cmd[-2:]==["--worker",name] and Path(cmd[-3]).name=="launch.py"


def aggregate(config,frozen):
    from datetime import datetime,timezone,timedelta
    import pandas as pd
    from society.analyze import holm
    summaries=[];interactions=[];profiles=[]
    for model in config["models"]:
        folder=OUTPUT/"models"/model
        for filename,target in [("network_tail_effects.csv",summaries),("private_schedule_interaction.csv",interactions),("validation_rates.csv",profiles)]:
            d=pd.read_csv(folder/filename);d.insert(0,"model",model);target.append(d)
    summary=pd.concat(summaries,ignore_index=True);summary["p_holm_all_models_conditions"]=holm(summary.p_signflip.to_numpy())
    summary.to_csv(OUTPUT/"all_models_network_effects.csv",index=False)
    interaction=pd.concat(interactions,ignore_index=True);interaction["p_holm_all_models_conditions"]=holm(interaction.p_signflip.to_numpy())
    interaction.to_csv(OUTPUT/"all_models_private_schedule_interactions.csv",index=False)
    pd.concat(profiles,ignore_index=True).to_csv(OUTPUT/"all_models_validation_rates.csv",index=False)
    lines=["# v3：先测个体响应、再预测与验证群体过程","",f"生成：{datetime.now(timezone(timedelta(hours=8))).isoformat()}","",
           "三个Qwen模型按固定样本全部收齐；共同q=.7与各自校准条件分列。校准、验证、序列与网络使用互不重叠的新任务。",
           "该新阶段没有跨家族模型，不能外推家族规律；原Mistral跨家族研究独立保留。","",
           "主要机制比较为：仅初始化提供私人线索与每轮提供私人线索之间，匹配扰动效应的差异。该干预改变信息可用性，因此正常协作性能必须同时报告。",
           "个体采纳率、纠错率以及数值前向预测均在验证和网络开始前冻结；预测器没有用验证或网络结果调参。","",
           "完整比较：all_models_network_effects.csv；all_models_private_schedule_interactions.csv；all_models_validation_rates.csv。",
           "每模型报告与完整请求/响应位于models/、profiles/、sequences/、networks/和cache/。"]
    (OUTPUT/"研究报告_zh.md").write_text("\n".join(lines),encoding="utf-8")
    import sqlite3
    caches={}
    for model in config["models"]:
        path=(OUTPUT/"cache"/(model+'.sqlite')).resolve()
        connection=sqlite3.connect(f'file:{path}?mode=ro',uri=True)
        if connection.execute('pragma integrity_check').fetchone()[0]!='ok':raise RuntimeError('V3 cache integrity failure')
        caches[model]={"successful_unique_calls":connection.execute('select count(*) from calls').fetchone()[0],
                       "failure_attempts_retained":connection.execute('select count(*) from failures').fetchone()[0]}
        connection.close()
    atomic_json(OUTPUT/"cache_integrity.json",caches)
    atomic_json(OUTPUT/"complete.json",{"design_hash":frozen["hash"],"models":config["models"],"finished":time.time(),"all_models_complete":True})


def main():
    config=load_config();models=model_definitions(config);frozen=freeze(config,models)
    print(json.dumps({"event":"v3_design_frozen","hash":frozen["hash"],"models":config["models"]}),flush=True)
    announced=set()
    while True:
        status={};failed=[]
        for model in config["models"]:
            folder=OUTPUT/"models"/model;name=name_for(model)
            exitfile=ROOT/"results/jobs"/(name+".exit.json")
            if (folder/"complete.json").exists():
                if managed_alive(name):
                    status[model]={"stage":"closing_cache"};continue
                if not exitfile.exists():
                    status[model]={"stage":"awaiting_worker_exit_record"};continue
                result=json.loads(exitfile.read_bytes())
                if result['exit_code']!=0:
                    failed.append(model);status[model]={"stage":"failed","exit":result};continue
                if json.loads((folder/'complete.json').read_bytes())['design_hash']!=frozen['hash']:
                    raise RuntimeError('Worker completion design hash mismatch')
                status[model]={"stage":"complete"};continue
            if exitfile.exists() and not managed_alive(name):
                result=json.loads(exitfile.read_bytes())
                if result["exit_code"]!=0:
                    failed.append(model);status[model]={"stage":"failed","exit":result};continue
                raise RuntimeError(f"Worker exited0 without complete manifest: {model}")
            if managed_alive(name):
                path=folder/"status.json"
                status[model]=json.loads(path.read_bytes()) if path.exists() else {"stage":"starting"}
                continue
            if model=="Qwen3-14B":
                path=ROOT/"results/environment/gpu_rotation_state.json"
                rotation=json.loads(path.read_bytes()) if path.exists() else {}
                if rotation.get("status")!="restored":
                    status[model]={"stage":"waiting_for_existing_service_restoration"}
                    if model not in announced:
                        print(json.dumps({"event":"v3_waiting_for_model","model":model,"reason":"Existing authorized Mistral study owns GPU1; no interruption"}),flush=True);announced.add(model)
                    continue
            if not ready(models[model]):
                status[model]={"stage":"waiting_for_model_endpoint"};continue
            launch(name,"society.response_v3",["--model",model]);status[model]={"stage":"starting"}
        atomic_json(OUTPUT/"campaign_status.json",{"updated":time.time(),"design_hash":frozen["hash"],"models":status,"failed_models":failed})
        if all(x.get("stage")=="complete" for x in status.values()):
            aggregate(config,frozen);print(json.dumps({"event":"v3_campaign_complete"}),flush=True);return
        if failed and all(x.get("stage") in ["complete","failed"] for x in status.values()):
            raise RuntimeError(f"V3 workers failed; original responses retained: {failed}")
        time.sleep(15)


if __name__=="__main__":main()

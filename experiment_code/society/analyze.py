from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

from .core import atomic_json, seed_for

METRICS = ["nonseed_tail_error", "indirect_tail_error", "tail_error", "final_error", "sustained_failure", "final_majority_wrong", "final_correct_majority"]
GROUPS = ["phase", "model", "scenario", "topology", "n", "reliability", "fraction", "duration", "placement", "temperature", "wording", "rounds", "memory"]
# Wording is cycled by replicate. Do not split the main report into one-task cells.
REPORT_GROUPS = [x for x in GROUPS if x != "wording"]


def mean_ci(values, key="ci", draws=10000):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"sample_n": 0, "estimate": None, "lo": None, "hi": None}
    if len(x) == 1:
        return {"sample_n": 1, "estimate":float(x[0]), "lo":float("nan"), "hi":float("nan")}
    rng = np.random.default_rng(seed_for("bootstrap-v1", key))
    means = x[rng.integers(0, len(x), (draws, len(x)))].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return {"sample_n":len(x),"estimate":float(x.mean()),"lo":float(lo),"hi":float(hi)}


def sign_flip_p(values, key="perm"):
    x=np.asarray(values,dtype=float)
    x=x[np.isfinite(x)]
    # Zero pairs do not affect the randomization distribution.
    x=x[np.abs(x)>1e-14]
    if len(x)==0:
        return 1.0
    obs=abs(x.sum())
    if len(x)<=16:
        sums=np.asarray([np.dot(sign,x) for sign in itertools.product([-1,1],repeat=len(x))])
        return float(np.mean(np.abs(sums)>=obs-1e-12))
    rng=np.random.default_rng(seed_for("signflip-v1",key))
    sums=(rng.choice([-1,1],(20000,len(x)))*x).sum(axis=1)
    return float((1+np.sum(np.abs(sums)>=obs-1e-12))/(len(sums)+1))


def holm(pvalues):
    order=np.argsort(pvalues)
    adjusted=np.empty(len(order))
    running=0
    for i,j in enumerate(order):
        running=max(running,min(1,(len(order)-i)*pvalues[j]))
        adjusted[j]=running
    return adjusted


def binomial_ci(successes,total):
    if not total:
        return [None,None]
    return [float(beta.ppf(0.025,successes,total-successes+1)) if successes else 0.0,
            float(beta.ppf(0.975,successes+1,total-successes)) if successes<total else 1.0]


def collect(output:Path):
    runs=[]
    trajectories=[]
    audit={"runs":0,"events":0,"pair_preperiod_mismatches":[],"no_relay_sham_mismatches":[],"no_relay_indirect_mismatches":[],"invalid_citations":0,"injection_outside_window":[],"missing_pair_arms":[]}
    paths=sorted((output/"raw").glob("*/*/*/*.json"))
    raw_index={}
    for path in paths:
        r=json.loads(path.read_text(encoding="utf-8"))
        audit["runs"]+=1
        audit["events"]+=len(r["events"])
        spec=r["spec"]
        row={**spec,"block_id":r["block_id"],"model":r["model"],"family":r["family"],"mode":r["mode"],"treatment":r["treatment"],**{k:v for k,v in r["metrics"].items() if not isinstance(v,list)},"path":str(path)}
        objective=spec['scenario'] in ['cascade','asch']
        row['final_correct_majority']=float(r['metrics']['final_error']<0.5) if objective else np.nan
        row['final_majority_tie']=float(r['metrics']['final_error']==0.5) if objective else np.nan
        private_error=np.mean([x!=r['task']['truth'] for x in r['task']['private']])
        row['private_signal_majority_correct']=float(private_error<0.5) if spec['scenario']=='cascade' else np.nan
        row['private_signal_majority_tie']=float(private_error==0.5) if spec['scenario']=='cascade' else np.nan
        runs.append(row)
        raw_index[(r["model"],r["block_id"],r["mode"],r["treatment"])]=r
        for t,(err,ns,far) in enumerate(zip(r["metrics"]["error_trajectory"],r["metrics"]["nonseed_trajectory"],r["metrics"]["indirect_trajectory"])):
            trajectories.append({**{k:row[k] for k in GROUPS},"block_id":r["block_id"],"replicate":spec["replicate"],"mode":r["mode"],"treatment":r["treatment"],"round":t,"error":err,"nonseed_error":ns,"indirect_error":far})
        for e in r["events"]:
            audit["invalid_citations"]+=len(e["outgoing"]["invalid_cites"])
            if e["direct_injected_exposure"] and not spec["onset"] <= e["round"] < spec["onset"]+spec["duration"]:
                audit["injection_outside_window"].append([r["model"],r["block_id"],r["mode"],e["round"]])
    for key,r in raw_index.items():
        model,block,mode,treated=key
        if treated:
            control=raw_index.get((model,block,mode,False))
            if control is None:
                audit["missing_pair_arms"].append(key)
                continue
            # Frozen/provenance are active from t0; each has its OWN matched control.
            for a,b in zip(r["events"],control["events"]):
                if a["round"]<r["spec"]["onset"] and a!=b:
                    audit["pair_preperiod_mismatches"].append([model,block,mode,a["round"],a["node"]])
                distance=a['distance_from_seeds']
                if mode=='no_relay' and not a['seed_node'] and (distance is None or distance>=2) and a!=b:
                    audit['no_relay_indirect_mismatches'].append([model,block,a['round'],a['node']])
        if mode=="no_relay" and not treated:
            natural=raw_index.get((model,block,"natural",False))
            if natural and r["events"]!=natural["events"]:
                audit["no_relay_sham_mismatches"].append([model,block])
    return pd.DataFrame(runs),pd.DataFrame(trajectories),audit


def paired_tables(frame):
    if frame.empty:
        return pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
    pairs=[]
    lookup={(r["model"],r["block_id"],r["mode"],r["treatment"]):r for r in frame.to_dict("records")}
    for row in frame.to_dict("records"):
        if not row["treatment"]:
            continue
        control=lookup.get((row["model"],row["block_id"],row["mode"],False))
        if control is None:
            continue
        pair={k:row[k] for k in GROUPS+["block_id","replicate","mode","family"]}
        for metric in METRICS:
            a,b=row[metric],control[metric]
            pair[metric+"_treated"]=a
            pair[metric+"_control"]=b
            pair[metric+"_delta"]=float(a)-float(b) if a is not None and b is not None else np.nan
        pair["control_correct_to_treated_wrong"] = bool(control["final_error"]<0.5 and row["final_error"]>0.5)
        pair["treatment_specific_sustained_failure"] = bool(row["sustained_failure"] and not control["sustained_failure"])
        pairs.append(pair)
    pairs=pd.DataFrame(pairs)
    summary=[]
    for key,g in pairs.groupby(REPORT_GROUPS+["mode"],dropna=False):
        labels=dict(zip(REPORT_GROUPS+["mode"],key))
        for metric in METRICS:
            stat=mean_ci(g[metric+"_delta"],key=str(key)+metric)
            failure_ci=binomial_ci(int(g["treatment_specific_sustained_failure"].sum()),len(g))
            summary.append({**labels,"metric":metric,**stat,"control_mean":g[metric+"_control"].mean(),"treated_mean":g[metric+"_treated"].mean(),"p_signflip":sign_flip_p(g[metric+"_delta"],str(key)+metric),"specific_sustained_rate":g["treatment_specific_sustained_failure"].mean(),"specific_sustained_lo":failure_ci[0],"specific_sustained_hi":failure_ci[1],"flip_rate":g["control_correct_to_treated_wrong"].mean()})
    summary=pd.DataFrame(summary)
    if len(summary):
        summary["p_holm_exploratory_family"]=np.nan
        # All metrics and conditions within each phase/model constitute one exploratory family.
        for _,idx in summary.groupby(["phase","model"]).groups.items():
            summary.loc[idx,"p_holm_exploratory_family"]=holm(summary.loc[idx,"p_signflip"].values)
    mechanisms=[]
    pidx={(r["model"],r["block_id"],r["mode"]):r for r in pairs.to_dict("records")}
    for row in pairs.to_dict("records"):
        if row["mode"]=="natural":
            continue
        natural=pidx.get((row["model"],row["block_id"],"natural"))
        if not natural:
            continue
        for metric in METRICS:
            mechanisms.append({**{k:row[k] for k in GROUPS+["replicate","block_id","mode"]},"metric":metric,"mechanism_delta":natural[metric+"_delta"]-row[metric+"_delta"]})
    mech=pd.DataFrame(mechanisms)
    mech_summary=[]
    if len(mech):
        for key,g in mech.groupby(REPORT_GROUPS+["mode","metric"],dropna=False):
            labels=dict(zip(REPORT_GROUPS+["mode","metric"],key))
            mech_summary.append({**labels,**mean_ci(g.mechanism_delta,key=str(key)),"p_signflip":sign_flip_p(g.mechanism_delta,str(key))})
    return pairs,summary,pd.DataFrame(mech_summary)


def micro_tables(output:Path,folder:Path):
    rows=[]
    for path in sorted((output/"micro").glob("*/*/*.json")):
        r=json.loads(path.read_text(encoding="utf-8"))
        rows.append({k:v for k,v in r.items() if k!="decision"}|{"choice":r["decision"]["choice"],"confidence":r["decision"]["confidence"],"call_key":r["decision"]["call_key"]})
    frame=pd.DataFrame(rows)
    if frame.empty:
        return
    frame.to_csv(folder/"micro_trials.csv",index=False)
    summary=[]
    groups=["phase","model","audience","reliability","wrong_support","semantics","dependence"]
    for key,g in frame.groupby(groups):
        summary.append({**dict(zip(groups,key)),**mean_ci(g.groupby('replicate').wrong.mean(),str(key))})
    pd.DataFrame(summary).to_csv(folder/"micro_summary.csv",index=False)
    # Paired dependence interventions; each replicate remains the independent unit.
    index=["phase","model","replicate","reliability","wrong_support","semantics","audience","truth"]
    wide=frame.pivot(index=index,columns="dependence",values="wrong").reset_index()
    contrasts=[]
    contrast_groups=["phase","model","reliability","wrong_support","semantics","audience"]
    for key,g in wide.groupby(contrast_groups):
        if "independent" not in g or "shared" not in g:
            continue
        delta=(g.assign(delta=g.independent.astype(float)-g.shared.astype(float))).groupby('replicate').delta.mean()
        contrasts.append({**dict(zip(contrast_groups,key)),"contrast":"independent_minus_shared",**mean_ci(delta,str(key)),"p_signflip":sign_flip_p(delta,str(key))})
    pd.DataFrame(contrasts).to_csv(folder/"micro_dependence_contrasts.csv",index=False)
    # This is conditional on taking the stated report model as valid, not a claim
    # that any ground-truth-wrong response is irrational or an act of conformity.
    valid=frame[frame.normative_p_a_if_reports_verified.notna()].copy()
    valid=valid[~np.isclose(valid.normative_p_a_if_reports_verified,0.5)]
    valid['conditional_map_disagreement']=(np.where(valid.normative_p_a_if_reports_verified>0.5,'A','B')!=valid.choice).astype(float)
    valid.to_csv(folder/'micro_conditional_benchmark_trials.csv',index=False)
    benchmark=[]
    for key,g in valid.groupby(['phase','model','dependence','wrong_support','audience']):
        clusters=g.groupby('replicate').conditional_map_disagreement.mean()
        benchmark.append({**dict(zip(['phase','model','dependence','wrong_support','audience'],key)),**mean_ci(clusters,str(key))})
    pd.DataFrame(benchmark).to_csv(folder/'micro_conditional_benchmark.csv',index=False)


def probe_tables(output,folder):
    count_rows=[]
    for path in (output/"source_count").glob("*/*/*.json"):
        row=json.loads(path.read_text())
        count_rows.append({k:v for k,v in row.items() if k not in ["decision","expected_sources"]})
    if count_rows:
        frame=pd.DataFrame(count_rows)
        frame.to_csv(folder/"source_count_trials.csv",index=False)
        frame.groupby(["phase","model","dependence","wrong_support"]).agg(exact_source_set=("exact_source_set","mean"),choice_accuracy=("choice_correct","mean"),replicates=("replicate","nunique")).reset_index().to_csv(folder/"source_count_summary.csv",index=False)
    replay_rows=[]
    for path in (output/"replay").glob("*/*/*.json"):
        if path.name=="selection.json":
            continue
        row=json.loads(path.read_text())
        replay_rows.append({k:v for k,v in row.items() if k!="decision"}|{"context_key":str(row["context_id"]),"choice":row["decision"]["choice"]})
    if replay_rows:
        frame=pd.DataFrame(replay_rows)
        frame.to_csv(folder/"replay_trials.csv",index=False)
        # First aggregate within original society replicate. Multiple snapshots are dependent.
        means=frame.groupby(["phase","model","replicate","lineage","audience"]).wrong.mean().reset_index()
        summary=[]
        for key,g in means.groupby(["phase","model","lineage","audience"]):
            summary.append({**dict(zip(["phase","model","lineage","audience"],key)),**mean_ci(g.wrong,str(key))})
        pd.DataFrame(summary).to_csv(folder/"replay_summary.csv",index=False)


def write_report(summary,audit,folder):
    lines=["# 实验结果状态报告", "", "本文件由真实已完成轨迹生成；不将运行中或失败轨迹补成结果。完整结果需以最终研究报告为准。", "", f"已完成 {audit['runs']} 条群体轨迹、{audit['events']} 个 Agent 决策。", "", "## 数据完整性", ""]
    for key,value in audit.items():
        if isinstance(value,list):
            lines.append(f"- {key}: {len(value)}")
    lines.extend(["",f"无效引用共 {audit['invalid_citations']} 条，保留原始输出供审计。", "", "## 主要配对结果", "", "指标：末三轮非种子节点错误率的干预减对照差异；95% bootstrap CI；小样本仅供探索。", "", "|阶段|模型|场景|网络|通信|配对数|对照|干预|差异 [95% CI]|", "|---|---|---|---|---|---:|---:|---:|---|" ])
    if not summary.empty:
        for row in summary[summary.metric=="nonseed_tail_error"].to_dict("records"):
            if row["scenario"] not in ["cascade","asch"]:
                continue
            lines.append(f"|{row['phase']}|{row['model']}|{row['scenario']}|{row['topology']}|{row['mode']}|{row['sample_n']}|{row['control_mean']:.3f}|{row['treated_mean']:.3f}|{row['estimate']:.3f} [{row['lo']:.3f}, {row['hi']:.3f}]|")
    lines.extend(["", "## 解释边界", "", "- 持续失效、普通错误扩散、错误多数分别记录；未观察到崩溃同样是研究结果。", "- 规范与公共物品场景仅为探索；偏好或自利行为不解释为事实错误。", "- 来源元数据反映有效明确引用，不保证模型实际使用信息的完整血缘。", "- frozen 会改变消息长度和论据形式；相关对照不能直接识别纯中介效应。", "- Qwen不同代际和规模仍属一个模型家族，跨家族验证需Llama/Mistral结果。", "- 所有推断以完整轨迹/任务为单位，不能把每轮每节点作为独立样本。", ""])
    (folder/"status_report_zh.md").write_text("\n".join(lines),encoding="utf-8")


def main(args):
    output=Path(args.output)
    folder=output/"analysis"
    folder.mkdir(parents=True,exist_ok=True)
    frame,trajectories,audit=collect(output)
    frame.to_csv(folder/"runs.csv",index=False)
    trajectories.to_csv(folder/"trajectories.csv",index=False)
    pairs,summary,mech=paired_tables(frame)
    pairs.to_csv(folder/"paired_runs.csv",index=False)
    summary.to_csv(folder/"paired_summary.csv",index=False)
    mech.to_csv(folder/"mechanism_summary.csv",index=False)
    atomic_json(folder/"integrity.json",audit)
    micro_tables(output,folder)
    probe_tables(output,folder)
    write_report(summary,audit,folder)
    print(json.dumps({"analysis":str(folder),"runs":audit["runs"],"integrity":{k:len(v) for k,v in audit.items() if isinstance(v,list)}}))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--output",default="results")
    main(p.parse_args())

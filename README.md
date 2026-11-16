# From Local Disruptions to Collective Collapse: Understanding the Fragility of LLM Agent Societies

Code, data and results for the paper (under double-blind review).
Project page: <https://anonymous.4open.science/w/llm-society-fragility/> (source in `docs/`).

**Summary.** Networked LLM agents repeatedly infer a hidden binary state from a private signal and their
neighbours' messages (the canonical social-learning game). Three of 24 agents send a false message for two
rounds and then stop (a *transient perturbation*). Across five open-weight LLMs and 1,150,848 agent decisions,
the **model and the message set the size of the shock**, whereas the **information structure sets its
persistence**: with a one-time private signal up to 46 pp of excess error remain three to five rounds later;
with recurrent signal access at most 2.2 pp remain; with peer relay blocked at most 0.4 pp remain in every
model and task.

## Repository layout

```
experiment_code/        Simulation code that produced the data (run against vLLM OpenAI-compatible endpoints)
  society/
    core.py               tasks, Watts-Strogatz / Erdos-Renyi networks, seeds
    client.py             async vLLM client with request cache and JSON-schema validation
    experiment.py         prompts and the false message
    response_v3.py        individual calibration probes and the first network runs (abstract task)
    structural_v4.py      signal access (rho), seed-count dose, full-history ablation (abstract task)
    semantic_v5.py        natural-language task (authority claim / neutral wording)
    structural_extract.py raw run JSON -> data/agent_rounds_all.parquet
  scripts/              job launchers and GPU orchestrators
  configs/              model registry and design configs
  tests/                unit tests (pytest)
analysis/               analysis of the extracted data
  v4_analysis.py          tail effects, bootstrap CIs, comparison with the frozen ABM prediction
  abm.py                  individual-rule agent-based model (generates the frozen prediction)
  structural_analysis.py  perturbation memory, effective local rules
  phase_diagram.py        resilience / learning trade-off simulation
  cross_family_abm.py     cross-family ABM (see note below)
  figures.py              exploratory figures
data/
  agent_rounds_all.parquet  every agent x round decision (1,150,848 rows; all models and conditions)
  agent_rounds_v3.parquet   v3 subset used to fit the frozen ABM prediction
  profiles_v3/              isolated-agent probes (Qwen models)
results/
  tables/                   analysis outputs (CSV)
  abm_frozen/               ABM predictions and rules frozen *before* the v4 runs (SHA-256 in FROZEN_SHA256.txt)
  completion_runs/          manifest and coverage of the final completion campaign
paper/
  scripts/                  compute every number, figure and table in the paper
    compute_numbers.py      -> numbers.json
    compute_extras.py       -> out/extras.json
    cross_family_abm_fixed.py -> out/*_fixed.csv
    make_figures_planA.py   -> paper/figures/
    make_tables_planA.py    -> paper/tables/
  figures/                  paper figures (PDF/PNG; fig1_overview.svg is editable)
  tables/                   LaTeX table bodies
docs/                       project web page
```

## Reproducing the paper's numbers, figures and tables

Python 3.10+.

```bash
pip install -r requirements.txt
python paper/scripts/compute_numbers.py      # every number quoted in the paper -> paper/scripts/numbers.json
python paper/scripts/compute_extras.py
python paper/scripts/make_figures_planA.py   # Figures 1-4
python paper/scripts/make_tables_planA.py    # Tables 1-2
```

All of these read only `data/` and `results/` and run in about a minute on a laptop.

### Data dictionary (`data/agent_rounds_all.parquet`)

| column | meaning |
|---|---|
| `source` | experiment generation: `v3`/`v4` abstract task, `v5` natural-language task |
| `model` | LLM |
| `schedule` | `initial_only` (one-time signal), `persistent` (recurrent), `in_history` (signal listed in history), `init_fullmem` (full answer history), `anchor25`/`anchor50` (rho = 0.25/0.5), `init_seed6`/`persist_seed6`/`init_seed1` (seed-count dose) |
| `anchor_fraction` | rho, share of agents with recurrent access |
| `block`, `replicate` | task id (note: `block` ids are reused across `source`; group by `source` as well) |
| `topology` | `WS` or `ER` |
| `arm` | `control`, `treated` (attacked), `no_relay_treated` (attacked, relay blocked), `treated_neutral` / `treated_authority` (natural-language wording variants) |
| `round`, `node` | round 0-8, agent 0-23 |
| `wrong` | agent's answer is wrong |
| `seed` | agent is one of the scripted seed agents |
| `private_correct` | the agent's private signal is correct |
| `n_inbox`, `n_wrong_inbox`, `n_injected_inbox` | number of received messages / wrong ones / false (injected) ones |
| `cites_planted` | the agent's own message mentions the injected claim |
| `seed_fraction` | share of seed agents (0.125 or 0.25) |

### Notes

- `analysis/cross_family_abm.py` computes the previous-round answer without separating `source`, so
  `results/tables/cross_family_abm.csv` and `persuasion_premium.csv` mix the abstract and natural-language
  tasks. The paper uses the corrected versions in `paper/scripts/out/*_fixed.csv`
  (`paper/scripts/cross_family_abm_fixed.py`).
- Llama-3.1-8B under xgrammar occasionally emits `"cites": ["]}`; because seeds are fixed this repeats on
  retry. Exactly this pattern was repaired to an empty citation list (8 of 78,228 calls, 0.01%).

## Re-running the simulations

The simulations call vLLM (0.14.1) OpenAI-compatible servers listed in `experiment_code/configs/models.json`
(models: Qwen2.5-7B-Instruct, Qwen3-14B, Qwen3-30B-A3B-GPTQ-Int4, GLM-4-9B-0414, Llama-3.1-8B-Instruct;
temperature 0.7, fixed sampling seeds). Example:

```bash
cd experiment_code
python -m society.structural_v4 --model Qwen2.5-7B-Instruct --schedules initial_only,persistent
python -m society.semantic_v5   --model Qwen2.5-7B-Instruct
python -m society.structural_extract
```

Paths of the Python and vLLM binaries in `experiment_code/scripts/*` refer to our server and need to be adapted.

## License

Code: MIT (see `LICENSE`). Data and results: CC BY 4.0.

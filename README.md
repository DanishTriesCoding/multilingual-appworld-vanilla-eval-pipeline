# Multilingual AppWorld Vanilla Evaluation Pipeline

A modular, robust, direct-inference (zero-shot) evaluation harness for the [AppWorld benchmark](https://github.com/StonyBrookNLP/appworld). The pipeline evaluates local open-weights LLMs (e.g., Qwen2.5, Llama-3, Mistral) operating as autonomous interactive coding agents in AppWorld's digital environment.

It interfaces with any OpenAI-compatible inference server (vLLM, SGLang, Ollama, TGI), natively supports cross-lingual instruction evaluations via custom JSON task mappings, guarantees process isolation across parallel rollouts, and provides sandbox-safe persistence against environment monkey-patches.

**Author:** Danish Javed
**Repo:** `github.com/DanishTriesCoding/multilingual-appworld-vanilla-eval-pipeline`

---

## Table of Contents

- [Introduction and Pipeline Overview](#introduction-and-pipeline-overview)
  - [Key Design Principles](#key-design-principles)
- [Repository Architecture & File Map](#repository-architecture--file-map)
- [Formal Evaluation Metrics](#formal-evaluation-metrics)
  - [Task Goal Completion (TGC)](#task-goal-completion-tgc)
  - [Scenario Goal Completion (SGC)](#scenario-goal-completion-sgc)
  - [Bootstrap Confidence Intervals](#bootstrap-confidence-intervals)
- [Setup & Installation](#setup--installation)
  - [Virtual Environment Setup](#virtual-environment-setup)
  - [AppWorld Installation and Data Sync](#appworld-installation-and-data-sync)
- [Subsampling vs. Full Split Execution](#subsampling-vs-full-split-execution)
- [Multilingual Evaluation Protocol](#multilingual-evaluation-protocol)
- [Configuration Reference](#configuration-reference)
- [CLI Usage Reference](#cli-usage-reference)
- [Troubleshooting & Diagnostic Guide](#troubleshooting--diagnostic-guide)
- [References](#references)

---

## Introduction and Pipeline Overview

The AppWorld benchmark tests autonomous agents across complex, multi-day digital tasks interacting with 9 simulated applications and ~457 APIs. Evaluating LLMs in this environment requires iterative interaction: the agent observes textual state feedback, generates executable Python code blocks, and completes tasks via designated supervisor API endpoints.

### Key Design Principles

- **Model & Server Agnostic** — Any server providing a standard `POST /v1/chat/completions` endpoint is supported.
- **Native Multilingual Evaluation** — Direct mapping of translated instructions (T<sub>target</sub>) to canonical AppWorld task IDs without mutating base data files.
- **Process-Level Sandbox Isolation** — AppWorld utilizes sandbox hooks and stateful globals. The pipeline executes rollouts via multiprocessing with `spawn` context and process recycling (`maxtasksperchild=1`).
- **Sandbox-Resilient Storage** — Low-level POSIX file descriptors (`os.open`, `os.write`) bypass AppWorld's process-wide `open()` read-only locks.
- **Statistically Sound Metrics** — Computes Task Goal Completion (avg@k, best@k, unbiased pass@k), Scenario Goal Completion (SGC), and 95% bootstrap confidence intervals.

---

## Repository Architecture & File Map

The repository separates environment interaction, agent policy, API communication, and metric reporting into isolated, modular components:

```
MULTILINGUAL-APPWORLD-VANILLA-EVAL-PIPELINE/
├── .gitignore                  # Git ignore rules (caches, local trajectories)
├── diag.py                     # Rapid single-task rollout diagnostic utility
├── export_instructions.py      # Utility to dump split task instructions to JSON
├── instructions_en.json        # Base extracted English task instructions map
├── pyproject.toml              # Build metadata & console CLI entrypoints
├── README.md                   # Repository overview and setup guide
├── requirements.txt            # Package dependencies (requests, PyYAML, appworld)
│
├── configs/
│   ├── smoke_dev.yaml          # Lightweight 4-task dev smoke test config
│   └── vanilla_qwen7b.yaml     # Benchmark evaluation config (Qwen2.5-7B)
│
├── prompts/
│   └── appworld_official.txt   # Canonical AppWorld prompt template
│
├── scripts/
│   ├── run_full_eval.sh        # Bash automation for full evaluation sweeps
│   └── selftest.py             # Offline, GPU-less pipeline validation suite
│
└── src/
    └── appworld_vanilla/
        ├── __init__.py         # Package initialization
        ├── agent.py            # Message history management & step proposal policy
        ├── cli.py              # CLI routers (check, sample, run, report, eval)
        ├── config.py           # Typed dataclasses & YAML configuration parser
        ├── env.py              # Isolated interface encapsulating appworld package
        ├── llm_client.py       # OpenAI-compatible HTTP client with retries
        ├── metrics.py          # avg@k, best@k, pass@k, SGC, & bootstrap CIs
        ├── parsers.py          # Code block extraction & output truncation
        ├── prompts.py          # Prompt constructors & multi-turn template parser
        ├── report.py           # Terminal summary renderer & safe CSV exporter
        ├── runner.py           # Single rollout execution & parallel sweep runner
        ├── sampler.py          # Scenario-stratified & randomized task selection
        └── storage.py          # Thread-safe POSIX JSONL result store & trajectory dumps
```

---

## Formal Evaluation Metrics

AppWorld groups tasks into scenarios where each scenario `S_j` contains 3 distinct, interdependent tasks `{t_j,1, t_j,2, t_j,3}`.

### Task Goal Completion (TGC)

Let `k` be the number of rollouts per task, and `y_i,s ∈ {0, 1}` denote the binary evaluation outcome of task `i` on rollout seed `s ∈ {1, ..., k}`.

**Average Success Rate (avg@k):**

```
avg@k = (1 / (N·k)) · Σ_i Σ_s y_i,s
```

**Best-of-k Success Rate (best@k):**

```
best@k = (1 / N) · Σ_i max_s( y_i,s )
```

**Unbiased pass@k (Codex Estimator):** When `n ≥ k` rollouts are generated and `c_i = Σ_s y_i,s` rollouts succeed for task `i`:

```
pass@k = (1 / N) · Σ_i [ 1 − C(n − c_i, k) / C(n, k) ]
```

### Scenario Goal Completion (SGC)

A scenario is solved under seed `s` if and only if all 3 constituent tasks succeed:

```
Y_scenario_j,s = Π_m y_j,m,s   (for m = 1..3)
```

- **Scenario avg@k:** `(1 / (|S|·k)) · Σ_j Σ_s Y_scenario_j,s`
- **Scenario best@k:** `(1 / |S|) · Σ_j Π_m max_s( y_j,m,s )`

### Bootstrap Confidence Intervals

Non-parametric 95% bootstrap confidence intervals are computed over `B = 2,000` resamples of task clusters with replacement, to ensure statistically sound comparisons between baseline models and multilingual setups.

---

## Setup & Installation

### Virtual Environment Setup

```bash
# Clone and enter directory
git clone https://github.com/DanishTriesCoding/multilingual-appworld-vanilla-eval-pipeline.git
cd multilingual-appworld-vanilla-eval-pipeline

# Initialize virtual environment (Python 3.10+)
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# Install editable package
pip install -e .
```

### AppWorld Installation and Data Sync

```bash
pip install appworld>=0.1.3
appworld install
appworld download data
```

---

## Subsampling vs. Full Split Execution

By default, executing `appworld-vanilla run -c <config>` without specifying a task list evaluates the entire dataset split specified in `run.split` (e.g., all 57 tasks in `dev`, 168 tasks in `test_normal`, or 585 tasks across the full `test` set).

For development, smoke testing, or budget-constrained pilots, generate and supply a subsample file:

```bash
# 1. Extract a stratified subsample preserving scenario triplets
appworld-vanilla sample -c configs/smoke_dev.yaml \
  --set sample.size=4 \
  --out task_ids_4.txt

# 2. Run strictly against the generated subsample list
appworld-vanilla run -c configs/smoke_dev.yaml \
  --set run.task_ids_file=task_ids_4.txt
```

---

## Multilingual Evaluation Protocol

Evaluating models on non-English task instructions proceeds in three steps:

1. **Export Canonical Instructions**

   ```bash
   python export_instructions.py
   ```

   Produces `instructions_en.json` with schema `{"<task_id>": "<instruction text>"}`.

2. **Generate Target Language Mapping**

   Translate `instructions_en.json` to the target language (e.g., Urdu, Chinese, Spanish) into `instructions_ur.json`, preserving exact task ID keys.

3. **Execute Sweep with Instruction Mapping**

   ```bash
   appworld-vanilla run -c configs/vanilla_qwen7b.yaml \
     --set run.instruction_map_file=instructions_ur.json \
     run.experiment_prefix=qwen7b_urdu_eval
   ```

---

## Configuration Reference

Evaluation configurations are specified via YAML files:

```yaml
llm:
  base_url: "http://localhost:8000/v1"   # Target OpenAI-compatible server
  api_key: "EMPTY"
  model: "Qwen/Qwen2.5-7B-Instruct"
  temperature: 0.7                        # Must be > 0 for valid avg@k != best@k
  top_p: 0.8
  max_tokens: 1024
  timeout: 300
  max_retries: 4
  extra_body:
    repetition_penalty: 1.05

agent:
  max_steps: 40                           # Max turns per rollout (40 recommended)
  output_char_limit: 3000                 # Context-saving head/tail truncation
  history_strategy: "sliding"             # Sliding context window
  keep_last_n_steps: 16
  include_supervisor_header: true
  prompt_variant: "zero_shot"             # "zero_shot" or "custom"
  prompt_path: null

run:
  experiment_prefix: "vanilla_eval"
  split: "test_normal"                    # dev | test_normal | test_challenge
  num_rollouts: 8                         # k parameter
  seed_base: 1000
  max_workers: 4                          # Multi-process parallelism
  output_dir: "results"
  resume: true
  task_ids_file: null                     # Custom subset file path (null = full split)
  instruction_map_file: null              # Multilingual mapping JSON
```

---

## CLI Usage Reference

```bash
# 1. Health check LLM endpoint and AppWorld installation
appworld-vanilla check -c configs/smoke_dev.yaml

# 2. Run an offline self-test (GPU-less & data-less)
python scripts/selftest.py

# 3. Sample 36 scenario-stratified tasks
appworld-vanilla sample -c configs/vanilla_qwen7b.yaml \
  --set sample.size=36 \
  --out task_ids_36.txt

# 4. Execute evaluation sweep
appworld-vanilla run -c configs/vanilla_qwen7b.yaml \
  --set run.task_ids_file=task_ids_36.txt

# 5. Display evaluation report and export CSV
appworld-vanilla report -c configs/vanilla_qwen7b.yaml

# 6. Run official AppWorld evaluator over generated trajectories
appworld-vanilla official-eval -c configs/vanilla_qwen7b.yaml
```

---

## Troubleshooting & Diagnostic Guide

| Symptom | Likely Cause / Fix |
|---|---|
| **All rollouts end with unparseable output** | The model is failing to output code fenced in ` ```python ` blocks. Increase `agent.max_empty_code_retries` or refine the system prompt. |
| **High occurrence of "max steps exhausted"** | Small zero-shot models often require ≥25 steps to explore API specifications and obtain authentication tokens. If `agent.max_steps` is set too low (e.g., 12), the model runs out of turn budget before completing goals. |
| **Evaluating entire split instead of a subset** | Ensure `--set run.task_ids_file=<filename>` is passed to `run`, or specify `task_ids_file` in the configuration YAML. |
| **`avg@k == best@k` identically** | Verify that `llm.temperature` is greater than 0 and that seeds vary across rollouts. |

---

## References

1. Chowdhury, H., et al. (2024). *AppWorld: A Controllable World of Apps and Execution Environment for Agent Evaluation.* In Proceedings of the Association for Computational Linguistics (ACL 2024).
2. Chen, M., et al. (2021). *Evaluating Large Language Models Trained on Code.* arXiv:2107.03374 (Codex pass@k formulation).

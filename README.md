# AppWorld — vanilla Qwen2.5-7B-Instruct evaluation pipeline

A modular harness for the **direct-inference baseline** on AppWorld: off-the-shelf
`Qwen2.5-7B-Instruct` acting in the environment, scored as **avg@8 / best@8**.
No RL, no self-questioning, no experience pool, no training loop of any kind.

It assumes you already have the model served locally behind an OpenAI-compatible
endpoint (vLLM, SGLang, TGI, llama.cpp, Ollama — anything with
`POST /v1/chat/completions`).

---


## Methodology & Prompts

This pipeline performs a **vanilla direct-inference run**. By "vanilla," we mean:
- **No advanced scaffolding**: There is no self-reflection step, no reinforcement learning, no retrieval-augmented generation (RAG), and no experience pool.
- **Direct code-as-action loop**: The model acts in the environment purely by reading the prompt, emitting a single fenced block of Python code, and receiving the execution output as the next observation. This ReAct-style loop continues until the model explicitly calls the supervisor's `complete_task()` API or hits a step limit.
- **Prompting**: By default, we use a custom zero-shot system prompt (`src/appworld_vanilla/prompts.py`) that explains the `apis` object, the supervisor, and the rules of the environment.

## Alignment with the Original AppWorld Paper

We heavily leverage the `StonyBrookNLP/appworld` library for the environment, tasks, and official evaluation script (`world.evaluate()`). This ensures that the environments are identical and the final metrics (e.g., avg@k, best@k) are perfectly comparable to published numbers.

However, we deviate from the original paper's setup in a few specific ways:
- **Custom Runner**: Instead of using their `jsonnet`-based experiment runner and complex baseline configs, we use our own modular Python loop. This provides a cleaner, more hackable pipeline.
- **Prompts**: As noted, our default built-in prompt is faithful in *spirit* to AppWorld's code-as-action agent, but it is **not** byte-identical. The official baselines use a long few-shot prompt which can move scores significantly.
- **Single Block Execution**: Our evaluator specifically enforces that *only the first fenced Python block per message is executed*. We truncate hallucinated follow-up blocks to keep the real environment as the sole source of truth.

To strictly match the paper's baseline performance, you must run the full test set and opt-in to their exact prompts (see [Matching published numbers](#matching-published-numbers)).

---

## Answers to the questions you asked

**Does `StonyBrookNLP/appworld` work for this?** Yes. It ships the environment
(9 apps, ~457 APIs), the task datasets, the evaluator, and an `experiments/`
package with baseline agent configs. This pipeline uses the *library* (`AppWorld`,
`load_task_ids`, `world.evaluate()`) rather than their experiment runner, so you
get a clean, hackable loop instead of jsonnet configs — but it writes into the same
`experiments/outputs/{experiment_name}/` layout, so `appworld evaluate` still works
on the results.

**How many tasks are there?** 750 total (250 scenarios × 3 tasks), split into
train / dev / test_normal / test_challenge. I'd rather you not trust a number I
half-remember here — the exact per-split counts have been reported inconsistently
in secondary sources. Run:

```bash
appworld-vanilla check -c configs/vanilla_qwen7b.yaml
```

and it prints the real count straight from `load_task_ids(split)`. Test set =
`test_normal` + `test_challenge`.

**Did the paper subsample?** The leaderboard protocol is the full test set, and
papers reporting AppWorld numbers are expected to follow it. If you want numbers
comparable to a published table, run both full test splits. If you're doing an
exploratory replication on a budget, subsample — but see
[Subsampling](#subsampling-and-how-much-it-costs-you) for what that does to your
error bars, because it matters a lot when the true score is ~2%.

**One correction on the earlier notes you were given:** the "≈1.8% avg@8 / ≈5.6%
best@8" vanilla figures were cited to the *Qwen2.5 technical report*. They aren't
from there — the Qwen tech report doesn't evaluate on AppWorld. Those come from
the AgentEvolver paper's own baseline row. Worth checking against the actual PDF
before you treat them as your reproduction target.

---

## Install

```bash
# 1. AppWorld itself
pip install appworld
appworld install
appworld download data          # requires git-lfs

# 2. this pipeline
cd appworld_vanilla_eval
pip install -e .
```

Point the config at your server, then verify both halves are alive:

```bash
appworld-vanilla check -c configs/vanilla_qwen7b.yaml
```

This probes `/v1/models`, sends one real completion, and loads the split. Fix
anything it complains about before running a sweep.

## Quickstart

```bash
# 1. Smoke test: 4 dev tasks x 2 rollouts x 12 steps. Do this first.
appworld-vanilla run -c configs/smoke_dev.yaml

# 2. Look at an actual trajectory to sanity-check the prompt and parsing
cat results/smoke_dev/trajectories/*/seed_7000.json | head -60

# 3. Optional: carve out a subsample
appworld-vanilla sample -c configs/vanilla_qwen7b.yaml \
  --set sample.size=60 --out task_ids_60.txt

# 4. Real run
appworld-vanilla run -c configs/vanilla_qwen7b.yaml \
  --set run.task_ids_file=task_ids_60.txt

# 5. Re-report without re-running (e.g. at a smaller k)
appworld-vanilla report -c configs/vanilla_qwen7b.yaml --k 4

# 6. AppWorld's own canonical evaluator, per seed
appworld-vanilla official-eval -c configs/vanilla_qwen7b.yaml
```

Full two-split sweep with an environment server:

```bash
./scripts/run_full_eval.sh configs/vanilla_qwen7b.yaml
```

Runs are **resumable** — kill it, rerun the same command, and it skips every
`(task_id, seed)` already in `rollouts.jsonl`.

---

## What a rollout actually does

Standard code-as-action loop. The model gets a system prompt describing the `apis`
object, then for each step emits one fenced Python block; the block runs in
AppWorld's persistent IPython session; the output comes back as the next user
message. The loop ends when the model calls `apis.supervisor.complete_task()`, the
step budget runs out, or the output can't be parsed. Either way, AppWorld's
evaluator scores the final world state.

Two details that matter more than they look:

- **Only the first fenced block per message is executed.** 7B models routinely
  hallucinate an execution output and a follow-up block in the same turn. Taking
  the first block keeps the environment the sole source of observations.
- **Sampling must be stochastic.** `temperature: 0.7` by default. At temperature 0
  all 8 rollouts are identical and `avg@8 == best@8`, which silently makes your
  numbers meaningless.

## Metrics

Per task, over `k` rollouts:

- **avg@k** — fraction of the k attempts that succeeded, averaged over tasks.
  "If you try k times, what share of attempts work?"
- **best@k** — 1 if any of the k attempts succeeded, averaged over tasks.
  "What share of tasks are solvable within k tries?"
- **unbiased pass@k** — the Codex estimator, `1 − C(n−c, k)/C(n, k)`. Identical to
  best@k when `n == k`; use it when you ran more rollouts than the k you're quoting.
- **95% CIs** — bootstrap over tasks, so you can see whether a difference is real.
- **Scenario-level (SGC-style)** — a scenario counts only if all its tasks pass.
  Computed via the `<scenario>_<n>` task-id convention; treat it as indicative and
  use `official-eval` for the number you'd actually quote.

Outputs land in `results/{experiment_prefix}/`:

```
rollouts.jsonl                    one line per (task, seed) — the source of truth
trajectories/{task}/seed_N.json   full messages, code, outputs, eval detail
per_task.csv                      per-task avg / best / pass@k
summary.json                      headline numbers
config.json                       exact config used
```

## Subsampling, and how much it costs you

The vanilla 7B baseline scores near the floor, and near-floor scores are where
subsampling hurts most. Rough 95% half-widths on **best@k** for a true rate of 5%:

| tasks sampled | ≈ 95% half-width | reads as |
|---|---|---|
| 30 | ±8 pts | 5% ± 8 — indistinguishable from zero |
| 60 | ±5.5 pts | still very wide |
| 168 (full test_normal) | ±3.3 pts | usable |
| 585 (full test set) | ±1.8 pts | comparable to published tables |

So: a 30-task pilot tells you the harness works, not what the score is. If you
want a number you'd put in a table, run at least full `test_normal`. Use
`strategy: scenario_stratified` when subsampling so scenarios stay intact and the
scenario-level metric means something.

**Budget.** Full test set = 585 tasks × 8 rollouts ≈ 4,700 rollouts. Vanilla 7B
rarely finishes early, so expect a mean near the step cap — call it 70k–115k
generations, each with a growing multi-turn context. Plan for a long run, raise
`run.max_workers` to whatever your server's batch throughput supports, and lean on
resume. The `dev` split with `num_rollouts: 2` is the right place to tune first.

## Configuration

Everything lives in the YAML; anything can be overridden inline with
`--set dotted.key=value` (values are parsed as YAML, so `null`, `true`, numbers and
lists all work).

| key | what it does |
|---|---|
| `llm.base_url` / `llm.model` | your local server; `model` must match `/v1/models` |
| `llm.temperature` / `top_p` | keep temperature > 0 |
| `llm.extra_body` | anything else your server takes, e.g. `repetition_penalty` |
| `agent.max_steps` | interaction budget per rollout (default 40) |
| `agent.history_strategy` | `sliding` (default) or `full` if context allows |
| `agent.output_char_limit` | truncates giant API dumps head-and-tail |
| `agent.prompt_variant` | `zero_shot`, or `custom` + `prompt_path` |
| `env.remote_environment_url` | use with `appworld serve environment` for parallelism |
| `run.num_rollouts` | k |
| `run.task_ids_file` | subsample list from `sample` |
| `run.max_workers` | concurrent rollouts |

### Matching published numbers

The built-in prompt (`src/appworld_vanilla/prompts.py`) is faithful in *spirit* to
AppWorld's code-as-action agent but is **not** byte-identical to theirs. Prompt
wording moves AppWorld scores by several points, and the official baselines use a
long few-shot prompt. To close that gap, copy the prompt out of the appworld repo
(`experiments/prompts/`, referenced from `experiments/configs/*.jsonnet`) into a
file and set:

```yaml
agent:
  prompt_variant: "custom"
  prompt_path: "prompts/appworld_official.txt"
```

Placeholders available: `{task_instruction}`, `{supervisor_block}`, `{max_steps}`.

You can also download their precomputed baseline trajectories
(`appworld download experiment-outputs`) and diff a few against yours — that's the
fastest way to spot a prompt or step-budget mismatch.

## Module map

Each file does one thing, so you can replace one without touching the rest.

| module | responsibility |
|---|---|
| `config.py` | typed dataclasses, YAML loading, dotted overrides |
| `llm_client.py` | OpenAI-compatible chat with retries; `ScriptedClient` for tests |
| `prompts.py` | prompt text and assembly — **edit this to change the agent's brief** |
| `parsers.py` | completion → (thought, code); output truncation |
| `env.py` | **the only file that imports `appworld`** — all version drift lands here |
| `agent.py` | policy: message history, context window management, one step |
| `runner.py` | one rollout; the parallel, resumable sweep |
| `sampler.py` | full / random / scenario-stratified task selection |
| `storage.py` | append-only JSONL + trajectory dumps + resume |
| `metrics.py` | avg@k, best@k, pass@k, bootstrap CIs, scenario grouping |
| `report.py` | terminal summary + CSV |
| `cli.py` | `check` / `sample` / `run` / `report` / `official-eval` |

Swapping in a different agent scaffold means writing one class with
`reset` / `propose` / `record` and pointing `runner.py` at it. Swapping the model
means changing `llm.model` and `llm.base_url` — nothing else.

Offline plumbing test (no GPU, no AppWorld data needed):

```bash
python scripts/selftest.py
```

## Troubleshooting

**Every rollout ends `unparseable_output`.** The model isn't emitting fenced
blocks. Read a trajectory's `raw_completion`. Usually the fix is prompt wording or
raising `max_empty_code_retries`.

**Every rollout hits `max_steps_exhausted` with 0 successes.** Expected to a large
degree for vanilla 7B — but check a trajectory anyway. If the model never gets past
`show_app_descriptions`, it's stuck in discovery and never logging in; that's a
prompt problem, not a model ceiling.

**Parallel runs are flaky or slow.** Start `appworld serve environment --port 8123`
and set `env.remote_environment_url`. Then raise `max_workers`.

**`world.evaluate()` errors.** AppWorld's evaluator API has shifted across
versions. `env.normalize_report()` handles dict/object/attribute shapes; if yours
differs, that one function is the only place to patch.

**Numbers look too good.** Confirm temperature > 0 and that seeds actually differ —
if avg@8 exactly equals best@8 across the board, your rollouts are duplicates.

Multilingual AppWorld Vanilla Evaluation PipelineA modular, robust, direct-inference (zero-shot) evaluation harness for the AppWorld benchmark (StonyBrookNLP/appworld).This pipeline is built to evaluate local open-weights LLMs (such as Qwen, Llama, Mistral, and DeepSeek) acting as autonomous code agents in AppWorld's interactive environment, scored via official avg@k and best@k metrics. It connects to any OpenAI-compatible inference endpoint (vLLM, SGLang, Ollama, TGI, llama.cpp) and cleanly supports multilingual evaluations through custom task instruction mappings.Key FeaturesModel & Server Agnostic: Connects to any endpoint exposing POST /v1/chat/completions.Multilingual Task Mapping: Native support for custom/translated instructions via JSON mappings (run.instruction_map_file) without code hacks.Robust Process Isolation: Uses multiprocessing with spawn and process recycling (maxtasksperchild=1) to prevent state pollution and memory leaks across parallel AppWorld rollouts.Sandbox-Safe Persistence: Built-in POSIX file descriptor I/O (os.open/os.write) to bypass AppWorld's process-wide open() read-only monkey-patches during JSONL logging and CSV report exports.Native Custom Prompt Support: Built-in parsing for official multi-turn template prompts (e.g., USER: / ASSISTANT: blocks with Jinja-style variable substitution).Official Metrics & Statistical Analysis: Computes avg@k, best@k, unbiased Codex estimator pass@k, scenario-level SGC metrics, and 95% bootstrap confidence intervals.Resumable Execution: Append-only JSONL trajectory logging with automatic rollout deduplication and resume capability per seed.Directory LayoutMULTILINGUAL-APPWORLD-VANILLA-EVAL-PIPELINE/
├── .gitignore
├── pyproject.toml
├── requirements.txt
├── README.md
├── instructions_en.json        # Pre-extracted English task instructions map
├── task_ids_4.txt              # Task subsample lists
├── task_ids_36.txt
├── task_ids_70.txt
│
├── configs/
│   ├── smoke_dev.yaml          # Lightweight 4-task dev smoke test
│   └── vanilla_qwen7b.yaml     # Main evaluation config (e.g., Qwen2.5-7B-Instruct)
│
├── prompts/
│   └── appworld_official.txt   # Canonical multi-turn AppWorld prompt template
│
├── scripts/
│   ├── run_full_eval.sh        # Bash automation for full evaluation sweeps
│   └── selftest.py             # Offline, GPU-less pipeline unit test
│
├── export_instructions.py      # Utility to dump task instructions to JSON
├── diag.py                     # Rapid single-task rollout diagnostic utility
│
└── src/
    └── appworld_vanilla/
        ├── __init__.py
        ├── agent.py            # Code agent policy & history management
        ├── cli.py              # CLI entrypoints (check, sample, run, report, official-eval)
        ├── config.py           # Dataclasses & YAML config parser
        ├── env.py              # AppWorld environment interface & report normalizer
        ├── llm_client.py       # OpenAI-compatible API client with retries
        ├── metrics.py          # avg@k, best@k, pass@k, SGC, & bootstrap CIs
        ├── parsers.py          # Code block extraction & output truncation
        ├── prompts.py          # Prompt builders & template message parsers
        ├── report.py           # Terminal summary renderer & safe CSV exporter
        ├── runner.py           # Single rollout execution & parallel sweep runner
        ├── sampler.py          # Task sampling strategies (scenario-stratified)
        └── storage.py          # Safe JSONL result store & trajectory saver
Prerequisites & Installation1. Environment SetupClone the repository and install dependencies in a Python 3.10+ virtual environment:git clone [https://github.com/DanishTriesCoding/multilingual-appworld-vanilla-eval-pipeline.git](https://github.com/DanishTriesCoding/multilingual-appworld-vanilla-eval-pipeline.git)
cd multilingual-appworld-vanilla-eval-pipeline

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -e .
2. AppWorld SetupInstall AppWorld's environment and download its task data:pip install appworld>=0.1.3
appworld install
appworld download data
Quickstart & Verification1. Run the Offline Pipeline Self-TestVerify that the core plumbing (config, parser, agent, runner, storage, metrics, and reporting) works without requiring a GPU or AppWorld dataset downloads:python scripts/selftest.py
2. Start Your Local LLM ServerServe your model using any OpenAI-compatible server. For example, using vllm:vllm serve Qwen/Qwen2.5-7B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 16384
3. Run a Health CheckTest the connection between the pipeline, your served LLM, and the AppWorld package:appworld-vanilla check -c configs/smoke_dev.yaml
4. Run a Fast End-to-End Smoke TestRun a 4-task, 2-rollout smoke run on the dev split:appworld-vanilla run -c configs/smoke_dev.yaml
Print the resulting report:appworld-vanilla report -c configs/smoke_dev.yaml
Running Full EvaluationsCLI CommandsThe package provides a unified CLI tool appworld-vanilla:# Health check LLM endpoint and AppWorld setup
appworld-vanilla check -c configs/vanilla_qwen7b.yaml

# Generate a scenario-stratified task list (e.g., 70 tasks)
appworld-vanilla sample -c configs/vanilla_qwen7b.yaml --out task_ids_70.txt

# Execute parallel rollout sweeps
appworld-vanilla run -c configs/vanilla_qwen7b.yaml

# Print summary metrics report & export CSV
appworld-vanilla report -c configs/vanilla_qwen7b.yaml

# Run official AppWorld evaluation script on generated trajectories
appworld-vanilla official-eval -c configs/vanilla_qwen7b.yaml
Command-Line OverridesOverride any configuration parameter dynamically using --set:appworld-vanilla run -c configs/vanilla_qwen7b.yaml \
  --set run.split=test_challenge \
        run.num_rollouts=8 \
        llm.temperature=0.7 \
        run.max_workers=8
Multilingual Evaluation WorkflowEvaluating agents on translated instructions is fully supported:Step 1: Export Base InstructionsExport the default English instructions for a given split or task list:python export_instructions.py
This produces instructions_en.json containing { "task_id": "instruction text" }.Step 2: Translate InstructionsTranslate instructions_en.json into your target language (e.g., instructions_ur.json, instructions_zh.json), keeping the task ID keys intact.Step 3: Run Multilingual SweepPass the translated instructions file via run.instruction_map_file:appworld-vanilla run -c configs/vanilla_qwen7b.yaml \
  --set run.instruction_map_file=instructions_ur.json \
        run.experiment_prefix=qwen7b_urdu_eval
The agent will receive the translated instruction while operating in the standard execution environment.Configuration ReferenceConfiguration files are written in YAML and mapped to typed dataclasses (src/appworld_vanilla/config.py).llm:
  base_url: "http://localhost:8000/v1"   # Base URL for OpenAI-compatible server
  api_key: "EMPTY"                       # API key (or "EMPTY" for local servers)
  model: "Qwen/Qwen2.5-7B-Instruct"      # Model identifier
  temperature: 0.7                       # > 0 required for stochastic rollouts (avg@k vs best@k)
  top_p: 0.8
  max_tokens: 1024
  timeout: 300
  max_retries: 4
  retry_backoff: 2.0
  send_seed: true
  extra_body:
    repetition_penalty: 1.05

agent:
  max_steps: 40                          # Maximum code blocks per rollout
  output_char_limit: 3000                # Head & tail truncation for REPL observations
  history_strategy: "sliding"            # "sliding" context window or "full"
  keep_last_n_steps: 16                  # Number of recent steps retained in sliding window
  include_supervisor_header: true
  prompt_variant: "zero_shot"            # "zero_shot" (built-in) or "custom"
  prompt_path: null                      # Path to custom template (e.g. prompts/appworld_official.txt)
  max_empty_code_retries: 2              # Nudge retries if model outputs no python fence

env:
  remote_environment_url: null           # Set e.g. "http://localhost:8123" if running remote AppWorld server
  extra_kwargs: {}

run:
  experiment_prefix: "vanilla_eval"      # Results saved in results/<experiment_prefix>_seed<seed>/
  split: "test_normal"                   # train | dev | test_normal | test_challenge
  num_rollouts: 8                        # k rollouts per task
  seed_base: 1000                        # Seed for rollout 0 is seed_base, rollout 1 is seed_base+1, etc.
  max_workers: 4                         # Parallel worker process count
  output_dir: "results"
  resume: true                           # Skip already completed (task_id, seed) rollouts
  task_ids_file: null                    # Custom task ID subset file (one ID per line)
  instruction_map_file: null             # Custom/translated instruction JSON map

sample:
  strategy: "scenario_stratified"        # "scenario_stratified" | "random" | "first_n"
  size: null                             # null -> full split
  seed: 0
Evaluation MetricsAppWorld evaluates tasks arranged in scenarios (each scenario contains 3 related tasks):Task Goal Completion (TGC):avg@k: Average success rate across all $k$ rollouts for a task.best@k: Probability that at least 1 of the $k$ rollouts succeeded.unbiased pass@k: Unbiased estimate of pass@k calculated using the Codex estimator when $n \ge k$.Scenario Goal Completion (SGC):Scenario avg@k: Average rate of completely solving all 3 tasks in a scenario.Scenario best@k: Percentage of scenarios where all 3 tasks were solved at least once across rollouts.Statistical Confidence:Computes non-parametric 95% bootstrap confidence intervals across task-level metrics.Architecture & Design PrinciplesSingle Environment Isolation Module (env.py): All dependencies on the upstream appworld library are encapsulated inside env.py.Decoupled Agent Policy (agent.py): The agent receives string observations and emits code strings. It carries no direct dependency on the environment or evaluator logic.Append-Only Result Store (storage.py): Writes individual trajectory files (seed_<seed>.json) and appends JSONL result records immediately upon completion.POSIX File Operations: Avoids open() write locks injected by AppWorld's sandbox evaluator.License & CitationDistributed under the MIT License.If you use this evaluation pipeline or the AppWorld benchmark, please cite:@inproceedings{appworld2024,
  title={AppWorld: A Controllable World of Apps and Execution Environment for Agent Evaluation},
  author={Chowdhury, Harsh and others},
  booktitle={ACL},
  year={2024}
}

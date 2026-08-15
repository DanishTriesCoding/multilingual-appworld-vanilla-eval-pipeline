"""Vanilla (direct-inference) AppWorld evaluation pipeline.

Modules are deliberately decoupled:
    config      - typed configuration loaded from YAML
    llm_client  - OpenAI-compatible chat client (your local vLLM/TGI/etc.)
    prompts     - prompt construction, swappable
    parsers     - LLM output -> executable code block
    env         - the ONLY module that touches the `appworld` package
    agent       - policy: message history + code proposal
    runner      - one (task_id, seed) rollout
    sampler     - which tasks to evaluate (full / subsample)
    storage     - append-only JSONL results + resume
    metrics     - avg@k, best@k, pass@k, bootstrap CIs
    report      - human-readable output
    cli         - entrypoints
"""

__version__ = "0.1.0"

"""Typed configuration objects, loadable from YAML."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, fields, is_dataclass, asdict
from pathlib import Path
from typing import Any, get_type_hints

import yaml


def _build(cls, data: dict[str, Any]):
    """Instantiate a (possibly nested) dataclass from a plain dict."""
    if data is None:
        return cls()
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}")
    # `from __future__ import annotations` turns field.type into a string,
    # so resolve real types once here.
    hints = get_type_hints(cls)
    kwargs = {}
    for name, value in data.items():
        ftype = hints.get(name)
        if is_dataclass(ftype) and isinstance(value, dict):
            kwargs[name] = _build(ftype, value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


@dataclass
class LLMConfig:
    """Points at your already-running local server (OpenAI-compatible)."""

    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"
    model: str = "Qwen/Qwen2.5-7B-Instruct"

    # Qwen2.5-Instruct's own recommended sampling defaults.
    # Sampling MUST be stochastic, otherwise all 8 rollouts are identical
    # and avg@8 == best@8 trivially.
    temperature: float = 0.7
    top_p: float = 0.8
    max_tokens: int = 1024
    stop: list[str] = field(default_factory=list)

    timeout: float = 300.0
    max_retries: int = 4
    retry_backoff: float = 2.0

    # Pass `seed` through to the server (vLLM supports it). Combined with a
    # per-rollout seed this makes individual rollouts reproducible.
    send_seed: bool = True
    # Anything else your server accepts, e.g. {"repetition_penalty": 1.05}
    extra_body: dict[str, Any] = field(default_factory=lambda: {"repetition_penalty": 1.05})


@dataclass
class AgentConfig:
    max_steps: int = 40
    # Truncate a single execution output before feeding it back to the model.
    output_char_limit: int = 3000
    # "full" keeps everything; "sliding" keeps system + task + last N exchanges.
    history_strategy: str = "sliding"
    keep_last_n_steps: int = 16
    include_supervisor_header: bool = True
    # "zero_shot" uses the built-in prompt; "custom" loads prompt_path verbatim.
    prompt_variant: str = "zero_shot"
    prompt_path: str | None = None
    # If the model emits no code block, retry the same step this many times
    # before aborting the rollout.
    max_empty_code_retries: int = 2


@dataclass
class EnvConfig:
    # Set this if you run `appworld serve environment --port 8123`
    # (recommended for parallel rollouts).
    remote_environment_url: str | None = None
    # Extra kwargs forwarded to AppWorld(...) if your version supports them.
    extra_kwargs: dict[str, Any] = field(default_factory=dict)
    # Ask AppWorld to persist its own logs/state for each rollout.
    ground_truth_mode: str | None = None


@dataclass
class RunConfig:
    experiment_prefix: str = "vanilla_qwen2p5_7b"
    split: str = "test_normal"
    num_rollouts: int = 8
    seed_base: int = 1000
    max_workers: int = 4
    output_dir: str = "results"
    resume: bool = True
    # Path written by `sample` (one task id per line). If null -> whole split.
    task_ids_file: str | None = None


@dataclass
class SampleConfig:
    strategy: str = "random"  # "random" | "scenario_stratified" | "first_n"
    size: int | None = None  # null -> full split
    seed: int = 0


@dataclass
class ExperimentConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    run: RunConfig = field(default_factory=RunConfig)
    sample: SampleConfig = field(default_factory=SampleConfig)

    @classmethod
    def load(cls, path: str | Path, overrides: dict[str, Any] | None = None) -> "ExperimentConfig":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        if overrides:
            raw = _deep_merge(raw, overrides)
        return _build(cls, raw)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def dump(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def parse_dotted_overrides(pairs: list[str]) -> dict[str, Any]:
    """--set llm.temperature=1.0 run.num_rollouts=4  ->  nested dict."""
    out: dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Bad override (expected key=value): {pair}")
        key, value = pair.split("=", 1)
        node = out
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(value)
    return out

"""Typed configuration objects, loadable from YAML."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, fields, is_dataclass
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
    """Configuration for any OpenAI-compatible server (vLLM, SGLang, TGI, Ollama, llama.cpp)."""

    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"
    model: str = "Qwen/Qwen2.5-7B-Instruct"

    temperature: float = 0.7
    top_p: float = 0.8
    max_tokens: int = 1024
    stop: list[str] = field(default_factory=list)

    timeout: float = 300.0
    max_retries: int = 4
    retry_backoff: float = 2.0

    send_seed: bool = True
    extra_body: dict[str, Any] = field(default_factory=lambda: {"repetition_penalty": 1.05})


@dataclass
class AgentConfig:
    max_steps: int = 40
    output_char_limit: int = 3000
    history_strategy: str = "sliding"
    keep_last_n_steps: int = 16
    include_supervisor_header: bool = True
    prompt_variant: str = "zero_shot"
    prompt_path: str | None = None
    max_empty_code_retries: int = 2


@dataclass
class EnvConfig:
    remote_environment_url: str | None = None
    extra_kwargs: dict[str, Any] = field(default_factory=dict)
    ground_truth_mode: str | None = None


@dataclass
class RunConfig:
    experiment_prefix: str = "vanilla_eval"
    split: str = "test_normal"
    num_rollouts: int = 8
    seed_base: int = 1000
    max_workers: int = 4
    output_dir: str = "results"
    resume: bool = True
    task_ids_file: str | None = None
    instruction_map_file: str | None = None


@dataclass
class SampleConfig:
    strategy: str = "scenario_stratified"
    size: int | None = None
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
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if overrides:
            raw = _deep_merge(raw, overrides)
        return _build(cls, raw)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def dump(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def parse_dotted_overrides(pairs: list[str]) -> dict[str, Any]:
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
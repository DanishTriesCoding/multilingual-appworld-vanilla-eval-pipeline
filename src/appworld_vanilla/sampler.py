"""Choose which AppWorld tasks to evaluate.

Full split, uniform random subsample, or scenario-stratified subsample (keeps
the 3 tasks of a scenario together, which is what SGC needs to be meaningful).
"""

from __future__ import annotations

import random
from collections import defaultdict

from .config import SampleConfig
from .env import scenario_id_of


def select_task_ids(all_ids: list[str], cfg: SampleConfig) -> list[str]:
    if cfg.size is None or cfg.size >= len(all_ids):
        return list(all_ids)
    if cfg.size <= 0:
        raise ValueError("sample.size must be positive or null")

    rng = random.Random(cfg.seed)

    if cfg.strategy == "first_n":
        return list(all_ids)[: cfg.size]

    if cfg.strategy == "random":
        return sorted(rng.sample(list(all_ids), cfg.size))

    if cfg.strategy == "scenario_stratified":
        by_scenario: dict[str, list[str]] = defaultdict(list)
        for task_id in all_ids:
            by_scenario[scenario_id_of(task_id)].append(task_id)
        scenarios = sorted(by_scenario)
        rng.shuffle(scenarios)
        picked: list[str] = []
        for scenario in scenarios:
            if len(picked) >= cfg.size:
                break
            picked.extend(sorted(by_scenario[scenario]))
        return sorted(picked[: cfg.size]) if len(picked) > cfg.size else sorted(picked)

    raise ValueError(f"Unknown sample.strategy: {cfg.strategy}")


def describe_selection(all_ids: list[str], selected: list[str]) -> dict:
    return {
        "split_size": len(all_ids),
        "selected": len(selected),
        "fraction": round(len(selected) / max(len(all_ids), 1), 4),
        "scenarios_in_selection": len({scenario_id_of(t) for t in selected}),
        "complete_scenarios": sum(
            1
            for _, tasks in _group(selected).items()
            if len(tasks) == len(_group(all_ids).get(_key(tasks), tasks))
        ),
    }


def _group(ids: list[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for task_id in ids:
        out.setdefault(scenario_id_of(task_id), []).append(task_id)
    return out


def _key(tasks: list[str]) -> str:
    return scenario_id_of(tasks[0])

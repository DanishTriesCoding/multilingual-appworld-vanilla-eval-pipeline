"""avg@k, best@k, unbiased pass@k, and bootstrap confidence intervals."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Iterable

from .env import scenario_id_of


def group_by_task(records: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        out[record["task_id"]].append(record)
    return dict(out)


def unbiased_pass_at_k(n: int, c: int, k: int) -> float:
    """P(at least one success in k draws without replacement from n samples).

    The Codex-paper estimator. Use it when you ran n != k rollouts and still
    want a k-comparable number; with n == k it reduces to `any(successes)`.
    """
    if k > n:
        raise ValueError(f"k={k} > n={n}")
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


def per_task_stats(records: Iterable[dict[str, Any]], k: int | None = None) -> list[dict[str, Any]]:
    stats = []
    for task_id, rollouts in sorted(group_by_task(records).items()):
        successes = [bool(r.get("success")) for r in rollouts]
        n = len(successes)
        c = sum(successes)
        kk = min(k or n, n)
        stats.append(
            {
                "task_id": task_id,
                "scenario_id": scenario_id_of(task_id),
                "n_rollouts": n,
                "n_success": c,
                "avg": c / n if n else 0.0,
                "best": 1.0 if c > 0 else 0.0,
                "pass_at_k": unbiased_pass_at_k(n, c, kk) if n else 0.0,
                "errors": sum(1 for r in rollouts if r.get("error")),
                "mean_steps": (sum(r.get("num_steps", 0) for r in rollouts) / n) if n else 0.0,
            }
        )
    return stats


def _bootstrap_ci(values: list[float], iters: int = 2000, seed: int = 0) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(iters):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return (means[int(0.025 * iters)], means[int(0.975 * iters) - 1])


def aggregate(records: list[dict[str, Any]], k: int | None = None) -> dict[str, Any]:
    """Task-level (TGC-style) and scenario-level (SGC-style) headline numbers."""
    tasks = per_task_stats(records, k=k)
    if not tasks:
        return {"n_tasks": 0, "n_rollouts": 0}

    avgs = [t["avg"] for t in tasks]
    bests = [t["best"] for t in tasks]
    passes = [t["pass_at_k"] for t in tasks]
    n_rollouts = sum(t["n_rollouts"] for t in tasks)
    kk = k or (n_rollouts // len(tasks))

    # Scenario level: a scenario counts as solved by a rollout index only if
    # every task in it succeeded. Computed per seed, then averaged.
    by_seed: dict[int, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        by_seed[int(record.get("seed", 0))][scenario_id_of(record["task_id"])].append(
            bool(record.get("success"))
        )
    scenario_avg_per_seed = []
    scenario_solved_any: dict[str, bool] = defaultdict(bool)
    for _, scenarios in by_seed.items():
        solved = [all(v) for v in scenarios.values()]
        if solved:
            scenario_avg_per_seed.append(sum(solved) / len(solved))
        for name, values in scenarios.items():
            scenario_solved_any[name] = scenario_solved_any[name] or all(values)

    ci_avg = _bootstrap_ci(avgs)
    ci_best = _bootstrap_ci(bests)

    return {
        "k": kk,
        "n_tasks": len(tasks),
        "n_rollouts": n_rollouts,
        "avg_at_k": sum(avgs) / len(avgs),
        "best_at_k": sum(bests) / len(bests),
        "unbiased_pass_at_k": sum(passes) / len(passes),
        "avg_at_k_ci95": ci_avg,
        "best_at_k_ci95": ci_best,
        "scenario_avg_at_k": (
            sum(scenario_avg_per_seed) / len(scenario_avg_per_seed)
            if scenario_avg_per_seed
            else None
        ),
        "scenario_best_at_k": (
            sum(scenario_solved_any.values()) / len(scenario_solved_any)
            if scenario_solved_any
            else None
        ),
        "n_scenarios": len(scenario_solved_any),
        "rollout_error_rate": sum(1 for r in records if r.get("error")) / max(len(records), 1),
        "mean_steps": sum(r.get("num_steps", 0) for r in records) / max(len(records), 1),
        "finish_reasons": _counts(r.get("finished_reason", "unknown") for r in records),
        "per_task": tasks,
    }


def _counts(values: Iterable[str]) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    for value in values:
        out[str(value)] += 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))

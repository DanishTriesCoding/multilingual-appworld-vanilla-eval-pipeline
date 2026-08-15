"""Human-readable rendering of evaluation results and CSV reporting."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:6.2f}%"


def render(summary: dict[str, Any], title: str = "AppWorld vanilla evaluation") -> str:
    if not summary.get("n_tasks"):
        return "No results yet."

    k = summary["k"]
    lines = [
        "",
        "=" * 62,
        title,
        "=" * 62,
        f"tasks evaluated      : {summary['n_tasks']}",
        f"rollouts             : {summary['n_rollouts']}  (k = {k})",
        "",
        "-- task level (TGC) ------------------------------------------",
        f"  avg@{k:<3}            : {_pct(summary['avg_at_k'])}   "
        f"95% CI [{_pct(summary['avg_at_k_ci95'][0]).strip()}, {_pct(summary['avg_at_k_ci95'][1]).strip()}]",
        f"  best@{k:<3}           : {_pct(summary['best_at_k'])}   "
        f"95% CI [{_pct(summary['best_at_k_ci95'][0]).strip()}, {_pct(summary['best_at_k_ci95'][1]).strip()}]",
        f"  unbiased pass@{k:<3}  : {_pct(summary['unbiased_pass_at_k'])}",
        "",
        "-- scenario level (SGC-style) --------------------------------",
        f"  scenarios            : {summary['n_scenarios']}",
        f"  avg@{k:<3}            : {_pct(summary['scenario_avg_at_k'])}",
        f"  best@{k:<3}           : {_pct(summary['scenario_best_at_k'])}",
        "",
        "-- diagnostics -----------------------------------------------",
        f"  rollout error rate   : {_pct(summary['rollout_error_rate'])}",
        f"  mean steps/rollout   : {summary['mean_steps']:.1f}",
        "  finish reasons:",
    ]
    for reason, count in summary["finish_reasons"].items():
        lines.append(f"    {count:>6}  {reason}")
    solved = [t for t in summary["per_task"] if t["n_success"] > 0]
    lines += [
        "",
        f"  tasks solved at least once: {len(solved)} / {summary['n_tasks']}",
    ]
    if solved:
        lines.append("  " + ", ".join(t["task_id"] for t in solved[:25]))
    lines.append("=" * 62)
    return "\n".join(lines)


def write_csv(summary: dict[str, Any], path: str | Path) -> Path:
    """Write per-task metrics bypassing AppWorld's standard file-hook protections."""
    target_path = Path(path)
    rows = summary.get("per_task", [])
    fields = [
        "task_id",
        "scenario_id",
        "n_rollouts",
        "n_success",
        "avg",
        "best",
        "pass_at_k",
        "errors",
        "mean_steps",
    ]
    out = [",".join(fields)]
    for row in rows:
        out.append(",".join(str(row.get(f, "")) for f in fields))

    fd = os.open(str(target_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.write(fd, ("\n".join(out) + "\n").encode("utf-8"))
    finally:
        os.close(fd)
    return target_path
"""Rollout execution: one (task_id, seed) -> one scored trajectory."""

from __future__ import annotations

import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from typing import Any, Callable

from . import env as envmod
from .agent import VanillaCodeAgent
from .config import ExperimentConfig
from .llm_client import build_client
from .storage import ResultStore


_INSTR_MAP = {}
if os.environ.get("INSTRUCTION_MAP"):
    import json as _json
    with open(os.environ["INSTRUCTION_MAP"]) as _f:
        _INSTR_MAP = _json.load(_f)


def experiment_name_for(cfg: ExperimentConfig, seed: int) -> str:
    """One AppWorld experiment per seed.

    This mirrors how AppWorld itself is organised (an experiment = one pass
    over a split) and means you can run the official CLI evaluator per seed:
        appworld evaluate <experiment_name> <split>
    """
    return f"{cfg.run.experiment_prefix}_seed{seed}"


def run_rollout(cfg: ExperimentConfig, task_id: str, seed: int) -> dict[str, Any]:
    started = time.time()
    client = build_client(cfg.llm)
    agent = VanillaCodeAgent(cfg=cfg.agent, client=client, seed=seed)

    record: dict[str, Any] = {
        "task_id": task_id,
        "seed": seed,
        "split": cfg.run.split,
        "experiment": experiment_name_for(cfg, seed),
        "success": False,
        "num_steps": 0,
        "finished_reason": "unknown",
        "error": None,
        "eval": None,
        "completion_tokens": 0,
        "prompt_tokens": 0,
        "wall_time_s": 0.0,
    }
    trajectory: dict[str, Any] = {"meta": dict(record), "steps": []}

    try:
        with envmod.open_world(
            task_id=task_id,
            experiment_name=experiment_name_for(cfg, seed),
            remote_environment_url=cfg.env.remote_environment_url,
            extra_kwargs=cfg.env.extra_kwargs,
        ) as world:
            instruction = _INSTR_MAP.get(task_id, envmod.get_instruction(world))
            supervisor = envmod.get_supervisor(world) if cfg.agent.include_supervisor_header else {}
            trajectory["instruction"] = instruction
            agent.reset(instruction, supervisor)

            for step_index in range(cfg.agent.max_steps):
                if envmod.task_completed(world):
                    record["finished_reason"] = "task_completed"
                    break

                parsed = agent.propose(step_index)
                if not parsed.ok:
                    record["finished_reason"] = f"unparseable_output: {parsed.reason}"
                    break

                try:
                    output = envmod.execute(world, parsed.code)
                except Exception as exc:  # noqa: BLE001 - env crash ends the rollout
                    record["finished_reason"] = "env_execute_error"
                    record["error"] = f"{type(exc).__name__}: {exc}"
                    break

                agent.record(step_index, parsed, output)
                record["num_steps"] = step_index + 1
            else:
                record["finished_reason"] = "max_steps_exhausted"

            if envmod.task_completed(world) and record["finished_reason"] == "unknown":
                record["finished_reason"] = "task_completed"

            # Score with AppWorld's own evaluator, regardless of how we exited.
            try:
                evaluation = envmod.evaluate(world)
                record["success"] = bool(evaluation["success"])
                record["eval"] = {
                    k: v for k, v in evaluation.items() if k != "raw"
                }
                trajectory["eval_raw"] = evaluation["raw"]
            except Exception as exc:  # noqa: BLE001
                record["error"] = (record["error"] or "") + f" | eval_error: {exc}"
                record["success"] = False

    except Exception as exc:  # noqa: BLE001 - never let one task kill the sweep
        record["finished_reason"] = "rollout_exception"
        record["error"] = f"{type(exc).__name__}: {exc}"
        trajectory["traceback"] = traceback.format_exc()

    record["completion_tokens"] = sum(s.completion_tokens for s in agent.steps)
    record["prompt_tokens"] = sum(s.prompt_tokens for s in agent.steps)
    record["num_steps"] = len(agent.steps)
    record["wall_time_s"] = round(time.time() - started, 2)

    trajectory["meta"] = dict(record)
    trajectory["steps"] = [asdict(s) for s in agent.steps]
    record["_trajectory"] = trajectory
    return record


def _worker(args):
    cfg, task_id, seed = args
    return run_rollout(cfg, task_id, seed)


def run_sweep(cfg, task_ids, store, on_result=None):
    import multiprocessing as mp
    seeds = [cfg.run.seed_base + i for i in range(cfg.run.num_rollouts)]
    jobs = [(t, s) for t in task_ids for s in seeds]
    done = store.completed_keys() if cfg.run.resume else set()
    pending = [j for j in jobs if j not in done]
    if not pending:
        return []
    payload = [(cfg, t, s) for t, s in pending]
    results = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=max(1, cfg.run.max_workers), maxtasksperchild=1) as pool:
        for record in pool.imap_unordered(_worker, payload):
            traj = record.pop("_trajectory", None)
            if traj:
                try:
                    store.save_trajectory(record["task_id"], record["seed"], traj)
                except Exception as exc:
                    print("[warn] trajectory write failed:", exc)
            store.append(record)
            results.append(record)
            if on_result:
                on_result(record)
    return results

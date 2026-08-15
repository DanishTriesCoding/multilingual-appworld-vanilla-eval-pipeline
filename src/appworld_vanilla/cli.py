"""Command line entrypoints.

    appworld-vanilla check   -c configs/vanilla_qwen7b.yaml
    appworld-vanilla sample  -c configs/vanilla_qwen7b.yaml
    appworld-vanilla run     -c configs/vanilla_qwen7b.yaml
    appworld-vanilla report  -c configs/vanilla_qwen7b.yaml
    appworld-vanilla official-eval -c configs/vanilla_qwen7b.yaml
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

from . import env as envmod
from .config import ExperimentConfig, parse_dotted_overrides
from .llm_client import OpenAICompatibleClient
from .metrics import aggregate
from .report import render, write_csv
from .runner import experiment_name_for, run_sweep
from .sampler import describe_selection, select_task_ids
from .storage import ResultStore


def _load_cfg(args) -> ExperimentConfig:
    overrides = parse_dotted_overrides(args.set or [])
    return ExperimentConfig.load(args.config, overrides)


def _task_ids(cfg: ExperimentConfig) -> list[str]:
    if cfg.run.task_ids_file:
        path = Path(cfg.run.task_ids_file)
        if path.exists():
            return [line.strip() for line in path.read_text().splitlines() if line.strip()]
        print(f"[warn] task_ids_file {path} missing; falling back to full split", file=sys.stderr)
    return envmod.load_split_task_ids(cfg.run.split)


# ---------------------------------------------------------------------- #


def cmd_check(args) -> int:
    cfg = _load_cfg(args)
    print("LLM endpoint:", cfg.llm.base_url)
    client = OpenAICompatibleClient(cfg.llm)
    info = client.health_check()
    print(json.dumps(info, indent=2))
    if info.get("served_models") and cfg.llm.model not in info["served_models"]:
        print(
            f"\n[warn] configured model '{cfg.llm.model}' is not in the served list "
            f"{info['served_models']}. Set llm.model to one of those.",
            file=sys.stderr,
        )
    try:
        ids = envmod.load_split_task_ids(cfg.run.split)
        print(f"\nAppWorld split '{cfg.run.split}': {len(ids)} tasks")
        print("first ids:", ids[:5])
    except Exception as exc:  # noqa: BLE001
        print(f"\n[error] AppWorld not usable: {exc}", file=sys.stderr)
        return 1
    return 0


def cmd_sample(args) -> int:
    cfg = _load_cfg(args)
    all_ids = envmod.load_split_task_ids(cfg.run.split)
    selected = select_task_ids(all_ids, cfg.sample)
    out = Path(args.out or f"task_ids_{cfg.run.split}_{len(selected)}.txt")
    out.write_text("\n".join(selected) + "\n")
    print(json.dumps(describe_selection(all_ids, selected), indent=2))
    print(f"\nwrote {len(selected)} task ids -> {out}")
    print(f"set run.task_ids_file: {out} in your config (or --set run.task_ids_file={out})")
    return 0


def cmd_run(args) -> int:
    cfg = _load_cfg(args)
    task_ids = _task_ids(cfg)
    store = ResultStore(cfg.run.output_dir, cfg.run.experiment_prefix)
    store.write_json("config.json", cfg.to_dict())
    store.write_json("task_ids.json", task_ids)

    total = len(task_ids) * cfg.run.num_rollouts
    already = len(store.completed_keys()) if cfg.run.resume else 0
    print(
        f"experiment : {cfg.run.experiment_prefix}\n"
        f"split      : {cfg.run.split}\n"
        f"tasks      : {len(task_ids)}\n"
        f"rollouts   : {cfg.run.num_rollouts} per task  -> {total} total "
        f"({already} already done)\n"
        f"workers    : {cfg.run.max_workers}\n"
        f"output     : {store.dir}\n"
    )
    if args.dry_run:
        print("dry run; exiting before any rollout.")
        return 0

    counter = {"n": already, "ok": 0}
    lock = threading.Lock()
    started = time.time()

    def progress(record):
        with lock:
            counter["n"] += 1
            counter["ok"] += int(bool(record.get("success")))
            elapsed = time.time() - started
            rate = counter["n"] / max(elapsed / 60, 1e-6)
            done_now = counter["n"] - already
            eta = (total - counter["n"]) / max(done_now / max(elapsed, 1e-6), 1e-9) if done_now else 0
            print(
                f"[{counter['n']}/{total}] {record['task_id']} seed={record['seed']} "
                f"{'PASS' if record.get('success') else 'fail'} "
                f"steps={record.get('num_steps')} reason={record.get('finished_reason')} "
                f"| {rate:.1f}/min eta={eta/60:.0f}m",
                flush=True,
            )

    run_sweep(cfg, task_ids, store, on_result=progress)

    records = store.load_rollouts()
    summary = aggregate(records, k=cfg.run.num_rollouts)
    store.write_json("summary.json", {k: v for k, v in summary.items() if k != "per_task"})
    write_csv(summary, store.dir / "per_task.csv")
    print(render(summary, title=f"{cfg.run.experiment_prefix} / {cfg.run.split}"))
    return 0


def cmd_report(args) -> int:
    cfg = _load_cfg(args)
    store = ResultStore(cfg.run.output_dir, cfg.run.experiment_prefix)
    records = store.load_rollouts()
    if args.k:
        wanted = set(range(cfg.run.seed_base, cfg.run.seed_base + args.k))
        records = [r for r in records if int(r.get("seed", -1)) in wanted]
    summary = aggregate(records, k=args.k or cfg.run.num_rollouts)
    store.write_json("summary.json", {k: v for k, v in summary.items() if k != "per_task"})
    write_csv(summary, store.dir / "per_task.csv")
    print(render(summary, title=f"{cfg.run.experiment_prefix} / {cfg.run.split}"))
    print(f"\nper-task csv: {store.dir / 'per_task.csv'}")
    return 0


def cmd_official_eval(args) -> int:
    """Shell out to AppWorld's own evaluator, once per seed-experiment.

    Our JSONL is the source of truth for avg@k / best@k; this gives you the
    benchmark's canonical TGC/SGC per individual pass, which is what you would
    quote or submit.
    """
    cfg = _load_cfg(args)
    codes = []
    for i in range(cfg.run.num_rollouts):
        name = experiment_name_for(cfg, cfg.run.seed_base + i)
        cmd = ["appworld", "evaluate", name, cfg.run.split]
        print("$", " ".join(cmd), flush=True)
        codes.append(subprocess.run(cmd, check=False).returncode)
    return 0 if all(c == 0 for c in codes) else 1


# ---------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="appworld-vanilla", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("-c", "--config", required=True, help="path to YAML config")
        p.add_argument(
            "--set",
            nargs="*",
            default=[],
            metavar="KEY=VALUE",
            help="dotted overrides, e.g. --set run.num_rollouts=2 llm.temperature=1.0",
        )
        return p

    common(sub.add_parser("check", help="probe the LLM endpoint and AppWorld install")).set_defaults(
        func=cmd_check
    )

    p_sample = common(sub.add_parser("sample", help="write a subsampled task id file"))
    p_sample.add_argument("--out", help="output path for the task id list")
    p_sample.set_defaults(func=cmd_sample)

    p_run = common(sub.add_parser("run", help="run the evaluation sweep"))
    p_run.add_argument("--dry-run", action="store_true")
    p_run.set_defaults(func=cmd_run)

    p_report = common(sub.add_parser("report", help="recompute metrics from stored rollouts"))
    p_report.add_argument("--k", type=int, help="report at a smaller k than was run")
    p_report.set_defaults(func=cmd_report)

    common(
        sub.add_parser("official-eval", help="run AppWorld's own CLI evaluator per seed")
    ).set_defaults(func=cmd_official_eval)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

"""Offline self-test: exercises everything except AppWorld and the LLM server.

    python scripts/selftest.py

Uses a scripted fake model and a fake environment, so it verifies the plumbing
(config -> agent -> parser -> runner -> storage -> metrics -> report) without
needing a GPU or the AppWorld data download.
"""

from __future__ import annotations

import json
import os
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from appworld_vanilla import env as envmod  # noqa: E402
from appworld_vanilla.config import ExperimentConfig, parse_dotted_overrides  # noqa: E402
from appworld_vanilla.metrics import aggregate, unbiased_pass_at_k  # noqa: E402
from appworld_vanilla.parsers import parse_action, truncate_output  # noqa: E402
from appworld_vanilla.prompts import load_official_messages  # noqa: E402
from appworld_vanilla.report import render, write_csv  # noqa: E402
from appworld_vanilla.sampler import select_task_ids  # noqa: E402
from appworld_vanilla.storage import ResultStore  # noqa: E402


def test_parser():
    a = parse_action(
        "Thought: look around.\n```python\nprint(apis.api_docs.show_app_descriptions())\n```"
    )
    assert a.ok and a.code.startswith("print(apis")
    assert a.thought == "Thought: look around."

    a_nofence = parse_action("apis.spotify.login()")
    assert a_nofence.ok and a_nofence.code == "apis.spotify.login()"

    a_bad = parse_action("Just text with no python block")
    assert not a_bad.ok

    truncated = truncate_output("1234567890" * 100, 20)
    assert "elided" in truncated or "..." in truncated
    assert len(truncated) < len("1234567890" * 100)
    print("  parser ok")


def test_config():
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write("run:\n  num_rollouts: 4\n  instruction_map_file: 'custom.json'\n")
        f_path = f.name
    try:
        cfg = ExperimentConfig.load(f_path, parse_dotted_overrides(["llm.temperature=0.2"]))
        assert cfg.run.num_rollouts == 4
        assert cfg.run.instruction_map_file == "custom.json"
        assert cfg.llm.temperature == 0.2
    finally:
        os.unlink(f_path)
    print("  config ok")


def test_prompts():
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("USER:\nHello {{ main_user.first_name }}\nTask: {{ instruction }}\nASSISTANT:\nUnderstood.")
        f_path = f.name
    try:
        msgs = load_official_messages(f_path, "Do task X", {"first_name": "Alice"})
        assert len(msgs) == 2
        assert "Hello Alice" in msgs[0]["content"]
        assert "Do task X" in msgs[0]["content"]
        assert msgs[1]["content"] == "Understood."
    finally:
        os.unlink(f_path)
    print("  prompts ok")


def test_sampler():
    all_ids = ["sc1_1", "sc1_2", "sc1_3", "sc2_1", "sc2_2", "sc2_3"]
    cfg = ExperimentConfig().sample
    cfg.size = 3
    cfg.strategy = "scenario_stratified"
    picked = select_task_ids(all_ids, cfg)
    assert len(picked) == 3
    scenarios = {t.split("_")[0] for t in picked}
    assert len(scenarios) == 1  # scenario_stratified keeps task triplets together
    print("  sampler ok")


def test_metrics():
    assert unbiased_pass_at_k(8, 2, 8) == 1.0
    assert unbiased_pass_at_k(8, 0, 8) == 0.0

    dummy_records = [
        {"task_id": "a_1", "seed": 1000, "success": True, "finished_reason": "task_completed"},
        {"task_id": "a_1", "seed": 1001, "success": False, "finished_reason": "max_steps_exhausted"},
        {"task_id": "a_2", "seed": 1000, "success": False, "finished_reason": "unparseable_output"},
        {"task_id": "a_2", "seed": 1001, "success": False, "finished_reason": "unparseable_output"},
    ]
    summary = aggregate(dummy_records, k=2)
    assert summary["n_tasks"] == 2
    assert summary["avg_at_k"] == 0.25
    assert summary["best_at_k"] == 0.5
    rendered = render(summary)
    assert "avg@2" in rendered
    print("  metrics & report ok")


def test_storage():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ResultStore(tmpdir, "test_exp")
        store.append({"task_id": "t1", "seed": 1, "success": True})
        store.save_trajectory("t1", 1, {"meta": {}, "steps": []})
        loaded = store.load_rollouts()
        assert len(loaded) == 1
        assert store.completed_keys() == {("t1", 1)}

        # Test safe CSV writing
        csv_path = Path(tmpdir) / "test.csv"
        write_csv({"per_task": [{"task_id": "t1", "avg": 1.0}]}, csv_path)
        assert csv_path.exists()
    print("  storage & csv writing ok")


def test_agent_and_runner():
    from appworld_vanilla import runner
    from appworld_vanilla.config import ExperimentConfig
    import contextlib

    class DeadWorld:
        task = type("T", (), {"instruction": "x", "supervisor": None})()

        def execute(self, code):
            raise RuntimeError("boom")

        def task_completed(self):
            return False

        def evaluate(self):
            return {"success": False, "failures": ["nope"]}

        def close(self):
            pass

    @contextlib.contextmanager
    def fake_open(**kwargs):
        yield DeadWorld()

    class Factory:
        def __call__(self, cfg):
            from appworld_vanilla.llm_client import ScriptedClient

            return ScriptedClient(["```python\nprint(1)\n```"])

    orig_open, orig_build = runner.envmod.open_world, runner.build_client
    runner.envmod.open_world = fake_open
    runner.build_client = Factory()
    try:
        rec = runner.run_rollout(ExperimentConfig(), "a_1", 1)
    finally:
        runner.envmod.open_world = orig_open
        runner.build_client = orig_build
    assert rec["success"] is False and rec["finished_reason"] == "env_execute_error"
    print("  error handling & runner ok")


if __name__ == "__main__":
    print("Running offline selftest...")
    test_parser()
    test_config()
    test_prompts()
    test_sampler()
    test_metrics()
    test_storage()
    test_agent_and_runner()
    print("\nALL SELFTESTS PASSED SUCCESSFULLY!")
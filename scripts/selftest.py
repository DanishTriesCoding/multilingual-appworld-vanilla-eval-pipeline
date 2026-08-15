"""Offline self-test: exercises everything except AppWorld and the LLM server.

    python scripts/selftest.py

Uses a scripted fake model and a fake environment, so it verifies the plumbing
(config -> agent -> parser -> runner -> storage -> metrics -> report) without
needing a GPU or the AppWorld data download.
"""

from __future__ import annotations

import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from appworld_vanilla import env as envmod  # noqa: E402
from appworld_vanilla.config import ExperimentConfig, parse_dotted_overrides  # noqa: E402
from appworld_vanilla.metrics import aggregate, unbiased_pass_at_k  # noqa: E402
from appworld_vanilla.parsers import parse_action, truncate_output  # noqa: E402
from appworld_vanilla.report import render  # noqa: E402
from appworld_vanilla.sampler import select_task_ids  # noqa: E402
from appworld_vanilla.storage import ResultStore  # noqa: E402


def test_parser():
    a = parse_action("Thought: look around.\n```python\nprint(apis.api_docs.show_app_descriptions())\n```")
    assert a.ok and a.code.startswith("print(apis")
    assert a.thought.startswith("Thought:")

    # Only the FIRST block is taken, hallucinated follow-ups ignored.
    b = parse_action("```python\nx=1\n```\nExecution output:\nfake\n```python\ny=2\n```")
    assert b.code == "x=1", b.code

    assert not parse_action("I am not sure what to do here, sorry.").ok
    assert not parse_action("").ok
    assert parse_action("print(apis.supervisor.show_profile())").ok  # unfenced code

    assert len(truncate_output("x" * 5000, 1000)) < 1200
    assert truncate_output("short", 1000) == "short"
    print("  parser ok")


def test_config():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "c.yaml"
        path.write_text("llm:\n  temperature: 0.3\nrun:\n  split: dev\n")
        cfg = ExperimentConfig.load(path, parse_dotted_overrides(["run.num_rollouts=3"]))
        assert cfg.llm.temperature == 0.3
        assert cfg.run.split == "dev"
        assert cfg.run.num_rollouts == 3
        assert cfg.agent.max_steps == 40  # default preserved
        assert isinstance(cfg.llm.extra_body, dict)
    print("  config ok")


def test_sampler():
    ids = [f"sc{i:03d}_{j}" for i in range(50) for j in (1, 2, 3)]
    from appworld_vanilla.config import SampleConfig

    full = select_task_ids(ids, SampleConfig(size=None))
    assert len(full) == 150

    r1 = select_task_ids(ids, SampleConfig(strategy="random", size=30, seed=1))
    r2 = select_task_ids(ids, SampleConfig(strategy="random", size=30, seed=1))
    assert r1 == r2 and len(r1) == 30, "sampling must be deterministic given a seed"

    strat = select_task_ids(ids, SampleConfig(strategy="scenario_stratified", size=30, seed=1))
    scenarios = {envmod.scenario_id_of(t) for t in strat}
    assert len(strat) == 30 and len(scenarios) == 10, (len(strat), len(scenarios))
    print("  sampler ok")


def test_metrics():
    assert abs(unbiased_pass_at_k(8, 0, 8) - 0.0) < 1e-9
    assert abs(unbiased_pass_at_k(8, 8, 8) - 1.0) < 1e-9
    assert abs(unbiased_pass_at_k(8, 1, 1) - 0.125) < 1e-9

    # Two scenarios x 3 tasks x 4 seeds, ~15% success.
    rng = random.Random(0)
    records = []
    for s in range(2):
        for t in (1, 2, 3):
            for seed in range(4):
                records.append(
                    {
                        "task_id": f"sc{s}_{t}",
                        "seed": seed,
                        "success": rng.random() < 0.15,
                        "num_steps": rng.randint(3, 20),
                        "finished_reason": "task_completed",
                    }
                )
    summary = aggregate(records, k=4)
    assert summary["n_tasks"] == 6 and summary["n_rollouts"] == 24
    assert 0.0 <= summary["avg_at_k"] <= summary["best_at_k"] <= 1.0
    assert summary["n_scenarios"] == 2
    lo, hi = summary["avg_at_k_ci95"]
    assert lo <= summary["avg_at_k"] <= hi

    # Degenerate all-fail case (what a real vanilla 7B run looks like a lot).
    zeros = [{"task_id": "a_1", "seed": i, "success": False, "num_steps": 40} for i in range(8)]
    z = aggregate(zeros, k=8)
    assert z["avg_at_k"] == 0.0 and z["best_at_k"] == 0.0
    assert render(z)
    print("  metrics ok")


def test_storage_and_resume():
    with tempfile.TemporaryDirectory() as tmp:
        store = ResultStore(tmp, "exp")
        store.append({"task_id": "a_1", "seed": 1, "success": True, "num_steps": 5})
        store.append({"task_id": "a_1", "seed": 2, "success": False, "num_steps": 9})
        store.save_trajectory("a_1", 1, {"steps": [{"code": "print(1)"}]})
        assert store.completed_keys() == {("a_1", 1), ("a_1", 2)}
        assert len(store.load_rollouts()) == 2
        # torn line tolerance
        with store.rollouts_path.open("a") as fh:
            fh.write('{"task_id": "a_1", "seed"')
        assert len(store.load_rollouts()) == 2
    print("  storage ok")


def test_agent_with_fake_env():
    """Full runner loop against a stubbed AppWorld."""
    from appworld_vanilla import runner
    from appworld_vanilla.config import ExperimentConfig

    class FakeWorld:
        def __init__(self):
            self.calls = []
            self.done = False
            self.task = type("T", (), {"instruction": "Play a song.", "supervisor": None})()

        def execute(self, code):
            self.calls.append(code)
            if "complete_task" in code:
                self.done = True
                return "Task marked complete."
            return "some output"

        def task_completed(self):
            return self.done

        def evaluate(self):
            return {"success": self.done, "failures": [] if self.done else ["x"]}

        def close(self):
            pass

    import contextlib

    worlds = []

    @contextlib.contextmanager
    def fake_open(**kwargs):
        w = FakeWorld()
        worlds.append(w)
        yield w

    scripted = [
        "Thought: explore.\n```python\nprint(apis.api_docs.show_app_descriptions())\n```",
        "Thought: done.\n```python\napis.supervisor.complete_task()\n```",
    ]

    class FakeClientFactory:
        def __call__(self, cfg):
            from appworld_vanilla.llm_client import ScriptedClient

            return ScriptedClient(scripted)

    orig_open, orig_build = runner.envmod.open_world, runner.build_client
    runner.envmod.open_world = fake_open
    runner.build_client = FakeClientFactory()
    try:
        cfg = ExperimentConfig()
        cfg.agent.max_steps = 6
        record = runner.run_rollout(cfg, "a_1", 1)
    finally:
        runner.envmod.open_world = orig_open
        runner.build_client = orig_build

    assert record["success"] is True, record
    assert record["num_steps"] == 2, record["num_steps"]
    assert record["finished_reason"] == "task_completed", record["finished_reason"]
    assert record["_trajectory"]["steps"][0]["code"].startswith("print(apis")
    print("  runner + agent ok")


def test_runner_survives_bad_model():
    """A model that never emits code must fail cleanly, not crash the sweep."""
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
    print("  error handling ok")


if __name__ == "__main__":
    print("running offline self-test")
    test_parser()
    test_config()
    test_sampler()
    test_metrics()
    test_storage_and_resume()
    test_agent_with_fake_env()
    test_runner_survives_bad_model()
    print("\nall self-tests passed")

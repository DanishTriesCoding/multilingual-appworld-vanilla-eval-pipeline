"""The only module that imports `appworld`.

Everything AppWorld-version-specific is isolated here, so if the upstream API
shifts you fix one file. All accessors are defensive: they try the documented
path first and degrade gracefully rather than killing a 4000-rollout run.
"""

from __future__ import annotations

import contextlib
from typing import Any, Iterator

_IMPORT_ERROR = (
    "Could not import `appworld`. Install it first:\n"
    "    pip install appworld\n"
    "    appworld install\n"
    "    appworld download data\n"
)


def _appworld():
    try:
        import appworld  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(_IMPORT_ERROR) from exc
    return appworld


def load_split_task_ids(split: str) -> list[str]:
    """Task ids for a split: train | dev | test_normal | test_challenge."""
    aw = _appworld()
    return list(aw.load_task_ids(split))


@contextlib.contextmanager
def open_world(
    task_id: str,
    experiment_name: str,
    remote_environment_url: str | None = None,
    extra_kwargs: dict[str, Any] | None = None,
) -> Iterator[Any]:
    aw = _appworld()
    kwargs: dict[str, Any] = {"task_id": task_id, "experiment_name": experiment_name}
    if remote_environment_url:
        kwargs["remote_environment_url"] = remote_environment_url
    kwargs.update(extra_kwargs or {})
    try:
        world = aw.AppWorld(**kwargs)
    except TypeError:
        # Older/newer signature rejected one of our optional kwargs.
        world = aw.AppWorld(task_id=task_id, experiment_name=experiment_name)
    try:
        yield world
    finally:
        with contextlib.suppress(Exception):
            world.close()


def get_instruction(world: Any) -> str:
    return str(getattr(world.task, "instruction", "") or "")


def get_supervisor(world: Any) -> dict[str, str]:
    """Best-effort name/email/phone of the task's supervisor, for the prompt."""
    sup = getattr(world.task, "supervisor", None)
    if sup is None:
        return {}
    out: dict[str, str] = {}
    for attr in ("first_name", "last_name", "email", "phone_number"):
        value = getattr(sup, attr, None)
        if value:
            out[attr.replace("_", " ")] = str(value)
    if isinstance(sup, dict):
        for key in ("first_name", "last_name", "email", "phone_number"):
            if sup.get(key):
                out[key.replace("_", " ")] = str(sup[key])
    return out


def execute(world: Any, code: str) -> str:
    return str(world.execute(code))


def task_completed(world: Any) -> bool:
    try:
        return bool(world.task_completed())
    except Exception:  # noqa: BLE001
        return False


def evaluate(world: Any) -> dict[str, Any]:
    """Run AppWorld's own evaluator for the current task.

    Returns {"success": bool, "num_passes": int, "num_failures": int, "raw": ...}
    """
    report = world.evaluate()
    return normalize_report(report)


def normalize_report(report: Any) -> dict[str, Any]:
    success = getattr(report, "success", None)
    passes = getattr(report, "passes", None)
    failures = getattr(report, "failures", None)

    if isinstance(report, dict):
        success = report.get("success", success)
        passes = report.get("passes", passes)
        failures = report.get("failures", failures)

    if success is None and failures is not None:
        success = len(failures) == 0

    raw: Any
    to_dict = getattr(report, "to_dict", None)
    if callable(to_dict):
        try:
            raw = to_dict()
        except Exception:  # noqa: BLE001
            raw = str(report)
    elif isinstance(report, dict):
        raw = report
    else:
        raw = str(report)

    return {
        "success": bool(success) if success is not None else False,
        "num_passes": len(passes) if hasattr(passes, "__len__") else None,
        "num_failures": len(failures) if hasattr(failures, "__len__") else None,
        "raw": raw,
    }


def scenario_id_of(task_id: str) -> str:
    """AppWorld task ids look like `<scenario>_<n>`; used for scenario-level (SGC).

    Heuristic, and treated as such: if it does not split, the task is its own
    scenario and SGC collapses to TGC for that group.
    """
    return task_id.rsplit("_", 1)[0] if "_" in task_id else task_id

"""Prompt construction and templating for vanilla and custom agent rollouts."""

from __future__ import annotations

import pathlib
import re
from typing import Any

from .config import AgentConfig

SYSTEM_PROMPT = """You are an autonomous agent that completes tasks for a user by writing Python code.

You are connected to a live sandbox containing that user's real digital life: their
phone, email, file system, shopping accounts, music service, calendar, payment app,
and so on. Every step, you write one short block of Python. It runs in a persistent
IPython session, and you are shown its output. You then write the next block.

## The `apis` object

A pre-loaded object named `apis` exposes every available API. You do not need to
import it. Discover your way down, one level at a time:

```python
print(apis.api_docs.show_app_descriptions())
```

```python
print(apis.api_docs.show_api_descriptions(app_name='spotify'))
```

```python
print(apis.api_docs.show_api_doc(app_name='spotify', api_name='login'))
```

Read the doc for an API before you call it. Guessing argument names wastes turns.

## The supervisor

The user who gave you the task is the "supervisor". These are always available:

```python
print(apis.supervisor.show_profile())
print(apis.supervisor.show_account_passwords())
print(apis.supervisor.show_addresses())
```

Use them to obtain credentials, email addresses, phone numbers and so on.
Never invent a credential, an id, or an email address. Look it up.

## Logging in

Most apps require an access token first:

```python
token = apis.spotify.login(username='...', password='...')['access_token']
```

Pass that token to subsequent calls for that app.

## Rules

1. Exactly ONE fenced ```python block per message. Everything after the first
   block is ignored.
2. Never write the execution output yourself. Emit the block, then stop and wait.
3. `print(...)` anything you need to see. Un-printed values are not shown.
4. Variables persist across blocks, so build up state incrementally.
5. Paginate. List APIs usually return a page at a time; keep requesting pages
   until you have everything before you filter or count.
6. Solve the task exactly as stated: no more, no less. Do not take extra
   liberties (do not delete things you were not asked to delete).
7. When, and only when, the task is fully done, finish with:

```python
apis.supervisor.complete_task()
```

   If the task asked you to return an answer, pass it:

```python
apis.supervisor.complete_task(answer=my_answer)
```

## Response format

A sentence or two of reasoning, then the single code block:

Thought: I need to see which apps exist before anything else.
```python
print(apis.api_docs.show_app_descriptions())
```
"""

FIRST_USER_TEMPLATE = """{supervisor_block}Task:
{task_instruction}

You have at most {max_steps} code blocks. Begin."""


def build_system_prompt(cfg: AgentConfig) -> str:
    if cfg.prompt_variant == "custom":
        if not cfg.prompt_path:
            raise ValueError("agent.prompt_variant='custom' requires agent.prompt_path")
        return pathlib.Path(cfg.prompt_path).read_text(encoding="utf-8")
    return SYSTEM_PROMPT


def build_first_user_message(
    cfg: AgentConfig, task_instruction: str, supervisor: dict | None
) -> str:
    supervisor_block = ""
    if cfg.include_supervisor_header and supervisor:
        lines = [f"- {k}: {v}" for k, v in supervisor.items() if v]
        if lines:
            supervisor_block = "You are working on behalf of:\n" + "\n".join(lines) + "\n\n"
    return FIRST_USER_TEMPLATE.format(
        supervisor_block=supervisor_block,
        task_instruction=task_instruction,
        max_steps=cfg.max_steps,
    )


def build_observation_message(output: str, step: int, max_steps: int) -> str:
    remaining = max_steps - step
    tail = ""
    if remaining <= 3:
        tail = (
            f"\n\n[{remaining} code blocks remaining. Finish the task and call "
            "apis.supervisor.complete_task().]"
        )
    return f"Execution output:\n{output}{tail}"


def load_official_messages(
    path: str | pathlib.Path,
    task_instruction: str,
    supervisor: dict[str, Any] | None,
    app_descriptions: str = "",
) -> list[dict[str, str]]:
    """Split AppWorld's react_code_agent prompt templates into proper chat turns."""
    raw = pathlib.Path(path).read_text(encoding="utf-8")
    sup = supervisor or {}
    subs = {
        "instruction": task_instruction,
        "app_descriptions": app_descriptions,
        "main_user.first_name": sup.get("first name", sup.get("first_name", "")),
        "main_user.last_name": sup.get("last name", sup.get("last_name", "")),
        "main_user.email": sup.get("email", ""),
        "main_user.phone_number": sup.get("phone number", sup.get("phone_number", "")),
    }
    for k, v in subs.items():
        raw = re.sub(r"\{\{\s*" + re.escape(k) + r"\s*\}\}", str(v), raw)

    msgs: list[dict[str, str]] = []
    role: str | None = None
    buf: list[str] = []

    for line in raw.split("\n"):
        stripped = line.strip()
        if stripped in ("USER:", "ASSISTANT:"):
            if role and buf:
                msgs.append({"role": role, "content": "\n".join(buf).strip()})
            role = "user" if stripped == "USER:" else "assistant"
            buf = []
        else:
            buf.append(line)
    if role and buf:
        msgs.append({"role": role, "content": "\n".join(buf).strip()})

    merged: list[dict[str, str]] = []
    for m in msgs:
        if not m["content"]:
            continue
        if merged and merged[-1]["role"] == m["role"]:
            merged[-1]["content"] += "\n" + m["content"]
        else:
            merged.append(m)
    return merged
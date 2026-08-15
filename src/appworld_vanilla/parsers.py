"""Turn a raw LLM completion into (thought, code)."""

from __future__ import annotations

import re
from dataclasses import dataclass

_FENCED = re.compile(r"```(?:python|py|ipython)?[ \t]*\r?\n(.*?)(?:```|\Z)", re.DOTALL)
_CODE_HINTS = ("print(", "apis.", "import ", "for ", "if ", "=", "def ")


@dataclass
class ParsedAction:
    thought: str
    code: str
    raw: str
    ok: bool
    reason: str = ""


def parse_action(raw: str) -> ParsedAction:
    """Take the FIRST fenced block. Anything before it is the thought.

    Taking the first block (not the last) matters: small models frequently
    hallucinate an execution output and a second code block in one turn.
    Everything after the first block is discarded, so the environment stays
    the single source of truth for observations.
    """
    text = (raw or "").strip()
    if not text:
        return ParsedAction("", "", raw, False, "empty completion")

    match = _FENCED.search(text)
    if match:
        code = match.group(1).strip()
        thought = text[: match.start()].strip()
        if not code:
            return ParsedAction(thought, "", raw, False, "empty code block")
        return ParsedAction(thought, code, raw, True)

    # No fence at all: accept it only if it plausibly *is* code.
    if any(hint in text for hint in _CODE_HINTS) and "\n\n" not in text[:200]:
        return ParsedAction("", text, raw, True, "unfenced code accepted")

    return ParsedAction(text, "", raw, False, "no code block found")


def truncate_output(text: str, limit: int) -> str:
    """Keep head and tail; drop the middle. Long API dumps blow up context."""
    if limit <= 0 or len(text) <= limit:
        return text
    head = limit * 2 // 3
    tail = limit - head
    dropped = len(text) - limit
    return f"{text[:head]}\n\n... [{dropped} characters truncated] ...\n\n{text[-tail:]}"

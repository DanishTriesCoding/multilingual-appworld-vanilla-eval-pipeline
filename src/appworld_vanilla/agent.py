"""The policy. Holds message history, proposes one code block per step.

Deliberately knows nothing about AppWorld or about rollout bookkeeping - it
receives observations as strings and returns code as strings. Swap this class
out (same three methods) to try a different agent scaffold.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import AgentConfig
from .llm_client import ChatClient
from .parsers import ParsedAction, parse_action, truncate_output
from .prompts import (
    build_first_user_message,
    build_observation_message,
    build_system_prompt,
    load_official_messages,
)


@dataclass
class Step:
    index: int
    thought: str
    code: str
    output: str
    raw_completion: str
    parse_ok: bool
    parse_reason: str
    completion_tokens: int
    prompt_tokens: int
    latency_s: float


@dataclass
class VanillaCodeAgent:
    cfg: AgentConfig
    client: ChatClient
    seed: int = 0
    messages: list[dict[str, str]] = field(default_factory=list)
    steps: list[Step] = field(default_factory=list)

    def reset(self, task_instruction: str, supervisor: dict | None) -> None:
        if self.cfg.prompt_variant == "custom" and self.cfg.prompt_path:
            self.messages = load_official_messages(
                self.cfg.prompt_path, task_instruction, supervisor
            )
        else:
            self.messages = [
                {"role": "system", "content": build_system_prompt(self.cfg)},
                {
                    "role": "user",
                    "content": build_first_user_message(
                        self.cfg, task_instruction, supervisor
                    ),
                },
            ]
        self.steps = []

    def _context(self) -> list[dict[str, str]]:
        if self.cfg.history_strategy == "full":
            return self.messages
        # Sliding window: keep initial messages (system + initial prompt), plus last 2*N turns
        head = self.messages[:2] if len(self.messages) >= 2 else self.messages[:1]
        tail = self.messages[len(head):]
        keep = self.cfg.keep_last_n_steps * 2
        if len(tail) <= keep:
            return self.messages
        elided = len(tail) - keep
        notice = {
            "role": "user",
            "content": f"[{elided} earlier messages elided to save context.]",
        }
        return head + [notice] + tail[-keep:]

    def propose(self, step_index: int) -> ParsedAction:
        """Query the LLM for the next action, retrying if no code block was emitted."""
        last: ParsedAction | None = None
        for attempt in range(self.cfg.max_empty_code_retries + 1):
            response = self.client.chat(
                self._context(),
                seed=self.seed * 10_000 + step_index * 10 + attempt,
            )
            parsed = parse_action(response.text)
            self._pending_usage = (
                response.prompt_tokens,
                response.completion_tokens,
                response.latency_s,
            )
            if parsed.ok:
                return parsed
            last = parsed
            # Add nudge to history for retry attempt
            self.messages.append({"role": "assistant", "content": response.text})
            self.messages.append(
                {
                    "role": "user",
                    "content": (
                        "That message contained no runnable code block. Reply with a "
                        "short thought followed by exactly one ```python block."
                    ),
                }
            )
        return last or ParsedAction("", "", "", False, "no completion")

    def record(self, step_index: int, parsed: ParsedAction, output: str) -> Step:
        prompt_tokens, completion_tokens, latency = getattr(self, "_pending_usage", (0, 0, 0.0))
        shown = truncate_output(output, self.cfg.output_char_limit)
        self.messages.append({"role": "assistant", "content": parsed.raw})
        self.messages.append(
            {
                "role": "user",
                "content": build_observation_message(shown, step_index + 1, self.cfg.max_steps),
            }
        )
        step = Step(
            index=step_index,
            thought=parsed.thought,
            code=parsed.code,
            output=shown,
            raw_completion=parsed.raw,
            parse_ok=parsed.ok,
            parse_reason=parsed.reason,
            completion_tokens=completion_tokens,
            prompt_tokens=prompt_tokens,
            latency_s=latency,
        )
        self.steps.append(step)
        return step
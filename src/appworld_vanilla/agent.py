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
from .prompts import build_first_user_message, build_observation_message, build_system_prompt


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
            from .prompts import load_official_messages
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

    # ------------------------------------------------------------------ #

    def _context(self) -> list[dict[str, str]]:
        if self.cfg.history_strategy == "full":
            return self.messages
        # "sliding": always keep the system prompt and the task statement,
        # then the most recent 2*N messages (assistant/user pairs).
        head, tail = self.messages[:2], self.messages[2:]
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
        """Ask the model for the next code block, retrying on unparseable output."""
        last: ParsedAction | None = None
        for attempt in range(self.cfg.max_empty_code_retries + 1):
            response = self.client.chat(self._context(), seed=self.seed * 10_000 + step_index * 10 + attempt)
            parsed = parse_action(response.text)
            self._pending_usage = (
                response.prompt_tokens,
                response.completion_tokens,
                response.latency_s,
            )
            if parsed.ok:
                return parsed
            last = parsed
            # Nudge and try again; the nudge stays in history only for the retry.
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

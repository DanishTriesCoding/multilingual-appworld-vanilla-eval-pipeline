import pathlib, re

p = pathlib.Path("src/appworld_vanilla/prompts.py")
s = p.read_text()

s += '''

def load_official_messages(path, task_instruction, supervisor, app_descriptions=""):
    """Split AppWorld's react_code_agent/instructions.txt into real turns."""
    raw = pathlib.Path(path).read_text()
    sup = supervisor or {}
    subs = {
        "instruction": task_instruction,
        "app_descriptions": app_descriptions,
        "main_user.first_name": sup.get("first name", ""),
        "main_user.last_name": sup.get("last name", ""),
        "main_user.email": sup.get("email", ""),
        "main_user.phone_number": sup.get("phone number", ""),
    }
    for k, v in subs.items():
        raw = re.sub(r"\\{\\{\\s*" + re.escape(k) + r"\\s*\\}\\}", str(v), raw)

    msgs, role, buf = [], None, []
    for line in raw.split("\\n"):
        if line.strip() in ("USER:", "ASSISTANT:"):
            if role and buf:
                msgs.append({"role": role, "content": "\\n".join(buf).strip()})
            role = "user" if line.strip() == "USER:" else "assistant"
            buf = []
        else:
            buf.append(line)
    if role and buf:
        msgs.append({"role": role, "content": "\\n".join(buf).strip()})

    merged = []
    for m in msgs:
        if not m["content"]:
            continue
        if merged and merged[-1]["role"] == m["role"]:
            merged[-1]["content"] += "\\n" + m["content"]
        else:
            merged.append(m)
    return merged
'''
if "import pathlib" not in s:
    s = "import pathlib\nimport re\n" + s
p.write_text(s)

a = pathlib.Path("src/appworld_vanilla/agent.py")
s = a.read_text()
old = s[s.index("    def reset("):s.index("    # ---")]
s = s.replace(old, '''    def reset(self, task_instruction: str, supervisor: dict | None) -> None:
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

''')
a.write_text(s)

c = pathlib.Path("src/appworld_vanilla/agent.py").read_text()
assert "load_official_messages" in c
print("VERIFIED")

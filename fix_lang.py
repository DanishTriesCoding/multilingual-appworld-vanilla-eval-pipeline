import pathlib
p = pathlib.Path("src/appworld_vanilla/runner.py")
s = p.read_text()
if "_INSTR_MAP" not in s:
    s = s.replace("def experiment_name_for", '''_INSTR_MAP = {}
if os.environ.get("INSTRUCTION_MAP"):
    import json as _json
    with open(os.environ["INSTRUCTION_MAP"]) as _f:
        _INSTR_MAP = _json.load(_f)


def experiment_name_for''', 1)
    s = s.replace("import time", "import os\nimport time", 1)
    s = s.replace(
        "            instruction = envmod.get_instruction(world)",
        "            instruction = _INSTR_MAP.get(task_id, envmod.get_instruction(world))")
    p.write_text(s)
assert "_INSTR_MAP.get(task_id" in p.read_text()
print("VERIFIED")

import json, os, sys
sys.path.insert(0, "src")
from appworld_vanilla import env as envmod

ids = open("task_ids_70.txt").read().split()
out = {}
for tid in ids:
    with envmod.open_world(tid, "export", None, {}) as w:
        out[tid] = envmod.get_instruction(w)
    print(tid, "ok")

fd = os.open("instructions_en.json", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
os.write(fd, json.dumps(out, ensure_ascii=False, indent=2).encode())
os.close(fd)
print("wrote", len(out), "instructions")

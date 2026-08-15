import sys, traceback
sys.path.insert(0, "src")
from appworld_vanilla.config import ExperimentConfig
from appworld_vanilla import env as envmod
from appworld_vanilla.llm_client import build_client
from appworld_vanilla.agent import VanillaCodeAgent

cfg = ExperimentConfig.load("configs/vanilla_qwen7b.yaml")
cfg.agent.max_steps = 3
task_id = open("task_ids_4.txt").read().split()[0]
print("task:", task_id)

try:
    with envmod.open_world(task_id, "diag", None, {}) as w:
        instr = envmod.get_instruction(w)
        print("INSTRUCTION:", instr[:300])
        agent = VanillaCodeAgent(cfg=cfg.agent, client=build_client(cfg.llm), seed=1)
        agent.reset(instr, envmod.get_supervisor(w))
        for i in range(3):
            p = agent.propose(i)
            print(f"\n--- step {i} parse_ok={p.ok} ---")
            print("CODE:", p.code[:300])
            out = envmod.execute(w, p.code)
            print("OUTPUT:", str(out)[:500])
            agent.record(i, p, out)
        print("\nEVAL:", envmod.evaluate(w))
except Exception:
    traceback.print_exc()

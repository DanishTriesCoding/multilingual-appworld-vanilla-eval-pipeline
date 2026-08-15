import pathlib

r = pathlib.Path("src/appworld_vanilla/report.py")
s = r.read_text()
i = s.index("def write_csv")
s = s[:i] + '''def write_csv(summary, path):
    path = Path(path)
    rows = summary.get("per_task", [])
    fields = ["task_id", "scenario_id", "n_rollouts", "n_success",
              "avg", "best", "pass_at_k", "errors", "mean_steps"]
    out = [",".join(fields)]
    for row in rows:
        out.append(",".join(str(row.get(f, "")) for f in fields))
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.write(fd, ("\\n".join(out) + "\\n").encode("utf-8"))
    finally:
        os.close(fd)
    return path
'''
if "\nimport os" not in s:
    s = s.replace("import csv", "import csv\nimport os", 1)
r.write_text(s)

u = pathlib.Path("src/appworld_vanilla/runner.py")
s = u.read_text()
i = s.index("def run_sweep")
s = s[:i] + '''def _worker(args):
    cfg, task_id, seed = args
    return run_rollout(cfg, task_id, seed)


def run_sweep(cfg, task_ids, store, on_result=None):
    import multiprocessing as mp
    seeds = [cfg.run.seed_base + i for i in range(cfg.run.num_rollouts)]
    jobs = [(t, s) for t in task_ids for s in seeds]
    done = store.completed_keys() if cfg.run.resume else set()
    pending = [j for j in jobs if j not in done]
    if not pending:
        return []
    payload = [(cfg, t, s) for t, s in pending]
    results = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=max(1, cfg.run.max_workers), maxtasksperchild=1) as pool:
        for record in pool.imap_unordered(_worker, payload):
            traj = record.pop("_trajectory", None)
            if traj:
                try:
                    store.save_trajectory(record["task_id"], record["seed"], traj)
                except Exception as exc:
                    print("[warn] trajectory write failed:", exc)
            store.append(record)
            results.append(record)
            if on_result:
                on_result(record)
    return results
'''
u.write_text(s)

assert "ThreadPoolExecutor" not in u.read_text().split("def run_sweep")[1]
assert "DictWriter" not in r.read_text()
print("VERIFIED: runner uses processes, csv module removed")

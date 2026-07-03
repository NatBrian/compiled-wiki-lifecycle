"""Trained-maintainer stage: crash-safe throttled job pool.

Runs a list of jobs with a concurrency cap, each fully detached (setsid) so a Claude crash
cannot kill them; relaunches a job if its process dies before its result file appears
(bounded retries), and exits when every result file exists. Run THIS launcher itself under
setsid so the whole batch survives.

Jobs file: JSON list of {"name","result","cmd"[, "env"{}]}.
  python run_pool.py jobs.json --maxc 6
"""
import argparse, json, os, signal, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
P4 = HERE                                          # bundled repo: no nested code/, scripts live here
LOGS = os.path.join(P4, "logs")
THREAD_CAPS = {"TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "2",
               "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
               "RAYON_NUM_THREADS": "2", "NUMBA_NUM_THREADS": "2"}


def launch(job):
    env = dict(os.environ)
    env.update(THREAD_CAPS)
    env.update(job.get("env", {}))
    log = open(os.path.join(LOGS, job["name"] + ".log"), "a")
    log.write(f"\n=== launch {time.ctime()} ===\n"); log.flush()
    p = subprocess.Popen(job["cmd"], cwd=P4, env=env, stdout=log, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, start_new_session=True)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("--maxc", type=int, default=6)
    ap.add_argument("--max_attempts", type=int, default=4)
    ap.add_argument("--poll", type=int, default=15)
    args = ap.parse_args()
    jobs = json.load(open(args.jobs))
    for j in jobs:
        j["result"] = j["result"] if os.path.isabs(j["result"]) else os.path.join(P4, j["result"])
    running = {}        # name -> Popen
    attempts = {j["name"]: 0 for j in jobs}
    done = lambda j: os.path.exists(j["result"])

    print(f"pool: {len(jobs)} jobs, maxc={args.maxc}", flush=True)
    while True:
        # reap finished/dead
        for name, p in list(running.items()):
            if p.poll() is not None:
                running.pop(name)
        pending = [j for j in jobs if not done(j) and j["name"] not in running]
        if not pending and not running:
            break
        # fill free slots
        for j in pending:
            if len(running) >= args.maxc:
                break
            if attempts[j["name"]] >= args.max_attempts:
                continue
            attempts[j["name"]] += 1
            running[j["name"]] = launch(j)
            print(f"  [{time.strftime('%H:%M:%S')}] launch {j['name']} "
                  f"(attempt {attempts[j['name']]}, running={len(running)})", flush=True)
        n_done = sum(1 for j in jobs if done(j))
        print(f"  progress {n_done}/{len(jobs)} done, {len(running)} running", flush=True)
        # all remaining exhausted attempts?
        if not running and all(done(j) or attempts[j["name"]] >= args.max_attempts for j in jobs):
            break
        time.sleep(args.poll)
    n_done = sum(1 for j in jobs if done(j))
    print(f"=== POOL DONE {n_done}/{len(jobs)} results present ===", flush=True)
    for j in jobs:
        if not done(j):
            print(f"  MISSING: {j['name']} -> {j['result']}", flush=True)


if __name__ == "__main__":
    main()

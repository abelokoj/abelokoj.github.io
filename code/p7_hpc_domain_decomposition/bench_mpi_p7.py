"""
bench_mpi_p7.py
===============
Runs wave_mpi_p7.py under mpirun for the correctness check and for strong-
and weak-scaling timings, and writes every result to bench_p7.json.

    python3 bench_mpi_p7.py            # full run
    python3 bench_mpi_p7.py --quick    # short smoke test

Each timing is repeated and the fastest repeat is kept, the usual convention
for wall-clock benchmarks on a machine that is also doing other work. Ranks are
pinned to physical cores while there are enough of them; beyond that the run
uses hardware threads, and the output records which placement was used.
"""
import argparse
import json
import os
import platform
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SOLVER = os.path.join(HERE, "wave_mpi_p7.py")


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()


def physical_cores():
    out = sh("lscpu -p=CORE,SOCKET | grep -v '^#' | sort -u | wc -l")
    return int(out) if out.isdigit() else os.cpu_count()


def mpirun(n, args, phys):
    if n <= phys:
        placement = ["--map-by", "core", "--bind-to", "core"]
        mode = "core"
    else:
        placement = ["--use-hwthread-cpus", "--map-by", "hwthread", "--bind-to", "hwthread"]
        mode = "hwthread"
    cmd = ["mpirun", "-n", str(n), *placement, sys.executable, SOLVER, *args]
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if p.returncode != 0:
        raise RuntimeError(f"mpirun failed for n={n}:\n{p.stderr[-2000:]}")
    rec = json.loads(p.stdout.strip().splitlines()[-1])
    rec["placement"] = mode
    return rec


def best_of(n, args, phys, repeats):
    runs = [mpirun(n, args, phys) for _ in range(repeats)]
    return min(runs, key=lambda r: r["wall_s"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args()

    phys = physical_cores()
    logical = os.cpu_count()
    ranks = [r for r in (1, 2, 4, 8, 12, 16) if r <= logical]

    n_strong, steps = (512, 50) if a.quick else (2048, 200)
    rows_per_rank = 64 if a.quick else 256
    repeats = 1 if a.quick else a.repeats

    import numpy, mpi4py
    out = {
        "system": {
            "cpu": sh("lscpu | grep 'Model name' | sed 's/.*: *//'"),
            "physical_cores": phys, "logical_cpus": logical,
            "memory": sh("free -h | awk '/Mem:/{print $2}'"),
            "kernel": platform.release(),
            "os": sh("lsb_release -ds"),
            "python": platform.python_version(),
            "numpy": numpy.__version__, "mpi4py": mpi4py.__version__,
            "mpi": sh("mpirun --version | head -1"),
        },
        "settings": {"strong_grid": n_strong, "steps": steps,
                     "rows_per_rank": rows_per_rank, "weak_ny": n_strong,
                     "repeats": repeats},
        "correctness": [], "strong": [], "weak": [],
    }
    print(json.dumps(out["system"], indent=2), flush=True)

    print("\ncorrectness (240 x 240, 300 steps, gathered and compared with the serial solver)")
    for n in ranks:
        r = mpirun(n, ["--check"], phys)
        out["correctness"].append(r)
        print(f"  P={n:2d}  max |u_mpi - u_serial| = {r['max_abs_diff_vs_serial']:.3e}", flush=True)

    print(f"\nstrong scaling ({n_strong} x {n_strong}, {steps} steps)")
    for n in ranks:
        r = best_of(n, ["--nx", str(n_strong), "--ny", str(n_strong), "--steps", str(steps)], phys, repeats)
        out["strong"].append(r)
        print(f"  P={n:2d} [{r['placement']:8s}]  wall {r['wall_s']:.3f} s  "
              f"comm {r['comm_s_max']:.3f} s  comp {r['comp_s_max']:.3f} s", flush=True)

    print(f"\nweak scaling ({rows_per_rank} rows per rank x {n_strong} columns, {steps} steps)")
    for n in ranks:
        r = best_of(n, ["--rows-per-rank", str(rows_per_rank), "--ny", str(n_strong),
                        "--steps", str(steps)], phys, repeats)
        out["weak"].append(r)
        print(f"  P={n:2d} [{r['placement']:8s}]  grid {r['nx']}x{r['ny']}  wall {r['wall_s']:.3f} s  "
              f"comm {r['comm_s_max']:.3f} s", flush=True)

    path = os.path.join(HERE, "bench_p7_quick.json" if a.quick else "bench_p7.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()

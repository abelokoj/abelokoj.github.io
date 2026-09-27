"""
table9_parallel.py
==================
Table 9 (Example 7.1, c = 8, t = 1) over the paper's full range
N = 20, 40, ..., 160, with each (scheme, N) solve run in its own process.

The stable time step scales like dx^3, so the cost of one solve grows like
N^4 and the finest grids dominate the run time. The solves are independent,
so running them in parallel cuts the wall time to roughly that of the
single largest case. Results are identical to the serial loop in
example_7_1_linear_kdv.py.

    python table9_parallel.py            # uses all cores
"""
import json
import os
from multiprocessing import Pool

import numpy as np

import example_7_1_linear_kdv as ex

C, T = 8, 1.0
NS = [20, 40, 60, 80, 100, 120, 140, 160]


def one(job):
    scheme, N = job
    solve = ex.solve_linear_kdv_TDCNCS if scheme == "TDCNCS" else ex.solve_linear_kdv_TDCCS
    out = solve(N, C, T)
    x, u = out[0], out[1]
    Linf, L1, L2 = ex.errors(u, np.sin(C * (x + T)))
    return scheme, N, Linf, L1, L2


if __name__ == "__main__":
    jobs = [(s, N) for s in ("TDCNCS", "TDCCS") for N in NS]
    jobs.sort(key=lambda j: -j[1])                      # largest first
    with Pool(min(len(jobs), os.cpu_count())) as pool:
        res = pool.map(one, jobs, chunksize=1)
    res.sort(key=lambda r: (r[0] != "TDCNCS", r[1]))
    rows, prev = [], {}
    print(f"{'Scheme':8s}{'N':>5s}{'Linf':>13s}{'rate':>8s}{'L1':>13s}{'rate':>8s}{'L2':>13s}{'rate':>8s}")
    for s, N, a, b, c in res:
        if s in prev:
            pN, pa, pb, pc = prev[s]
            d = np.log(N / pN)
            r = [np.log(pa / a) / d, np.log(pb / b) / d, np.log(pc / c) / d]
        else:
            r = [float("nan")] * 3
        print(f"{s:8s}{N:5d}{a:13.4e}{r[0]:8.4f}{b:13.4e}{r[1]:8.4f}{c:13.4e}{r[2]:8.4f}")
        rows.append(dict(scheme=s, N=N, Linf=a, L1=b, L2=c, rate_Linf=r[0], rate_L1=r[1], rate_L2=r[2]))
        prev[s] = (N, a, b, c)
    with open("data/table9_full.json", "w") as f:
        json.dump(rows, f, indent=1)
    print("wrote data/table9_full.json")

"""
table11_parallel.py
===================
Table 11 (Example 7.5, 2D linear dispersion, t = 1) over the paper's full
range N = 10, 15, ..., 40, with each (scheme, N) solve in its own process.

The 2D time step still scales like dx^3 and each step touches N^2 points
(four coupled fields for TDCCS), so the cost grows like N^5 and the
largest TDCCS grids dominate. Results are identical to the serial
table11() in example_7_5_2d_linear.py.

    python table11_parallel.py
"""
import json
import os
from multiprocessing import Pool

import numpy as np

import example_7_5_2d_linear as ex

NS = [10, 15, 20, 25, 30, 35, 40]
SCHEMES = ("TDCCS",)          # TDCNCS rows come from the serial run (fast)
T = 1.0


def one(job):
    scheme, N = job
    solve = ex.solve_2d_linear_TDCNCS if scheme == "TDCNCS" else ex.solve_2d_linear_TDCCS
    X, Y, u = solve(N, T=T)
    Linf, L1, L2 = ex.errors(u, np.sin(X + Y + 2 * T))
    return scheme, N, Linf, L1, L2


if __name__ == "__main__":
    jobs = [(s, N) for s in SCHEMES for N in NS]
    jobs.sort(key=lambda j: (-(j[1] ** 5) * (2 if j[0] == "TDCCS" else 1)))
    with Pool(min(len(jobs), os.cpu_count())) as pool:
        res = pool.map(one, jobs, chunksize=1)
    res.sort(key=lambda r: (r[0] != "TDCNCS", r[1]))
    rows, prev = [], {}
    for s, N, a, b, c in res:
        if s in prev:
            pN, pa, pb, pc = prev[s]
            d = np.log(N / pN)
            r = [np.log(pa / a) / d, np.log(pb / b) / d, np.log(pc / c) / d]
        else:
            r = [float("nan")] * 3
        print(f"{s:8s}{N:5d}{a:13.4e}{r[0]:8.4f}{b:13.4e}{r[1]:8.4f}{c:13.4e}{r[2]:8.4f}", flush=True)
        rows.append(dict(scheme=s, N=N, Linf=a, L1=b, L2=c, rate_Linf=r[0], rate_L1=r[1], rate_L2=r[2]))
        prev[s] = (N, a, b, c)
    with open("data/table11_full.json", "w") as f:
        json.dump(rows, f, indent=1)
    print("wrote data/table11_full.json")

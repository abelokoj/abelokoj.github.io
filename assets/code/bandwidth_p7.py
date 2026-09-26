"""
bandwidth_p7.py
===============
A STREAM-style triad, a[:] = b + s*c, run independently on every MPI rank
with no communication. If total memory bandwidth were unlimited, the
aggregate rate would grow in proportion to the number of ranks; where it
flattens is the memory-bandwidth ceiling of the machine.

    mpirun -n 8 --map-by core --bind-to core python3 bandwidth_p7.py
"""
import json

import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
N = 4_000_000                      # 32 MB per array, far larger than any cache
a, b, c = np.zeros(N), np.ones(N), np.full(N, 2.0)
s = 3.0
tmp = np.empty(N)

best = np.inf
for _ in range(10):
    comm.Barrier()
    t0 = MPI.Wtime()
    np.multiply(c, s, out=tmp)
    np.add(b, tmp, out=a)
    t = MPI.Wtime() - t0
    best = min(best, comm.allreduce(t, op=MPI.MAX))

# Bytes moved per rank: multiply reads c, writes tmp; add reads b and tmp, writes a.
bytes_per_rank = 5 * N * 8
total_gbs = comm.Get_size() * bytes_per_rank / best / 1e9
if comm.Get_rank() == 0:
    print(json.dumps({"ranks": comm.Get_size(), "aggregate_GB_s": total_gbs}))

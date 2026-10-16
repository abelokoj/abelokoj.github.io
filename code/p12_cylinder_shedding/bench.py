import time
import numpy as np
import lbm_core as L
import numba

N = 40
nx, ny = 22 * N + 1, 41 * N // 10
solid, (lx, ly, ld, lq) = L.cylinder_geometry(nx, ny, 2.0 * N, 2.0 * N, 0.5 * N)
f = np.zeros((9, nx, ny))
L.init_equilibrium(f, np.ones((nx, ny)), np.zeros((nx, ny)), np.zeros((nx, ny)))
fp = f.copy()
omp, omm = L.trt_rates(0.04 * N / 100)
prof = np.full(ny, 0.04)
for k in range(3):
    L.collide_trt(f, fp, solid, omp, omm); L.stream(fp, f, solid, False); L.bouzidi(fp, f, lx, ly, ld, lq); L.zou_he(f, prof, 1.0)
print("threads", numba.get_num_threads())
for name, fn in [("collide", lambda: L.collide_trt(f, fp, solid, omp, omm)), ("stream", lambda: L.stream(fp, f, solid, False)), ("bouzidi", lambda: L.bouzidi(fp, f, lx, ly, ld, lq)), ("zouhe", lambda: L.zou_he(f, prof, 1.0))]:
    t = time.perf_counter()
    for k in range(200):
        fn()
    dt = (time.perf_counter() - t) / 200
    print(f"{name:8s} {dt * 1e3:8.3f} ms  ({nx * ny / dt / 1e6:7.1f} MLUPS)")

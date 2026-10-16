"""Print the numbers quoted in the post from the saved runs."""
import numpy as np

from pub_style import load_data, has_data

for tag in ("main", "coarse"):
    if not has_data(f"kh_rollup_{tag}"):
        continue
    d = load_data(f"kh_rollup_{tag}")
    m = d["meta"]
    t, th, A1, A2, E, Z = d["t"], d["theta"], d["A1"], d["A2"], d["E"], d["Z"]
    print(f"== {tag}: {m['nx']} x {m['ny']}, alpha = {m['alpha']:.4f}, wi1 = {m['wi1']:.5f}, wi2 = {m['wi2']:.5f}")
    print(f"  A1 peak at t = {t[np.argmax(A1)]:.1f} ({A1.max():.4f}); A2 peak at t = {t[np.argmax(A2)]:.1f} ({A2.max():.4f})")
    # measured growth rates in the linear stage
    for name, A, lo, hi in (("A1", A1, 5.0, 25.0), ("A2", A2, 5.0, 40.0)):
        s = (t >= lo) & (t <= hi)
        print(f"  {name} fitted rate over [{lo}, {hi}] = {np.polyfit(t[s], np.log(A[s]), 1)[0]:.5f}")
    print("  theta/theta0:", " ".join(f"{tt}:{np.interp(tt, t, th) / th[0]:.3f}" for tt in range(0, 201, 10)))
    print(f"  E: {E[0]:.6f} -> {E[-1]:.6f} ({(E[-1] / E[0] - 1) * 100:.3f} %), Z: {Z[0]:.5f} -> {Z[-1]:.5f}")
    print(f"  max budget residual {d['budget_resid'][2:-2].max():.2e}, umax peak {d['umax'].max():.3f}")

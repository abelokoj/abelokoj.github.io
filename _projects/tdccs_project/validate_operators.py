"""
validate_operators.py
=====================
Grid-refinement check of the eighth-order third-derivative operators on a
smooth periodic function, f(x) = sin(kx) on [0, 2*pi), whose third
derivative is known exactly: f'''(x) = -k^3 cos(kx).

This is the check quoted in Section 3 of the blog post. For each grid it
prints the maximum error of the computed third derivative and the observed
order p = log(e1/e2) / log(N2/N1).

    python validate_operators.py
"""
import numpy as np

import tdccs_lib as T

L = 2 * np.pi


def observed_order(e1, e2, n1, n2):
    return np.log(e1 / e2) / np.log(n2 / n1)


def run(k, Ns):
    print(f"\nf = sin({k} x): maximum error of f''' (nodes and centers for TDCCS)")
    print(f"{'N':>6s} {'TDCNCS-T8':>12s} {'rate':>6s} {'TDCCS-T8':>12s} {'rate':>6s}")
    prev = None
    for N in Ns:
        dx = L / N
        xn = np.arange(N) * dx
        xh = xn + dx / 2
        exact_n = -k**3 * np.cos(k * xn)
        exact_h = -k**3 * np.cos(k * xh)

        e_ncs = np.abs(T.third_derivative_TDCNCS(np.sin(k * xn), dx, T.TDCNCS["T8"]) - exact_n).max()
        f3n, f3h = T.third_derivative_TDCCS(np.sin(k * xn), np.sin(k * xh), dx, T.TDCCS["T8"])
        e_ccs = max(np.abs(f3n - exact_n).max(), np.abs(f3h - exact_h).max())

        if prev is None:
            print(f"{N:6d} {e_ncs:12.3e} {'':>6s} {e_ccs:12.3e} {'':>6s}")
        else:
            pN, p_ncs, p_ccs = prev
            print(f"{N:6d} {e_ncs:12.3e} {observed_order(p_ncs, e_ncs, pN, N):6.2f} "
                  f"{e_ccs:12.3e} {observed_order(p_ccs, e_ccs, pN, N):6.2f}")
        prev = (N, e_ncs, e_ccs)


if __name__ == "__main__":
    for k in (1, 2, 4):
        run(k, [10, 20, 40, 80, 160, 320, 640])

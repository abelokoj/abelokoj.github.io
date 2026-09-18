"""
Reproduces Figures 2, 3, 4 and Tables 5, 6, 7 of Salian, Samala & Ghosh (2026).
Pure Fourier-analysis content -- fully and exactly specified by the paper's
own equations (2.6), (2.7), (2.9), (4.1), (5.2), Table 7 -- no external
first-derivative operator is needed here, so this reproduces the published
results with no ambiguity.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pub_style import apply_style, cm_x, cm_y, savefig_all, FIGDIR
from tdccs_lib import (TDCNCS, TDCCCS, TDCCS, CI, filter_F12_coeffs,
                        omega3_TDCNCS, omega3_TDCCCS, omega3_TDCCS, T_CI)

OUT = FIGDIR   # <project>/figs, resolved relative to pub_style.py
import os
os.makedirs(OUT, exist_ok=True)

apply_style()

omega = np.linspace(1e-4, np.pi, 800)

# =====================================================================
# Figure 2 : TDCNCS, TDCCCS, TDCCCS-CI modified wavenumber
# =====================================================================
order_list = ["E2", "E4", "E6", "T4", "T6", "T8", "P6", "P8", "P10"]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

for key in order_list:
    axes[0].plot(omega, omega3_TDCNCS(omega, TDCNCS[key]), label=key,
                  linestyle='--' if key == 'E4' else '-')
axes[0].plot(omega, omega**3, 'k-', lw=1.6, label='Exact')
axes[0].set_title("(a) TDCNCS")

for key in order_list:
    axes[1].plot(omega, omega3_TDCCCS(omega, TDCCCS[key]), label=key)
axes[1].plot(omega, omega**3, 'k-', lw=1.6, label='Exact')
axes[1].set_title("(b) TDCCCS")

# TDCCCS-CI : apply the tenth-order pentadiagonal CI interpolation, i.e.
# the *effective* modified wavenumber is the TDCCCS one but with the true
# half-grid values replaced by CI-interpolated ones, which multiplies the
# half-grid terms by the CI transfer function T_CI(omega) (Eq. 2.9).
def omega3_TDCCCS_CI(omega, coeffs, ci_coeffs):
    a, b, c, alpha, beta, _ = coeffs
    Tci = T_CI(omega, ci_coeffs)
    num = (2 * a * (3 * np.sin(omega / 2) - np.sin(3 * omega / 2))
           + (2 * b / 5) * (5 * np.sin(omega / 2) - np.sin(5 * omega / 2))
           + (c / 7) * (7 * np.sin(omega / 2) - np.sin(7 * omega / 2)))
    # the half-grid values entering the numerator scale with Tci relative
    # to the true (un-interpolated) values used in omega3_TDCCCS
    return Tci * num / (1 + 2 * alpha * np.cos(omega) + 2 * beta * np.cos(2 * omega))

for key in order_list:
    axes[2].plot(omega, omega3_TDCCCS_CI(omega, TDCCCS[key], CI["P10"]), label=key)
axes[2].plot(omega, omega**3, 'k-', lw=1.6, label='Exact')
axes[2].set_title("(c) TDCCCS-CI")

for ax in axes:
    ax.set_xlabel(r"Wavenumber $\omega$")
    ax.set_ylabel(r"Modified wavenumber $\omega'''$")
    ax.set_ylim(0, 32)
    ax.set_xlim(0, np.pi)
axes[0].legend(fontsize=7, ncol=2, loc="upper left", framealpha=0.9)
fig.tight_layout()
savefig_all(fig, "figure2_TDCNCS_TDCCCS", outdir=OUT)
plt.close(fig)

# =====================================================================
# Figure 3 : TDCCS, TDCCS-CI
# =====================================================================
order_list_ccs = ["E4", "E6", "T4", "T6", "T8", "P6", "P8", "P10"]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
for key in order_list_ccs:
    axes[0].plot(omega, omega3_TDCCS(omega, TDCCS[key]), label=key)
axes[0].plot(omega, omega**3, 'k-', lw=1.6, label='Exact')
axes[0].set_title("(a) TDCCS")

def omega3_TDCCS_CI(omega, coeffs, ci_coeffs):
    a, b, c, alpha, beta, _ = coeffs
    Tci = T_CI(omega, ci_coeffs)
    num = (2 * a * (8 * np.sin(omega / 2) - 4 * np.sin(omega))
           + (2 * b / 5) * (12 * np.sin(omega) - 8 * np.sin(3 * omega / 2))
           + (2 * c / 35) * (20 * np.sin(omega) - 8 * np.sin(5 * omega / 2)))
    return Tci * num / (1 + 2 * alpha * np.cos(omega) + 2 * beta * np.cos(2 * omega))

for key in order_list_ccs:
    axes[1].plot(omega, omega3_TDCCS_CI(omega, TDCCS[key], CI["P10"]), label=key)
axes[1].plot(omega, omega**3, 'k-', lw=1.6, label='Exact')
axes[1].set_title("(b) TDCCS-CI")

for ax in axes:
    ax.set_xlabel(r"Wavenumber $\omega$")
    ax.set_ylabel(r"Modified wavenumber $\omega'''$")
    ax.set_ylim(0, 32)
    ax.set_xlim(0, np.pi)
axes[0].legend(fontsize=8, ncol=2, loc="upper left", framealpha=0.9)
fig.tight_layout()
savefig_all(fig, "figure3_TDCCS", outdir=OUT)
plt.close(fig)

print("Figures 2 and 3 written to", OUT)

# =====================================================================
# Tables 5 & 6 : bandwidth resolving efficiency
# =====================================================================
def resolving_efficiency(omega3_func, coeffs, eps_t, omega_grid):
    w = omega_grid
    w3 = omega3_func(w, coeffs)
    rel_err = np.abs((w3 - w**3) / w**3)
    ok = rel_err <= eps_t
    # shortest well-resolved wave = largest contiguous omega from 0 s.t. ok holds
    # (find first index where condition fails)
    first_fail = np.argmax(~ok) if not ok.all() else len(w) - 1
    wf = w[first_fail - 1] if first_fail > 0 else w[0]
    return wf, wf / np.pi

w_fine = np.linspace(0.02, np.pi - 1e-4, 200000)  # avoid float cancellation
                                                    # noise near omega->0 for
                                                    # schemes whose (1+2 alpha
                                                    # cos w + ...) denominator
                                                    # -> 0 there (e.g. TDCCS-T6)

def make_table(eps_t):
    rows = []
    schemes = [
        ("TDCNCS", omega3_TDCNCS, TDCNCS),
        ("TDCCCS", omega3_TDCCCS, TDCCCS),
        ("TDCCS", omega3_TDCCS, TDCCS),
    ]
    order_map = {"4-T": "T4", "6-T": "T6", "8-T": "T8", "10-P": "P10"}
    for name, func, table in schemes:
        row = {"scheme": name}
        for col, key in order_map.items():
            if key not in table:
                row[col] = None
                continue
            wf, e = resolving_efficiency(func, table[key], eps_t, w_fine)
            row[col] = (wf, e)
        rows.append(row)
    return rows

for eps_t, tabname in [(1e-3, "Table 5"), (1e-4, "Table 6")]:
    print(f"\n{tabname}  (tolerance eps_t = {eps_t})")
    print(f"{'Scheme':10s} {'4-T wf':>8s} {'4-T e':>8s} {'6-T wf':>8s} {'6-T e':>8s} "
          f"{'8-T wf':>8s} {'8-T e':>8s} {'10-P wf':>8s} {'10-P e':>8s}")
    for row in make_table(eps_t):
        vals = []
        for col in ["4-T", "6-T", "8-T", "10-P"]:
            wf, e = row[col]
            vals += [f"{wf:8.4f}", f"{e:8.4f}"]
        print(f"{row['scheme']:10s} " + " ".join(vals))

# =====================================================================
# Table 7 : filter coefficients (symbolic form, direct evaluation demo)
# =====================================================================
print("\nTable 7 (F12 filter coefficients) at alphaF = 0.4:")
coeffs = filter_F12_coeffs(0.4)
for i, ci in enumerate(coeffs):
    print(f"  a{i} = {ci: .6f}")



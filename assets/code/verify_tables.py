"""
Verify Tables 1, 2, and 4 (TDCNCS, TDCCCS, TDCCS coefficients) against the
order conditions derived symbolically in derive_order_conditions.py.

For a scheme claimed to be order-N accurate, the D[3], D[5], ..., D[N+1]
conditions must vanish EXACTLY (using exact rationals), and the D[N+3]
condition gives the leading truncation-error constant Q in
    Q * f^{(N+3)}(x) * h^N + O(h^{N+2}).
"""
import sympy as sp
from derive_order_conditions import (
    build_equation, K, D, h,
)

a, b, c, alpha, beta = sp.symbols('a b c alpha beta')
half = sp.Rational(1, 2)

# Rebuild the three raw (LHS-RHS) expressions (symbolic, exact) --------------
lhs_nodes = {-2: beta, -1: alpha, 0: sp.Integer(1), 1: alpha, 2: beta}
rhs_ncs = [
    (a, [(1, 2), (-2, 1), (2, -1), (-1, -2)], 2 * h**3),
    (b, [(1, 3), (-3, 1), (3, -1), (-1, -3)], 8 * h**3),
    (c, [(1, 4), (-4, 1), (4, -1), (-1, -4)], 20 * h**3),
]
expr_ncs = build_equation(lhs_nodes, rhs_ncs)

rhs_ccc = [
    (a, [(1, sp.Rational(3, 2)), (-3, half), (3, -half), (-1, -sp.Rational(3, 2))], h**3),
    (b, [(1, sp.Rational(5, 2)), (-5, half), (5, -half), (-1, -sp.Rational(5, 2))], 5 * h**3),
    (c, [(1, sp.Rational(7, 2)), (-7, half), (7, -half), (-1, -sp.Rational(7, 2))], 14 * h**3),
]
expr_ccc = build_equation(lhs_nodes, rhs_ccc)

rhs_ccs = [
    (a, [(4, 1), (-8, half), (8, -half), (-4, -1)], h**3),
    (b, [(8, sp.Rational(3, 2)), (-12, 1), (12, -1), (-8, -sp.Rational(3, 2))], 5 * h**3),
    (c, [(8, sp.Rational(5, 2)), (-20, 1), (20, -1), (-8, -sp.Rational(5, 2))], 35 * h**3),
]
expr_ccs = build_equation(lhs_nodes, rhs_ccs)

SCHEMES = {"TDCNCS": expr_ncs, "TDCCCS": expr_ccc, "TDCCS": expr_ccs}

# Published coefficient tables (Tables 1, 2, 4) -------------------------------
R = sp.Rational
TABLE1_TDCNCS = {  # (a, b, c, alpha, beta, order)
    "E2": (1, 0, 0, 0, 0, 2),
    "E4": (2, -1, 0, 0, 0, 4),
    "E6": (R(169, 60), R(-12, 5), R(7, 12), 0, 0, 6),
    "T4": (2, 0, 0, R(1, 2), 0, 4),
    "T6": (2, R(-1, 8), 0, R(7, 16), 0, 6),
    "T8": (R(2367, 1180), R(-167, 1180), R(1, 236), R(205, 472), 0, 8),
    "P6": (R(40, 21), 0, 0, R(4, 9), R(1, 126), 6),
    "P8": (R(160, 83), R(-5, 166), 0, R(147, 332), R(-1, 166), 8),
    "P10": (R(18221, 5478), R(-1846, 913), R(5, 66), R(799, 2739), R(-557, 5478), 10),
}
TABLE2_TDCCCS = {
    "E2": (1, 0, 0, 0, 0, 2),
    "E4": (R(13, 8), R(-5, 8), 0, 0, 0, 4),
    "E6": (R(1299, 640), R(-499, 384), R(259, 960), 0, 0, 6),
    "T4": (R(4, 3), 0, 0, R(1, 6), 0, 4),
    "T6": (R(205, 166), R(35, 166), 0, R(37, 166), 0, 6),
    "T8": (R(1058279, 975200), R(96627, 195040), R(-24787, 487600), R(3229, 12190), 0, 8),
    "P6": (R(320, 233), 0, 0, R(134, 699), R(-7, 1398), 6),
    "P8": (R(49720, 79903), R(91400, 79903), 0, R(28838, 79903), R(3541, 159806), 8),
    "P10": (R(55463611, 150617762), R(677644345, 451853286), R(6301771, 225926643),
             R(93443398, 225926643), R(15505921, 451853286), 10),
}
TABLE4_TDCCS = {
    "E4": (R(13, 8), R(-5, 8), 0, 0, 0, 4),
    "E6": (R(361, 192), R(-129, 128), R(49, 384), 0, 0, 6),
    "T4": (R(8, 7), 0, 0, R(1, 14), 0, 4),
    "T6": (5, -5, 0, R(-1, 2), 0, 6),
    "T8": (R(58021, 14120), R(-109007, 28240), R(1029, 28240), R(-1261, 3530), 0, 8),
    "P6": (R(320, 273), 0, 0, R(74, 819), R(-1, 234), 6),
    "P8": (R(19640, 4621), R(-353000, 87799), 0, R(-33746, 87799), R(-147, 175598), 8),
    "P10": (R(74390155, 19635801), R(-45752035, 13090534), R(4684435, 39271602),
             R(-5803114, 19635801), R(74747, 39271602), 10),
}

ALL = {"TDCNCS": TABLE1_TDCNCS, "TDCCCS": TABLE2_TDCCCS, "TDCCS": TABLE4_TDCCS}


def leading_error_const(expr, order):
    """Coefficient of D[order+3], i.e. the truncation-error constant Q."""
    k = order + 3
    coef = sp.nsimplify(expr.coeff(D[k], 1))
    return sp.simplify(coef / h**(k - 3))


print(f"{'Scheme':8s} {'Variant':6s} {'Order':>5s}  exact? (all lower D-conditions)   leading Q")
print("-" * 80)
for scheme_name, table in ALL.items():
    expr = SCHEMES[scheme_name]
    for variant, (av, bv, cv, alv, bev, order) in table.items():
        subs = {a: av, b: bv, c: cv, alpha: alv, beta: bev}
        ok = True
        for k in range(3, order + 3, 2):  # D3, D5, ..., D_{order+1} must vanish
            val = sp.simplify(expr.coeff(D[k], 1).subs(subs))
            if val != 0:
                ok = False
        Q = leading_error_const(expr, order).subs(subs)
        Qf = float(Q)
        print(f"{scheme_name:8s} {variant:6s} {order:5d}   {'OK' if ok else 'FAIL':>5s}"
              f"                         Q = {Qf: .6e}")



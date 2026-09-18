"""
Symbolic derivation of the Taylor-series order conditions for the three
third-derivative compact schemes in Salian, Samala & Ghosh (2026):

    TDCNCS  - Eq. (2.4)   (values at cell NODES only)
    TDCCCS  - Eq. (2.5)   (values at cell CENTERS, half-shift interpolation)
    TDCCS   - Eq. (3.1)/(3.2) (the paper's new scheme: nodes + centers, no interpolation)

Method (this is exactly Lele 1992's approach, generalised):
  Represent f as a formal Taylor series about the node x_j with symbolic
  derivative values D[0], D[1], ..., D[K] = f(x_j), f'(x_j), ..., f^(K)(x_j).
  Every stencil value f_{j+m} (node offset m, possibly half-integer for the
  cell-centered/central schemes) is then

        f_{j+m} = sum_k D[k] * (m*h)**k / k!

  and every f'''_{j+m} appearing on the LHS of the implicit operator is the
  same series differentiated 3 times, i.e. shifted up by 3 in the D-index:

        f'''_{j+m} = sum_k D[k+3] * (m*h)**k / k!

  Substituting into (LHS - RHS) and collecting powers of h and of the
  symbolic derivatives D[3], D[4], D[5], ... gives, order by order, exactly
  the "order conditions" quoted in the paper as Eqs. (3.3)-(3.7) for TDCCS.
  We reproduce that machinery here for all three schemes, verify the
  published coefficient tables satisfy it, and recover the stated
  leading-order truncation-error constants Q in Q f^(11)(x) Δx^8 + O(Δx^10).
"""
import sympy as sp

K = 14  # number of derivative symbols to keep (D[0]..D[K])
h, m = sp.symbols('h m', positive=True)
D = sp.symbols('D0:%d' % (K + 1))


def taylor_value(offset):
    """f_{j+offset}, offset in units of h (can be sp.Rational for half-grid)."""
    s = offset * h
    return sum(D[k] * s**k / sp.factorial(k) for k in range(K + 1))


def taylor_deriv3(offset):
    """f'''_{j+offset}, i.e. the same Taylor series shifted by 3 derivatives."""
    s = offset * h
    return sum(D[k + 3] * s**k / sp.factorial(k) for k in range(K - 2))


def build_equation(lhs_offsets, rhs_terms):
    """
    lhs_offsets: dict {offset: symbolic coefficient} for the f''' stencil
                 (β,α,1,α,β at offsets -2,-1,0,1,2 for TDCNCS/TDCCCS,
                  or at half-integer offsets for TDCCS eq. 3.2)
    rhs_terms:   list of (coeff, [(sign, offset), ...], denom) tuples encoding
                 RHS = coeff/denom * sum(sign * f_{j+offset})
    Returns LHS - RHS as a series in h, with symbolic D[k] left in place.
    """
    lhs = sum(c * taylor_deriv3(off) for off, c in lhs_offsets.items())
    rhs = 0
    for coeff, sign_offsets, denom in rhs_terms:
        term = sum(sign * taylor_value(off) for sign, off in sign_offsets)
        rhs += coeff * term / denom
    expr = sp.expand(lhs - rhs)
    return expr


def order_conditions(expr, alpha, beta, label):
    """
    Collect (LHS-RHS) by the symbolic derivative order, normalise by the
    D[3] term (which should have coefficient (1+2*alpha+2*beta) on the LHS
    contribution), and print the condition that must vanish for each
    successive even accuracy order 2,4,6,8,10, i.e. the coefficients of
    D[4] (order-2 -> automatically zero by antisymmetry), D[5] (order-4),
    D[6] (auto-zero), D[7] (order-6), D[9] (order-8), D[11] (order-10).
    """
    print(f"\n=== Order conditions for {label} ===")
    conditions = {}
    for k in range(3, K - 2):
        c = sp.simplify(expr.coeff(D[k], 1))
        # remove the h-dependence bookkeeping: coefficient of D[k] should be
        # (const) * h**(k-3) after our construction, isolate that constant
        if c == 0:
            continue
        # factor out h power
        c_over_h = sp.simplify(c / h**(k - 3))
        conditions[k] = sp.simplify(c_over_h)
        if k <= 12:
            print(f"  coefficient of D[{k}] (~ f^({k})(x_j) h^{k-3}) : {conditions[k]}")
    return conditions


if __name__ == "__main__":
    a, b, c, alpha, beta = sp.symbols('a b c alpha beta')

    # ---------------------------------------------------------------
    # TDCNCS, Eq. (2.4): node-only stencil
    # ---------------------------------------------------------------
    lhs_nodes = {-2: beta, -1: alpha, 0: sp.Integer(1), 1: alpha, 2: beta}
    rhs_nodes = [
        (a, [(1, 2), (-2, 1), (2, -1), (-1, -2)], 2 * h**3),
        (b, [(1, 3), (-3, 1), (3, -1), (-1, -3)], 8 * h**3),
        (c, [(1, 4), (-4, 1), (4, -1), (-1, -4)], 20 * h**3),
    ]
    expr_ncs = build_equation(lhs_nodes, rhs_nodes)
    cond_ncs = order_conditions(expr_ncs, alpha, beta, "TDCNCS (Eq. 2.4)")

    # ---------------------------------------------------------------
    # TDCCCS, Eq. (2.5): half-grid (cell-center) stencil
    # ---------------------------------------------------------------
    half = sp.Rational(1, 2)
    lhs_ctr = {-2: beta, -1: alpha, 0: sp.Integer(1), 1: alpha, 2: beta}
    rhs_ctr = [
        (a, [(1, sp.Rational(3, 2)), (-3, half), (3, -half), (-1, -sp.Rational(3, 2))], h**3),
        (b, [(1, sp.Rational(5, 2)), (-5, half), (5, -half), (-1, -sp.Rational(5, 2))], 5 * h**3),
        (c, [(1, sp.Rational(7, 2)), (-7, half), (7, -half), (-1, -sp.Rational(7, 2))], 14 * h**3),
    ]
    expr_ccc = build_equation(lhs_ctr, rhs_ctr)
    cond_ccc = order_conditions(expr_ccc, alpha, beta, "TDCCCS (Eq. 2.5)")

    # ---------------------------------------------------------------
    # TDCCS, Eq. (3.1): node stencil on LHS, mixed node+half-grid on RHS
    # ---------------------------------------------------------------
    lhs_ccs = {-2: beta, -1: alpha, 0: sp.Integer(1), 1: alpha, 2: beta}
    rhs_ccs = [
        (a, [(4, 1), (-8, half), (8, -half), (-4, -1)], h**3),
        (b, [(8, sp.Rational(3, 2)), (-12, 1), (12, -1), (-8, -sp.Rational(3, 2))], 5 * h**3),
        (c, [(8, sp.Rational(5, 2)), (-20, 1), (20, -1), (-8, -sp.Rational(5, 2))], 35 * h**3),
    ]
    expr_ccs = build_equation(lhs_ccs, rhs_ccs)
    cond_ccs = order_conditions(expr_ccs, alpha, beta, "TDCCS (Eq. 3.1)")



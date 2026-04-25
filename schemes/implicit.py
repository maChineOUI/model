import numpy as np


def _thomas(a, d, c, b):
    """Solve tridiagonal system using Thomas algorithm.

    a: sub-diagonal (length n, a[0] unused)
    d: main diagonal (length n)
    c: super-diagonal (length n, c[-1] unused)
    b: right-hand side (length n)
    Returns solution x of length n.
    """
    n = len(d)
    c_ = np.empty(n)
    b_ = np.empty(n)
    x  = np.empty(n)

    c_[0] = c[0] / d[0]
    b_[0] = b[0] / d[0]
    for i in range(1, n):
        denom = d[i] - a[i] * c_[i - 1]
        c_[i] = c[i] / denom
        b_[i] = (b[i] - a[i] * b_[i - 1]) / denom

    x[-1] = b_[-1]
    for i in range(n - 2, -1, -1):
        x[i] = b_[i] - c_[i] * x[i + 1]
    return x


def solve_implicit(U0, Nx, Nt, r, alpha, dt, store_all=False):
    # BTCS matrix is constant: pre-compute Thomas forward-sweep coefficients once.
    diag  = 1.0 + 2.0 * r + alpha * dt   # scalar — same for every row
    sub   = -r
    sup   = -r

    # Forward sweep: build c_pre (modified super-diagonal) and pivot sequence.
    c_pre = np.empty(Nx)
    pivot = np.empty(Nx)
    pivot[0] = diag
    c_pre[0] = sup / pivot[0]
    for i in range(1, Nx):
        pivot[i] = diag - sub * c_pre[i - 1]
        c_pre[i] = sup / pivot[i]

    U = U0.copy()
    history = [U.copy()] if store_all else None
    b_ = np.empty(Nx)

    for _ in range(Nt):
        # Forward sweep on RHS only (matrix part already pre-computed)
        b_[0] = U[0] / pivot[0]
        for i in range(1, Nx):
            b_[i] = (U[i] - sub * b_[i - 1]) / pivot[i]

        # Back substitution
        U[-1] = b_[-1]
        for i in range(Nx - 2, -1, -1):
            U[i] = b_[i] - c_pre[i] * U[i + 1]

        if store_all:
            history.append(U.copy())

    return (U, np.array(history)) if store_all else U

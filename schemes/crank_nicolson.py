import numpy as np


def _build_rhs(U, r, alpha, dt):
    """Compute B*U for CN right-hand side (vectorised)."""
    c = r / 2.0
    d = 1.0 - r - alpha * dt / 2.0
    rhs = d * U
    rhs[:-1] += c * U[1:]
    rhs[1:]  += c * U[:-1]
    return rhs


def solve_crank_nicolson(U0, Nx, Nt, r, alpha, dt, store_all=False):
    c = r / 2.0
    diag = 1.0 + r + alpha * dt / 2.0
    sub = -c
    sup = -c

    # CN matrix is constant: pre-compute Thomas forward-sweep coefficients once.
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
        rhs = _build_rhs(U, r, alpha, dt)

        # Forward sweep on RHS only (matrix part already pre-computed).
        b_[0] = rhs[0] / pivot[0]
        for i in range(1, Nx):
            b_[i] = (rhs[i] - sub * b_[i - 1]) / pivot[i]

        # Back substitution.
        U[-1] = b_[-1]
        for i in range(Nx - 2, -1, -1):
            U[i] = b_[i] - c_pre[i] * U[i + 1]

        if store_all:
            history.append(U.copy())

    return (U, np.array(history)) if store_all else U

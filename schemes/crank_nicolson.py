import numpy as np
from .implicit import _thomas


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
    diag  = np.full(Nx, 1.0 + r + alpha * dt / 2.0)
    lower = np.full(Nx, -c)
    upper = np.full(Nx, -c)

    U = U0.copy()
    history = [U.copy()] if store_all else None

    for _ in range(Nt):
        rhs = _build_rhs(U, r, alpha, dt)
        U = _thomas(lower, diag, upper, rhs)
        if store_all:
            history.append(U.copy())

    return (U, np.array(history)) if store_all else U

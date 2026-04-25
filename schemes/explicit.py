import numpy as np


def step_explicit(U, r, alpha, dt):
    laplacian = np.empty_like(U)
    laplacian[1:-1] = U[:-2] - 2.0 * U[1:-1] + U[2:]
    laplacian[0]    =  0.0   - 2.0 * U[0]    + U[1]
    laplacian[-1]   = U[-2]  - 2.0 * U[-1]   + 0.0
    return U + r * laplacian - alpha * dt * U


def solve_explicit(U0, Nx, Nt, r, alpha, dt, store_all=False):
    if r > 0.5:
        print(f"Warning: stability condition violated (r={r:.4f} > 0.5)")

    U = U0.copy()
    history = [U.copy()] if store_all else None

    for step in range(Nt):
        with np.errstate(over="ignore", invalid="ignore"):
            U = step_explicit(U, r, alpha, dt)
        if not np.all(np.isfinite(U)):
            print(f"  FTCS diverged at step {step + 1} (NaN/Inf detected), stopping early.")
            if store_all:
                nan_frame = np.full(Nx, np.nan)
                history.extend([nan_frame] * (Nt - step - 1))
            break
        if store_all:
            history.append(U.copy())

    return (U, np.array(history)) if store_all else U

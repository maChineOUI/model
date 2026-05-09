import statistics
import time

import numpy as np

from schemes.crank_nicolson import solve_crank_nicolson, _build_rhs
from schemes.implicit import _thomas
from utils.grid import make_grid
from utils.initial_conditions import get_initial_condition


def solve_crank_nicolson_baseline(U0, Nx, Nt, r, alpha, dt):
    """Original CN implementation: rebuild Thomas coefficients at every step."""
    c = r / 2.0
    diag = np.full(Nx, 1.0 + r + alpha * dt / 2.0)
    lower = np.full(Nx, -c)
    upper = np.full(Nx, -c)

    U = U0.copy()
    for _ in range(Nt):
        rhs = _build_rhs(U, r, alpha, dt)
        U = _thomas(lower, diag, upper, rhs)
    return U


def time_median(fn, repeats=5):
    timings = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        timings.append(time.perf_counter() - start)
    return statistics.median(timings), result


def main():
    Nx = 1200
    Nt = 800
    T = 0.001
    alpha = 1.0
    repeats = 5

    dx, dt, x, r = make_grid(Nx, Nt, T)
    x_interior = x[1:-1]
    U0 = get_initial_condition("f2", x_interior)

    baseline_time, baseline_u = time_median(
        lambda: solve_crank_nicolson_baseline(U0, Nx, Nt, r, alpha, dt),
        repeats=repeats,
    )
    optimized_time, optimized_u = time_median(
        lambda: solve_crank_nicolson(U0, Nx, Nt, r, alpha, dt, store_all=False),
        repeats=repeats,
    )

    max_diff = float(np.max(np.abs(baseline_u - optimized_u)))
    speedup = baseline_time / optimized_time

    full_history_bytes = (Nt + 1) * Nx * np.dtype(float).itemsize
    selected_snapshots = 5
    snapshot_bytes = (selected_snapshots + 1) * Nx * np.dtype(float).itemsize
    memory_ratio = full_history_bytes / snapshot_bytes

    store_time, (_, history) = time_median(
        lambda: solve_crank_nicolson(U0, Nx, Nt, r, alpha, dt, store_all=True),
        repeats=3,
    )
    no_store_time, _ = time_median(
        lambda: solve_crank_nicolson(U0, Nx, Nt, r, alpha, dt, store_all=False),
        repeats=3,
    )

    print("Benchmark configuration")
    print(f"  Nx={Nx}, Nt={Nt}, T={T}, alpha={alpha}, r={r:.6f}, repeats={repeats}")
    print()
    print("Thomas factor pre-computation")
    print(f"  baseline CN median time:  {baseline_time:.4f} s")
    print(f"  optimized CN median time: {optimized_time:.4f} s")
    print(f"  speedup:                  {speedup:.2f}x")
    print(f"  max difference:           {max_diff:.3e}")
    print()
    print("History storage")
    print(f"  no-history median time:   {no_store_time:.4f} s")
    print(f"  full-history median time: {store_time:.4f} s")
    print(f"  stored history shape:     {history.shape}")
    print(f"  full history memory:      {full_history_bytes / 1024**2:.2f} MiB")
    print(f"  5 snapshots memory:       {snapshot_bytes / 1024**2:.2f} MiB")
    print(f"  memory reduction:         {memory_ratio:.1f}x")


if __name__ == "__main__":
    main()

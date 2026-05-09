import numpy as np
import matplotlib.pyplot as plt
from utils.initial_conditions import exact_solution_f1, exact_solution_f2

_EXACT_FN = {"f1": exact_solution_f1, "f2": exact_solution_f2}


def plot_snapshots(history, x_interior, dt, title, snapshot_steps=None, ax=None):
    Nt = len(history) - 1
    if snapshot_steps is None:
        snapshot_steps = [0, Nt // 4, Nt // 2, 3 * Nt // 4, Nt]

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(7, 4))

    for step in snapshot_steps:
        t = step * dt
        ax.plot(x_interior, history[step], label=f"t={t:.3f}")

    ax.set_xlabel("x")
    ax.set_ylabel("u(x,t)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    if own_fig:
        plt.tight_layout()
        return fig


def plot_scheme_comparison(results, x_interior, dt, alpha, initial_condition,
                           snapshot_steps=None, params=None, save_path=None):
    schemes = list(results.keys())
    Nt = len(next(iter(results.values()))[1]) - 1

    if snapshot_steps is None:
        snapshot_steps = [0, Nt // 4, Nt // 2, 3 * Nt // 4, Nt]

    n_times = len(snapshot_steps)
    n_schemes = len(schemes)
    fig, axes = plt.subplots(n_times, n_schemes,
                             figsize=(4 * n_schemes, 3 * n_times),
                             sharey="row", sharex=True)
    if n_times == 1:
        axes = axes[np.newaxis, :]
    if n_schemes == 1:
        axes = axes[:, np.newaxis]

    exact_fn = _EXACT_FN.get(initial_condition)

    for col, scheme in enumerate(schemes):
        _, history = results[scheme]
        for row, step in enumerate(snapshot_steps):
            ax = axes[row, col]
            t = step * dt
            ax.plot(x_interior, history[step], color="steelblue")
            if exact_fn is not None:
                u_exact = exact_fn(x_interior, t, alpha)
                ax.plot(x_interior, u_exact, "r--", linewidth=1, label="exact")
                ax.legend(fontsize=7)
            ax.set_title(f"{scheme}  t={t:.3f}", fontsize=9)
            ax.grid(True, alpha=0.3)
            if col == 0:
                ax.set_ylabel("u(x,t)")
            if row == n_times - 1:
                ax.set_xlabel("x")

    title = f"Scheme comparison — IC={initial_condition}, α={alpha}"
    if params:
        title += f" — {params}"
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig


def plot_validation(history, x_interior, dt, alpha, initial_condition="f1",
                    title="Validation", save_path=None):
    exact_fn = _EXACT_FN[initial_condition]
    Nt = len(history) - 1
    snapshot_steps = [Nt // 4, Nt // 2, 3 * Nt // 4, Nt]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    for step in snapshot_steps:
        t = step * dt
        line, = ax.plot(x_interior, history[step], label=f"num t={t:.3f}")
        u_exact = exact_fn(x_interior, t, alpha)
        ax.plot(x_interior, u_exact, "--", color=line.get_color(), linewidth=1)
    ax.set_title("Numerical vs exact")
    ax.set_xlabel("x")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    dx = x_interior[1] - x_interior[0]
    l2_errors = []
    times = []
    for step in range(1, Nt + 1):
        t = step * dt
        frame = history[step]
        if not np.all(np.isfinite(frame)):
            break
        u_exact = exact_fn(x_interior, t, alpha)
        l2 = np.sqrt(dx * np.sum((frame - u_exact) ** 2))
        l2_errors.append(l2)
        times.append(t)
    ax.semilogy(times, l2_errors, color="steelblue")
    ax.set_title("L2 error over time")
    ax.set_xlabel("t")
    ax.set_ylabel("L2 error")
    ax.grid(True, alpha=0.3)

    fig.suptitle(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig

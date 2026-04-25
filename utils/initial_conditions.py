import numpy as np


def f1(x):
    return np.sin(2 * np.pi * x)


def f2(x):
    return np.where(x <= 0.5, 2 * x, 2 * (1 - x))


def get_initial_condition(name, x_interior):
    if name == "f1":
        return f1(x_interior)
    elif name == "f2":
        return f2(x_interior)
    else:
        raise ValueError(f"Unknown initial condition: {name}")


def exact_solution_f1(x, t, alpha):
    return np.exp(-(4 * np.pi**2 + alpha) * t) * np.sin(2 * np.pi * x)


def exact_solution_f2(x, t, alpha, K=50):
    """Truncated Fourier series solution for f2 (tent function), K odd modes."""
    u = np.zeros_like(x, dtype=float)
    for k in range(K):
        n = 2 * k + 1
        bn = 8.0 * ((-1) ** k) / (np.pi ** 2 * n ** 2)
        u += bn * np.exp(-(n ** 2 * np.pi ** 2 + alpha) * t) * np.sin(n * np.pi * x)
    return u

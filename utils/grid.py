import numpy as np


def make_grid(Nx, Nt, T):
    dx = 1.0 / (Nx + 1)
    dt = T / Nt
    x = np.linspace(0, 1, Nx + 2)  # x_0 ... x_{Nx+1}
    r = dt / dx**2
    return dx, dt, x, r

from utils.grid import make_grid
from utils.initial_conditions import get_initial_condition
from schemes.explicit import solve_explicit
from schemes.implicit import solve_implicit
from schemes.crank_nicolson import solve_crank_nicolson


def run_solver(scheme, Nx, Nt, T, alpha, initial_condition, store_all=False):
    dx, dt, x, r = make_grid(Nx, Nt, T)
    x_interior = x[1:-1]  # x_1 ... x_Nx
    U0 = get_initial_condition(initial_condition, x_interior)

    if scheme == "explicit":
        result = solve_explicit(U0, Nx, Nt, r, alpha, dt, store_all)
    elif scheme == "implicit":
        result = solve_implicit(U0, Nx, Nt, r, alpha, dt, store_all)
    elif scheme == "crank-nicolson":
        result = solve_crank_nicolson(U0, Nx, Nt, r, alpha, dt, store_all)
    else:
        raise ValueError(f"Unknown scheme: {scheme}")

    return result, x_interior, dx, dt, r

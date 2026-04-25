import argparse
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.grid import make_grid
from utils.initial_conditions import get_initial_condition, exact_solution_f1, exact_solution_f2
from utils.solver import run_solver
from utils.visualization import plot_snapshots, plot_scheme_comparison, plot_validation


def parse_args():
    p = argparse.ArgumentParser(description="PDE solver: u_t = u_xx - alpha*u")
    p.add_argument("--scheme", choices=["explicit", "implicit", "crank-nicolson"],
                   default="crank-nicolson")
    p.add_argument("--Nx",    type=int,   default=100)
    p.add_argument("--Nt",    type=int,   default=500)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--T",     type=float, default=0.5)
    p.add_argument("--initial", choices=["f1", "f2"], default="f1")
    p.add_argument("--compare", action="store_true",
                   help="Run all three schemes and produce comparison plot")
    p.add_argument("--validate", action="store_true",
                   help="Show validation against analytical solution (f1 only)")
    p.add_argument("--save-prefix", default="output",
                   help="Prefix for saved figures")
    return p.parse_args()


def print_summary(scheme, Nx, Nt, T, alpha, ic, r):
    print(f"\n{'='*50}")
    print(f"  PDE Solver: u_t = u_xx - {alpha}*u")
    print(f"  Scheme:     {scheme}")
    print(f"  Grid:       Nx={Nx}, Nt={Nt}, T={T}")
    print(f"  IC:         {ic}")
    print(f"  r = dt/dx²: {r:.4f}")
    if scheme == "explicit" and r > 0.5:
        print(f"  *** UNSTABLE: r={r:.4f} > 0.5 ***")
    print(f"{'='*50}\n")


def run_single(args):
    result, x_interior, dx, dt, r = run_solver(
        args.scheme, args.Nx, args.Nt, args.T, args.alpha, args.initial,
        store_all=True
    )
    U_final, history = result
    print_summary(args.scheme, args.Nx, args.Nt, args.T, args.alpha, args.initial, r)

    fig = plot_snapshots(history, x_interior, dt,
                         title=f"{args.scheme} | IC={args.initial} | α={args.alpha}")
    path = f"{args.save_prefix}_{args.scheme}_{args.initial}.png"
    fig.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close(fig)

    if args.validate:
        fig_v = plot_validation(history, x_interior, dt, args.alpha,
                                initial_condition=args.initial,
                                title=f"Validation — {args.scheme} | IC={args.initial}")
        path_v = f"{args.save_prefix}_{args.scheme}_validation.png"
        fig_v.savefig(path_v, dpi=150)
        print(f"Saved: {path_v}")
        plt.close(fig_v)

        T_final = args.Nt * dt
        exact_fn = exact_solution_f1 if args.initial == "f1" else exact_solution_f2
        u_exact = exact_fn(x_interior, T_final, args.alpha)
        l2 = np.sqrt(dx * np.sum((U_final - u_exact) ** 2))
        linf = np.max(np.abs(U_final - u_exact))
        print(f"Final L2 error:   {l2:.6e}")
        print(f"Final Linf error: {linf:.6e}")


def run_comparison(args):
    schemes = ["explicit", "implicit", "crank-nicolson"]
    results = {}
    x_interior = dx = dt = None
    for scheme in schemes:
        result, x_interior, dx, dt, r = run_solver(
            scheme, args.Nx, args.Nt, args.T, args.alpha, args.initial,
            store_all=True
        )
        print_summary(scheme, args.Nx, args.Nt, args.T, args.alpha, args.initial, r)
        results[scheme] = result  # (U_final, history)

    fig = plot_scheme_comparison(results, x_interior, dt, args.alpha,
                                 args.initial,
                                 save_path=f"{args.save_prefix}_compare_{args.initial}.png")
    print(f"Saved: {args.save_prefix}_compare_{args.initial}.png")
    plt.close(fig)

    T_final = args.Nt * dt
    exact_fn = exact_solution_f1 if args.initial == "f1" else exact_solution_f2
    print(f"\nFinal errors vs exact solution ({args.initial}):")
    print(f"{'scheme':20s}  {'L2':>12s}  {'Linf':>12s}")
    for scheme, (U_final, _) in results.items():
        if not np.all(np.isfinite(U_final)):
            print(f"  {scheme:18s}  {'NaN':>12s}  {'NaN':>12s}")
            continue
        u_exact = exact_fn(x_interior, T_final, args.alpha)
        l2   = np.sqrt(dx * np.sum((U_final - u_exact) ** 2))
        linf = np.max(np.abs(U_final - u_exact))
        print(f"  {scheme:18s}  {l2:12.6e}  {linf:12.6e}")


def main():
    args = parse_args()
    if args.compare:
        run_comparison(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()

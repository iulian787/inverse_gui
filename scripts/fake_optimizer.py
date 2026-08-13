#!/usr/bin/env python3
"""Stand-in for the AI4NS optimizer, for developing the GUI without checkpoints.

The EffPropNet checkpoints (models/fm_multi_store/*.pt) ship with neither repo and
argparse hard-fails without at least one, so on a machine that does not have them
nothing real can run. This script mirrors the CLI surface and artifact format of both
entry points closely enough to develop and test everything except the solve itself:

    run_inverse_design_fm_multi_ac.py   -> single-point (default)
    run_pareto_epsilon_fm_multi_ac.py   -> pareto (selected by --pareto_steps)

The one behaviour worth being fussy about is HOW progress is printed. The real
per-iteration table comes from IPOPT's C++ Journalist writing to C-level stdout, not
from Python -- `python -u` and PYTHONUNBUFFERED do not reach glibc's FILE* buffering
inside libipopt.so. So this script prints through ctypes printf with no flush, which
reproduces the real buffering: line-buffered on a tty, 4 KB block-buffered on a pipe.
A Runner that streams correctly here will stream correctly against the real optimizer;
one that only works against Python-level prints will appear frozen in production.

Point [paths].ai4ns_repo at this repo and both [scripts] entries at this file.

    python scripts/fake_optimizer.py --ckpt_elastic_fm fake.pt \
        --E "target 200000" --nu "target 0.25" --output_dir /tmp/fakerun
"""

import argparse
import ctypes
import os
import random
import sys
import time

import numpy as np

# Mirrors utils/optimization/constants.py
ALL_SCALAR_PROPS = [
    'E', 'nu', 'E_xx', 'E_yy', 'G_xy', 'nu_xy', 'nu_yx',
    'kappa', 'kappa_x', 'kappa_y',
    'alpha', 'alpha_xx', 'alpha_yy', 'alpha_xy', 'rho',
]
PROP_DEFAULTS = {
    'E': 240000.0, 'nu': 0.22, 'E_xx': 240000.0, 'E_yy': 240000.0, 'G_xy': 98000.0,
    'nu_xy': 0.22, 'nu_yx': 0.22, 'kappa': 178.5, 'kappa_x': 178.5, 'kappa_y': 178.5,
    'alpha': 1.3e-5, 'alpha_xx': 1.3e-5, 'alpha_yy': 1.3e-5, 'alpha_xy': 1e-6, 'rho': 0.5,
}

_libc = ctypes.CDLL(None)


def cprint(line=""):
    """Print via libc, deliberately unflushed -- see module docstring."""
    _libc.printf(b"%s\n", ctypes.c_char_p(line.encode()))


def parse_directive(spec):
    """'target 2e5' | 'range 1e5 3e5' | 'max' | 'min' | 'free' -> (mode, values)."""
    if not spec:
        return None
    parts = spec.split()
    mode = parts[0]
    return mode, [float(x) for x in parts[1:]]


def build_parser():
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument('--ckpt_elastic_fm', type=str, default=None)
    p.add_argument('--ckpt_thermal_conductivity_fm', type=str, default=None)
    p.add_argument('--ckpt_thermal_expansion_fm', type=str, default=None)
    p.add_argument('--act', type=str, default=None)

    p.add_argument('--data_dir', type=str, default=None)
    p.add_argument('--phase_props_json', type=str, default=None)
    for short in ['E_A', 'nu_A', 'rho_A', 'kappa_A', 'alpha_A',
                  'E_B', 'nu_B', 'rho_B', 'kappa_B', 'alpha_B', 'interface_kappa_AB']:
        p.add_argument(f'--{short}', type=float, default=None)

    for name in ALL_SCALAR_PROPS:
        p.add_argument(f'--{name}', type=str, default=None, dest=name, metavar='DIRECTIVE')
    for name in ALL_SCALAR_PROPS:
        p.add_argument(f'--weight_{name}', type=float, default=1.0, dest=f'weight_{name}')
    for name, ref in PROP_DEFAULTS.items():
        p.add_argument(f'--{name}_ref', type=float, default=ref, dest=f'{name}_ref')

    # Defaults mirror utils/optimization/filters.py:101-105 exactly.
    p.add_argument('--ac_epsi', type=float, default=2.0)
    p.add_argument('--ac_lambda', type=float, default=40.0)
    p.add_argument('--ac_dt', type=float, default=0.1)
    p.add_argument('--ac_steps', type=int, default=10)
    p.add_argument('--ac_sharpen_beta', type=float, default=0.0)

    p.add_argument('--target_tol', type=float, default=0.001)
    p.add_argument('--enforce_isotropy', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--restarts', type=int, default=1)
    p.add_argument('--ipopt_max_iter', type=int, default=500)
    p.add_argument('--ipopt_tol', type=float, default=1e-3)
    p.add_argument('--ipopt_constr_viol_tol', type=float, default=1e-3)
    p.add_argument('--ipopt_print', type=int, default=5)
    p.add_argument('--ipopt_lbfgs_history', type=int, default=50)
    p.add_argument('--ipopt_scaling', type=str, default='gradient-based')
    p.add_argument('--ipopt_mu_strategy', type=str, default='adaptive')
    p.add_argument('--ipopt_acceptable_tol', type=float, default=1e-3)
    p.add_argument('--ipopt_acceptable_iter', type=int, default=5)
    p.add_argument('--ipopt_recalc_y', action='store_true', default=True)
    p.add_argument('--ipopt_alpha_for_y', type=str, default='safer-min-dual-infeas')

    p.add_argument('--vf_min', type=float, default=0.05)
    p.add_argument('--vf_max', type=float, default=0.95)
    p.add_argument('--beam_width', type=int, default=0)
    p.add_argument('--beam_depth', type=int, default=2)
    p.add_argument('--beam_c', type=float, default=0.1)

    p.add_argument('--output_dir', type=str, default='./plots/inverse_design_fm_multi_ac')

    p.add_argument('--fenics_validate', action='store_true')
    for f in ['E_A', 'nu_A', 'E_B', 'nu_B', 'kappa_A', 'kappa_B', 'alpha_A', 'alpha_B']:
        p.add_argument(f'--fenics_{f}', type=float, default=None)
    p.add_argument('--fenics_h5_props', type=str, default=None)
    p.add_argument('--fenics_h5_sample_idx', type=int, default=0)
    p.add_argument('--fenics_output_dir', type=str, default=None)
    p.add_argument('--fenics_conda_env', type=str, default='fenics_env')

    # Pareto-only upstream; its presence is what selects pareto mode here.
    p.add_argument('--pareto_steps', type=int, default=None)

    # Fake-only knobs for exercising Runner failure paths.
    # Env defaults keep the test suite fast without every call site passing flags.
    p.add_argument('--iters', type=int, default=int(os.environ.get('FAKE_ITERS', 25)),
                   help='[fake] iterations to emit')
    p.add_argument('--iter_delay', type=float,
                   default=float(os.environ.get('FAKE_ITER_DELAY', 1.0)),
                   help='[fake] seconds per iteration')
    p.add_argument('--crash', action='store_true', help='[fake] die mid-solve with exit 3')
    p.add_argument('--hang', action='store_true', help='[fake] never terminate; test cancel')
    return p


def active_directives(args):
    out = {}
    for name in ALL_SCALAR_PROPS:
        spec = getattr(args, name, None)
        if spec:
            out[name] = parse_directive(spec)
    return out or {'E': ('target', [PROP_DEFAULTS['E']])}


def synthetic_mask(rng, n=128):
    """Blobby binary field that looks like a spinodal microstructure."""
    field = rng.normal(size=(n, n))
    fx = np.fft.fftfreq(n)[:, None]
    fy = np.fft.fftfreq(n)[None, :]
    r = np.sqrt(fx ** 2 + fy ** 2)
    band = np.exp(-((r - 0.06) ** 2) / (2 * 0.02 ** 2))
    smoothed = np.real(np.fft.ifft2(np.fft.fft2(field) * band))
    return (smoothed > np.median(smoothed)).astype(np.float64)


def achieved(mode, vals, ref, rng):
    """A plausible achieved value for a directive, close-but-not-exact."""
    if mode == 'target':
        return vals[0] * (1.0 + rng.normal(0, 0.02))
    if mode == 'range':
        return (vals[0] + vals[1]) / 2 * (1.0 + rng.normal(0, 0.02))
    if mode == 'max':
        return ref * rng.uniform(1.1, 1.4)
    if mode == 'min':
        return ref * rng.uniform(0.6, 0.9)
    return ref


def emit_header(args, mode_label):
    cprint(f"  Device: cpu")
    cprint(f"  Mode:   {mode_label}")
    cprint(f"  Output: {args.output_dir}")
    cprint("")


def emit_ipopt_banner():
    cprint("This is Ipopt version 3.14.16, running with linear solver MUMPS 5.7.3.")
    cprint("")
    cprint("iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls")


def emit_ipopt_rows(n_iters, delay, obj0=1.0, start=0):
    """Emit n_iters rows of IPOPT's print_level>=5 table, C-buffered. Banner is separate
    so a multi-restart or multi-grid-point run prints one banner per solve, not per row."""
    obj = obj0
    for i in range(n_iters):
        k = start + i
        obj *= random.uniform(0.72, 0.93)
        inf_pr = obj * random.uniform(0.05, 0.4)
        inf_du = obj * random.uniform(0.01, 0.2)
        cprint(f"{k:>4d}  {obj:.7e} {inf_pr:.2e} {inf_du:.2e} "
               f"{-1.0 - k * 0.1:5.1f} {obj * 3:.2e}    -  "
               f"{random.uniform(0.5, 1.0):.2e} {random.uniform(0.5, 1.0):.2e}   1")
        time.sleep(delay)
    return obj


def write_single_point(args, directives, rng, final_obj, loss_hist):
    os.makedirs(args.output_dir, exist_ok=True)
    mask = synthetic_mask(rng)
    payload = {
        'optimized_material': mask,
        'optimized_material_expanded': np.block([[mask, mask[:, ::-1]],
                                                 [mask[::-1, :], mask[::-1, ::-1]]]),
        'loss_hist': np.asarray(loss_hist),
        'final_loss': float(final_obj),
        'ac_epsi': args.ac_epsi, 'ac_lambda': args.ac_lambda, 'ac_dt': args.ac_dt,
        'ac_steps': args.ac_steps, 'ac_sharpen_beta': args.ac_sharpen_beta,
    }
    for name, (mode, vals) in directives.items():
        ref = getattr(args, f'{name}_ref', PROP_DEFAULTS.get(name, 1.0))
        val = achieved(mode, vals, ref, rng)
        payload[f'effective_{name}'] = val
        payload[f'directive_{name}'] = mode
        if mode == 'target':
            payload[f'target_{name}'] = vals[0]
        elif mode == 'range':
            payload[f'range_{name}_lo'] = vals[0]
            payload[f'range_{name}_hi'] = vals[1]
        n = len(loss_hist)
        payload[f'hist_{name}'] = val * (1 + np.linspace(0.5, 0, n) * rng.normal(1, .2, n))

    path = os.path.join(args.output_dir, 'inverse_result_fm_multi_ac.npz')
    np.savez_compressed(path, **payload)
    cprint(f"\nSaved: {path}")

    try:                                   # optional: absent in the GUI venv
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(9, 4))
        ax[0].semilogy(loss_hist); ax[0].set_title('convergence')
        ax[1].imshow(mask, cmap='gray', origin='lower')
        ax[1].set_title(f'VF = {mask.mean():.3f}')
        png = os.path.join(args.output_dir, 'inverse_design_fm_multi_ac_result.png')
        fig.savefig(png, dpi=150); plt.close(fig)
        cprint(f"Saved: {png}")
    except ImportError:
        cprint("[fake] matplotlib unavailable; skipped PNG")


def write_pareto(args, directives, rng, n):
    os.makedirs(args.output_dir, exist_ok=True)
    prop_names = [p for p in directives if p != 'rho'] or ['E']
    props = np.full((n, len(prop_names)), np.nan)
    masks = np.zeros((n, 128, 128), dtype=np.uint8)
    rho = np.linspace(0.25, 0.75, n)
    for i in range(n):
        m = synthetic_mask(rng)
        masks[i] = m.astype(np.uint8)
        for j, name in enumerate(prop_names):
            ref = getattr(args, f'{name}_ref', PROP_DEFAULTS.get(name, 1.0))
            props[i, j] = ref * (0.6 + 1.1 * rho[i]) * (1 + rng.normal(0, 0.03))

    rank = np.zeros(n, dtype=np.int64)
    rank[rng.choice(n, size=max(1, n // 4), replace=False)] = 1   # some dominated
    path = os.path.join(args.output_dir, 'pareto_results.npz')
    np.savez_compressed(
        path,
        rho_cost=rho,
        rho_directive_mode='min',
        pareto_rank=rank,
        crowding_distance=rng.uniform(0, 1, n),
        feasible=rng.random(n) > 0.15,
        statuses=np.array(['Solve_Succeeded'] * n),
        prop_names=np.array(prop_names),
        props=props,
        microstructures=masks,
    )
    cprint(f"\nSaved: {path}")
    cprint("[note] the real pareto script writes no PNG; plotting is a separate step")


def main():
    args = build_parser().parse_args()
    if not any([args.ckpt_elastic_fm, args.ckpt_thermal_conductivity_fm,
                args.ckpt_thermal_expansion_fm]):
        build_parser().error("Provide at least one --ckpt_*_fm checkpoint.")

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    directives = active_directives(args)
    is_pareto = args.pareto_steps is not None

    emit_header(args, 'pareto (epsilon-constraint)' if is_pareto else 'single-point')

    if args.hang:
        cprint("[fake] --hang: sleeping forever; cancel me")
        while True:
            time.sleep(1)

    if is_pareto:
        n = max(2, args.pareto_steps)
        n_obj = sum(1 for p, (m, _) in directives.items()
                    if m in ('max', 'min') and p != 'rho') or 1
        cprint(f"\n  Estimated solves: {args.restarts} feasibility + "
               f"({n_obj} payoff rows + {n} grid pts ({n}^{n_obj}))"
               f" × {args.restarts} restarts = {args.restarts * (n_obj + n)} total")
        # Stage banners match upstream byte for byte, including the U+2500 rules --
        # a fake that prints its own format would not exercise the real parser.
        for stage, title in enumerate([
            'Constraint Feasibility Check',
            'Payoff Table',
            'Epsilon-Constraint Pareto Sweep (ρ-objective)',
            'Non-Dominated Sorting',
        ]):
            cprint("\n" + "─" * 65)
            cprint(f"Stage {stage}: {title}")
            cprint("─" * 65)
            if stage == 2:
                cprint(f"\n  Epsilon grid: [{n}] points per property "
                       f"({n} combinations × {args.restarts} restarts "
                       f"= {n * args.restarts} solves)")
                for g in range(n):
                    cprint(f"\n  Grid point {g + 1}/{n}: "
                           f"{{'eps': {0.1 * (g + 1):.4g}}}")
                    emit_ipopt_banner()
                    emit_ipopt_rows(max(2, args.iters // n), args.iter_delay)
                    if args.crash and g == n // 2:
                        cprint("[fake] --crash"); sys.exit(3)
            else:
                time.sleep(args.iter_delay)
        write_pareto(args, directives, rng, n)
    else:
        hist = []
        for r in range(args.restarts):
            if args.restarts > 1:
                cprint(f"\n[restart {r}]")
            emit_ipopt_banner()
            obj = 1.0
            for k in range(args.iters):
                obj = emit_ipopt_rows(1, args.iter_delay, obj0=obj, start=k)
                hist.append(obj)
                if args.crash and k == args.iters // 2:
                    cprint("[fake] --crash"); sys.exit(3)
        write_single_point(args, directives, rng, hist[-1], hist)

    if args.fenics_validate:
        cprint(f"\n[FEniCS] would run in env={args.fenics_conda_env} (fake: skipped)")

    cprint("\nDone.")


if __name__ == '__main__':
    main()

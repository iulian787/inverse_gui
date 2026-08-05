"""Pareto solve-count estimator.

Reproduces run_pareto_epsilon_fm_multi_ac.py:1062-1067 verbatim, so the number the
form shows matches what `--dry_run` prints. This matters: the grid is a Cartesian
product, so the count is (2N-1)^k and blows up fast. Two objectives at the default
pareto_steps=10 is 1092 solves; three is ~20,700.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .schema import RunConfig, RunMode


@dataclass(frozen=True)
class CostEstimate:
    pts_per_prop: int
    n_objectives: int
    n_constraints: int
    total_grid: int
    stage0_solves: int
    total_solves: int

    @property
    def upstream_line(self) -> str:
        """The same sentence the script prints, for eyeball comparison with --dry_run."""
        return (f'{self.stage0_solves} feasibility + ({self.n_objectives} payoff rows '
                f'+ {self.total_grid} grid pts '
                f'({self.pts_per_prop}^{self.n_objectives}))'
                f' = {self.total_solves} total')


def points_per_prop(n_steps: int) -> int:
    """|union(linspace(0,1,N), logspace(-6,0,N))| -- upstream's own expression.

    The two grids share only the endpoint 1.0, so this is 2N-1, not 2N-2.
    Measured: N=10 -> 19.
    """
    if n_steps <= 1:
        return 1
    return int(len(np.union1d(np.linspace(0, 1, n_steps),
                              np.logspace(-6, 0, n_steps))))


def estimate(cfg: RunConfig) -> CostEstimate | None:
    """None for single-point mode, where the count is simply `restarts`."""
    if cfg.mode is not RunMode.PARETO:
        return None
    n_obj = len(cfg.objective_props())
    n_con = len(cfg.constraint_props())
    pts = points_per_prop(cfg.pareto_steps)
    total_grid = pts ** n_obj if n_obj > 0 else 1
    stage0 = cfg.restarts if n_con > 0 else 0
    total = stage0 + (n_obj + total_grid) * cfg.restarts
    return CostEstimate(
        pts_per_prop=pts, n_objectives=n_obj, n_constraints=n_con,
        total_grid=total_grid, stage0_solves=stage0, total_solves=total,
    )


def eta_seconds(est: CostEstimate, seconds_per_solve: float) -> float:
    return est.total_solves * seconds_per_solve


def format_duration(seconds: float) -> str:
    if seconds < 90:
        return f'{seconds:.0f}s'
    if seconds < 5400:
        return f'{seconds / 60:.0f}m'
    if seconds < 172800:
        return f'{seconds / 3600:.1f}h'
    return f'{seconds / 86400:.1f}d'

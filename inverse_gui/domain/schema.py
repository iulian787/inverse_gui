"""RunConfig: the validated object the form produces and the Runner consumes.

There is no upstream schema to reuse -- the design deck's claim that this is "built
from the existing Pydantic schema" is wrong; `grep -rn pydantic` over amit_AI4NS
returns nothing. This is authored against the two scripts' argparse surfaces.

Defaults here mirror upstream exactly, INCLUDING the places where the two scripts
disagree (target_tol, restarts, output_dir) -- see MODE_DEFAULTS.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from enum import Enum

from .directives import Directive, Mode


class RunMode(str, Enum):
    SINGLE = 'single_point'
    PARETO = 'pareto'

    @property
    def label(self) -> str:
        return 'Single-point' if self is RunMode.SINGLE else 'Pareto (ε-constraint)'


# Flags whose default differs between the two scripts.
MODE_DEFAULTS: dict[RunMode, dict[str, object]] = {
    RunMode.SINGLE: {
        'target_tol': 0.001,
        'restarts': 1,
        'output_dir': './plots/inverse_design_fm_multi_ac',
    },
    RunMode.PARETO: {
        'target_tol': 0.02,
        'restarts': 3,
        'output_dir': './plots/pareto_fm_multi_ac',
    },
}


@dataclass
class Checkpoints:
    elastic: str = ''
    thermal_conductivity: str = ''
    thermal_expansion: str = ''

    def loaded_physics(self) -> frozenset[str]:
        out = set()
        if self.elastic.strip():
            out.add('elastic')
        if self.thermal_conductivity.strip():
            out.add('thermal_conductivity')
        if self.thermal_expansion.strip():
            out.add('thermal_expansion')
        return frozenset(out)

    def path_for(self, physics: str) -> str:
        return getattr(self, physics, '')


@dataclass
class PhaseProps:
    """Per-phase overrides. None means "leave upstream's default alone"."""
    a: dict[str, float | None] = field(default_factory=dict)   # keyed by flag stem, e.g. 'E_A'
    b: dict[str, float | None] = field(default_factory=dict)
    interface_kappa_AB: float | None = None
    data_dir: str = ''
    phase_props_json: str = ''


@dataclass
class AllenCahn:
    """utils/optimization/filters.py:101-105 defaults."""
    epsi: float = 2.0
    lam: float = 40.0
    dt: float = 0.1
    steps: int = 10
    sharpen_beta: float = 0.0


@dataclass
class Ipopt:
    max_iter: int = 500
    tol: float = 1e-3
    constr_viol_tol: float = 1e-3
    print_level: int = 5
    lbfgs_history: int = 50
    scaling: str = 'gradient-based'
    mu_strategy: str = 'adaptive'
    acceptable_tol: float = 1e-3
    acceptable_iter: int = 5
    alpha_for_y: str = 'safer-min-dual-infeas'
    # --ipopt_recalc_y is store_true with default=True upstream, so it is always on
    # and cannot be disabled from the CLI. Emitting it is a no-op; we never do.


SCALING_CHOICES = ('gradient-based', 'none', 'user-scaling')
MU_STRATEGY_CHOICES = ('monotone', 'adaptive')
ALPHA_FOR_Y_CHOICES = ('primal', 'bound-mult', 'min', 'max', 'full',
                       'min-dual-infeas', 'safer-min-dual-infeas')


@dataclass
class Beam:
    width: int = 0      # 0 disables beam search
    depth: int = 2
    c: float = 0.1


@dataclass
class Fenics:
    validate: bool = False
    conda_env: str = 'fenics_env'
    output_dir: str = ''
    h5_props: str = ''
    h5_sample_idx: int = 0
    # per-phase overrides, separate from the model phase props and not kept in sync
    props: dict[str, float | None] = field(default_factory=dict)


@dataclass
class RunConfig:
    mode: RunMode = RunMode.SINGLE
    checkpoints: Checkpoints = field(default_factory=Checkpoints)
    directives: dict[str, Directive] = field(default_factory=dict)
    # Which families are shown per-component rather than isotropic. Affects only
    # which directives the form lets you set, not the emitted argv.
    anisotropic: frozenset[str] = frozenset()
    phase: PhaseProps = field(default_factory=PhaseProps)
    ac: AllenCahn = field(default_factory=AllenCahn)
    ipopt: Ipopt = field(default_factory=Ipopt)
    beam: Beam = field(default_factory=Beam)
    fenics: Fenics = field(default_factory=Fenics)

    seed: int = 42
    restarts: int = 1
    target_tol: float = 0.001
    enforce_isotropy: bool = False
    act: str = ''                      # leave empty: the checkpoint carries its own
    output_dir: str = './plots/inverse_design_fm_multi_ac'

    # single-point only
    vf_min: float = 0.05
    vf_max: float = 0.95

    # Pareto only
    pareto_steps: int = 10
    bin_viol_tol: float = 0.05
    feasibility_threshold: float = 1e-3
    dry_run: bool = False

    # ---------------------------------------------------------------- helpers

    def active_directives(self) -> dict[str, Directive]:
        return {p: d for p, d in self.directives.items() if d.active}

    def objective_props(self) -> list[str]:
        """Non-rho max/min props -- the epsilon-swept axes in Pareto mode."""
        return sorted(
            p for p, d in self.active_directives().items()
            if d.mode.is_objective and p != 'rho'
        )

    def constraint_props(self) -> list[str]:
        """Non-rho target/range props -- hard constraints (or soft, if infeasible)."""
        return sorted(
            p for p, d in self.active_directives().items()
            if d.mode.is_constraint and p != 'rho'
        )

    def derived_vf_bounds(self) -> tuple[float, float]:
        """Pareto derives vf bounds from the rho directive; single-point does not.

        Mirrors run_pareto_epsilon_fm_multi_ac.py:66-75 (vf_bounds_from_rho).
        """
        if self.mode is RunMode.SINGLE:
            return self.vf_min, self.vf_max
        d = self.directives.get('rho')
        if d is None or not d.active or d.mode in (Mode.MAX, Mode.MIN):
            return 0.05, 0.95
        if d.mode is Mode.TARGET and d.value is not None:
            return d.value * (1 - self.target_tol), d.value * (1 + self.target_tol)
        if d.mode is Mode.RANGE and d.value is not None and d.hi is not None:
            return d.value, d.hi
        return 0.05, 0.95

    @classmethod
    def for_mode(cls, mode: RunMode, **kwargs) -> 'RunConfig':
        """Build a config with the correct defaults for `mode`.

        Prefer this over RunConfig(mode=...) — the dataclass field defaults are the
        single-point ones, so a directly-constructed Pareto config would carry
        target_tol=0.001 and restarts=1 rather than 0.02 and 3.
        """
        base = {'mode': mode, **MODE_DEFAULTS[mode]}
        base.update(kwargs)
        return cls(**base)  # type: ignore[arg-type]

    def apply_mode_defaults(self) -> 'RunConfig':
        """Return a copy with the mode's canonical defaults for the diverging flags.

        Called when the user flips the mode toggle, so target_tol/restarts/output_dir
        track the script actually being launched instead of silently keeping the
        other script's value.
        """
        return dataclasses.replace(self, **MODE_DEFAULTS[self.mode])  # type: ignore[arg-type]

"""The rules engine: RunConfig -> list[Issue].

Every rule here corresponds to a way the upstream scripts fail, and most of them
fail *silently* or with a misleading message. The comments name the failure mode,
because "why does the form complain about this" is otherwise unanswerable.

Severity contract:
  ERROR   blocks Launch
  WARNING allows Launch, shown amber
  INFO    allows Launch, collapsed by default
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum

from . import properties as props
from .directives import Mode
from .schema import RunConfig, RunMode


class Severity(str, Enum):
    ERROR = 'error'
    WARNING = 'warning'
    INFO = 'info'


@dataclass(frozen=True)
class Issue:
    code: str
    severity: Severity
    message: str
    # NB: this attribute shadows dataclasses.field inside the class body, so the
    # defaults below must be literals rather than field(default_factory=...).
    field: str = ''            # form field key, for inline placement
    remedy: str = ''           # what the user should do
    fix_props: tuple[str, ...] = ()   # for one-click "set these to free" fixes

    @property
    def blocking(self) -> bool:
        return self.severity is Severity.ERROR


def validate(cfg: RunConfig, *, check_files: bool = True) -> list[Issue]:
    out: list[Issue] = []
    loaded = cfg.checkpoints.loaded_physics()
    available = props.available_props(loaded)
    active = cfg.active_directives()

    _check_checkpoints(cfg, loaded, out, check_files)
    _check_gating(cfg, available, active, out)
    _check_directive_values(cfg, active, out)
    _check_isotropic_conflicts(cfg, active, out)
    _check_refs_and_weights(cfg, active, out)
    _check_phase_props(cfg, loaded, out)
    _check_vf(cfg, out)
    _check_mode_specific(cfg, active, out)
    _check_fenics(cfg, out)
    _check_early_exit(cfg, active, out)
    return out


# ------------------------------------------------------------------ A. checkpoints

def _check_checkpoints(cfg, loaded, out, check_files):
    if not loaded:
        out.append(Issue(
            'NO_CHECKPOINT', Severity.ERROR,
            'No checkpoint provided. At least one is required.',
            field='checkpoints',
            remedy='Set at least one of the three --ckpt_*_fm paths in section A. '
                   'Both scripts call parser.error() without one.',
        ))
        return
    if not check_files:
        return
    for fam in props.FAMILIES:
        path = cfg.checkpoints.path_for(fam.physics).strip()
        if path and not os.path.exists(path):
            out.append(Issue(
                'CKPT_MISSING', Severity.ERROR,
                f'{fam.label} checkpoint not found: {path}',
                field=f'ckpt.{fam.physics}',
                # The Pareto script checks for checkpoints only AFTER the load loop,
                # so a bad path gives a traceback rather than a clean argparse error.
                remedy='Fix the path, or clear it to disable this physics.',
            ))


# ------------------------------------------------------------------ B. gating

def _check_gating(cfg, available, active, out):
    """The single highest-value check.

    The Pareto script warns and drops an ungated directive. The single-point script
    does NOT gate at all: a target/range on an unpredictable property becomes a
    constraint row with g=0.0 and an all-zero Jacobian (solver.py:138,150), so it can
    never be satisfied. Ipopt reports local infeasibility and the user has no way to
    know a missing checkpoint caused it. A max/min is silently dropped instead.
    """
    ungated = [p for p in active if p not in available]
    if not ungated:
        return
    constraints = [p for p in ungated if active[p].mode.is_constraint]
    objectives = [p for p in ungated if active[p].mode.is_objective]

    if constraints:
        needed = sorted({props.physics_for_prop(p) or '?' for p in constraints})
        out.append(Issue(
            'UNGATED_CONSTRAINT', Severity.ERROR,
            f"Target/range set on {', '.join(sorted(constraints))}, but no "
            f"{', '.join(needed)} checkpoint is loaded.",
            field='directives',
            remedy=('This does not error upstream in single-point mode — it becomes a '
                    'constraint the solver can never satisfy (g=0 with zero gradient), '
                    'reported as an Ipopt convergence failure. Load the checkpoint or '
                    'set these directives to free.'),
            fix_props=tuple(sorted(constraints)),
        ))
    if objectives:
        out.append(Issue(
            'UNGATED_OBJECTIVE', Severity.ERROR,
            f"max/min set on {', '.join(sorted(objectives))}, but the required "
            'checkpoint is not loaded.',
            field='directives',
            remedy=('Single-point silently drops the objective term and optimises '
                    'something else; Pareto prints a warning and drops it. Either way '
                    'the run does not do what you asked.'),
            fix_props=tuple(sorted(objectives)),
        ))


# ------------------------------------------------------------------ B. values

def _check_directive_values(cfg, active, out):
    for prop, d in sorted(active.items()):
        if not d.complete:
            out.append(Issue(
                'DIRECTIVE_INCOMPLETE', Severity.ERROR,
                f'{prop}: {d.mode.value} needs '
                f'{d.mode.n_values} value(s).',
                field=f'directive.{prop}',
                # An empty value string raises IndexError upstream, not a clean error.
                remedy='Fill in the value, or set the directive to free.',
            ))
            continue
        if d.mode is Mode.RANGE and d.value is not None and d.hi is not None:
            if d.value >= d.hi:
                out.append(Issue(
                    'RANGE_INVERTED', Severity.ERROR,
                    f'{prop}: range low ({d.value:g}) must be below high ({d.hi:g}).',
                    field=f'directive.{prop}',
                    remedy=('Upstream accepts this silently and produces cl > cu in '
                            'cyipopt, which is infeasible by construction.'),
                ))
        if d.mode is Mode.TARGET and d.value == 0:
            out.append(Issue(
                'TARGET_ZERO', Severity.WARNING,
                f'{prop}: target 0 gives a degenerate zero-width band.',
                field=f'directive.{prop}',
            ))
        _check_training_range(prop, d, out)


def _check_training_range(prop, d, out):
    rng = props.training_range_for(prop)
    if rng is None:
        return
    lo, hi = rng
    vals = [v for v in (d.value, d.hi) if v is not None]
    outside = [v for v in vals if not (lo <= v <= hi)]
    if outside:
        out.append(Issue(
            'OUTSIDE_TRAINING', Severity.WARNING,
            f'{prop}: {", ".join(f"{v:g}" for v in outside)} is outside the '
            f'phase-endpoint range [{lo:g}, {hi:g}].',
            field=f'directive.{prop}',
            remedy=('An achievable effective property lies between the two phase '
                    'values. Check section C, or expect infeasibility.'),
        ))


# ------------------------------------------------------------------ B. isotropy

def _check_isotropic_conflicts(cfg, active, out):
    """Defence in depth: the form makes this structurally impossible, but a loaded
    preset or a hand-edited config can still trip it.

    Upstream (directives.py:36-45) overwrites a component only if it is still free,
    so setting both --E and --E_xx silently gives E_xx the anisotropic value and E_yy
    the isotropic one, with no warning.
    """
    for iso, comps in props.ISOTROPIC_EXPAND.items():
        if iso not in active:
            continue
        clash = [c for c in comps if c in active]
        if clash:
            out.append(Issue(
                'ISO_AND_COMPONENT', Severity.ERROR,
                f'Both --{iso} and --{", --".join(clash)} are set.',
                field=f'directive.{iso}',
                remedy=(f'Upstream would keep {", ".join(clash)} and apply --{iso} to '
                        f'the remaining component, deleting --{iso}. Pick one level.'),
                fix_props=(iso,),
            ))

    if cfg.enforce_isotropy:
        e, nu = active.get('E'), active.get('nu')
        ok = (e is not None and e.mode is Mode.TARGET
              and nu is not None and nu.mode is Mode.TARGET
              and 'G_xy' not in active)
        if not ok:
            out.append(Issue(
                'ENFORCE_ISOTROPY_NOOP', Severity.WARNING,
                '--enforce_isotropy has no effect here.',
                field='enforce_isotropy',
                remedy=('It fires only when E and nu are both *target* directives '
                        '(not range) and G_xy is free.'),
            ))


# ------------------------------------------------------------------ B. refs/weights

def _check_refs_and_weights(cfg, active, out):
    for prop, d in sorted(cfg.directives.items()):
        ref = d.ref if d.ref is not None else props.PROP_DEFAULTS.get(prop)
        if ref is not None and ref <= 0:
            out.append(Issue(
                'REF_NONPOSITIVE', Severity.ERROR,
                f'{prop}_ref must be positive (got {ref:g}).',
                field=f'ref.{prop}',
                remedy=('ref=0 raises ZeroDivisionError; a negative ref inverts the '
                        'constraint bounds and is always infeasible.'),
            ))
            continue
        # Scale mismatch: constraints are normalised as p/ref, and Ipopt's
        # constr_viol_tol is applied to the normalised value. A target far below ref
        # turns a 1e-3 absolute tolerance into a huge relative one.
        if d.active and d.value is not None and ref and d.value > 0:
            ratio = d.value / ref
            if ratio < 0.05 or ratio > 20:
                out.append(Issue(
                    'REF_SCALE', Severity.WARNING,
                    f'{prop}: target {d.value:g} is far from ref {ref:g} '
                    f'({ratio:.3g}x).',
                    field=f'ref.{prop}',
                    remedy=f'Set {prop}_ref near {d.value:g} so Ipopt sees O(1) values.',
                    fix_props=(prop,),
                ))

    if cfg.mode is RunMode.PARETO:
        weighted = [p for p, d in cfg.directives.items() if d.weight != 1.0]
        if weighted:
            out.append(Issue(
                'WEIGHTS_IGNORED_PARETO', Severity.INFO,
                f'Weights on {", ".join(sorted(weighted))} are ignored in Pareto mode.',
                field='directives',
                remedy='The Pareto script has no --weight_* flags; they are not emitted.',
            ))
    else:
        misapplied = [
            p for p, d in active.items()
            if d.weight != 1.0 and d.mode.is_constraint and cfg.beam.width == 0
        ]
        if misapplied:
            out.append(Issue(
                'WEIGHT_ON_CONSTRAINT', Severity.INFO,
                f'Weights on {", ".join(sorted(misapplied))} have no effect.',
                field='directives',
                remedy=('Weights apply to max/min objective terms. For target/range '
                        'they matter only when beam search is on (--beam_width > 0).'),
            ))


# ------------------------------------------------------------------ C. phase props

def _check_phase_props(cfg, loaded, out):
    """FmMultiAdapter raises KeyError if a required bulk key is missing for A or B."""
    required = props.required_phase_flags(loaded)
    by_key = {pf.moose_key: pf for pf in props.PHASE_FLAGS}
    using_source = bool(cfg.phase.data_dir.strip() or cfg.phase.phase_props_json.strip())

    for key in sorted(required):
        pf = by_key.get(key)
        if pf is None:
            continue
        for phase, store in (('A', cfg.phase.a), ('B', cfg.phase.b)):
            stem = pf.flag_a if phase == 'A' else pf.flag_b
            if store.get(stem) is not None:
                continue
            if props.default_phase_value(key, phase) is not None or using_source:
                continue
            out.append(Issue(
                'PHASE_PROP_MISSING', Severity.ERROR,
                f'Phase {phase} is missing {key}, required by '
                f'{", ".join(sorted(loaded))}.',
                field=f'phase.{stem}',
                remedy='FmMultiAdapter raises KeyError at load time without it.',
            ))

    if 'interfacial_conductivity' in props.required_interface_keys(loaded):
        if cfg.phase.interface_kappa_AB is None and not using_source:
            # There is a hardcoded default (1e7), so this is informational.
            out.append(Issue(
                'INTERFACE_DEFAULTED', Severity.INFO,
                'interface_kappa_AB not set; upstream default 1e7 will be used.',
                field='phase.interface_kappa_AB',
            ))

    dead = [pf for pf in props.PHASE_FLAGS if pf.dead]
    used_dead = [
        pf.moose_key for pf in dead
        if cfg.phase.a.get(pf.flag_a) is not None or cfg.phase.b.get(pf.flag_b) is not None
    ]
    if used_dead:
        out.append(Issue(
            'DEAD_PHASE_FLAG', Severity.INFO,
            f'{", ".join(used_dead)} is set but read by no fm_multi physics.',
            field='phase.density',
            remedy=('density appears in the hardcoded defaults but in no '
                    'bulk_props_keys, so FmMultiAdapter never reads it.'),
        ))

    if cfg.phase.phase_props_json.strip() and cfg.phase.data_dir.strip():
        out.append(Issue(
            'PHASE_SOURCE_SHADOWED', Severity.WARNING,
            'phase_props_json takes precedence; data_dir will not be read.',
            field='phase.data_dir',
            remedy=('Resolution is first-match-wins, not a merge: JSON, then the H5 in '
                    'data_dir, then hardcoded defaults. CLI overrides apply last.'),
        ))


# ------------------------------------------------------------------ D. vf

def _check_vf(cfg, out):
    if cfg.mode is not RunMode.SINGLE:
        return
    if not (0.0 <= cfg.vf_min < cfg.vf_max <= 1.0):
        out.append(Issue(
            'VF_BOUNDS', Severity.ERROR,
            f'Volume-fraction bounds must satisfy 0 <= min < max <= 1 '
            f'(got {cfg.vf_min:g}, {cfg.vf_max:g}).',
            field='vf',
            remedy='vf_ref = (min+max)/2; both zero raises ZeroDivisionError.',
        ))
    rho = cfg.directives.get('rho')
    if rho is not None and rho.active and rho.mode.is_constraint:
        out.append(Issue(
            'VF_AND_RHO', Severity.INFO,
            'Both a rho directive and vf bounds are active.',
            field='vf',
            remedy=('They constrain different quantities: the rho directive bounds the '
                    'AC-filtered field, vf_min/vf_max bound the raw design variable. '
                    'Keep them consistent.'),
        ))


# ------------------------------------------------------------------ mode-specific

def _check_mode_specific(cfg, active, out):
    if cfg.target_tol == 0:
        out.append(Issue(
            'TARGET_TOL_ZERO', Severity.WARNING,
            '--target_tol=0 makes target directives point constraints.',
            field='target_tol',
            remedy='Upstream warns that this may cause Ipopt infeasibility; use > 1e-4.',
        ))
    if cfg.restarts < 1:
        out.append(Issue(
            'RESTARTS_INVALID', Severity.ERROR,
            f'restarts must be >= 1 (got {cfg.restarts}).', field='restarts',
        ))
    if cfg.mode is RunMode.PARETO and cfg.pareto_steps < 0:
        out.append(Issue(
            'PARETO_STEPS_INVALID', Severity.ERROR,
            f'pareto_steps must be >= 0 (got {cfg.pareto_steps}).', field='pareto_steps',
        ))
    if cfg.act.strip() and cfg.act.strip() not in ('relu', 'gelu', 'silu', 'leaky_relu'):
        out.append(Issue(
            'ACT_INVALID', Severity.ERROR,
            f'--act {cfg.act!r} is not a known activation.',
            field='act',
            remedy=('Upstream silently falls back to ReLU on an unknown name, giving '
                    'wrong predictions if the checkpoint used silu. Leave it empty — '
                    'the checkpoint carries its own.'),
        ))


# ------------------------------------------------------------------ F. fenics

def _check_fenics(cfg, out):
    if not cfg.fenics.validate:
        return
    if cfg.fenics.h5_props.strip() and any(v is not None for v in cfg.fenics.props.values()):
        out.append(Issue(
            'FENICS_H5_SHADOWS', Severity.WARNING,
            'fenics_h5_props overrides the individual --fenics_* values.',
            field='fenics.h5_props',
            remedy=('Upstream returns early once the H5 is given, so the per-phase '
                    'fields are never read.'),
        ))
    out.append(Issue(
        'FENICS_ENV_REQUIRED', Severity.WARNING,
        f'Validation requires the conda env {cfg.fenics.conda_env!r} and conda on '
        "the child's PATH.",
        field='fenics.validate',
        remedy=('This is not skipped gracefully: the call has no try/except and runs '
                'after the solve but before artifacts are written, so a missing env '
                'destroys a completed run.'),
    ))


# ------------------------------------------------------------------ early exits

def _check_early_exit(cfg, active, out):
    if not active:
        out.append(Issue(
            'NO_ACTIVE_DIRECTIVES', Severity.ERROR,
            'No active property directives.',
            field='directives',
            remedy=('Single-point prints "No active property directives — nothing to '
                    'optimise." and exits without writing anything.'),
        ))
        return

    if cfg.mode is RunMode.PARETO:
        if cfg.pareto_steps == 0:
            out.append(Issue(
                'PARETO_STEPS_ZERO', Severity.WARNING,
                'pareto_steps=0 runs the payoff table only and writes no npz.',
                field='pareto_steps',
                remedy='The run will succeed but produce no design set to plot.',
            ))
        if not cfg.objective_props():
            out.append(Issue(
                'PARETO_DEGENERATE', Severity.WARNING,
                'No non-rho max/min objective — this degenerates to a single solve.',
                field='directives',
                remedy=('Without an epsilon axis there is no front. Add a max or min '
                        'directive, or use single-point mode.'),
            ))
        if cfg.dry_run:
            out.append(Issue(
                'DRY_RUN', Severity.INFO,
                'Dry run: prints the solve estimate and exits without solving.',
                field='dry_run',
            ))


def blocking(issues: list[Issue]) -> bool:
    return any(i.blocking for i in issues)


def by_field(issues: list[Issue]) -> dict[str, list[Issue]]:
    out: dict[str, list[Issue]] = {}
    for i in issues:
        out.setdefault(i.field, []).append(i)
    return out

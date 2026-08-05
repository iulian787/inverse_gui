"""RunConfig -> argv.

Only emits flags that differ from the upstream default, so the generated command
stays readable and reviewable. The one exception is the checkpoints, which are
always emitted because at least one is mandatory.

Never emits a flag with an empty value: an empty directive string raises IndexError
in the upstream parser rather than a clean error.
"""

from __future__ import annotations

import shlex

from . import properties as props
from .directives import Mode
from .schema import AllenCahn, Beam, Ipopt, RunConfig, RunMode


def _add(argv: list[str], flag: str, value: object, default: object) -> None:
    """Append --flag value, but only when it differs from upstream's default."""
    if value is None or value == default:
        return
    argv += [f'--{flag}', _fmt(value)]


def _fmt(v: object) -> str:
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return repr(int(v)) if v == int(v) and abs(v) < 1e16 else repr(v)
    return str(v)


def build(cfg: RunConfig) -> list[str]:
    """Argument list for the chosen script, excluding the interpreter and script path."""
    argv: list[str] = []

    # --- A. checkpoints -----------------------------------------------------
    for fam in props.FAMILIES:
        path = cfg.checkpoints.path_for(fam.physics).strip()
        if path:
            argv += [f'--{fam.ckpt_flag}', path]
    if cfg.act.strip():
        argv += ['--act', cfg.act.strip()]

    # --- C. phase property sourcing ----------------------------------------
    if cfg.phase.data_dir.strip():
        argv += ['--data_dir', cfg.phase.data_dir.strip()]
    if cfg.phase.phase_props_json.strip():
        argv += ['--phase_props_json', cfg.phase.phase_props_json.strip()]
    for pf in props.PHASE_FLAGS:
        for stem, store in ((pf.flag_a, cfg.phase.a), (pf.flag_b, cfg.phase.b)):
            val = store.get(stem)
            if val is not None:
                argv += [f'--{stem}', _fmt(val)]
    if cfg.phase.interface_kappa_AB is not None:
        argv += ['--interface_kappa_AB', _fmt(cfg.phase.interface_kappa_AB)]

    # --- B. directives, weights, refs --------------------------------------
    for prop in props.ALL_SCALAR_PROPS:
        d = cfg.directives.get(prop)
        if d is None:
            continue
        rendered = d.render()
        if rendered is not None:
            argv += [f'--{prop}', rendered]
        # weights are single-point only, and only affect max/min
        if cfg.mode is RunMode.SINGLE and d.weight != 1.0:
            argv += [f'--weight_{prop}', _fmt(d.weight)]
        if d.ref is not None and d.ref != props.PROP_DEFAULTS[prop]:
            argv += [f'--{prop}_ref', _fmt(d.ref)]

    if cfg.enforce_isotropy:
        argv += ['--enforce_isotropy']

    # --- D. Allen-Cahn ------------------------------------------------------
    ac_def = AllenCahn()
    _add(argv, 'ac_epsi', cfg.ac.epsi, ac_def.epsi)
    _add(argv, 'ac_lambda', cfg.ac.lam, ac_def.lam)
    _add(argv, 'ac_dt', cfg.ac.dt, ac_def.dt)
    _add(argv, 'ac_steps', cfg.ac.steps, ac_def.steps)
    _add(argv, 'ac_sharpen_beta', cfg.ac.sharpen_beta, ac_def.sharpen_beta)

    # --- D. optimization / run control -------------------------------------
    # target_tol and restarts differ per script, so compare against THIS mode's default.
    from .schema import MODE_DEFAULTS
    md = MODE_DEFAULTS[cfg.mode]
    _add(argv, 'target_tol', cfg.target_tol, md['target_tol'])
    _add(argv, 'restarts', cfg.restarts, md['restarts'])
    _add(argv, 'seed', cfg.seed, 42)

    ip_def = Ipopt()
    _add(argv, 'ipopt_max_iter', cfg.ipopt.max_iter, ip_def.max_iter)
    _add(argv, 'ipopt_tol', cfg.ipopt.tol, ip_def.tol)
    _add(argv, 'ipopt_constr_viol_tol', cfg.ipopt.constr_viol_tol, ip_def.constr_viol_tol)
    _add(argv, 'ipopt_print', cfg.ipopt.print_level, ip_def.print_level)
    _add(argv, 'ipopt_lbfgs_history', cfg.ipopt.lbfgs_history, ip_def.lbfgs_history)
    _add(argv, 'ipopt_scaling', cfg.ipopt.scaling, ip_def.scaling)
    _add(argv, 'ipopt_mu_strategy', cfg.ipopt.mu_strategy, ip_def.mu_strategy)
    _add(argv, 'ipopt_acceptable_tol', cfg.ipopt.acceptable_tol, ip_def.acceptable_tol)
    _add(argv, 'ipopt_acceptable_iter', cfg.ipopt.acceptable_iter, ip_def.acceptable_iter)
    _add(argv, 'ipopt_alpha_for_y', cfg.ipopt.alpha_for_y, ip_def.alpha_for_y)

    beam_def = Beam()
    _add(argv, 'beam_width', cfg.beam.width, beam_def.width)
    _add(argv, 'beam_depth', cfg.beam.depth, beam_def.depth)
    _add(argv, 'beam_c', cfg.beam.c, beam_def.c)

    # --- mode-specific ------------------------------------------------------
    if cfg.mode is RunMode.SINGLE:
        _add(argv, 'vf_min', cfg.vf_min, 0.05)
        _add(argv, 'vf_max', cfg.vf_max, 0.95)
    else:
        # Always emitted, even at its default: it is the defining parameter of the
        # sweep, so a generated command that omits it reads as if no sweep were
        # configured. (It is also how a single stand-in script can tell the two
        # modes apart, which the real pair does by filename.)
        argv += ['--pareto_steps', _fmt(cfg.pareto_steps)]
        _add(argv, 'bin_viol_tol', cfg.bin_viol_tol, 0.05)
        _add(argv, 'feasibility_threshold', cfg.feasibility_threshold, 1e-3)
        if cfg.dry_run:
            argv += ['--dry_run']

    argv += ['--output_dir', cfg.output_dir]

    # --- F. FEniCS validation ----------------------------------------------
    if cfg.fenics.validate:
        argv += ['--fenics_validate']
        _add(argv, 'fenics_conda_env', cfg.fenics.conda_env, 'fenics_env')
        if cfg.fenics.output_dir.strip():
            argv += ['--fenics_output_dir', cfg.fenics.output_dir.strip()]
        if cfg.fenics.h5_props.strip():
            # h5_props overrides all the individual --fenics_* flags upstream
            # (early return at run_inverse_design_fm_multi_ac.py:237-240), so the
            # per-phase values are deliberately not emitted alongside it.
            argv += ['--fenics_h5_props', cfg.fenics.h5_props.strip()]
            _add(argv, 'fenics_h5_sample_idx', cfg.fenics.h5_sample_idx, 0)
        else:
            for key, val in sorted(cfg.fenics.props.items()):
                if val is not None:
                    argv += [f'--fenics_{key}', _fmt(val)]

    return argv


def script_name(cfg: RunConfig, scripts_cfg) -> str:
    """The script filename for this mode, from the [scripts] config section."""
    return (scripts_cfg.single_point if cfg.mode is RunMode.SINGLE
            else scripts_cfg.pareto)


def preview(interpreter: str, script: str, argv: list[str], *, width: int = 88) -> str:
    """A copy-pasteable, line-wrapped command string."""
    parts = [shlex.quote(interpreter), '-u', shlex.quote(script)] + [
        shlex.quote(a) for a in argv
    ]
    lines: list[str] = []
    cur = parts[0]
    i = 1
    while i < len(parts):
        # keep --flag and its value on the same line
        chunk = parts[i]
        if chunk.startswith('--') and i + 1 < len(parts) and not parts[i + 1].startswith('--'):
            chunk = f'{chunk} {parts[i + 1]}'
            i += 2
        else:
            i += 1
        if len(cur) + len(chunk) + 1 > width:
            lines.append(cur + ' \\')
            cur = '    ' + chunk
        else:
            cur = f'{cur} {chunk}'
    lines.append(cur)
    return '\n'.join(lines)


def run_script(interpreter: str, script: str, argv: list[str], cwd: str,
               env_extra: dict[str, str]) -> str:
    """A standalone shell script reproducing the run outside the GUI."""
    exports = '\n'.join(
        f'export {k}={shlex.quote(v)}' for k, v in sorted(env_extra.items())
    )
    return f"""#!/usr/bin/env bash
# Generated by inverse_gui. Reproduces this run without the GUI.
set -euo pipefail

cd {shlex.quote(cwd)}

{exports}

{preview(interpreter, script, argv)}
"""

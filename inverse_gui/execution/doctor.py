"""Preflight checks.

Given how many things must line up before a real run works -- two conda envs, an
interpreter path, checkpoint paths that must be configured, conda reachable from the child
-- this turns each failure from a stack trace into a sentence plus the command that
fixes it.

Pure functions returning Check objects; the UI only renders them.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from . import env as env_mod


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str = ''
    remedy: str = ''
    critical: bool = True     # False => a missing optional capability


def _ok(name, detail='') -> Check:
    return Check(name, True, detail)


def run_all(cfg, *, probe_imports: bool = True) -> list[Check]:
    out = [
        check_ai4ns_repo(cfg),
        check_scripts(cfg),
        check_interpreter(cfg),
        check_conda(cfg),
        check_activation(cfg),
        check_runs_dir(cfg),
        check_checkpoints(cfg),
        # The FEniCS probe spawns `conda run` (~2s), so it rides the same flag as
        # the solver import probe rather than firing on every cheap call.
        check_fenics_env(cfg, probe=probe_imports),
    ]
    if probe_imports:
        out.insert(4, check_solver_imports(cfg))
    return out


def check_ai4ns_repo(cfg) -> Check:
    p = cfg.ai4ns_repo
    if p.is_dir():
        return _ok('Optimizer repo', str(p))
    return Check('Optimizer repo', False, f'{p} does not exist',
                 'Set [paths].ai4ns_repo in config.toml, or clone amit_AI4NS next '
                 'to this repo.')


def check_scripts(cfg) -> Check:
    repo = cfg.ai4ns_repo
    missing = [s for s in (cfg.scripts.single_point, cfg.scripts.pareto)
               if not (repo / s).exists()]
    if not missing:
        return _ok('Entry scripts', 'both present')
    return Check('Entry scripts', False, f"missing: {', '.join(missing)}",
                 'Check [scripts] in config.toml. To develop without the real '
                 'optimizer, point both at scripts/fake_optimizer.py.')


def check_interpreter(cfg) -> Check:
    se = env_mod.build(cfg)
    if not se.python:
        return Check('Solver interpreter', False, 'not configured',
                     'Run scripts/probe_solver_env.py --env cenv, or set '
                     '[solver].python in config.toml.')
    if not os.access(se.python, os.X_OK):
        return Check('Solver interpreter', False, f'{se.python} is not executable',
                     'Rebuild the env with ./scripts/setup.sh, then re-run '
                     'scripts/probe_solver_env.py.')
    return _ok('Solver interpreter', se.python)


def check_solver_imports(cfg) -> Check:
    """Importing cyipopt is the real test; a missing IPOPT library fails here."""
    se = env_mod.build(cfg)
    if not se.python or not os.access(se.python, os.X_OK):
        return Check('Solver imports', False, 'interpreter unavailable',
                     'Fix the interpreter check first.', critical=False)
    try:
        r = subprocess.run(
            [se.python, '-c', 'import cyipopt, torch, numpy'],
            capture_output=True, text=True, timeout=90, env=se.env,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return Check('Solver imports', False, str(exc), '', critical=False)
    if r.returncode == 0:
        return _ok('Solver imports', 'cyipopt, torch, numpy')
    tail = (r.stderr or '').strip().splitlines()[-1:] or ['']
    return Check('Solver imports', False, tail[0],
                 'conda env create -f env/cenv.yml', critical=False)


def check_conda(cfg) -> Check:
    conda = env_mod.find_conda(cfg.solver.conda_exe.strip())
    if conda:
        return _ok('conda', conda)
    return Check('conda', False, 'not found',
                 'Needed only for FEniCS validation and interpreter discovery. '
                 'Set [solver].conda_exe if it lives somewhere unusual.',
                 critical=False)


def check_activation(cfg) -> Check:
    p = cfg.activation_json
    data = env_mod.load_activation(p)
    if data.get('activation_env'):
        n = len(data['activation_env'])
        return _ok('Activation delta', f'{n} var(s) from {p.name}')
    return Check('Activation delta', False, f'{p} missing or empty',
                 'python scripts/probe_solver_env.py --env cenv  '
                 '(captures MKL_INTERFACE_LAYER, which numpy/scipy/torch need)',
                 critical=False)


def check_runs_dir(cfg) -> Check:
    p = cfg.runs_dir
    try:
        p.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return Check('Runs directory', False, str(exc),
                     'Set [paths].runs_dir to a writable location.')
    if not os.access(p, os.W_OK):
        return Check('Runs directory', False, f'{p} is not writable',
                     'Set [paths].runs_dir to a writable location.')
    free_gb = shutil.disk_usage(p).free / 1e9
    if free_gb < 1:
        return Check('Runs directory', False, f'{free_gb:.1f} GB free at {p}',
                     'Free space, or point [paths].runs_dir at a larger disk.')
    return _ok('Runs directory', f'{p} ({free_gb:.0f} GB free)')


def check_checkpoints(cfg) -> Check:
    paths = {k: v for k, v in (
        ('elastic', cfg.checkpoints.elastic),
        ('thermal_conductivity', cfg.checkpoints.thermal_conductivity),
        ('thermal_expansion', cfg.checkpoints.thermal_expansion),
    ) if v.strip()}
    if not paths:
        return Check('Checkpoints', False, 'none configured',
                     'Put the EffPropNet .pt files (shipped with neither repo) in '
                     '<ai4ns_repo>/models/fm_multi_store and set their absolute '
                     'paths in [checkpoints] in config.toml. Without them, point '
                     '[scripts] at scripts/fake_optimizer.py.', critical=False)
    missing = [k for k, v in paths.items() if not Path(v).expanduser().exists()]
    if missing:
        return Check('Checkpoints', False, f"not found: {', '.join(missing)}",
                     'Fix the paths in section A, or in [checkpoints] in '
                     'config.toml to make it stick.', critical=False)
    return _ok('Checkpoints', f"{len(paths)} configured")


# Existence is not usability, and the difference is expensive here. The upstream
# spec is incomplete: fenics_validation/mesh.py:7 imports dolfinx_mpc (the periodic
# BCs every solver uses) and output.py:7 imports pandas, while
# environment_fenics.yml lists neither. A name-only check goes green on an env that
# raises at import -- and upstream calls it with no try/except AFTER the solve and
# BEFORE artifacts are written, so the false green costs a completed run. Probe the
# import the way upstream will invoke it: conda run -n <env>, cwd = the ai4ns root.
_FENICS_PROBE = 'import fenics_validation.validate'


def check_fenics_env(cfg, *, probe: bool = True) -> Check:
    """Gates the validation toggle. Upstream does NOT skip gracefully without it."""
    if not fenics_env_exists(cfg):
        return Check('FEniCS env', False, f'{cfg.fenics.conda_env} not found',
                     'conda env create -f '
                     '<ai4ns>/fenics_validation/environment_fenics.yml  '
                     '(deferred by default; validation stays disabled)',
                     critical=False)
    if not probe:
        return _ok('FEniCS env', f'{cfg.fenics.conda_env} (present, not probed)')
    ok, detail = fenics_import_probe(cfg)
    if ok:
        return _ok('FEniCS env', f'{cfg.fenics.conda_env}: fenics_validation imports')
    return Check('FEniCS env', False,
                 f'{cfg.fenics.conda_env} present but unusable — {detail}',
                 _fenics_remedy(cfg, detail), critical=False)


def fenics_import_probe(cfg, *, timeout: float = 120) -> tuple[bool, str]:
    """(ok, detail) for importing the upstream validation package inside the env."""
    conda = env_mod.find_conda(cfg.solver.conda_exe.strip())
    if not conda:
        return False, 'conda not found'
    repo = cfg.ai4ns_repo
    if not (repo / 'fenics_validation').is_dir():
        return False, f'no fenics_validation package under {repo}'
    cmd = [conda, 'run', '-n', cfg.fenics.conda_env, '--no-capture-output',
           'python', '-c', _FENICS_PROBE]
    try:
        # cwd matters: nothing is installed, so the package is importable only
        # from the repo root -- the same cwd the optimizer child runs in.
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout, cwd=repo)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    if r.returncode == 0:
        return True, ''
    return False, _last_meaningful(r.stderr)


def _last_meaningful(stderr: str) -> str:
    """The child's own error, not conda's epilogue.

    `conda run` appends its own "ERROR conda.cli.main_run:execute(125): ... failed."
    line after the child's traceback, so the literal last line names conda rather
    than the missing module.
    """
    lines = [ln.strip() for ln in (stderr or '').splitlines() if ln.strip()]
    for ln in reversed(lines):
        if not ln.startswith('ERROR conda'):
            return ln
    return lines[-1] if lines else 'unknown error'


def _fenics_remedy(cfg, detail: str) -> str:
    missing = re.search(r"No module named '([\w.]+)'", detail)
    if missing:
        return (f'conda install -n {cfg.fenics.conda_env} -c conda-forge '
                f'{missing.group(1)}  (environment_fenics.yml omits dolfinx_mpc '
                'and pandas, which fenics_validation imports at package level)')
    return ('Repair the env before enabling validation — upstream has no '
            'try/except around the call, and it runs after the solve but before '
            'artifacts are written.')


def fenics_env_exists(cfg) -> bool:
    conda = env_mod.find_conda(cfg.solver.conda_exe.strip())
    if not conda:
        return False
    try:
        r = subprocess.run([conda, 'env', 'list'], capture_output=True,
                           text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return False
    name = cfg.fenics.conda_env
    for line in r.stdout.splitlines():
        if line.strip().startswith('#'):
            continue
        parts = line.split()
        if not parts:
            continue
        # Named envs list as "<name> [*] <prefix>"; an env conda knows only by
        # location lists as the bare prefix. `conda run -n` resolves a name by
        # looking for that basename under envs_dirs, so matching the prefix
        # basename too is the more faithful test. A false positive from either
        # is now caught by the import probe instead of at the end of a solve.
        if parts[0] == name or Path(parts[-1]).name == name:
            return True
    return False


def blocking(checks: list[Check]) -> list[Check]:
    return [c for c in checks if c.critical and not c.ok]

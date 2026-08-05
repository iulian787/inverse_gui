"""Preflight checks.

Given how many things must line up before a real run works -- two conda envs, an
interpreter path, checkpoints that do not exist yet, conda reachable from the child
-- this turns each failure from a stack trace into a sentence plus the command that
fixes it.

Pure functions returning Check objects; the UI only renders them.
"""

from __future__ import annotations

import os
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
        check_fenics_env(cfg),
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
                     'The EffPropNet .pt files are not in the repo and must be '
                     'obtained separately. Until then, point [scripts] at '
                     'scripts/fake_optimizer.py.', critical=False)
    missing = [k for k, v in paths.items() if not Path(v).expanduser().exists()]
    if missing:
        return Check('Checkpoints', False, f"not found: {', '.join(missing)}",
                     'Fix the paths in section A.', critical=False)
    return _ok('Checkpoints', f"{len(paths)} configured")


def check_fenics_env(cfg) -> Check:
    """Gates the validation toggle. Upstream does NOT skip gracefully without it."""
    if not fenics_env_exists(cfg):
        return Check('FEniCS env', False, f'{cfg.fenics.conda_env} not found',
                     'conda env create -f '
                     '<ai4ns>/fenics_validation/environment_fenics.yml  '
                     '(deferred by default; validation stays disabled)',
                     critical=False)
    return _ok('FEniCS env', cfg.fenics.conda_env)


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
        if parts and parts[0] == name:
            return True
    return False


def blocking(checks: list[Check]) -> list[Check]:
    return [c for c in checks if c.critical and not c.ok]

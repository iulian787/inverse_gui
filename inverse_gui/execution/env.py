"""Child-process environment assembly.

Three things this must get right, all measured — see docs/environment.md:

1. Merge the conda activation delta captured by scripts/probe_solver_env.py. We
   launch <prefix>/bin/python directly rather than via `conda run` (which orphans
   the child on cancel), so activate.d hooks never fire. cenv ships three, and
   MKL_INTERFACE_LAYER=LP64,GNU is load-bearing for numpy/scipy/torch's BLAS.
2. Put conda's bindir on the child's PATH. The optimizer itself shells out to
   `conda run -n fenics_env` for validation, with no try/except, after the solve
   completes but before artifacts are written.
3. STRIP the GUI venv from the child. os.environ here carries VIRTUAL_ENV and a
   PATH starting with .venv/bin; leaking those into a conda interpreter produces
   import errors that look like a broken solver environment.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

# Variables that belong to the GUI's own venv and must not reach the child.
_VENV_VARS = ('VIRTUAL_ENV', 'PYTHONHOME', 'PYTHONPATH', 'PYTHONSTARTUP',
              'UV_PROJECT_ENVIRONMENT', '__PYVENV_LAUNCHER__')


@dataclass(frozen=True)
class SolverEnv:
    python: str
    env: dict[str, str]
    conda_exe: str | None


def load_activation(path: Path) -> dict:
    """Read env/<name>.activation.json, or {} if it has not been generated."""
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}


def find_conda(explicit: str = '') -> str | None:
    for cand in (explicit, os.environ.get('CONDA_EXE', ''), shutil.which('conda') or ''):
        if cand and os.path.exists(cand):
            return cand
    for prefix in ('~/miniconda3', '~/anaconda3', '~/miniforge3', '/opt/conda'):
        cand = os.path.expanduser(f'{prefix}/bin/conda')
        if os.path.exists(cand):
            return cand
    return None


def _strip_venv(base: dict[str, str], venv_dir: Path | None) -> dict[str, str]:
    out = {k: v for k, v in base.items() if k not in _VENV_VARS}
    if venv_dir is not None:
        venv_bin = str(venv_dir / 'bin')
        parts = [p for p in out.get('PATH', '').split(os.pathsep)
                 if p and os.path.normpath(p) != os.path.normpath(venv_bin)]
        out['PATH'] = os.pathsep.join(parts)
    return out


def build(cfg, *, base_env: dict[str, str] | None = None,
          venv_dir: Path | None = None) -> SolverEnv:
    """Assemble the interpreter path and environment for the solver subprocess."""
    base = dict(os.environ if base_env is None else base_env)
    venv = venv_dir if venv_dir is not None else _detect_venv(base)
    env = _strip_venv(base, venv)

    activation = load_activation(cfg.activation_json)
    python = cfg.solver.python.strip() or activation.get('python', '')
    env.update(activation.get('activation_env', {}))

    conda = find_conda(cfg.solver.conda_exe.strip())
    if conda:
        bindir = str(Path(conda).parent)
        parts = env.get('PATH', '').split(os.pathsep)
        if bindir not in parts:
            env['PATH'] = os.pathsep.join([bindir, *[p for p in parts if p]])

    env.update({
        'PYTHONUNBUFFERED': '1',
        'PYTHONFAULTHANDLER': '1',
        # run_inverse_design_fm_multi_ac.py:47 is a bare `import matplotlib.pyplot`
        # with no use('Agg'); without this it can raise at import in a headless child.
        'MPLBACKEND': 'Agg',
        'OMP_NUM_THREADS': str(cfg.solver.threads),
        'MKL_NUM_THREADS': str(cfg.solver.threads),
        'OPENBLAS_NUM_THREADS': str(cfg.solver.threads),
    })
    if cfg.solver.device == 'cpu':
        # There is no --device flag upstream; the device probe is a try/except that
        # falls back to CPU, so this is the only way to force it.
        env['CUDA_VISIBLE_DEVICES'] = ''

    return SolverEnv(python=python, env=env, conda_exe=conda)


def _detect_venv(base: dict[str, str]) -> Path | None:
    v = base.get('VIRTUAL_ENV')
    return Path(v) if v else None


def notable(se: SolverEnv) -> dict[str, str]:
    """The subset worth showing in the UI and writing into the generated run.sh."""
    keys = ('MPLBACKEND', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
            'CUDA_VISIBLE_DEVICES', 'MKL_INTERFACE_LAYER', 'PYTHONUNBUFFERED')
    return {k: se.env[k] for k in keys if k in se.env}

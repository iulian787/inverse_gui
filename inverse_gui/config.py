"""Layered configuration.

Precedence, lowest first: dataclass defaults < config.toml < INVERSE_GUI_* env vars.
Env-vars-last matters for HPC and CI, where editing a file is not an option.

Env var naming: INVERSE_GUI_<SECTION>_<KEY>, uppercase, e.g.
    INVERSE_GUI_SOLVER_PYTHON=/opt/conda/envs/cenv/bin/python
    INVERSE_GUI_PATHS_AI4NS_REPO=/scratch/amit_AI4NS
"""

from __future__ import annotations

import dataclasses
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

ENV_PREFIX = 'INVERSE_GUI_'
REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class PathsCfg:
    ai4ns_repo: str = '../amit_AI4NS'
    runs_dir: str = './runs'
    ckpt_dir: str = ''


@dataclass
class ScriptsCfg:
    single_point: str = 'run_inverse_design_fm_multi_ac.py'
    pareto: str = 'run_pareto_epsilon_fm_multi_ac.py'
    pareto_plot: str = 'plots/plot_pareto_results.py'


@dataclass
class SolverCfg:
    python: str = ''
    conda_exe: str = ''
    conda_env: str = 'cenv'
    activation_json: str = 'env/cenv.activation.json'
    stream_mode: str = 'pty'      # pty | pipe_stdbuf | pipe
    device: str = 'cpu'           # cpu | auto
    threads: int = 4


@dataclass
class FenicsCfg:
    enabled: bool = False
    conda_env: str = 'fenics_env'


@dataclass
class CheckpointsCfg:
    elastic: str = ''
    thermal_conductivity: str = ''
    thermal_expansion: str = ''


@dataclass
class RunnerCfg:
    # Kill live runs when the Streamlit process exits. start_new_session=True means
    # Ctrl-C in the terminal does NOT reach the child, so without this every dev
    # server restart leaks a torch+IPOPT process.
    kill_on_exit: bool = True
    term_grace_seconds: float = 5.0
    tail_lines: int = 200
    poll_interval: str = '1s'


@dataclass
class UiCfg:
    # Above this many estimated Pareto solves, Launch is gated behind an explicit
    # acknowledgement checkbox.
    cost_warn_solves: int = 500
    seconds_per_solve: float = 8.0     # seed for the ETA; refined from run history


@dataclass
class Config:
    paths: PathsCfg = field(default_factory=PathsCfg)
    scripts: ScriptsCfg = field(default_factory=ScriptsCfg)
    solver: SolverCfg = field(default_factory=SolverCfg)
    fenics: FenicsCfg = field(default_factory=FenicsCfg)
    checkpoints: CheckpointsCfg = field(default_factory=CheckpointsCfg)
    runner: RunnerCfg = field(default_factory=RunnerCfg)
    ui: UiCfg = field(default_factory=UiCfg)

    source: str = '<defaults>'

    # ---------------------------------------------------------------- resolution

    def resolve(self, value: str) -> Path:
        """Resolve a possibly-relative path against the repo root, not the cwd.

        Streamlit's cwd is wherever the user launched from, so relative paths in
        config.toml must anchor to the repo or they break on the second machine.
        """
        p = Path(value).expanduser()
        return p if p.is_absolute() else (REPO_ROOT / p).resolve()

    @property
    def ai4ns_repo(self) -> Path:
        return self.resolve(self.paths.ai4ns_repo)

    @property
    def runs_dir(self) -> Path:
        return self.resolve(self.paths.runs_dir)

    @property
    def activation_json(self) -> Path:
        return self.resolve(self.solver.activation_json)


def _coerce(current, raw: str):
    if isinstance(current, bool):
        return raw.strip().lower() in ('1', 'true', 'yes', 'on')
    if isinstance(current, int) and not isinstance(current, bool):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    return raw


def _apply(section, values: dict) -> None:
    known = {f.name for f in dataclasses.fields(section)}
    for key, val in values.items():
        if key in known:
            setattr(section, key, val)


def env_overrides(cfg: Config, environ: dict | None = None) -> list[str]:
    """Apply INVERSE_GUI_* overrides in place. Returns the keys that were applied."""
    environ = os.environ if environ is None else environ
    applied: list[str] = []
    for sec_name in ('paths', 'scripts', 'solver', 'fenics', 'checkpoints',
                     'runner', 'ui'):
        section = getattr(cfg, sec_name)
        for f in dataclasses.fields(section):
            env_key = f'{ENV_PREFIX}{sec_name.upper()}_{f.name.upper()}'
            if env_key in environ:
                cur = getattr(section, f.name)
                setattr(section, f.name, _coerce(cur, environ[env_key]))
                applied.append(env_key)
    return applied


def load(path: str | Path | None = None, *, environ: dict | None = None) -> Config:
    """Load config.toml (if present) and apply env overrides."""
    cfg = Config()
    candidate = Path(path) if path else (REPO_ROOT / 'config.toml')
    if candidate.exists():
        with open(candidate, 'rb') as fh:
            data = tomllib.load(fh)
        for sec_name in ('paths', 'scripts', 'solver', 'fenics', 'checkpoints',
                         'runner', 'ui'):
            if sec_name in data:
                _apply(getattr(cfg, sec_name), data[sec_name])
        cfg.source = str(candidate)
    env_overrides(cfg, environ)
    return cfg

"""Session state and the cached singletons.

Two rules hold the streaming design together:

1. session_state holds only widget values and the current run_id. Everything about
   a run lives in the registry (fast, volatile) and on disk (slow, durable), so
   losing a session loses nothing but scroll position.
2. The registry lives in st.cache_resource, NOT in a module global. Streamlit's
   watcher evicts every watched local module from sys.modules on any source save;
   a module global would be emptied mid-run, leaving a live optimizer with no Stop
   button. cache_resource survives that.
"""

from __future__ import annotations

import streamlit as st

from ..config import load as load_config
from ..domain.schema import RunConfig, RunMode
from ..execution.registry import Registry
from ..execution.runner import SubprocessRunner

RUN_KEY = 'active_run_id'
CFG_KEY = 'run_config'


@st.cache_resource(show_spinner=False)
def get_config():
    return load_config()


@st.cache_resource(show_spinner=False)
def get_registry() -> Registry:
    """Process-global, survives module eviction and session churn."""
    return Registry(kill_on_exit=get_config().runner.kill_on_exit,
                    grace=get_config().runner.term_grace_seconds)


@st.cache_resource(show_spinner=False)
def get_runner() -> SubprocessRunner:
    reg = get_registry()
    reg.install_hooks()
    return SubprocessRunner(get_config(), reg)


def get_run_config() -> RunConfig:
    """The form's working copy. Seeded from config.toml on first use."""
    if CFG_KEY not in st.session_state:
        cfg = get_config()
        rc = RunConfig.for_mode(RunMode.SINGLE)
        rc.checkpoints.elastic = cfg.checkpoints.elastic
        rc.checkpoints.thermal_conductivity = cfg.checkpoints.thermal_conductivity
        rc.checkpoints.thermal_expansion = cfg.checkpoints.thermal_expansion
        rc.fenics.conda_env = cfg.fenics.conda_env
        st.session_state[CFG_KEY] = rc
    return st.session_state[CFG_KEY]


def set_run_config(rc: RunConfig) -> None:
    st.session_state[CFG_KEY] = rc


def active_run_id() -> str | None:
    """Prefer the URL, so a refresh past the session TTL still reattaches."""
    from_url = st.query_params.get('run')
    if from_url:
        st.session_state[RUN_KEY] = from_url
        return from_url
    return st.session_state.get(RUN_KEY)


def set_active_run(run_id: str | None) -> None:
    if run_id:
        st.session_state[RUN_KEY] = run_id
        st.query_params['run'] = run_id
    else:
        st.session_state.pop(RUN_KEY, None)
        st.query_params.pop('run', None)

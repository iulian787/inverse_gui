"""Preflight panel.

Cached because check_solver_imports spawns the solver interpreter (~2s). The
Re-check button clears the cache, which is what you want after fixing something.
"""

from __future__ import annotations

import streamlit as st

from ..execution import doctor
from .components import check_row


@st.cache_data(show_spinner='Checking environment…', ttl=300)
def _cached_checks(_cfg, nonce: int):
    return doctor.run_all(_cfg)


def checks(cfg):
    return _cached_checks(cfg, st.session_state.get('doctor.nonce', 0))


def render(cfg) -> None:
    result = checks(cfg)
    bad = doctor.blocking(result)
    warn = [c for c in result if not c.ok and not c.critical]

    if bad:
        st.error(f'{len(bad)} blocking problem(s)', icon='🔴')
    elif warn:
        st.warning(f'Ready, with {len(warn)} limitation(s)', icon='🟡')
    else:
        st.success('All checks passed', icon='✅')

    for c in result:
        check_row(c)

    if st.button('Re-check'):
        st.session_state['doctor.nonce'] = st.session_state.get('doctor.nonce', 0) + 1
        st.rerun()

    st.caption(f'Config: `{cfg.source}`')


def fenics_available(cfg) -> bool:
    for c in checks(cfg):
        if c.name == 'FEniCS env':
            return c.ok
    return False

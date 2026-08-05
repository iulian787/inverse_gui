"""Run history.

A stub in the sense that compare/overlay is not built, but a real one: the
persistence format already exists because the runner needs it for reattach, so
listing and reloading past runs is genuinely useful now and is not a rewrite later.
"""

from __future__ import annotations

import time

import streamlit as st

from ..execution import runstore
from . import state as state_mod

_ICON = {
    runstore.STATE_DONE: '✅', runstore.STATE_FAILED: '🔴',
    runstore.STATE_CANCELLED: '⏹', runstore.STATE_RUNNING: '🟢',
    runstore.STATE_CANCELLING: '🟠', runstore.STATE_ORPHANED: '👻',
}


def render(cfg) -> None:
    runs = runstore.scan(cfg.runs_dir, limit=50)
    if not runs:
        st.caption('No runs yet.')
        return

    active = state_mod.active_run_id()
    for st_row in runs:
        cols = st.columns([0.5, 3, 2, 1.5, 1])
        cols[0].markdown(_ICON.get(st_row.state, '•'))
        label = f'`{st_row.run_id}`'
        if st_row.run_id == active:
            label += ' ← open'
        cols[1].markdown(label)
        cols[2].caption(time.strftime('%Y-%m-%d %H:%M',
                                      time.localtime(st_row.started_at)))
        cols[3].caption(f'{st_row.duration:.0f}s')
        if cols[4].button('Open', key=f'hist.{st_row.run_id}'):
            state_mod.set_active_run(st_row.run_id)
            st.rerun()

    st.caption('Comparing designs across runs (masks side by side, convergence '
               'overlay) is not built yet.')

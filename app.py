"""Inverse-design GUI.

    .venv/bin/streamlit run app.py

Three panes, as in the design deck: Input · Run/progress · Design space, with a
run-history strip. The mode toggle at the top re-renders the targets, tuning and
results sections; everything else is shared.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from inverse_gui.domain import validate as validate_mod          # noqa: E402
from inverse_gui.domain.schema import MODE_DEFAULTS, RunMode     # noqa: E402
from inverse_gui.ui import design_space, doctor_pane, history, run_pane  # noqa: E402
from inverse_gui.ui import state as state_mod                    # noqa: E402
from inverse_gui.ui.form import sections                         # noqa: E402

st.set_page_config(page_title='Inverse-design GUI', page_icon='◧', layout='wide')


def main() -> None:
    cfg = state_mod.get_config()
    rc = state_mod.get_run_config()

    _header(rc, cfg)
    run_pane.render_active_strip(cfg)

    left, middle, right = st.columns([1.15, 1.0, 1.25], gap='medium')

    # Validated twice, deliberately. It is a pure function costing microseconds.
    # The first pass feeds inline per-field hints, which must be present on the very
    # first render before any widget has been touched. The second pass runs after the
    # form has written this interaction's values into rc, so the Launch gate reflects
    # what the user is looking at instead of lagging one interaction behind.
    with left:
        st.subheader('Input')
        sections.render(rc, cfg, validate_mod.by_field(validate_mod.validate(rc)),
                        fenics_available=doctor_pane.fenics_available(cfg))

    issues = validate_mod.validate(rc)

    with middle:
        st.subheader('Run')
        run_pane.render_launch(rc, cfg, issues)
        run_id = state_mod.active_run_id()
        if run_id:
            st.divider()
            run_pane.render_run(run_id, cfg)

    with right:
        st.subheader('Design space')
        run_id = state_mod.active_run_id()
        if run_id:
            design_space.render_for_run(run_id, cfg)
        else:
            st.caption('Launch a run, or open one from the history below.')

    st.divider()
    with st.expander('Run history', expanded=False):
        history.render(cfg)
    with st.expander('Environment check', expanded=False):
        doctor_pane.render(cfg)


def _header(rc, cfg) -> None:
    cols = st.columns([3, 3, 2])
    cols[0].markdown('### AI4NS · Inverse-design GUI')

    labels = {RunMode.SINGLE: RunMode.SINGLE.label, RunMode.PARETO: RunMode.PARETO.label}
    chosen = cols[1].segmented_control(
        'Mode', list(labels.values()), default=labels[rc.mode],
        key='mode.toggle', label_visibility='collapsed',
    ) or labels[rc.mode]
    new_mode = next(m for m, l in labels.items() if l == chosen)
    if new_mode is not rc.mode:
        # target_tol, restarts and output_dir genuinely differ between the two
        # scripts; carrying the other script's values over would be wrong.
        rc.mode = new_mode
        for key, val in MODE_DEFAULTS[new_mode].items():
            setattr(rc, key, val)
        # Assign the widget keys rather than deleting them. Deleting works under
        # AppTest but not in a browser, where the frontend re-sends the old value
        # on the next rerun and it wins over the `value=` argument.
        for widget_key in ('restarts', 'target_tol'):
            if widget_key in MODE_DEFAULTS[new_mode]:
                st.session_state[widget_key] = MODE_DEFAULTS[new_mode][widget_key]
        st.rerun()

    script = (cfg.scripts.single_point if rc.mode is RunMode.SINGLE
              else cfg.scripts.pareto)
    if 'fake_optimizer' in script:
        cols[2].warning('fake optimizer', icon='🧪')
    else:
        cols[2].caption(f'`{script}`')


main()

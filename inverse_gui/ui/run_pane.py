"""Launch, live progress, and Stop.

The live pane is an @st.fragment(run_every=...) so a tick redraws only the log,
not the whole ~40-widget form. Two call sites rather than one, because run_every is
bound at decoration time: once a run reaches a terminal state we render through the
static branch and the client-side timer stops re-arming.
"""

from __future__ import annotations

import streamlit as st

from ..domain import argv as argv_mod
from ..domain import cost as cost_mod
from ..domain import validate as validate_mod
from ..domain.schema import RunConfig, RunMode
from ..execution import runstore
from . import state as state_mod
from .components import issue_list

_STATE_BADGE = {
    runstore.STATE_RUNNING: ('🟢', 'running'),
    runstore.STATE_CANCELLING: ('🟠', 'cancelling…'),
    runstore.STATE_DONE: ('✅', 'done'),
    runstore.STATE_FAILED: ('🔴', 'failed'),
    runstore.STATE_CANCELLED: ('⏹', 'cancelled'),
    runstore.STATE_ORPHANED: ('👻', 'orphaned'),
}


# ---------------------------------------------------------------- launch controls

def render_launch(rc: RunConfig, cfg, issues: list) -> None:
    runner = state_mod.get_runner()
    blocking = validate_mod.blocking(issues)

    issue_list(issues, show_info=True)
    _cost_panel(rc, cfg)

    with st.expander('Command preview', expanded=False):
        try:
            full_argv, interpreter, _ = runner.plan(rc)
            st.code(argv_mod.preview(interpreter, full_argv[2], full_argv[3:]),
                    language='bash')
            st.caption('`--output_dir` is rewritten to this run\'s own directory on '
                       'launch, so runs never share artifacts.')
        except Exception as exc:                       # config not yet usable
            st.warning(f'Cannot build the command yet: {exc}')

    acknowledged = True
    est = cost_mod.estimate(rc)
    if est and est.total_solves > cfg.ui.cost_warn_solves:
        acknowledged = st.checkbox(
            f'I understand this launches ~{est.total_solves:,} solves',
            key='cost.ack')

    cols = st.columns([1, 1, 3])
    launch = cols[0].button('▶ Launch', type='primary', width='stretch',
                            disabled=blocking or not acknowledged)
    if rc.mode is RunMode.PARETO:
        if cols[1].button('Dry run', width='stretch', disabled=blocking,
                          help='Adds --dry_run: prints the solve estimate and exits '
                               'without solving. The cheapest way to validate.'):
            _launch(rc, dry_run=True)
    if blocking:
        cols[2].caption('Fix the errors above to enable Launch.')

    if launch:
        _launch(rc, dry_run=False)


def _launch(rc: RunConfig, *, dry_run: bool) -> None:
    import copy
    launch_cfg = copy.deepcopy(rc)
    launch_cfg.dry_run = dry_run and rc.mode is RunMode.PARETO
    run_id = state_mod.get_runner().submit(launch_cfg)
    state_mod.set_active_run(run_id)
    st.rerun()


def _cost_panel(rc: RunConfig, cfg) -> None:
    est = cost_mod.estimate(rc)
    if est is None:
        st.caption(f'Single-point: {rc.restarts} restart(s).')
        return
    eta = cost_mod.format_duration(
        cost_mod.eta_seconds(est, cfg.ui.seconds_per_solve))
    cols = st.columns(4)
    cols[0].metric('ε points / axis', est.pts_per_prop)
    cols[1].metric('grid points', f'{est.total_grid:,}')
    cols[2].metric('total solves', f'{est.total_solves:,}')
    cols[3].metric('rough ETA', eta)
    st.caption(f'Upstream will print: `{est.upstream_line}`')
    if est.total_solves > cfg.ui.cost_warn_solves:
        st.warning(
            f'The ε grid is a Cartesian product: {est.pts_per_prop} points on each of '
            f'{est.n_objectives} axes. Adding one more max/min objective would make '
            f'it ~{est.total_solves * est.pts_per_prop:,} solves.', icon='🟡')


# ---------------------------------------------------------------- run display

def render_run(run_id: str, cfg) -> None:
    runner = state_mod.get_runner()
    snap = runner.snapshot(run_id)
    if snap is None:
        st.info('This run no longer exists.')
        state_mod.set_active_run(None)
        return

    if snap.state in (runstore.STATE_RUNNING, runstore.STATE_CANCELLING):
        _live_pane(run_id, cfg)
    else:
        _static_pane(run_id, cfg)


@st.fragment(run_every='1s')
def _live_pane(run_id: str, cfg) -> None:
    runner = state_mod.get_runner()
    snap = runner.snapshot(run_id)
    if snap is None:
        return
    _header(snap)
    if snap.is_terminal:
        # Leave the timer branch so run_every stops re-arming.
        st.rerun()
        return
    _controls(run_id, snap)
    _log(snap, cfg)


def _static_pane(run_id: str, cfg) -> None:
    snap = state_mod.get_runner().snapshot(run_id)
    if snap is None:
        return
    _header(snap)
    _controls(run_id, snap)
    _log(snap, cfg)


def _header(snap) -> None:
    icon, label = _STATE_BADGE.get(snap.state, ('•', snap.state))
    cols = st.columns([2, 3, 2])
    cols[0].markdown(f'### {icon} {label}')
    cols[1].caption(snap.progress.headline)
    if snap.returncode is not None:
        cols[2].caption(f'exit code {snap.returncode}')
    if snap.degraded:
        st.caption('Reattached from disk — progress replayed from the log. '
                   'Stop still works via the process group.')

    frac = snap.progress.fraction
    if frac is not None and not snap.is_terminal:
        st.progress(frac)
    _sweep_coverage(snap.progress)
    if snap.state == runstore.STATE_ORPHANED:
        st.warning('This run was marked running but its process is gone — most '
                   'likely the app was killed with SIGKILL.', icon='👻')


def _sweep_coverage(progress) -> None:
    """Per-grid-point feasibility, as the sweep produces it.

    Not the designs themselves: upstream writes its npz once at the end and never
    prints an achieved property vector per grid point, so there is nothing to plot
    in the scatter until the run finishes. Coverage is what the log actually
    carries -- see CLAUDE.md.
    """
    summary = progress.sweep_summary
    if not summary:
        return
    st.caption(summary)
    # One cell per recent grid point: filled where at least one restart was
    # feasible. Cheap enough to redraw every second, and it makes a barren stretch
    # of the grid obvious while there is still time to cancel.
    cells = ''.join('▰' if feasible else '▱' for _, feasible, _ in progress.recent)
    if cells:
        st.caption(cells)


def _controls(run_id: str, snap) -> None:
    cols = st.columns([1, 1, 4])
    if not snap.is_terminal:
        if cols[0].button('⏹ Stop', key=f'stop.{run_id}', width='stretch'):
            state_mod.get_runner().cancel(run_id)
            st.rerun()
    else:
        if cols[0].button('Clear', key=f'clear.{run_id}', width='stretch'):
            state_mod.set_active_run(None)
            st.rerun()
    cols[2].caption(f'`{run_id}`')


def _log(snap, cfg) -> None:
    text = '\n'.join(snap.tail) or '(no output yet)'
    st.code(text, language='text', height=340)
    st.caption(f'last {len(snap.tail)} lines of {cfg.runner.tail_lines} kept')
    if snap.progress.warnings:
        with st.expander(f'{len(snap.progress.warnings)} warning(s) from the solver'):
            for w in snap.progress.warnings:
                st.caption(w)
    if snap.progress.fenics:
        # Validation runs after the solve, so these lines are the only sign of life
        # during a phase that can take longer than the optimisation itself -- and if
        # a physics solver fails, upstream only says so here.
        with st.expander(f'FEniCS validation ({len(snap.progress.fenics)} line(s))',
                         expanded=not snap.is_terminal):
            for line in snap.progress.fenics:
                st.caption(line)


# ---------------------------------------------------------------- active runs strip

def render_active_strip(cfg) -> None:
    """Always-visible list of live runs.

    Without this a run started before a hot-reload or in another browser session is
    invisible, and the only way to stop it is `kill`.
    """
    live = runstore.scan_live(cfg.runs_dir)
    if not live:
        return
    active = state_mod.active_run_id()
    with st.container(border=True):
        st.caption(f'⏵ {len(live)} run(s) in progress')
        for st_row in live:
            cols = st.columns([3, 3, 1, 1])
            cols[0].markdown(f'`{st_row.run_id}`')
            cols[1].caption(f'{st_row.mode} · {st_row.duration:.0f}s · pgid '
                            f'{st_row.pgid}')
            if st_row.run_id != active:
                if cols[2].button('Attach', key=f'att.{st_row.run_id}'):
                    state_mod.set_active_run(st_row.run_id)
                    st.rerun()
            if cols[3].button('Stop', key=f'kill.{st_row.run_id}'):
                state_mod.get_runner().cancel(st_row.run_id)
                st.rerun()

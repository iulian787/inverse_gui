"""The design-space view: every candidate as a point, click to inspect it.

Click rather than hover: Plotly hover events do not round-trip to Python in
Streamlit at all, whereas `on_select="rerun"` gives real selection events. The
tooltip still carries the numbers, so hovering is informative and clicking opens
the microstructure.

The one non-obvious correctness requirement: designs are split across several
traces (front / dominated / infeasible) so the legend filters them, and with
multiple traces Plotly's point_index is TRACE-relative. Every point therefore
carries its DesignSet index in customdata, and selection reads that, never the
point index.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from ..artifacts.model import Design, DesignSet
from .components import mask_image

_FRONT = '#2f6fed'
_DOMINATED = '#9aa7b8'
_INFEASIBLE = '#c23b3b'


def render(ds: DesignSet, *, key_prefix: str, target_tol: float = 0.02,
           run_id: str = '') -> None:
    choices = ds.axis_choices()
    if not choices:
        st.info('This artifact has no plottable properties.')
        return

    dx, dy = ds.default_axes()
    cols = st.columns([1, 1, 2])
    x = cols[0].selectbox('x axis', choices, index=choices.index(dx),
                          key=f'{key_prefix}.x')
    y = cols[1].selectbox('y axis', choices,
                          index=choices.index(dy) if dy in choices else 0,
                          key=f'{key_prefix}.y')
    summary = ds.summary()
    cols[2].caption(f"{summary['total']} designs · {summary['feasible']} feasible · "
                    f"{summary['front']} on the front")

    fig = _figure(ds, x, y, target_tol=target_tol)
    event = st.plotly_chart(fig, key=f'{key_prefix}.scatter',
                            on_select='rerun', selection_mode='points',
                            width='stretch')

    idx = _selected_index(event)
    if idx is None:
        st.caption('Click a point to see its microstructure and how it did against '
                   'the criteria.')
        return
    _detail(ds, ds.designs[idx], target_tol=target_tol, run_id=run_id)


def _selected_index(event) -> int | None:
    """Read the DesignSet index out of a selection event.

    Streamlit's return value supports both attribute and mapping access depending
    on version, so try both rather than assuming one.
    """
    selection = None
    for get in (lambda: event['selection'], lambda: event.selection):
        try:
            selection = get()
            break
        except (TypeError, KeyError, AttributeError):
            continue
    if not selection:
        return None
    try:
        points = selection['points']
    except (TypeError, KeyError):
        points = getattr(selection, 'points', None)
    if not points:
        return None

    point = points[0]
    cd = point.get('customdata') if hasattr(point, 'get') else None
    if isinstance(cd, (list, tuple)) and cd:
        return int(cd[0])
    if isinstance(cd, (int, float)):
        return int(cd)
    return None


def _figure(ds: DesignSet, x: str, y: str, *, target_tol: float) -> go.Figure:
    groups = {
        'Pareto front': (_FRONT, 9,
                         [d for d in ds.designs if d.feasible and d.rank == 0]),
        'dominated': (_DOMINATED, 7,
                      [d for d in ds.designs if d.feasible and d.rank != 0]),
        'infeasible': (_INFEASIBLE, 7,
                       [d for d in ds.designs if not d.feasible]),
    }
    fig = go.Figure()
    for name, (colour, size, designs) in groups.items():
        pts = [d for d in designs
               if _value(d, x) is not None and _value(d, y) is not None]
        if not pts:
            continue
        fig.add_trace(go.Scatter(
            x=[_value(d, x) for d in pts],
            y=[_value(d, y) for d in pts],
            mode='markers',
            name=f'{name} ({len(pts)})',
            marker=dict(size=size, color=colour,
                        line=dict(width=1, color='rgba(0,0,0,.35)'),
                        symbol='circle' if name != 'infeasible' else 'x'),
            # The DesignSet index, so selection survives the trace split.
            customdata=[[d.index] for d in pts],
            hovertemplate=(f'<b>%{{customdata[0]}}</b><br>{x}=%{{x:.4g}}<br>'
                           f'{y}=%{{y:.4g}}<extra>' + name + '</extra>'),
        ))

    _add_target_overlay(fig, ds, x, y, target_tol)
    fig.update_layout(
        xaxis_title=x, yaxis_title=y,
        margin=dict(l=10, r=10, t=30, b=10), height=440,
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        # NOT dragmode='select': that makes a plain click draw a zero-area selection
        # box, which selects nothing. Leave the default so a click picks the point;
        # box and lasso are still available from the mode bar.
        clickmode='event+select',
    )
    return fig


def _add_target_overlay(fig, ds: DesignSet, x: str, y: str, tol: float) -> None:
    """Draw the requested band, so the plot answers 'did it hit the target'."""
    cx = ds.criterion_for(x)
    cy = ds.criterion_for(y)
    bx = cx.band(tol) if cx else None
    by = cy.band(tol) if cy else None
    if bx:
        fig.add_vrect(x0=bx[0], x1=bx[1], fillcolor='rgba(47,111,237,.10)',
                      line_width=0, annotation_text=f'{x} target',
                      annotation_position='top left')
    if by:
        fig.add_hrect(y0=by[0], y1=by[1], fillcolor='rgba(47,111,237,.10)',
                      line_width=0)


def _value(d: Design, prop: str):
    return d.final_loss if prop == 'final_loss' else d.get(prop)


def _detail(ds: DesignSet, design: Design, *, target_tol: float,
            run_id: str = '') -> None:
    st.divider()
    st.markdown(f'#### Design {design.label}')
    left, right = st.columns([1, 1.5])

    with left:
        vf = design.volume_fraction
        mask_image(design.mask,
                   caption=f'128×128 mask' + (f' · phase A = {vf:.3f}' if vf else ''))
        meta = []
        if design.status:
            meta.append(f'status `{design.status}`')
        meta.append(f'rank {design.rank}')
        meta.append('feasible' if design.feasible else '**infeasible**')
        if design.final_loss is not None:
            meta.append(f'loss {design.final_loss:.4g}')
        if design.has_fenics:
            meta.append('FEniCS validated')
        st.caption(' · '.join(meta))
        if run_id:
            from . import compare
            compare.pin_button(run_id, design)

    with right:
        _criteria_table(ds, design, target_tol)

    if ds.loss_hist is not None and len(ds.loss_hist):
        with st.expander('Convergence', expanded=False):
            st.line_chart({'loss': np.asarray(ds.loss_hist, dtype=float)},
                          height=200)


def _criteria_table(ds: DesignSet, design: Design, target_tol: float) -> None:
    """Achieved vs requested, plus ground truth when validation ran.

    The surrogate column is what the optimizer solved against; the FEniCS column is
    what the PDE says the same mask actually does. The gap between them is the
    number that decides whether to trust the design, so it gets its own column
    rather than a separate panel.
    """
    rows, line = criteria_rows(ds, design, target_tol)
    if line:
        st.markdown(line)
    st.dataframe(rows, hide_index=True, width='stretch')
    if design.has_fenics:
        st.caption('FEniCS values are recomputed from the run\'s own '
                   '`fenics_<physics>_results.npz`, using upstream\'s homogenisation '
                   'so they match the number the CLI prints.')


def criteria_rows(ds: DesignSet, design: Design,
                  target_tol: float) -> tuple[list[dict], str]:
    """Rows for the detail table, plus its one-line summary. Pure, so it is tested.

    Two different errors share this table and must not be confused: `error` is the
    surrogate against what the user *asked for*, `Δ` is the surrogate against what
    the PDE *says it got*. A design can hit its target and still be wrong.
    """
    has_gt = design.has_fenics
    # A physics FEniCS solved may expose a property the surrogate never wrote.
    extra = [p for p in sorted(design.fenics_props) if p not in ds.prop_names]

    rows: list[dict] = []
    met_count = checked = 0
    deltas: list[float] = []
    for prop in [*ds.prop_names, *extra]:
        achieved = design.get(prop)
        crit = ds.criterion_for(prop)
        met, err = (crit.check(achieved, target_tol) if crit else (None, None))
        if met is not None:
            checked += 1
            met_count += int(met)
        row = {'property': prop, 'NN' if has_gt else 'achieved': _fmt(achieved)}
        if has_gt:
            truth = design.fenics_props.get(prop)
            delta = _relative(achieved, truth)
            if delta is not None:
                deltas.append(delta)
            row['FEniCS'] = _fmt(truth)
            row['Δ'] = '' if delta is None else f'{delta * 100:.1f}%'
        row['requested'] = _requested(crit)
        row['error'] = '' if err is None else f'{err * 100:.1f}%'
        row[''] = '' if met is None else ('✓' if met else '✗')
        rows.append(row)

    parts = []
    if checked:
        parts.append(f'**meets {met_count}/{checked}**')
    if deltas:
        parts.append(f'max surrogate error vs FEniCS **{max(deltas) * 100:.1f}%**')
    return rows, ' · '.join(parts)


def _relative(nn, truth) -> float | None:
    """|NN − FEniCS| / |FEniCS|, ground truth in the denominator."""
    if nn is None or truth is None or truth == 0:
        return None
    return abs(nn - truth) / abs(truth)


def _requested(crit) -> str:
    if crit is None:
        return ''
    if crit.target is not None:
        return f'target {crit.target:.4g}'
    if crit.lo is not None and crit.hi is not None:
        return f'range {crit.lo:.4g}–{crit.hi:.4g}'
    return crit.mode


def _fmt(v) -> str:
    if v is None:
        return '—'
    return f'{v:.4g}'


def render_for_run(run_id: str, cfg) -> None:
    """Load a run's artifacts and plot them, explaining any absence precisely."""
    from ..artifacts.loader import ArtifactError, load_run
    from ..execution import runstore

    status = runstore.read_status(cfg.runs_dir, run_id)
    art_dir = runstore.artifact_dir(cfg.runs_dir, run_id)
    try:
        ds = load_run(art_dir)
    except ArtifactError as exc:
        st.error(str(exc), icon='🔴')
        return

    if ds is None:
        _explain_missing(status)
        return
    # The band around a `target` directive is target_tol wide, and that value is a
    # property of the run, not of the viewer. Using a fixed default here would draw
    # a 2% band around a run solved to 0.1% and make a miss look like a hit.
    render(ds, key_prefix=f'ds.{run_id}', run_id=run_id,
           target_tol=_run_target_tol(cfg.runs_dir, run_id))


def _run_target_tol(runs_root, run_id: str, default: float = 0.02) -> float:
    import json
    from ..execution import runstore
    path = runstore.run_dir(runs_root, run_id) / 'config.json'
    try:
        return float(json.loads(path.read_text())['target_tol'])
    except (OSError, ValueError, KeyError, TypeError):
        return default


def _explain_missing(status) -> None:
    """'No artifact yet', 'none by design', and 'failed' look identical on disk."""
    from ..execution import runstore
    if status is None:
        st.info('No artifacts.')
    elif status.state in (runstore.STATE_RUNNING, runstore.STATE_CANCELLING):
        st.info('No artifacts yet — they are written when the solve finishes.')
    elif status.state == runstore.STATE_FAILED:
        st.error(f'The run failed (exit code {status.returncode}); no artifacts were '
                 'written. Check the log above.', icon='🔴')
    elif status.state == runstore.STATE_CANCELLED:
        st.info('Run cancelled before it wrote artifacts.')
    elif status.state == runstore.STATE_DONE:
        st.info('This run wrote no design set. That is expected for `--dry_run` and '
                'for `--pareto_steps 0`, which runs the payoff table only.')
    else:
        st.info('No artifacts.')

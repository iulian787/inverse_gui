"""Compare designs across runs: masks side by side, properties, convergence.

Pinning rather than multi-select: the designs worth comparing are found by clicking
around several runs' scatters minutes apart, so the selection has to survive
navigating away. Pins live in session_state and hold only (run_id, design index) --
the artifacts are re-read on render, so a pin cannot go stale against a re-run.

Everything above the Streamlit calls is pure and tested directly: which properties
differ, and by how much, is the actual content of this view.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import streamlit as st

from ..artifacts.model import Design, DesignSet
from .components import mask_image

PINS_KEY = 'compare.pins'
MAX_PINS = 4


@dataclass(frozen=True)
class Pin:
    run_id: str
    index: int

    @property
    def key(self) -> str:
        return f'{self.run_id}#{self.index}'


# ------------------------------------------------------------------ pin state

# Pins live in the URL as well as session_state, for the same reason the active run
# does (see ui/state.active_run_id): session_state dies on a browser refresh, and the
# whole point of this view is to collect designs across several runs over minutes.
# The URL is the only place that survives, and a pin is two short strings.
PARAM = 'pins'


def _encode(items: list[Pin]) -> str:
    return ','.join(p.key for p in items)


def _decode(raw: str) -> list[Pin]:
    out: list[Pin] = []
    for chunk in (raw or '').split(','):
        run_id, _, index = chunk.partition('#')
        if not run_id or not index.isdigit():
            continue                      # hand-edited URL; ignore the bad entry
        pin = Pin(run_id, int(index))
        if pin not in out and len(out) < MAX_PINS:
            out.append(pin)
    return out


def pins() -> list[Pin]:
    """Session state is the working copy; the URL is what survives a refresh."""
    if PINS_KEY not in st.session_state:
        st.session_state[PINS_KEY] = _decode(st.query_params.get(PARAM, ''))
    return list(st.session_state[PINS_KEY])


def _store(items: list[Pin]) -> None:
    st.session_state[PINS_KEY] = items
    if items:
        st.query_params[PARAM] = _encode(items)
    elif PARAM in st.query_params:
        del st.query_params[PARAM]


def is_pinned(run_id: str, index: int) -> bool:
    return any(p.run_id == run_id and p.index == index for p in pins())


def toggle_pin(run_id: str, index: int) -> None:
    current = pins()
    match = [p for p in current if p.run_id == run_id and p.index == index]
    if match:
        current.remove(match[0])
    elif len(current) < MAX_PINS:
        current.append(Pin(run_id, index))
    _store(current)


def clear_pins() -> None:
    _store([])


def pin_button(run_id: str, design: Design) -> None:
    """Rendered inside the design detail, where the user is already looking."""
    pinned = is_pinned(run_id, design.index)
    full = len(pins()) >= MAX_PINS and not pinned
    if st.button('📌 Pinned — click to remove' if pinned else '📌 Pin for compare',
                 key=f'pin.{run_id}.{design.index}', width='stretch', disabled=full,
                 help=f'Compare up to {MAX_PINS} designs from any runs.'
                      if not full else f'{MAX_PINS} designs already pinned.'):
        toggle_pin(run_id, design.index)
        st.rerun()


# ------------------------------------------------------------------ pure part

@dataclass
class Pinned:
    """A resolved pin: the design plus the set it came from."""
    pin: Pin
    ds: DesignSet
    design: Design

    @property
    def label(self) -> str:
        return f'{self.pin.run_id.replace("run_", "")} · {self.design.label}'


def comparison_rows(items: list[Pinned]) -> list[dict]:
    """One row per property, one column per pinned design, plus the spread.

    The spread is what the view is for -- with four columns of four-digit numbers
    the eye cannot find the property that actually differs. Relative to the mean, so
    it is comparable across E (1e5) and alpha (1e-5); blank when a property is
    missing from any pin, because a spread over a subset invites the wrong read.
    """
    names: list[str] = []
    for item in items:
        for prop in item.ds.prop_names:
            if prop not in names and item.design.get(prop) is not None:
                names.append(prop)

    rows = []
    for prop in names:
        values = [i.design.get(prop) for i in items]
        row = {'property': prop}
        for item, value in zip(items, values):
            row[item.label] = '—' if value is None else f'{value:.4g}'
        row['spread'] = _spread(values)
        rows.append(row)
    return rows


def _spread(values) -> str:
    present = [v for v in values if v is not None]
    if len(present) < 2 or len(present) != len(values):
        return ''
    mean = sum(present) / len(present)
    if mean == 0:
        return ''
    return f'{(max(present) - min(present)) / abs(mean) * 100:.1f}%'


def convergence_series(items: list[Pinned]) -> dict[str, list[float]]:
    """Loss histories padded to a common length so they can share one chart.

    NaN padding, not zero or the last value: a run that converged in 40 iterations
    should stop being drawn at 40, not flatline to the width of the longest run.
    """
    series = {i.label: np.asarray(i.ds.loss_hist, dtype=float).reshape(-1)
              for i in items if i.ds.loss_hist is not None and len(i.ds.loss_hist)}
    if not series:
        return {}
    width = max(len(v) for v in series.values())
    return {k: [*v, *([float('nan')] * (width - len(v)))] for k, v in series.items()}


# ------------------------------------------------------------------ rendering

def render(cfg) -> None:
    items = _resolve(cfg, pins())
    if not items:
        st.caption('No designs pinned. Open a run, click a point in the design '
                   'space, and use **📌 Pin for compare**.')
        return

    cols = st.columns([6, 1])
    cols[0].caption(f'{len(items)} of {MAX_PINS} pinned')
    if cols[1].button('Clear', key='compare.clear'):
        clear_pins()
        st.rerun()

    for col, item in zip(st.columns(len(items)), items):
        with col:
            st.markdown(f'**{item.label}**')
            # Fixed width, not 'stretch': with two pins a stretched mask fills the
            # pane and pushes the table -- the actual comparison -- off screen.
            mask_image(item.design.mask, caption=_caption(item.design), width=230)
            if st.button('Unpin', key=f'unpin.{item.pin.key}', width='stretch'):
                toggle_pin(item.pin.run_id, item.pin.index)
                st.rerun()

    st.dataframe(comparison_rows(items), hide_index=True, width='stretch')

    series = convergence_series(items)
    if len(series) > 1:
        st.caption('Convergence')
        st.line_chart(series, height=220)
    elif series:
        st.caption('Only one pinned run has a convergence history; nothing to '
                   'overlay. Pareto runs record no per-iteration loss.')


def _caption(design: Design) -> str:
    bits = [f'rank {design.rank}']
    vf = design.volume_fraction
    if vf is not None:
        bits.append(f'VF {vf:.3f}')
    if design.final_loss is not None:
        bits.append(f'loss {design.final_loss:.3g}')
    if design.has_fenics:
        bits.append('FEniCS ✓')
    return ' · '.join(bits)


def _resolve(cfg, wanted: list[Pin]) -> list[Pinned]:
    """Load each pinned run, dropping pins whose run or design has gone away."""
    from ..artifacts.loader import ArtifactError, load_run
    from ..execution import runstore

    out: list[Pinned] = []
    for pin in wanted:
        try:
            ds = load_run(runstore.artifact_dir(cfg.runs_dir, pin.run_id))
        except ArtifactError:
            ds = None
        if ds is None or pin.index >= len(ds.designs):
            st.caption(f'`{pin.run_id}` #{pin.index} is no longer available.')
            continue
        out.append(Pinned(pin, ds, ds.designs[pin.index]))
    return out

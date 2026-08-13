"""Design-space figure construction and selection decoding.

The trap being guarded: designs are split into several traces so the legend can
filter them, and Plotly's point_index is trace-relative. If selection ever reads
point_index instead of customdata, clicking a dominated point opens a different
design's microstructure -- a silent, plausible-looking wrong answer.
"""

from __future__ import annotations

import numpy as np
import pytest

from inverse_gui.artifacts.model import Criterion, Design, DesignSet
from inverse_gui.ui import design_space as dsp


def make_set() -> DesignSet:
    designs = []
    for i in range(6):
        designs.append(Design(
            index=i, label=f'#{i}',
            props={'rho': 0.3 + 0.1 * i, 'E': 200000.0 + 1000 * i},
            mask=np.zeros((128, 128), dtype=np.uint8),
            rank=0 if i % 2 == 0 else 1,
            feasible=i != 5,
        ))
    return DesignSet(kind='pareto', prop_names=['rho', 'E'], designs=designs)


def test_traces_are_split_by_rank_and_feasibility():
    fig = dsp._figure(make_set(), 'rho', 'E', target_tol=0.02)
    names = [t.name for t in fig.data]
    assert any('Pareto front' in n for n in names)
    assert any('dominated' in n for n in names)
    assert any('infeasible' in n for n in names)


def test_every_point_carries_its_designset_index():
    fig = dsp._figure(make_set(), 'rho', 'E', target_tol=0.02)
    seen = []
    for trace in fig.data:
        if trace.customdata is None:
            continue
        seen.extend(int(cd[0]) for cd in trace.customdata)
    assert sorted(seen) == list(range(6))


def test_customdata_index_survives_the_trace_split():
    """The dominated trace's first point is design 1, not design 0."""
    fig = dsp._figure(make_set(), 'rho', 'E', target_tol=0.02)
    dominated = next(t for t in fig.data if 'dominated' in t.name)
    assert int(dominated.customdata[0][0]) == 1


def test_selection_decodes_customdata():
    event = {'selection': {'points': [{'customdata': [4], 'point_index': 0}]}}
    assert dsp._selected_index(event) == 4


@pytest.mark.parametrize('event', [
    None, {}, {'selection': {}}, {'selection': {'points': []}},
    {'selection': {'points': [{}]}},
])
def test_selection_handles_empty_events(event):
    assert dsp._selected_index(event) is None


def test_dragmode_is_not_select():
    """dragmode='select' turns a plain click into a zero-area box that selects nothing."""
    fig = dsp._figure(make_set(), 'rho', 'E', target_tol=0.02)
    assert fig.layout.dragmode != 'select'


def test_target_band_uses_the_supplied_tolerance():
    ds = make_set()
    ds.criteria = [Criterion(prop='E', mode='range', target=200000.0)]
    tight = dsp._figure(ds, 'E', 'rho', target_tol=0.001)
    loose = dsp._figure(ds, 'E', 'rho', target_tol=0.05)

    def width(fig):
        rect = next(s for s in fig.layout.shapes if s.type == 'rect')
        return abs(rect.x1 - rect.x0)

    assert width(loose) > width(tight) * 10


def test_the_fenics_column_appears_only_when_validation_ran():
    ds = make_set()
    plain, _ = dsp.criteria_rows(ds, ds.designs[0], 0.02)
    assert 'achieved' in plain[0] and 'FEniCS' not in plain[0]

    ds.designs[0].fenics_props = {'E': 202000.0}
    validated, _ = dsp.criteria_rows(ds, ds.designs[0], 0.02)
    assert 'NN' in validated[0] and 'achieved' not in validated[0]
    assert 'FEniCS' in validated[0] and 'Δ' in validated[0]


def test_the_delta_is_measured_against_ground_truth():
    """1% of the FEniCS value, not of the surrogate's -- the denominator matters."""
    ds = make_set()
    d = ds.designs[0]
    d.props['E'] = 202000.0
    d.fenics_props = {'E': 200000.0}
    rows, line = dsp.criteria_rows(ds, d, 0.02)
    row = next(r for r in rows if r['property'] == 'E')
    assert row['Δ'] == '1.0%'
    assert 'max surrogate error vs FEniCS **1.0%**' in line


def test_a_property_only_fenics_computed_still_gets_a_row():
    """FEniCS reports the full anisotropic set even when the run targeted E alone."""
    ds = make_set()
    ds.designs[0].fenics_props = {'E': 2e5, 'G_xy': 77000.0}
    rows, _ = dsp.criteria_rows(ds, ds.designs[0], 0.02)
    extra = next(r for r in rows if r['property'] == 'G_xy')
    assert extra['NN'] == '—' and extra['FEniCS'] == '7.7e+04'
    assert extra['Δ'] == ''            # nothing to compare against


def test_target_error_and_fenics_delta_are_separate_columns():
    """Hitting the target says nothing about agreeing with the PDE."""
    ds = make_set()
    ds.criteria = [Criterion(prop='E', mode='range', target=200000.0)]
    d = ds.designs[0]
    d.props['E'] = 200000.0            # exactly on target
    d.fenics_props = {'E': 180000.0}   # but the PDE disagrees by 11%
    row = next(r for r in dsp.criteria_rows(ds, d, 0.02)[0] if r['property'] == 'E')
    assert row['error'] == '0.0%' and row[''] == '✓'
    assert row['Δ'] == '11.1%'


def test_points_with_missing_values_are_dropped_not_zeroed():
    ds = make_set()
    ds.designs[0].props.pop('E')
    fig = dsp._figure(ds, 'rho', 'E', target_tol=0.02)
    total = sum(len(t.x) for t in fig.data if t.x is not None)
    assert total == 5

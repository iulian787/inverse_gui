"""Cross-run compare: pin bookkeeping, the diff table, and the overlay.

The pure functions carry the meaning of this view -- which property differs, by how
much, and where each convergence curve should stop -- so they are tested directly
rather than through AppTest, which can only assert that widgets exist.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from inverse_gui.artifacts.model import Design, DesignSet
from inverse_gui.ui import compare


def make_pinned(run_id: str, *, E: float, rho: float = 0.4, loss=None,
                index: int = 0, extra: dict | None = None) -> compare.Pinned:
    props = {'E': E, 'rho': rho, **(extra or {})}
    design = Design(index=index, label=f'#{index}', props=props,
                    mask=np.zeros((4, 4), dtype=np.uint8), final_loss=loss)
    ds = DesignSet(kind='single_point', prop_names=list(props), designs=[design],
                   loss_hist=None if loss is None else np.linspace(1.0, loss, 5))
    return compare.Pinned(compare.Pin(run_id, index), ds, design)


# ------------------------------------------------------------------ pin state

class FakeState(dict):
    """st.session_state and st.query_params are both dict-like; that is all we need."""


@pytest.fixture(autouse=True)
def session(monkeypatch):
    state, params = FakeState(), FakeState()
    monkeypatch.setattr(compare.st, 'session_state', state)
    monkeypatch.setattr(compare.st, 'query_params', params)
    return state


@pytest.fixture
def url(monkeypatch):
    """The query-param store, for the refresh-survival tests."""
    return compare.st.query_params


def test_pinning_is_a_toggle():
    compare.toggle_pin('run_a', 0)
    assert compare.is_pinned('run_a', 0)
    compare.toggle_pin('run_a', 0)
    assert not compare.is_pinned('run_a', 0)


def test_the_same_index_in_two_runs_is_two_pins():
    """Pins are (run, index) -- design #0 of two runs are different designs."""
    compare.toggle_pin('run_a', 0)
    compare.toggle_pin('run_b', 0)
    assert len(compare.pins()) == 2
    assert compare.is_pinned('run_b', 0)


def test_pins_stop_at_the_cap():
    for i in range(compare.MAX_PINS + 3):
        compare.toggle_pin('run_a', i)
    assert len(compare.pins()) == compare.MAX_PINS


def test_unpinning_frees_a_slot():
    for i in range(compare.MAX_PINS):
        compare.toggle_pin('run_a', i)
    compare.toggle_pin('run_a', 0)
    compare.toggle_pin('run_b', 9)
    assert compare.is_pinned('run_b', 9)
    assert len(compare.pins()) == compare.MAX_PINS


def test_clear_empties_everything():
    compare.toggle_pin('run_a', 0)
    compare.clear_pins()
    assert compare.pins() == []


def test_pins_survive_a_refresh(session, url):
    """A browser refresh starts a new session; the URL is what carries them over."""
    compare.toggle_pin('run_a', 0)
    compare.toggle_pin('run_b', 2)
    assert url[compare.PARAM] == 'run_a#0,run_b#2'

    session.clear()                                   # what a refresh looks like
    assert compare.pins() == [compare.Pin('run_a', 0), compare.Pin('run_b', 2)]


def test_clearing_removes_the_query_param(session, url):
    compare.toggle_pin('run_a', 0)
    compare.clear_pins()
    assert compare.PARAM not in url
    session.clear()
    assert compare.pins() == []


@pytest.mark.parametrize('raw, expected', [
    ('', []),
    ('garbage', []),
    ('run_a#notanumber', []),
    ('run_a#0,,run_b#1', [('run_a', 0), ('run_b', 1)]),
    ('run_a#0,run_a#0', [('run_a', 0)]),                       # deduped
    ('a#0,b#1,c#2,d#3,e#4,f#5', [('a', 0), ('b', 1), ('c', 2), ('d', 3)]),  # capped
])
def test_a_hand_edited_url_cannot_break_the_view(session, url, raw, expected):
    url[compare.PARAM] = raw
    assert compare.pins() == [compare.Pin(r, i) for r, i in expected]


# ------------------------------------------------------------------ diff table

def test_each_pin_gets_its_own_column():
    items = [make_pinned('run_a', E=2.0e5), make_pinned('run_b', E=2.2e5)]
    rows = compare.comparison_rows(items)
    row = next(r for r in rows if r['property'] == 'E')
    assert row[items[0].label] == '2e+05'
    assert row[items[1].label] == '2.2e+05'


def test_spread_is_relative_so_it_is_comparable_across_scales():
    """10% on E and 10% on alpha must read the same, or the column is useless."""
    items = [make_pinned('run_a', E=1.0e5, extra={'alpha': 1.0e-5}),
             make_pinned('run_b', E=1.1e5, extra={'alpha': 1.1e-5})]
    rows = {r['property']: r['spread'] for r in compare.comparison_rows(items)}
    assert rows['E'] == rows['alpha'] != ''


def test_identical_designs_have_no_spread():
    items = [make_pinned('run_a', E=2.0e5), make_pinned('run_b', E=2.0e5)]
    row = next(r for r in compare.comparison_rows(items) if r['property'] == 'E')
    assert row['spread'] == '0.0%'


def test_a_property_missing_from_one_pin_has_no_spread():
    """A spread over a subset of the columns reads as a spread over all of them."""
    items = [make_pinned('run_a', E=2.0e5, extra={'kappa': 150.0}),
             make_pinned('run_b', E=2.2e5)]
    rows = {r['property']: r for r in compare.comparison_rows(items)}
    assert rows['kappa']['spread'] == ''
    assert rows['kappa'][items[1].label] == '—'
    assert rows['E']['spread'] != ''


def test_a_single_pin_yields_no_spread():
    rows = compare.comparison_rows([make_pinned('run_a', E=2.0e5)])
    assert all(r['spread'] == '' for r in rows)


def test_properties_from_every_pin_appear():
    items = [make_pinned('run_a', E=2.0e5, extra={'kappa': 150.0}),
             make_pinned('run_b', E=2.2e5, extra={'alpha': 1.1e-5})]
    names = {r['property'] for r in compare.comparison_rows(items)}
    assert {'E', 'rho', 'kappa', 'alpha'} == names


# ------------------------------------------------------------------ overlay

def test_shorter_histories_are_nan_padded_not_flatlined():
    """A run that converged early must stop being drawn, not run on as a flat line."""
    a = make_pinned('run_a', E=2e5, loss=0.1)
    b = make_pinned('run_b', E=2e5, loss=0.2)
    b.ds.loss_hist = np.linspace(1.0, 0.2, 9)          # longer than a's 5

    series = compare.convergence_series([a, b])
    assert len(set(len(v) for v in series.values())) == 1, 'lengths must match'
    tail = series[a.label][5:]
    assert tail and all(math.isnan(v) for v in tail)
    assert not any(math.isnan(v) for v in series[b.label])


def test_runs_without_a_history_are_omitted():
    """Pareto runs record no per-iteration loss; they simply do not appear."""
    with_hist = make_pinned('run_a', E=2e5, loss=0.1)
    without = make_pinned('run_b', E=2e5)
    series = compare.convergence_series([with_hist, without])
    assert list(series) == [with_hist.label]


def test_no_histories_is_an_empty_chart_not_a_crash():
    assert compare.convergence_series([make_pinned('run_a', E=2e5)]) == {}

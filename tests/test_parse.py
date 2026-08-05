"""Line classifier and progress reducer.

The real-IPOPT fixtures matter more than the fake ones: fake_optimizer.py emits a
clean table, while real IPOPT prints '-' in several columns on iteration 0 and
suffixes restoration iterations with a letter.
"""

from __future__ import annotations

import pytest

from inverse_gui.parse.lines import EventKind, classify
from inverse_gui.parse.progress import ProgressReducer, replay

# Verbatim from a real IPOPT print_level=5 table.
REAL_IPOPT = """This is Ipopt version 3.14.16, running with linear solver MUMPS 5.7.3.

iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
   0  1.0000000e+00 1.00e+00 1.00e+00  -1.0 0.00e+00    -  0.00e+00 0.00e+00   0
   1  9.1234567e-01 8.20e-01 3.10e-02  -1.7 1.15e+00    -  9.90e-01 1.00e+00f  1
  12r 4.4000000e-01 2.00e-03 9.99e+02  -2.3 0.00e+00   3.2 0.00e+00 0.00e+00R  1
"""


def kinds(text: str):
    return [classify(l).kind for l in text.splitlines()]


def test_real_ipopt_iteration_rows_parse():
    ks = kinds(REAL_IPOPT)
    assert ks.count(EventKind.IPOPT_ITER) == 3


def test_iteration_zero_with_dashes():
    e = classify('   0  1.0000000e+00 1.00e+00 1.00e+00  -1.0 0.00e+00    -  '
                 '0.00e+00 0.00e+00   0')
    assert e.kind is EventKind.IPOPT_ITER
    assert e.data['iter'] == 0 and e.data['objective'] == 1.0


def test_restoration_iteration_suffix():
    e = classify('  12r 4.4000000e-01 2.00e-03 9.99e+02  -2.3 0.00e+00   3.2 '
                 '0.00e+00 0.00e+00R  1')
    assert e.kind is EventKind.IPOPT_ITER
    assert e.data['iter'] == 12 and e.data['phase'] == 'r'


def test_header_is_not_an_iteration():
    e = classify('iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) '
                 'alpha_du alpha_pr  ls')
    assert e.kind is EventKind.IPOPT_BANNER


@pytest.mark.parametrize('line', [
    '  Saved: /tmp/x/pareto_results.npz',
    'Total solutions: 12  |  Feasible: 9',
    '',
    '─────────────────',
])
def test_non_iteration_lines_are_not_misread(line):
    assert classify(line).kind is not EventKind.IPOPT_ITER


def test_stage_banner_with_box_drawing():
    e = classify('Stage 2: Epsilon-Constraint Pareto Sweep (ρ-objective)')
    assert e.kind is EventKind.STAGE
    assert e.data['stage'] == 2
    assert 'Epsilon-Constraint' in e.data['title']


def test_box_drawing_rule_is_ignored():
    assert classify('─' * 65).kind is EventKind.OTHER


def test_grid_total_gives_denominator():
    e = classify('  Epsilon grid: [19, 19] points per property '
                 '(361 combinations × 3 restarts = 1083 solves)')
    assert e.kind is EventKind.GRID_TOTAL
    assert e.data['combinations'] == 361 and e.data['solves'] == 1083


def test_grid_point_tick():
    e = classify("  Grid point 41/361: {'E': 250000.0, 'kappa': 180.0}")
    assert e.kind is EventKind.GRID_POINT
    assert (e.data['index'], e.data['total']) == (41, 361)


def test_estimate_line():
    e = classify('  Estimated solves: 3 feasibility + (2 payoff rows + 361 grid pts '
                 '(19^2)) × 3 restarts = 1092 total')
    assert e.kind is EventKind.ESTIMATE and e.data['total_solves'] == 1092


def test_payoff_achieved():
    e = classify('  [Payoff] kappa: achieved=1.9800e+02')
    assert e.kind is EventKind.PAYOFF and e.data['prop'] == 'kappa'


def test_fenics_line():
    e = classify('  [FEniCS] Running elastic solver (env=fenics_env)...')
    assert e.kind is EventKind.FENICS


# ------------------------------------------------------------------ reducer

def test_reducer_tracks_pareto_progress():
    r = ProgressReducer()
    for line in [
        '  Device: cpu',
        'Stage 2: Epsilon-Constraint Pareto Sweep',
        '  Epsilon grid: [19] points per property '
        '(19 combinations × 3 restarts = 57 solves)',
        '  Grid point 7/19: {}',
    ]:
        r.feed(classify(line))
    s = r.state
    assert s.device == 'cpu' and s.stage == 2
    assert (s.grid_index, s.grid_total) == (7, 19)
    assert abs(s.fraction - 7 / 19) < 1e-9
    assert 'sweep' in s.headline


def test_new_stage_resets_grid_counters():
    r = ProgressReducer()
    for line in ['Stage 2: Sweep', '  Grid point 9/19: {}', 'Stage 3: Sorting']:
        r.feed(classify(line))
    assert r.state.grid_index == 0 and r.state.grid_total == 0


def test_reducer_tracks_single_point_iterations():
    r = ProgressReducer()
    for line in REAL_IPOPT.splitlines():
        r.feed(classify(line))
    assert r.state.iters_total == 3
    assert r.state.iter == 12
    assert r.state.fraction is None       # no denominator in single-point mode


def test_done_sets_fraction_to_one():
    r = ProgressReducer()
    r.feed(classify('Done.'))
    assert r.state.done and r.state.fraction == 1.0


def test_replay_is_equivalent_to_streaming():
    """The reattach path must reconstruct exactly what streaming produced."""
    lines = REAL_IPOPT.splitlines() + ['  Saved: /tmp/r.npz', 'Done.']
    streamed = ProgressReducer()
    for line in lines:
        streamed.feed(classify(line))
    assert replay(lines) == streamed.state

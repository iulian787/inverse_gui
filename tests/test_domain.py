"""Domain-layer tests.

Each validation test names the upstream failure mode it guards against. If one of
these starts failing, check whether upstream changed before changing the test.
"""

from __future__ import annotations

import pytest

from inverse_gui.domain import argv as A
from inverse_gui.domain import cost
from inverse_gui.domain import properties as P
from inverse_gui.domain import validate as V
from inverse_gui.domain.directives import Directive, Mode, expand_isotropic, parse
from inverse_gui.domain.schema import Checkpoints, RunConfig, RunMode


def cfg_for(mode=RunMode.SINGLE, **kw) -> RunConfig:
    c = RunConfig.for_mode(mode, **kw)
    c.checkpoints = Checkpoints(elastic='/dev/null')
    c.directives = {'E': Directive('E', Mode.TARGET, value=200000.0)}
    return c


def codes(issues) -> set[str]:
    return {i.code for i in issues}


# ------------------------------------------------------------------ directives

@pytest.mark.parametrize('spec,mode,value,hi', [
    ('max', Mode.MAX, None, None),
    ('min', Mode.MIN, None, None),
    ('free', Mode.FREE, None, None),
    ('target 200000', Mode.TARGET, 200000.0, None),
    ('range 1e5 3e5', Mode.RANGE, 1e5, 3e5),
])
def test_parse_roundtrip(spec, mode, value, hi):
    d = parse(spec)
    assert (d.mode, d.value, d.hi) == (mode, value, hi)


@pytest.mark.parametrize('bad', ['', 'target', 'range 1e5', 'Max', 'maximize',
                                 'target 1 2', 'range 1 2 3'])
def test_parse_rejects_malformed(bad):
    with pytest.raises(ValueError):
        parse(bad)


def test_render_never_emits_empty_string():
    """An empty value raises IndexError in the upstream parser, not ValueError."""
    assert Directive('E', Mode.TARGET, value=None).render() is None
    assert Directive('E', Mode.RANGE, value=1.0, hi=None).render() is None
    assert Directive('E', Mode.FREE).render() is None


def test_render_is_float_roundtrippable():
    assert parse(Directive('a', Mode.TARGET, value=1e-5).render()).value == 1e-5
    assert Directive('a', Mode.TARGET, value=200000.0).render() == 'target 200000'


def test_isotropic_expansion_halves_weight_and_drops_alias():
    """directives.py:36-45 — E becomes E_xx AND E_yy independently, at half weight."""
    got = expand_isotropic(
        {'E': Directive('E', Mode.TARGET, value=2e5, weight=2.0)},
        iso_map=P.ISOTROPIC_EXPAND,
    )
    assert 'E' not in got
    assert got['E_xx'].value == 2e5 and got['E_yy'].value == 2e5
    assert got['E_xx'].weight == 1.0


def test_explicit_component_wins_over_isotropic():
    got = expand_isotropic(
        {'E': Directive('E', Mode.TARGET, value=2e5),
         'E_xx': Directive('E_xx', Mode.TARGET, value=3e5)},
        iso_map=P.ISOTROPIC_EXPAND,
    )
    assert got['E_xx'].value == 3e5      # explicit survives
    assert got['E_yy'].value == 2e5      # isotropic fills the gap


# ------------------------------------------------------------------ gating

def test_available_props_tracks_checkpoints():
    assert P.available_props(set()) == frozenset({'rho'})
    el = P.available_props({'elastic'})
    assert 'E' in el and 'G_xy' in el and 'kappa' not in el


def test_ungated_constraint_is_blocking():
    """Single-point does no gating: g=0.0 with zero gradient => never satisfiable."""
    c = cfg_for()
    c.directives['kappa'] = Directive('kappa', Mode.TARGET, value=180.0)
    issues = V.validate(c, check_files=False)
    assert 'UNGATED_CONSTRAINT' in codes(issues)
    assert V.blocking(issues)


def test_ungated_objective_is_blocking():
    c = cfg_for()
    c.directives['alpha'] = Directive('alpha', Mode.MAX)
    assert 'UNGATED_OBJECTIVE' in codes(V.validate(c, check_files=False))


def test_gated_property_passes():
    c = cfg_for()
    c.checkpoints = Checkpoints(elastic='/dev/null', thermal_conductivity='/dev/null')
    c.directives['kappa'] = Directive('kappa', Mode.TARGET, value=180.0)
    assert 'UNGATED_CONSTRAINT' not in codes(V.validate(c, check_files=False))


def test_rho_is_always_available():
    c = cfg_for()
    c.directives['rho'] = Directive('rho', Mode.TARGET, value=0.5)
    assert 'UNGATED_CONSTRAINT' not in codes(V.validate(c, check_files=False))


# ------------------------------------------------------------------ other rules

def test_no_checkpoint_blocks():
    c = cfg_for()
    c.checkpoints = Checkpoints()
    assert 'NO_CHECKPOINT' in codes(V.validate(c, check_files=False))


def test_inverted_range_blocks():
    """Upstream accepts L>H silently and produces cl>cu in cyipopt."""
    c = cfg_for()
    c.directives['E'] = Directive('E', Mode.RANGE, value=3e5, hi=1e5)
    assert 'RANGE_INVERTED' in codes(V.validate(c, check_files=False))


def test_nonpositive_ref_blocks():
    c = cfg_for()
    c.directives['E'] = Directive('E', Mode.TARGET, value=2e5, ref=0.0)
    assert 'REF_NONPOSITIVE' in codes(V.validate(c, check_files=False))


def test_ref_scale_mismatch_warns():
    c = cfg_for()
    c.checkpoints = Checkpoints(thermal_expansion='/dev/null')
    c.directives = {'alpha': Directive('alpha', Mode.TARGET, value=1e-7)}
    assert 'REF_SCALE' in codes(V.validate(c, check_files=False))


def test_iso_and_component_conflict_blocks():
    c = cfg_for()
    c.directives['E_xx'] = Directive('E_xx', Mode.TARGET, value=3e5)
    assert 'ISO_AND_COMPONENT' in codes(V.validate(c, check_files=False))


def test_no_active_directives_blocks():
    c = cfg_for()
    c.directives = {}
    assert 'NO_ACTIVE_DIRECTIVES' in codes(V.validate(c, check_files=False))


def test_vf_bounds_validated():
    c = cfg_for(vf_min=0.9, vf_max=0.1)
    assert 'VF_BOUNDS' in codes(V.validate(c, check_files=False))


def test_outside_training_range_warns():
    c = cfg_for()
    c.directives['E'] = Directive('E', Mode.TARGET, value=9e5)   # > 410000
    assert 'OUTSIDE_TRAINING' in codes(V.validate(c, check_files=False))


def test_pareto_degenerate_warns():
    c = cfg_for(RunMode.PARETO)
    assert 'PARETO_DEGENERATE' in codes(V.validate(c, check_files=False))


def test_bad_act_blocks():
    c = cfg_for(act='swish')
    assert 'ACT_INVALID' in codes(V.validate(c, check_files=False))


# ------------------------------------------------------------------ vf derivation

def test_pareto_derives_vf_from_rho_target():
    c = cfg_for(RunMode.PARETO, target_tol=0.02)
    c.directives['rho'] = Directive('rho', Mode.TARGET, value=0.5)
    lo, hi = c.derived_vf_bounds()
    assert (round(lo, 4), round(hi, 4)) == (0.49, 0.51)


def test_pareto_derives_vf_from_rho_range():
    c = cfg_for(RunMode.PARETO)
    c.directives['rho'] = Directive('rho', Mode.RANGE, value=0.3, hi=0.7)
    assert c.derived_vf_bounds() == (0.3, 0.7)


def test_pareto_vf_defaults_when_rho_is_maxmin():
    c = cfg_for(RunMode.PARETO)
    c.directives['rho'] = Directive('rho', Mode.MIN)
    assert c.derived_vf_bounds() == (0.05, 0.95)


# ------------------------------------------------------------------ cost

def test_points_per_prop_matches_upstream():
    """linspace and logspace share only the endpoint 1.0, so this is 2N-1."""
    assert cost.points_per_prop(1) == 1
    assert cost.points_per_prop(5) == 9
    assert cost.points_per_prop(10) == 19


def test_cost_two_objectives_is_1092():
    """The number the form shows must match what --dry_run prints."""
    c = cfg_for(RunMode.PARETO, restarts=3, pareto_steps=10)
    c.checkpoints = Checkpoints(elastic='/dev/null', thermal_conductivity='/dev/null')
    c.directives = {'E': Directive('E', Mode.MAX),
                    'kappa': Directive('kappa', Mode.MAX),
                    'nu': Directive('nu', Mode.TARGET, value=0.25)}
    est = cost.estimate(c)
    assert (est.pts_per_prop, est.total_grid, est.total_solves) == (19, 361, 1092)


def test_cost_is_none_for_single_point():
    assert cost.estimate(cfg_for(RunMode.SINGLE)) is None


# ------------------------------------------------------------------ argv

def test_argv_emits_only_non_defaults():
    c = cfg_for()
    got = A.build(c)
    assert '--ipopt_max_iter' not in got      # at default
    assert '--seed' not in got                # at default
    assert got.count('--output_dir') == 1


def test_argv_mode_defaults_are_mode_relative():
    """restarts=3 is the Pareto default, so it must not be emitted in Pareto mode."""
    assert '--restarts' not in A.build(cfg_for(RunMode.PARETO, restarts=3))
    assert '--restarts' in A.build(cfg_for(RunMode.SINGLE, restarts=3))


def test_argv_weights_only_in_single_point():
    for mode, expected in ((RunMode.SINGLE, True), (RunMode.PARETO, False)):
        c = cfg_for(mode)
        c.directives['E'] = Directive('E', Mode.MAX, weight=2.0)
        assert ('--weight_E' in A.build(c)) is expected


def test_argv_vf_only_single_pareto_only_steps():
    single = A.build(cfg_for(RunMode.SINGLE, vf_min=0.2))
    assert '--vf_min' in single and '--pareto_steps' not in single
    pareto = A.build(cfg_for(RunMode.PARETO, pareto_steps=4))
    assert '--pareto_steps' in pareto and '--vf_min' not in pareto


def test_argv_directive_value_is_one_token():
    """argparse takes the whole 'target 200000' as a single value."""
    got = A.build(cfg_for())
    assert got[got.index('--E') + 1] == 'target 200000'


def test_argv_omits_incomplete_directive():
    c = cfg_for()
    c.directives['nu'] = Directive('nu', Mode.TARGET, value=None)
    assert '--nu' not in A.build(c)


def test_argv_h5_props_suppresses_per_phase_fenics():
    c = cfg_for()
    c.fenics.validate = True
    c.fenics.h5_props = '/tmp/p.h5'
    c.fenics.props = {'E_A': 70000.0}
    got = A.build(c)
    assert '--fenics_h5_props' in got and '--fenics_E_A' not in got


def test_preview_is_shell_quoted():
    c = cfg_for()
    text = A.preview('/x/python', '/y/run.py', A.build(c))
    assert "'target 200000'" in text

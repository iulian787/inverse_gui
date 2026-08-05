"""Artifact loading, exercised against real files written by the fake optimizer."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from inverse_gui.artifacts.loader import ArtifactError, find_artifacts, load_run
from inverse_gui.artifacts.model import Criterion

REPO = Path(__file__).resolve().parent.parent
FAKE = REPO / 'scripts' / 'fake_optimizer.py'


def run_fake(out: Path, *extra: str) -> None:
    subprocess.run(
        [sys.executable, str(FAKE), '--ckpt_elastic_fm', 'f.pt',
         '--E', 'target 200000', '--nu', 'range 0.2 0.3',
         '--output_dir', str(out), *extra],
        check=True, capture_output=True,
        env={**os.environ, 'FAKE_ITERS': '4', 'FAKE_ITER_DELAY': '0'},
    )


@pytest.fixture(scope='module')
def single_dir(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp('single')
    run_fake(d)
    return d


@pytest.fixture(scope='module')
def pareto_dir(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp('pareto')
    run_fake(d, '--kappa', 'max', '--pareto_steps', '5')
    return d


# ------------------------------------------------------------------ single point

def test_single_point_loads(single_dir):
    ds = load_run(single_dir)
    assert ds is not None and ds.kind == 'single_point'
    assert len(ds) == 1
    d = ds.designs[0]
    assert d.mask is not None and d.mask.shape == (128, 128)
    assert 'E' in d.props


def test_single_point_recovers_criteria(single_dir):
    ds = load_run(single_dir)
    by_prop = {c.prop: c for c in ds.criteria}
    assert by_prop['E'].target == 200000.0
    assert (by_prop['nu'].lo, by_prop['nu'].hi) == (0.2, 0.3)


def test_single_point_has_convergence_history(single_dir):
    ds = load_run(single_dir)
    assert ds.loss_hist is not None and len(ds.loss_hist) > 0
    assert 'E' in ds.designs[0].prop_history


# ------------------------------------------------------------------ pareto

def test_pareto_loads_all_designs(pareto_dir):
    ds = load_run(pareto_dir)
    assert ds.kind == 'pareto'
    assert len(ds) == 5
    assert all(d.mask is not None and d.mask.shape == (128, 128) for d in ds.designs)


def test_pareto_design_index_matches_position(pareto_dir):
    """The plot puts design.index in customdata; it must index back correctly."""
    ds = load_run(pareto_dir)
    for i, d in enumerate(ds.designs):
        assert d.index == i


def test_pareto_carries_rank_and_feasibility(pareto_dir):
    ds = load_run(pareto_dir)
    summary = ds.summary()
    assert summary['total'] == 5
    assert 0 <= summary['front'] <= 5


def test_pareto_axes_default_to_rho_vs_other(pareto_dir):
    ds = load_run(pareto_dir)
    x, y = ds.default_axes()
    assert x == 'rho' and y != 'rho'


def test_nan_padded_props_are_dropped(tmp_path):
    """props is NaN-padded upstream; a NaN must be absent, not plotted as 0."""
    np.savez(tmp_path / 'pareto_results.npz',
             prop_names=np.array(['E', 'kappa']),
             props=np.array([[1.0, np.nan], [2.0, 3.0]]),
             rho_cost=np.array([0.4, 0.6]),
             microstructures=np.zeros((2, 128, 128), dtype=np.uint8),
             pareto_rank=np.array([0, 1]))
    ds = load_run(tmp_path)
    assert 'kappa' not in ds.designs[0].props
    assert ds.designs[1].props['kappa'] == 3.0


# ------------------------------------------------------------------ misc

def test_empty_dir_returns_none(tmp_path):
    assert load_run(tmp_path) is None


def test_find_artifacts_reports_what_exists(single_dir):
    found = find_artifacts(single_dir)
    assert 'single' in found and 'pareto' not in found


def test_corrupt_npz_raises_artifact_error(tmp_path):
    (tmp_path / 'pareto_results.npz').write_bytes(b'not an npz')
    with pytest.raises(ArtifactError):
        load_run(tmp_path)


# ------------------------------------------------------------------ criteria

def test_criterion_target_band_and_error():
    c = Criterion(prop='E', mode='range', target=200000.0)
    met, err = c.check(202000.0, target_tol=0.02)
    assert met is True and abs(err - 0.01) < 1e-9


def test_criterion_outside_band_fails():
    c = Criterion(prop='E', mode='range', target=200000.0)
    met, err = c.check(250000.0, target_tol=0.02)
    assert met is False and abs(err - 0.25) < 1e-9


def test_criterion_range_uses_explicit_bounds():
    c = Criterion(prop='nu', mode='range', lo=0.2, hi=0.3)
    assert c.check(0.25)[0] is True
    assert c.check(0.35)[0] is False

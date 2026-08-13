"""FEniCS ground-truth results: conversion, discovery, and attachment.

The conversion is a port of upstream's `_fenics_aniso_props`, so the tests pin it
against a *known inverse* rather than a recorded output: build C_hom from a plane
stress (E, nu) and require the reader to recover exactly those numbers. A tidier
but different inverse would then fail here instead of silently disagreeing with the
table the CLI prints.

Discovery is tested against all three real layouts (flat, per-restart, Pareto),
because each one comes from a different branch upstream and only the flat one is
what you get on a casual first run.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from inverse_gui.artifacts import fenics as fx
from inverse_gui.artifacts.loader import load_run
from inverse_gui.artifacts.model import Design, DesignSet

REPO = Path(__file__).resolve().parent.parent
FAKE = REPO / 'scripts' / 'fake_optimizer.py'


# ------------------------------------------------------------------ conversion

def plane_stress(E: float, nu: float) -> np.ndarray:
    c = E / (1 - nu ** 2)
    return np.array([[c, c * nu, 0.0], [c * nu, c, 0.0], [0.0, 0.0, c * (1 - nu) / 2]])


def guarded(E: float) -> float:
    """E as upstream reports it: 1/(S + 1e-8), not 1/S.

    The guard is negligible for compliant materials and NOT negligible for stiff
    ones -- at 210 GPa, 1/E is 4.76e-6 and the epsilon is 0.2% of it. The port keeps
    the guard on purpose, so this is the number the panel must show to agree with
    the CLI's own comparison table.
    """
    return 1.0 / (1.0 / E + 1e-8)


def test_elastic_conversion_recovers_the_moduli_it_was_built_from():
    props = fx.props_from_arrays(C_hom=plane_stress(210000.0, 0.3))
    assert props['E_xx'] == pytest.approx(guarded(210000.0), rel=1e-6)
    assert props['E_yy'] == pytest.approx(guarded(210000.0), rel=1e-6)
    # nu is scaled by the guarded E_xx, so it inherits exactly the same bias.
    assert props['nu_xy'] == pytest.approx(0.3 * guarded(210000.0) / 210000.0,
                                           rel=1e-6)
    assert props['nu_yx'] == pytest.approx(props['nu_xy'], rel=1e-9)
    # Plane-stress shear modulus, the one value that is not simply read back.
    assert props['G_xy'] == pytest.approx(guarded(210000.0 / (2 * 1.3)), rel=1e-6)
    assert props['E'] == pytest.approx(0.5 * (props['E_xx'] + props['E_yy']))


def test_the_upstream_compliance_guard_is_preserved():
    """A cleaner inverse would read ~0.2% higher and silently disagree with the CLI."""
    props = fx.props_from_arrays(C_hom=plane_stress(210000.0, 0.3))
    bias = (210000.0 - props['E_xx']) / 210000.0
    assert 0.001 < bias < 0.003, 'the 1e-8 guard was dropped or changed'


def test_anisotropic_input_is_not_averaged_away():
    C = plane_stress(210000.0, 0.3)
    C[1, 1] *= 0.5                                  # soft in y
    props = fx.props_from_arrays(C_hom=C)
    assert props['E_yy'] < props['E_xx']
    assert props['E'] == pytest.approx(0.5 * (props['E_xx'] + props['E_yy']))


def test_conductivity_and_expansion_conversions():
    props = fx.props_from_arrays(kappa_hom=np.diag([160.0, 140.0]),
                                 hom_strain=np.array([1.2e-5, 1.0e-5, 3e-8]))
    assert props['kappa_x'] == 160.0 and props['kappa_y'] == 140.0
    assert props['kappa'] == pytest.approx(150.0)
    assert props['alpha_xx'] == pytest.approx(1.2e-5)
    assert props['alpha_xy'] == pytest.approx(3e-8)
    assert props['alpha'] == pytest.approx(1.1e-5)


def test_nothing_in_nothing_out():
    assert fx.props_from_arrays() == {}


def test_every_value_is_a_plain_float():
    """np.float64 leaks into st.dataframe and JSON badly; keep the boundary clean."""
    props = fx.props_from_arrays(C_hom=plane_stress(2e5, 0.25),
                                 kappa_hom=np.diag([1.0, 2.0]))
    assert all(type(v) is float for v in props.values())


# ------------------------------------------------------------------ discovery

def write_tree(root: Path, physics: str, **arrays) -> None:
    d = root / physics
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / fx.result_name(physics), **arrays)


def test_flat_layout_is_read_as_the_best_design(tmp_path):
    write_tree(tmp_path / 'fenics', 'elastic', C_hom=plane_stress(2e5, 0.3))
    got = fx.load_single_point(tmp_path)
    assert set(got) == {'best'}
    assert got['best']['E_xx'] == pytest.approx(guarded(2e5), rel=1e-6)


def test_the_three_physics_merge_into_one_dict(tmp_path):
    root = tmp_path / 'fenics'
    write_tree(root, 'elastic', C_hom=plane_stress(2e5, 0.3))
    write_tree(root, 'thermal_conductivity', kappa_hom=np.diag([150.0, 150.0]))
    write_tree(root, 'thermal_expansion',
               homogenized_strain=np.array([1e-5, 1e-5, 0.0]))
    props = fx.load_tree(root)
    assert {'E', 'kappa', 'alpha'} <= set(props)


def test_a_physics_that_failed_is_skipped_not_fatal(tmp_path):
    """Upstream prints and continues when a solver fails, leaving no npz."""
    root = tmp_path / 'fenics'
    write_tree(root, 'elastic', C_hom=plane_stress(2e5, 0.3))
    (root / 'thermal_conductivity').mkdir(parents=True)      # started, wrote nothing
    props = fx.load_tree(root)
    assert 'E' in props and 'kappa' not in props


def test_a_truncated_npz_is_skipped_not_raised(tmp_path):
    root = tmp_path / 'fenics'
    (root / 'elastic').mkdir(parents=True)
    (root / 'elastic' / fx.result_name('elastic')).write_bytes(b'PK\x03\x04 garbage')
    assert fx.load_tree(root) == {}


def test_missing_tree_is_empty_not_an_error(tmp_path):
    assert fx.load_single_point(tmp_path) == {}
    assert fx.load_pareto(tmp_path) == {}


# ------------------------------------------------------------------ attachment

def single_set(**kw) -> DesignSet:
    designs = [Design(index=0, label='best', props={'E': 1.9e5}, final_loss=0.5),
               Design(index=1, label='restart 0', props={'E': 1.8e5}, final_loss=0.9),
               Design(index=2, label='restart 1', props={'E': 1.9e5}, final_loss=0.5)]
    return DesignSet(kind='single_point', prop_names=['E'], designs=designs, **kw)


def test_restart_trees_attach_to_their_own_designs(tmp_path):
    root = tmp_path / 'fenics'
    write_tree(root / 'restart0', 'elastic', C_hom=plane_stress(1.7e5, 0.3))
    write_tree(root / 'restart1', 'elastic', C_hom=plane_stress(1.95e5, 0.3))
    ds = single_set()
    fx.attach(ds, tmp_path)

    assert ds.designs[1].fenics_props['E_xx'] == pytest.approx(guarded(1.7e5), rel=1e-6)
    assert ds.designs[2].fenics_props['E_xx'] == pytest.approx(guarded(1.95e5), rel=1e-6)


def test_the_best_design_inherits_from_its_matching_restart(tmp_path):
    """With restarts > 1 upstream writes no flat tree; best == the restart that won.

    Without this the design a user clicks first shows an empty FEniCS column even
    though validation ran on exactly that mask.
    """
    root = tmp_path / 'fenics'
    write_tree(root / 'restart1', 'elastic', C_hom=plane_stress(1.95e5, 0.3))
    ds = single_set()
    fx.attach(ds, tmp_path)

    assert ds.designs[0].has_fenics                       # final_loss 0.5 == restart 1
    assert ds.designs[0].fenics_props == ds.designs[2].fenics_props
    assert not ds.designs[1].has_fenics                   # loss 0.9 lost; untouched


def pareto_set(n=4) -> DesignSet:
    designs = [Design(index=i, label=f'#{i}', props={'rho': 0.1 * i},
                      rank=0 if i % 2 == 0 else 1) for i in range(n)]
    return DesignSet(kind='pareto', prop_names=['rho'], designs=designs)


def test_pareto_summary_is_keyed_by_solution_index(tmp_path):
    """solution_idx indexes the whole result set, not the rank-0 subset."""
    summary = [{'solution_idx': 2, 'fenics': {'E': 2.05e5}, 'nn': {'E': 2.0e5}}]
    np.savez_compressed(tmp_path / fx.SUMMARY_NAME,
                        fenics_summary=np.array(json.dumps(summary), dtype=object))
    ds = pareto_set()
    assert fx.attach(ds, tmp_path) == 1
    assert ds.designs[2].fenics_props == {'E': 2.05e5}
    assert not ds.designs[0].has_fenics


def test_pareto_falls_back_to_rank0_directories(tmp_path):
    """No summary npz: rank0_<i> is the i-th rank-0 design, not design i."""
    root = tmp_path / 'fenics'
    write_tree(root / 'rank0_000', 'elastic', C_hom=plane_stress(2.0e5, 0.3))
    write_tree(root / 'rank0_001', 'elastic', C_hom=plane_stress(2.2e5, 0.3))
    ds = pareto_set()
    fx.attach(ds, tmp_path)

    assert ds.designs[0].fenics_props['E_xx'] == pytest.approx(guarded(2.0e5), rel=1e-6)
    assert ds.designs[2].fenics_props['E_xx'] == pytest.approx(guarded(2.2e5), rel=1e-6)
    assert not ds.designs[1].has_fenics and not ds.designs[3].has_fenics


def test_a_corrupt_summary_falls_back_instead_of_losing_everything(tmp_path):
    (tmp_path / fx.SUMMARY_NAME).write_bytes(b'not an npz')
    write_tree(tmp_path / 'fenics' / 'rank0_000', 'elastic',
               C_hom=plane_stress(2.0e5, 0.3))
    ds = pareto_set()
    assert fx.attach(ds, tmp_path) == 1


def test_attach_is_a_no_op_without_results(tmp_path):
    ds = single_set()
    assert fx.attach(ds, tmp_path) == 0
    assert not ds.has_fenics


# ------------------------------------------------- end to end via the fake CLI

def run_fake(out: Path, *extra: str) -> None:
    subprocess.run(
        [sys.executable, str(FAKE), '--ckpt_elastic_fm', 'f.pt',
         '--ckpt_thermal_conductivity_fm', 'f.pt',
         '--E', 'target 200000', '--nu', 'target 0.25', '--kappa', 'target 150',
         '--output_dir', str(out), '--fenics_validate', *extra],
        check=True, capture_output=True,
        env={**os.environ, 'FAKE_ITERS': '3', 'FAKE_ITER_DELAY': '0'},
    )


def test_single_point_run_loads_with_ground_truth(tmp_path):
    run_fake(tmp_path)
    ds = load_run(tmp_path)
    best = ds.designs[0]
    assert ds.has_fenics and best.has_fenics
    # Same physics the checkpoints enabled, and close to the surrogate's own answer.
    assert {'E', 'nu', 'kappa'} <= set(best.fenics_props)
    assert best.fenics_props['E'] == pytest.approx(best.get('E'), rel=0.2)


def test_pareto_run_validates_only_the_front(tmp_path):
    run_fake(tmp_path, '--pareto_steps', '4')
    ds = load_run(tmp_path)
    validated = [d for d in ds.designs if d.has_fenics]
    assert validated, 'no rank-0 solution picked up its FEniCS results'
    assert all(d.rank == 0 for d in validated)


def test_validation_off_leaves_no_fenics_anywhere(tmp_path):
    subprocess.run(
        [sys.executable, str(FAKE), '--ckpt_elastic_fm', 'f.pt',
         '--E', 'target 200000', '--output_dir', str(tmp_path)],
        check=True, capture_output=True,
        env={**os.environ, 'FAKE_ITERS': '3', 'FAKE_ITER_DELAY': '0'},
    )
    ds = load_run(tmp_path)
    assert not ds.has_fenics

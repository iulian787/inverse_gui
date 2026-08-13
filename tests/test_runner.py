"""SubprocessRunner end-to-end against scripts/fake_optimizer.py.

Covers the two paths that matter operationally: the live one, and the degraded one
the app falls back to after a hot-reload or a Streamlit restart.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pytest

from inverse_gui import config as config_mod
from inverse_gui.domain.directives import Directive, Mode
from inverse_gui.domain.schema import Checkpoints, RunConfig, RunMode
from inverse_gui.execution import runstore
from inverse_gui.execution.ptyproc import pgid_alive
from inverse_gui.execution.registry import Registry
from inverse_gui.execution.runner import SubprocessRunner

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture
def runner(tmp_path):
    cfg = config_mod.Config()
    cfg.paths.ai4ns_repo = str(REPO)
    cfg.paths.runs_dir = str(tmp_path / 'runs')
    cfg.scripts.single_point = 'scripts/fake_optimizer.py'
    cfg.scripts.pareto = 'scripts/fake_optimizer.py'
    cfg.solver.python = sys.executable          # the GUI venv can run the fake
    cfg.solver.activation_json = 'env/does-not-exist.json'
    cfg.runner.term_grace_seconds = 3.0
    return SubprocessRunner(cfg, Registry(kill_on_exit=False))


def make_cfg(mode=RunMode.SINGLE, **kw) -> RunConfig:
    c = RunConfig.for_mode(mode, **kw)
    c.checkpoints = Checkpoints(elastic='fake.pt')
    c.directives = {'E': Directive('E', Mode.TARGET, value=200000.0)}
    return c


def wait_terminal(runner, run_id, timeout=90):
    deadline = time.time() + timeout
    while time.time() < deadline:
        snap = runner.snapshot(run_id)
        if snap and snap.is_terminal:
            return snap
        time.sleep(0.2)
    pytest.fail(f'run {run_id} did not finish within {timeout}s')


def test_single_point_run_completes_and_writes_artifacts(runner):
    cfg = make_cfg()
    cfg.ipopt.print_level = 5
    run_id = runner.submit(cfg)
    snap = wait_terminal(runner, run_id)

    assert snap.state == runstore.STATE_DONE
    assert snap.returncode == 0
    assert snap.progress.done
    assert snap.progress.iters_total > 0

    art = runstore.artifact_dir(runner.runs_root, run_id)
    assert (art / 'inverse_result_fm_multi_ac.npz').exists()


def test_run_dir_contains_reproduction_files(runner):
    run_id = runner.submit(make_cfg())
    wait_terminal(runner, run_id)
    d = runstore.run_dir(runner.runs_root, run_id)
    for name in ('command.txt', 'run.sh', 'config.json', 'stdout.log', 'status.json'):
        assert (d / name).exists(), f'missing {name}'
    assert 'fake_optimizer.py' in (d / 'command.txt').read_text()


def test_output_dir_is_redirected_into_the_run(runner):
    """Whatever the form says, artifacts must land in this run's directory."""
    cfg = make_cfg()
    cfg.output_dir = '/tmp/should-not-be-used'
    run_id = runner.submit(cfg)
    wait_terminal(runner, run_id)
    st = runstore.read_status(runner.runs_root, run_id)
    assert st.artifact_dir.endswith(f'{run_id}/artifacts')
    assert '/tmp/should-not-be-used' not in ' '.join(st.argv)


def test_fenics_output_dir_is_redirected_too(runner):
    """Validation results must not escape the run directory either.

    Upstream defaults --fenics_output_dir to <output_dir>/fenics, which is already
    inside the run -- but an explicit value in section F would send ground truth to
    a shared path, where the results panel cannot find it and two runs overwrite
    each other's numbers.
    """
    cfg = make_cfg()
    cfg.fenics.validate = True
    cfg.fenics.output_dir = '/tmp/shared-fenics'
    run_id = runner.submit(cfg)
    wait_terminal(runner, run_id)

    argv = ' '.join(runstore.read_status(runner.runs_root, run_id).argv)
    assert '/tmp/shared-fenics' not in argv
    assert f'{run_id}/artifacts/fenics' in argv


def test_fenics_results_land_where_the_loader_looks(runner):
    """End to end: validation on, then the design set comes back with ground truth."""
    from inverse_gui.artifacts.loader import load_run
    cfg = make_cfg()
    cfg.fenics.validate = True
    run_id = runner.submit(cfg)
    wait_terminal(runner, run_id)

    ds = load_run(runstore.artifact_dir(runner.runs_root, run_id))
    assert ds is not None and ds.has_fenics
    assert ds.designs[0].fenics_props


def test_pareto_run_writes_pareto_npz(runner):
    cfg = make_cfg(RunMode.PARETO, pareto_steps=4)
    cfg.directives['kappa'] = Directive('kappa', Mode.MAX)
    run_id = runner.submit(cfg)
    snap = wait_terminal(runner, run_id)
    assert snap.state == runstore.STATE_DONE
    art = runstore.artifact_dir(runner.runs_root, run_id)
    assert (art / 'pareto_results.npz').exists()
    assert snap.progress.stage == 3


def test_nonzero_exit_is_reported_as_failed(runner):
    """No checkpoint => upstream parser.error => exit 2, which must surface as failed."""
    cfg = make_cfg()
    cfg.checkpoints = Checkpoints()
    run_id = runner.submit(cfg)
    snap = wait_terminal(runner, run_id)
    assert snap.state == runstore.STATE_FAILED
    assert snap.returncode == 2


def test_cancel_leaves_no_survivors(runner, monkeypatch):
    # Make the child long-lived, or it finishes before we can cancel it.
    monkeypatch.setenv('FAKE_ITERS', '2000')
    monkeypatch.setenv('FAKE_ITER_DELAY', '0.5')

    run_id = runner.submit(make_cfg())
    st = runstore.read_status(runner.runs_root, run_id)
    time.sleep(1.0)
    assert pgid_alive(st.pgid)

    runner.cancel(run_id)
    snap = wait_terminal(runner, run_id, timeout=20)

    assert snap.state == runstore.STATE_CANCELLED
    assert not pgid_alive(st.pgid)


def test_degraded_snapshot_after_registry_loss(runner):
    """Simulates a hot-reload: the in-memory handle is gone, the run is not."""
    run_id = runner.submit(make_cfg())
    wait_terminal(runner, run_id)
    live = runner.snapshot(run_id)

    runner.registry.drop(run_id)                 # what module eviction does
    degraded = runner.snapshot(run_id)

    assert degraded is not None and degraded.degraded
    assert degraded.state == live.state
    # Progress replayed from the log must match what streaming produced.
    assert degraded.progress.done == live.progress.done
    assert degraded.progress.iters_total == live.progress.iters_total


def test_scan_live_marks_dead_runs_orphaned(runner, tmp_path):
    run_id = runner.submit(make_cfg())
    wait_terminal(runner, run_id)

    # Forge a status file claiming to be running under a pgid that cannot exist.
    st = runstore.read_status(runner.runs_root, run_id)
    st.state = runstore.STATE_RUNNING
    st.pgid = 999_999
    runstore.write_status(runner.runs_root, st)

    assert runstore.scan_live(runner.runs_root) == []
    assert runstore.read_status(runner.runs_root, run_id).state == runstore.STATE_ORPHANED


def test_snapshot_of_unknown_run_is_none(runner):
    assert runner.snapshot('run_does_not_exist') is None


def test_plan_does_not_launch_anything(runner):
    argv, interpreter, env = runner.plan(make_cfg())
    assert argv[0] == interpreter and argv[1] == '-u'
    assert argv[2].endswith('fake_optimizer.py')
    assert env['MPLBACKEND'] == 'Agg'
    assert not subprocess.run(['pgrep', '-f', 'fake_optimizer.py --ckpt'],
                              capture_output=True).stdout

"""fenics_env as a conda environment: detection, usability, and reachability.

Three separate things have to hold before the validation toggle may be enabled, and
each one has failed in a different way:

1. conda must be able to *resolve the name* -- upstream hardcodes
   `conda run -n fenics_env`, which only finds envs whose directory basename sits in
   a registered envs_dir.
2. the env must actually *import the validation package*. environment_fenics.yml
   omits dolfinx_mpc and pandas, so a by-the-book env satisfies (1) and still dies at
   import. That matters more here than elsewhere: upstream calls validation with no
   try/except, after the solve and before artifacts are written.
3. conda must be reachable *from the child's PATH*, because the optimizer -- not the
   GUI -- is what shells out to it, and the GUI strips its own venv from that PATH.

The unit tests below fake `conda env list` so they run anywhere. The integration
tests drive the real env and skip when it is absent.
"""

from __future__ import annotations

import functools
import shutil
import subprocess
import types
from pathlib import Path

import pytest

from inverse_gui import config as config_mod
from inverse_gui.execution import doctor
from inverse_gui.execution import env as env_mod

REPO = Path(__file__).resolve().parent.parent

ENV_LIST = """\
# conda environments:
#
base                  *  /home/u/miniconda3
cenv                     /big/disk/envs/cenv
fenics_env               /big/disk/envs/fenics_env
"""


# --------------------------------------------------------------------- helpers

def _cfg(**fenics):
    cfg = config_mod.Config()
    cfg.paths.ai4ns_repo = str(REPO)          # has no fenics_validation/
    for k, v in fenics.items():
        setattr(cfg.fenics, k, v)
    return cfg


def _fake_run(monkeypatch, *, env_list=ENV_LIST, probe=None, calls=None):
    """Stub subprocess.run for both `conda env list` and the `conda run` probe."""
    def run(cmd, **kw):
        if calls is not None:
            calls.append(cmd)
        if cmd[1:3] == ['env', 'list']:
            return types.SimpleNamespace(returncode=0, stdout=env_list, stderr='')
        if probe is None:
            raise AssertionError(f'unexpected subprocess call: {cmd}')
        if isinstance(probe, Exception):
            raise probe
        return probe
    monkeypatch.setattr(doctor.env_mod, 'find_conda', lambda *a, **k: '/fake/conda')
    monkeypatch.setattr(doctor.subprocess, 'run', run)


def _probe_result(returncode, stderr=''):
    return types.SimpleNamespace(returncode=returncode, stdout='', stderr=stderr)


# The traceback conda actually produces, epilogue included -- the last line names
# conda, not the missing module, which is the whole reason _last_meaningful exists.
MPC_STDERR = """\
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "/x/amit_AI4NS/fenics_validation/__init__.py", line 3, in <module>
    from .mesh import create_pixel_mesh
  File "/x/amit_AI4NS/fenics_validation/mesh.py", line 7, in <module>
    import dolfinx_mpc
ModuleNotFoundError: No module named 'dolfinx_mpc'
ERROR conda.cli.main_run:execute(125): `conda run python -c import \
fenics_validation.validate` failed. (See above for error)
"""


# ------------------------------------------------------- 1. resolving the name

def test_the_named_env_is_found(monkeypatch):
    _fake_run(monkeypatch)
    assert doctor.fenics_env_exists(_cfg())


def test_a_longer_name_is_not_a_match(monkeypatch):
    """fenics_env_old must not satisfy a request for fenics_env."""
    _fake_run(monkeypatch, env_list='fenics_env_old   /big/disk/envs/fenics_env_old\n')
    assert not doctor.fenics_env_exists(_cfg())


def test_an_env_known_only_by_location_counts(monkeypatch):
    """conda run -n resolves by directory basename, so the bare-prefix form counts."""
    _fake_run(monkeypatch, env_list='                   /big/disk/envs/fenics_env\n')
    assert doctor.fenics_env_exists(_cfg())


def test_the_active_marker_does_not_break_parsing(monkeypatch):
    _fake_run(monkeypatch, env_list='fenics_env  *  /big/disk/envs/fenics_env\n')
    assert doctor.fenics_env_exists(_cfg())


def test_no_conda_is_a_missing_env_not_a_crash(monkeypatch):
    monkeypatch.setattr(doctor.env_mod, 'find_conda', lambda *a, **k: None)
    assert not doctor.fenics_env_exists(_cfg())


def test_a_hanging_conda_is_a_missing_env_not_a_crash(monkeypatch):
    monkeypatch.setattr(doctor.env_mod, 'find_conda', lambda *a, **k: '/fake/conda')

    def hang(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, 30)

    monkeypatch.setattr(doctor.subprocess, 'run', hang)
    assert not doctor.fenics_env_exists(_cfg())


# ------------------------------------------------------------ 2. usability

def test_a_present_but_unimportable_env_fails_the_check(monkeypatch, tmp_path):
    (tmp_path / 'fenics_validation').mkdir()
    cfg = _cfg()
    cfg.paths.ai4ns_repo = str(tmp_path)
    _fake_run(monkeypatch, probe=_probe_result(1, MPC_STDERR))

    c = doctor.check_fenics_env(cfg)
    assert not c.ok
    # The cause, not conda's epilogue.
    assert "No module named 'dolfinx_mpc'" in c.detail
    assert 'ERROR conda' not in c.detail
    # A missing optional capability, not a blocker: the run itself is still fine.
    assert not c.critical
    assert doctor.blocking([c]) == []


def test_the_remedy_names_the_install_command(monkeypatch, tmp_path):
    (tmp_path / 'fenics_validation').mkdir()
    cfg = _cfg(conda_env='fe')                   # the name is configurable...
    cfg.paths.ai4ns_repo = str(tmp_path)
    _fake_run(monkeypatch, env_list='fe   /big/disk/envs/fe\n',
              probe=_probe_result(1, MPC_STDERR))

    remedy = doctor.check_fenics_env(cfg).remedy
    # ...so the remedy must be copy-pasteable, not hardcoded to fenics_env.
    assert 'conda install -n fe' in remedy and 'dolfinx_mpc' in remedy


def test_an_importable_env_passes(monkeypatch, tmp_path):
    (tmp_path / 'fenics_validation').mkdir()
    cfg = _cfg()
    cfg.paths.ai4ns_repo = str(tmp_path)
    _fake_run(monkeypatch, probe=_probe_result(0))

    c = doctor.check_fenics_env(cfg)
    assert c.ok and 'fenics_validation' in c.detail


def test_a_hanging_probe_is_reported_not_raised(monkeypatch, tmp_path):
    (tmp_path / 'fenics_validation').mkdir()
    cfg = _cfg()
    cfg.paths.ai4ns_repo = str(tmp_path)
    _fake_run(monkeypatch, probe=subprocess.TimeoutExpired('conda', 120))

    c = doctor.check_fenics_env(cfg)
    assert not c.ok and not c.critical


def test_a_repo_without_the_package_does_not_spawn_conda(monkeypatch):
    """Nothing to import means nothing to probe; say so instead of shelling out."""
    calls: list = []
    _fake_run(monkeypatch, calls=calls)          # probe=None => any run() asserts
    c = doctor.check_fenics_env(_cfg())          # REPO has no fenics_validation/
    assert not c.ok and 'fenics_validation' in c.detail
    assert [c_ for c_ in calls if c_[1:3] != ['env', 'list']] == []


def test_probe_imports_false_skips_the_conda_run(monkeypatch):
    """run_all's cheap mode must stay cheap -- the pane relies on it."""
    calls: list = []
    _fake_run(monkeypatch, calls=calls)
    names = {c.name: c for c in doctor.run_all(_cfg(), probe_imports=False)}
    assert names['FEniCS env'].ok
    assert 'not probed' in names['FEniCS env'].detail
    assert all(c[1:3] == ['env', 'list'] for c in calls)


# ------------------------------------------------- 3. reachable from the child

def test_the_child_gets_conda_on_its_path(tmp_path):
    """The optimizer, not the GUI, runs `conda run -n fenics_env`.

    env.build strips the GUI venv from PATH; if it did not also prepend conda's
    bindir, upstream would raise FileNotFoundError after the solve.
    """
    conda = env_mod.find_conda()
    if not conda:
        pytest.skip('conda not installed')
    cfg = config_mod.Config()
    base = {'PATH': f'{tmp_path}/.venv/bin:/usr/bin', 'VIRTUAL_ENV': f'{tmp_path}/.venv'}
    se = env_mod.build(cfg, base_env=base, venv_dir=tmp_path / '.venv')

    assert shutil.which('conda', path=se.env['PATH'])
    assert 'VIRTUAL_ENV' not in se.env
    assert f'{tmp_path}/.venv/bin' not in se.env['PATH'].split(':')


# ------------------------------------------------------- integration: real env

@functools.lru_cache(maxsize=1)
def _real_cfg():
    return config_mod.load()


@functools.lru_cache(maxsize=1)
def _env_present() -> bool:
    return doctor.fenics_env_exists(_real_cfg())


@functools.lru_cache(maxsize=1)
def _imports(module: str) -> bool:
    """Does `module` import inside the real fenics_env?"""
    conda = env_mod.find_conda()
    if not conda or not _env_present():
        return False
    r = subprocess.run(
        [conda, 'run', '-n', _real_cfg().fenics.conda_env, '--no-capture-output',
         'python', '-c', f'import {module}'],
        capture_output=True, text=True, timeout=180,
        cwd=_real_cfg().ai4ns_repo,
    )
    return r.returncode == 0


needs_env = pytest.mark.skipif(not _env_present(),
                               reason='fenics_env not built (optional; ./scripts/setup.sh --fenics)')


@needs_env
def test_the_real_env_is_a_working_dolfinx_env():
    """The env's reason for existing: dolfinx + its PETSc/MPI stack import."""
    assert _imports('dolfinx'), 'fenics_env exists but dolfinx does not import'
    assert _imports('petsc4py.PETSc')
    assert _imports('mpi4py.MPI')


@needs_env
def test_upstream_can_reach_the_real_env_through_the_child_path():
    """The exact call shape from run_inverse_design_fm_multi_ac.py:283-286.

    Bare 'conda', resolved from the child's PATH, not an absolute path.
    """
    cfg = _real_cfg()
    se = env_mod.build(cfg)
    r = subprocess.run(
        ['conda', 'run', '-n', cfg.fenics.conda_env, '--no-capture-output',
         'python', '-c', 'print("reached")'],
        capture_output=True, text=True, timeout=180, env=se.env,
    )
    assert r.returncode == 0, r.stderr
    assert 'reached' in r.stdout


@needs_env
@pytest.mark.skipif(not _imports('dolfinx_mpc'),
                    reason='dolfinx_mpc missing -- see the companion test')
def test_the_doctor_passes_a_complete_env():
    c = doctor.check_fenics_env(_real_cfg())
    assert c.ok, c.detail


@needs_env
@pytest.mark.skipif(_imports('dolfinx_mpc'),
                    reason='dolfinx_mpc present -- see the companion test')
def test_the_doctor_catches_the_incomplete_upstream_env():
    """environment_fenics.yml omits dolfinx_mpc, so a by-the-book env fails here.

    This is the false green the name-only check used to produce: env resolves,
    toggle goes live, and the run dies after the solve. Delete this test the day
    upstream fixes the yml -- its companion above takes over automatically.
    """
    c = doctor.check_fenics_env(_real_cfg())
    assert not c.ok
    assert 'dolfinx_mpc' in c.detail
    assert 'conda install' in c.remedy
    assert not c.critical, 'a broken optional env must warn, never block a run'

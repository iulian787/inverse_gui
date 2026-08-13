"""Full loop through the UI: fill the form, launch, wait, plot.

Exercises the seam the unit tests cannot: the app's own runner singleton actually
spawning the fake optimizer, and the design-space pane loading what it wrote.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

REPO = Path(__file__).resolve().parent.parent
APP = REPO / 'app.py'


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv('INVERSE_GUI_PATHS_RUNS_DIR', str(tmp_path / 'runs'))
    monkeypatch.setenv('INVERSE_GUI_PATHS_AI4NS_REPO', str(REPO))
    monkeypatch.setenv('INVERSE_GUI_SCRIPTS_SINGLE_POINT', 'scripts/fake_optimizer.py')
    monkeypatch.setenv('INVERSE_GUI_SCRIPTS_PARETO', 'scripts/fake_optimizer.py')
    monkeypatch.setenv('FAKE_ITERS', '5')
    monkeypatch.setenv('FAKE_ITER_DELAY', '0.05')
    # The form checks that checkpoint paths exist, so give it a real file.
    (tmp_path / 'elastic.pt').write_bytes(b'not really a checkpoint')
    return tmp_path


def fresh_app() -> AppTest:
    # Clear the cached singletons so each test gets its own runner and config.
    import streamlit as st
    st.cache_resource.clear()
    st.cache_data.clear()
    return AppTest.from_file(str(APP), default_timeout=60)


def launch(at: AppTest, ckpt: Path) -> str:
    """Fill in the minimum valid config, then click Launch."""
    at.text_input(key='ckpt.elastic').set_value(str(ckpt)).run()
    # A run with no active directive is rejected: upstream would print
    # "No active property directives" and exit without writing anything.
    at.selectbox(key='dir.E.mode').set_value('target').run()

    launch_btn = next(b for b in at.button if 'Launch' in b.label)
    assert not launch_btn.disabled, (
        'Launch blocked; errors: '
        + '; '.join(e.value for e in at.error)
    )
    launch_btn.click().run()
    run_id = at.session_state['active_run_id'] if 'active_run_id' in at.session_state else None
    assert run_id, 'no run id was recorded'
    return run_id


def wait_for(runs_dir: Path, run_id: str, timeout: float = 60) -> None:
    from inverse_gui.execution import runstore
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = runstore.read_status(runs_dir, run_id)
        if st and st.is_terminal:
            return
        time.sleep(0.2)
    pytest.fail(f'{run_id} did not finish')


def test_single_point_run_end_to_end(env):
    at = fresh_app()
    at.run()
    assert not at.exception

    run_id = launch(at, env / "elastic.pt")
    runs_dir = env / 'runs'
    wait_for(runs_dir, run_id)

    at.run()
    assert not at.exception, [str(e) for e in at.exception]

    from inverse_gui.execution import runstore
    status = runstore.read_status(runs_dir, run_id)
    assert status.state == runstore.STATE_DONE

    # The design space must have loaded the artifact and offered axis pickers.
    keys = {sb.key for sb in at.selectbox if sb.key}
    assert any(k and k.endswith('.x') for k in keys), f'no axis picker rendered: {keys}'


def test_design_space_detail_after_click(env):
    """Selection reads customdata, so index 0 must resolve to design 0."""
    at = fresh_app()
    at.run()
    run_id = launch(at, env / "elastic.pt")
    wait_for(env / 'runs', run_id)
    at.run()

    from inverse_gui.artifacts.loader import load_run
    from inverse_gui.execution import runstore
    ds = load_run(runstore.artifact_dir(env / 'runs', run_id))
    assert ds is not None and len(ds) >= 1
    assert ds.designs[0].mask is not None


def test_run_appears_in_history(env):
    at = fresh_app()
    at.run()
    run_id = launch(at, env / "elastic.pt")
    wait_for(env / 'runs', run_id)
    at.run()
    assert any(run_id in (c.value or '') for c in at.caption) or \
        any(run_id in (m.value or '') for m in at.markdown)


def test_pinned_designs_render_in_the_compare_panel(env):
    """Pins survive a rerun and resolve against artifacts on disk.

    The click itself cannot be simulated -- AppTest has no plotly selection -- so
    this drives the state the click produces and checks everything downstream of it.
    """
    from inverse_gui.ui.compare import PINS_KEY, Pin

    at = fresh_app()
    at.run()
    run_id = launch(at, env / 'elastic.pt')
    wait_for(env / 'runs', run_id)

    at.session_state[PINS_KEY] = [Pin(run_id, 0)]
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    assert at.session_state[PINS_KEY], 'the pin was dropped on rerun'
    assert any('1 of' in (c.value or '') for c in at.caption), \
        'the compare panel did not report its pin count'


def test_a_pin_whose_run_vanished_is_reported_not_fatal(env):
    from inverse_gui.ui.compare import PINS_KEY, Pin

    at = fresh_app()
    at.session_state[PINS_KEY] = [Pin('run_20200101_000000_sp', 3)]
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    assert any('no longer available' in (c.value or '') for c in at.caption)


def test_reproduction_files_are_written(env):
    at = fresh_app()
    at.run()
    run_id = launch(at, env / "elastic.pt")
    wait_for(env / 'runs', run_id)
    d = env / 'runs' / run_id
    assert (d / 'run.sh').exists() and (d / 'command.txt').exists()
    assert 'fake_optimizer' in (d / 'command.txt').read_text()

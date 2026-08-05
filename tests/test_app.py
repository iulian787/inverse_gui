"""App-level smoke tests using Streamlit's own headless harness.

These catch render-time exceptions (bad widget args, wrong API signatures) that
import checks miss, without needing a browser.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

REPO = Path(__file__).resolve().parent.parent
APP = REPO / 'app.py'


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv('INVERSE_GUI_PATHS_RUNS_DIR', str(tmp_path / 'runs'))
    monkeypatch.setenv('INVERSE_GUI_PATHS_AI4NS_REPO', str(REPO))
    monkeypatch.setenv('INVERSE_GUI_SCRIPTS_SINGLE_POINT', 'scripts/fake_optimizer.py')
    monkeypatch.setenv('INVERSE_GUI_SCRIPTS_PARETO', 'scripts/fake_optimizer.py')
    at = AppTest.from_file(str(APP), default_timeout=60)
    return at


def test_app_renders_without_exception(app):
    app.run()
    assert not app.exception, [str(e) for e in app.exception]


def test_app_shows_the_three_panes(app):
    app.run()
    headers = [h.value for h in app.subheader]
    assert 'Input' in headers and 'Run' in headers and 'Design space' in headers


def test_launch_is_blocked_without_a_checkpoint(app):
    """NO_CHECKPOINT is an error, so Launch must be disabled."""
    app.run()
    launch = [b for b in app.button if 'Launch' in b.label]
    assert launch and launch[0].disabled


def test_error_explains_the_missing_checkpoint(app):
    app.run()
    text = ' '.join(e.value for e in app.error)
    assert 'checkpoint' in text.lower()


def test_setting_a_checkpoint_reveals_elastic_targets(app):
    """Physics gating is structural: rows exist only when the checkpoint is set."""
    app.run()
    before = {sb.key for sb in app.selectbox if sb.key}
    app.text_input(key='ckpt.elastic').set_value('fake.pt').run()
    after = {sb.key for sb in app.selectbox if sb.key}

    new = after - before
    assert 'dir.E.mode' in new, f'no elastic rows appeared: {new}'
    assert 'dir.G_xy.mode' in new
    # thermal properties must stay hidden without their own checkpoints
    assert not any(k.startswith('dir.kappa') or k.startswith('dir.alpha')
                   for k in after)


def test_removing_a_checkpoint_blocks_rather_than_discards(app):
    """Removing a checkpoint under a set directive must be loud, not silent.

    The directive is deliberately kept -- discarding the user's target without
    telling them would be worse -- but it is now ungated, which upstream turns into
    a constraint with g=0 and zero gradient. So Launch must be blocked.
    """
    from inverse_gui.domain import validate as V

    app.run()
    app.text_input(key='ckpt.elastic').set_value('fake.pt').run()
    app.selectbox(key='dir.E.mode').set_value('target').run()
    assert app.session_state['run_config'].directives['E'].active

    app.text_input(key='ckpt.elastic').set_value('').run()
    rc = app.session_state['run_config']
    assert rc.directives['E'].active, 'the target should be preserved, not dropped'

    codes = {i.code for i in V.validate(rc, check_files=False)}
    assert 'UNGATED_CONSTRAINT' in codes or 'NO_CHECKPOINT' in codes
    assert next(b for b in app.button if 'Launch' in b.label).disabled


def test_mode_toggle_switches_defaults(app):
    """target_tol is 0.001 single-point and 0.02 Pareto; the toggle must follow."""
    app.run()
    assert app.number_input(key='target_tol').value == pytest.approx(0.001)
    app.segmented_control(key='mode.toggle').set_value('Pareto (ε-constraint)').run()
    assert app.number_input(key='target_tol').value == pytest.approx(0.02)
    assert app.number_input(key='restarts').value == 3


def test_pareto_shows_the_cost_panel(app):
    app.run()
    app.text_input(key='ckpt.elastic').set_value('fake.pt').run()
    app.segmented_control(key='mode.toggle').set_value('Pareto (ε-constraint)').run()
    labels = [m.label for m in app.metric]
    assert 'total solves' in labels


def test_history_and_doctor_render(app):
    app.run()
    assert not app.exception
    captions = ' '.join(c.value for c in app.caption)
    assert 'No runs yet' in captions, captions[-300:]

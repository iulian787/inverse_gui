import os

import pytest


@pytest.fixture(autouse=True)
def no_local_checkpoints(monkeypatch):
    """Isolate the suite from the developer's config.toml.

    Section A of the form is seeded from [checkpoints], so on a machine that has the
    real .pt files configured the app opens with checkpoints already set -- and every
    test that asserts on the unconfigured state ("Launch is blocked", "setting a
    checkpoint reveals the elastic rows") fails for a reason that has nothing to do
    with the code. Env vars win over the file, so blanking them here pins the
    starting state regardless of the machine.
    """
    for fam in ('ELASTIC', 'THERMAL_CONDUCTIVITY', 'THERMAL_EXPANSION'):
        monkeypatch.setenv(f'INVERSE_GUI_CHECKPOINTS_{fam}', '')


@pytest.fixture(autouse=True)
def fast_fake_optimizer(monkeypatch):
    """Keep the fake optimizer quick everywhere except the timing test.

    Its own defaults (25 iterations at 1s) mirror a realistic solve, which makes
    the suite take minutes. Tests that assert on timing set these explicitly.
    """
    monkeypatch.setenv('FAKE_ITERS', os.environ.get('FAKE_ITERS', '4'))
    monkeypatch.setenv('FAKE_ITER_DELAY', os.environ.get('FAKE_ITER_DELAY', '0.02'))

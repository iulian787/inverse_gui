import os

import pytest


@pytest.fixture(autouse=True)
def fast_fake_optimizer(monkeypatch):
    """Keep the fake optimizer quick everywhere except the timing test.

    Its own defaults (25 iterations at 1s) mirror a realistic solve, which makes
    the suite take minutes. Tests that assert on timing set these explicitly.
    """
    monkeypatch.setenv('FAKE_ITERS', os.environ.get('FAKE_ITERS', '4'))
    monkeypatch.setenv('FAKE_ITER_DELAY', os.environ.get('FAKE_ITER_DELAY', '0.02'))

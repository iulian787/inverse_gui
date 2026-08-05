"""PTY streaming and cancellation.

test_lines_arrive_incrementally is the regression test for the whole PTY decision
(docs/environment.md, finding 2). It asserts *timing*, not just content, because a
pipe-based implementation passes every content assertion and then renders the live
log pane empty for the entire run.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from inverse_gui.execution.ptyproc import PtyProcess, pgid_alive

REPO = Path(__file__).resolve().parent.parent
FAKE = REPO / 'scripts' / 'fake_optimizer.py'


def fake_argv(*extra: str) -> list[str]:
    return [sys.executable, '-u', str(FAKE),
            '--ckpt_elastic_fm', 'f.pt', '--E', 'target 200000', *extra]


@pytest.fixture
def logfile(tmp_path) -> Path:
    return tmp_path / 'stdout.log'


def test_lines_arrive_incrementally(tmp_path, logfile):
    """Iteration rows must arrive as they are printed, not batched at exit.

    fake_optimizer prints through ctypes printf with no flush, exactly as IPOPT's
    C++ Journalist does, so this exercises the real buffering behaviour.
    """
    stamps: list[float] = []
    t0 = time.monotonic()

    def on_line(line: str) -> None:
        if line.strip()[:1].isdigit():
            stamps.append(time.monotonic() - t0)

    p = PtyProcess(
        fake_argv('--output_dir', str(tmp_path / 'out'),
                  '--iters', '4', '--iter_delay', '1.0'),
        cwd=REPO, env=dict(os.environ), log_path=logfile, on_line=on_line,
    )
    p.start()
    assert p.wait(timeout=60) == 0

    assert len(stamps) == 4, f'expected 4 iteration rows, got {stamps}'
    # The decisive assertion: the first row must land ~1s in, not at exit (~4s).
    assert stamps[0] < 2.0, f'first row arrived at {stamps[0]:.2f}s — output is batched'
    assert stamps[-1] - stamps[0] > 2.0, 'rows are not spread over time'


def test_log_file_is_written(tmp_path, logfile):
    p = PtyProcess(
        fake_argv('--output_dir', str(tmp_path / 'out'),
                  '--iters', '2', '--iter_delay', '0.05'),
        cwd=REPO, env=dict(os.environ), log_path=logfile,
    )
    p.start()
    p.wait(timeout=60)
    text = logfile.read_text()
    assert 'Ipopt version' in text and 'Done.' in text
    # Raw mode means no \r\n mangling in the captured log.
    assert '\r\n' not in text


def test_crash_surfaces_real_exit_code(tmp_path, logfile):
    p = PtyProcess(
        fake_argv('--output_dir', str(tmp_path / 'out'),
                  '--iters', '4', '--iter_delay', '0.02', '--crash'),
        cwd=REPO, env=dict(os.environ), log_path=logfile,
    )
    p.start()
    assert p.wait(timeout=60) == 3


def test_missing_checkpoint_is_argparse_exit_2(tmp_path, logfile):
    p = PtyProcess(
        [sys.executable, '-u', str(FAKE), '--output_dir', str(tmp_path / 'out')],
        cwd=REPO, env=dict(os.environ), log_path=logfile,
    )
    p.start()
    assert p.wait(timeout=60) == 2


def test_cancel_leaves_no_survivors(tmp_path, logfile):
    """--hang never exits on its own; killpg must take out the whole group."""
    p = PtyProcess(
        fake_argv('--output_dir', str(tmp_path / 'out'), '--hang'),
        cwd=REPO, env=dict(os.environ), log_path=logfile,
    )
    spawned = p.start()
    time.sleep(1.5)
    assert pgid_alive(spawned.pgid)

    p.cancel(grace=3.0)

    assert not p.running()
    assert not pgid_alive(spawned.pgid), 'process group survived cancel'
    survivors = subprocess.run(
        ['pgrep', '-f', 'fake_optimizer.py --ckpt'],
        capture_output=True, text=True,
    ).stdout.split()
    assert not survivors, f'orphaned processes: {survivors}'


def test_cancel_is_idempotent(tmp_path, logfile):
    p = PtyProcess(
        fake_argv('--output_dir', str(tmp_path / 'out'), '--hang'),
        cwd=REPO, env=dict(os.environ), log_path=logfile,
    )
    p.start()
    time.sleep(1.0)
    p.cancel(grace=2.0)
    p.cancel(grace=2.0)      # must not raise
    assert p.cancelled

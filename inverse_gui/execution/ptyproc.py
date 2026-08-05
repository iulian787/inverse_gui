"""Spawn a child on a PTY and drain it from a daemon thread.

Why a PTY rather than a pipe: the IPOPT iteration table is written by IPOPT's C++
Journalist to C-level stdout. `python -u` and PYTHONUNBUFFERED only affect CPython's
io layer, not glibc's FILE* buffering inside libipopt.so. Measured (docs/environment.md):
against a pipe, four rows emitted at 1s intervals all arrived together at exit;
against a PTY they arrived at 1s intervals.

The reader thread does nothing but read, append to a log file, and hand lines to a
callback. Everything expensive happens elsewhere. If this thread stalls, the tty
line-discipline buffer (4096 bytes) fills and the child blocks in write() — the
optimizer would appear to hang.

This module must never import streamlit: it runs off the script-run thread, where
st.session_state silently returns a process-global mock shared across all sessions.
"""

from __future__ import annotations

import codecs
import errno
import fcntl
import os
import select
import signal
import struct
import subprocess
import termios
import threading
import tty
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

READ_SIZE = 65536
SELECT_TIMEOUT = 0.2


@dataclass
class Spawned:
    proc: subprocess.Popen
    pid: int
    pgid: int


class PtyProcess:
    """A child process whose stdout+stderr are read from a pseudo-terminal.

    Lifecycle: start() -> (thread drains, invoking on_line) -> wait()/cancel().
    All public methods are safe to call from the Streamlit script thread; the
    reader thread only touches its own state and the callbacks.
    """

    def __init__(
        self,
        argv: list[str],
        *,
        cwd: str | Path,
        env: dict[str, str],
        log_path: str | Path,
        on_line: Callable[[str], None] | None = None,
        on_exit: Callable[[int], None] | None = None,
    ) -> None:
        self.argv = list(argv)
        self.cwd = str(cwd)
        self.env = dict(env)
        self.log_path = Path(log_path)
        self._on_line = on_line
        self._on_exit = on_exit

        self._proc: subprocess.Popen | None = None
        self._master: int | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._returncode: int | None = None
        self._cancelled = False
        self._finished = threading.Event()

    # ---------------------------------------------------------------- lifecycle

    def start(self) -> Spawned:
        master, slave = os.openpty()

        # Raw mode: disables ICANON (which imposes a 4095-byte line limit) and
        # ONLCR (which would turn every \n into \r\n in the captured log).
        tty.setraw(slave)
        # Default winsize is 0x0; libraries that consult it (rich, click, tqdm)
        # emit garbage or hard-wrap the IPOPT table.
        fcntl.ioctl(master, termios.TIOCSWINSZ, struct.pack('HHHH', 50, 200, 0, 0))

        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        proc = subprocess.Popen(
            self.argv,
            cwd=self.cwd,
            env=self.env,
            stdin=subprocess.DEVNULL,   # a child that reads stdin would hang forever
            stdout=slave,
            stderr=subprocess.STDOUT,
            close_fds=True,
            start_new_session=True,     # own process group => killpg works on the tree
        )
        # Must close our copy of the slave, or read() on the master never sees EOF.
        os.close(slave)

        self._proc = proc
        self._master = master
        self._thread = threading.Thread(
            target=self._drain, name=f'pty-{proc.pid}', daemon=True,
        )
        self._thread.start()
        return Spawned(proc=proc, pid=proc.pid, pgid=os.getpgid(proc.pid))

    def _drain(self) -> None:
        assert self._master is not None and self._proc is not None
        master = self._master
        # A 64 KiB read can split a multibyte sequence; a split U+2500 would corrupt
        # a Pareto stage banner and silently freeze the stage tracker.
        decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
        pending = ''
        try:
            with open(self.log_path, 'ab', buffering=0) as log:
                while True:
                    try:
                        ready, _, _ = select.select([master], [], [], SELECT_TIMEOUT)
                    except (OSError, ValueError):
                        break
                    if ready:
                        try:
                            chunk = os.read(master, READ_SIZE)
                        except OSError as exc:
                            # On Linux, EIO on the master IS end-of-file: it means the
                            # last slave fd closed. Not an error.
                            if exc.errno != errno.EIO:
                                raise
                            chunk = b''
                        if not chunk:
                            break
                        log.write(chunk)
                        pending = self._emit(decoder.decode(chunk), pending)
                    elif self._proc.poll() is not None:
                        # Drain whatever is still buffered after exit.
                        if not self._final_read(master, decoder, log, pending):
                            break
                        pending = ''
                        break
                tail = decoder.decode(b'', final=True)
                if pending or tail:
                    self._deliver(pending + tail)
        finally:
            try:
                os.close(master)
            except OSError:
                pass
            rc = self._proc.wait()
            with self._lock:
                self._returncode = rc
            self._finished.set()
            if self._on_exit:
                self._on_exit(rc)

    def _final_read(self, master, decoder, log, pending: str) -> bool:
        try:
            chunk = os.read(master, READ_SIZE)
        except OSError:
            chunk = b''
        if not chunk:
            if pending:
                self._deliver(pending)
            return False
        log.write(chunk)
        self._emit(decoder.decode(chunk), pending)
        return False

    def _emit(self, text: str, pending: str) -> str:
        """Split on \\n and \\r, deliver complete lines, return the remainder.

        \\r is a terminator too: a bare-CR progress redraw would otherwise accumulate
        into one multi-megabyte "line".
        """
        buf = pending + text
        buf = buf.replace('\r\n', '\n')
        out: list[str] = []
        start = 0
        for i, ch in enumerate(buf):
            if ch in '\n\r':
                out.append(buf[start:i])
                start = i + 1
        for line in out:
            self._deliver(line)
        return buf[start:]

    def _deliver(self, line: str) -> None:
        if self._on_line is not None:
            try:
                self._on_line(line)
            except Exception:      # a bad callback must never kill the reader
                pass

    # ---------------------------------------------------------------- control

    @property
    def returncode(self) -> int | None:
        with self._lock:
            return self._returncode

    @property
    def cancelled(self) -> bool:
        with self._lock:
            return self._cancelled

    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def wait(self, timeout: float | None = None) -> int | None:
        self._finished.wait(timeout)
        return self.returncode

    def cancel(self, grace: float = 5.0) -> None:
        """Kill the whole process group. Idempotent.

        killpg, not terminate(): the optimizer spawns its own children (`conda run
        -n fenics_env`), and signalling only the direct child leaves them running.
        """
        if self._proc is None:
            return
        with self._lock:
            self._cancelled = True
        kill_group(self._proc.pid, grace=grace)
        self._finished.wait(grace + 2.0)


def kill_group(pid: int, *, grace: float = 5.0) -> bool:
    """SIGTERM the process group, then SIGKILL. True if anything was signalled."""
    try:
        pgid = os.getpgid(pid)
    except ProcessLookupError:
        return False
    return kill_pgid(pgid, grace=grace)


def kill_pgid(pgid: int, *, grace: float = 5.0) -> bool:
    import time
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return False
    except PermissionError:
        return False

    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        if not pgid_alive(pgid):
            return True
        time.sleep(0.1)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    return True


def pgid_alive(pgid: int) -> bool:
    """Signal 0 probes existence without delivering anything."""
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True

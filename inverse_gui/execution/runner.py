"""The Runner seam: submit / stream / result / cancel.

The GUI talks only to this. Swapping the subprocess backend for an in-process or
remote one later means implementing this protocol, not touching UI code.

No streamlit import: everything here runs partly on a daemon thread, where
st.session_state silently returns a process-global mock shared across sessions.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ..domain import argv as argv_mod
from ..domain.schema import RunConfig, RunMode
from ..parse.lines import classify
from ..parse.progress import ProgressReducer, ProgressState, replay
from . import env as env_mod
from . import runstore
from .ptyproc import PtyProcess, kill_pgid
from .runstore import RunStatus


@dataclass
class Snapshot:
    """What the UI renders. Cheap to produce; taken under a short lock."""
    run_id: str
    state: str
    progress: ProgressState
    tail: list[str]
    seq: int                      # monotonically increases as lines arrive
    returncode: int | None
    degraded: bool = False        # rebuilt from disk; no live in-memory handle

    @property
    def is_terminal(self) -> bool:
        return self.state in runstore.TERMINAL


class RunHandle:
    """A live run: the PTY process, its rolling tail, and its parsed progress."""

    def __init__(self, run_id: str, runs_root: Path, status: RunStatus,
                 tail_lines: int = 200) -> None:
        self.run_id = run_id
        self.runs_root = Path(runs_root)
        self.status = status
        self._lock = threading.Lock()
        self._tail: deque[str] = deque(maxlen=tail_lines)
        self._reducer = ProgressReducer()
        self._seq = 0
        self.proc: PtyProcess | None = None

    # ------------------------------------------------------- reader-thread side

    def _on_line(self, line: str) -> None:
        event = classify(line)
        with self._lock:
            self._tail.append(line)
            self._reducer.feed(event)
            self._seq += 1

    def _on_exit(self, returncode: int) -> None:
        with self._lock:
            cancelled = self.proc.cancelled if self.proc else False
            if cancelled:
                state = runstore.STATE_CANCELLED
            elif returncode == 0:
                state = runstore.STATE_DONE
            else:
                state = runstore.STATE_FAILED
            self.status.state = state
            self.status.returncode = returncode
            self.status.ended_at = time.time()
        runstore.write_status(self.runs_root, self.status)

    # ------------------------------------------------------------- UI-thread side

    def snapshot(self) -> Snapshot:
        with self._lock:
            return Snapshot(
                run_id=self.run_id,
                state=self.status.state,
                progress=self._reducer.state,
                tail=list(self._tail),
                seq=self._seq,
                returncode=self.status.returncode,
            )

    def cancel(self, grace: float = 5.0) -> None:
        with self._lock:
            if self.status.is_terminal:
                return
            self.status.state = runstore.STATE_CANCELLING
        runstore.write_status(self.runs_root, self.status)
        if self.proc is not None:
            self.proc.cancel(grace=grace)
        elif self.status.pgid:
            kill_pgid(self.status.pgid, grace=grace)


class Runner(Protocol):
    def submit(self, cfg: RunConfig) -> str: ...
    def snapshot(self, run_id: str) -> Snapshot | None: ...
    def cancel(self, run_id: str) -> None: ...


class SubprocessRunner:
    """Launches the optimizer as a direct child on a PTY.

    Deliberately NOT `conda run`: measured, `conda run` builds a 4-deep process tree
    and Popen.terminate() kills only the outermost, so cancel() would report success
    while the optimizer kept running. See docs/environment.md.
    """

    def __init__(self, cfg, registry) -> None:
        self.cfg = cfg
        self.registry = registry
        self.runs_root = cfg.runs_dir

    # ------------------------------------------------------------------ submit

    def plan(self, run_cfg: RunConfig) -> tuple[list[str], str, dict[str, str]]:
        """Resolve interpreter, script and environment without launching anything."""
        se = env_mod.build(self.cfg)
        script = argv_mod.script_name(run_cfg, self.cfg.scripts)
        script_path = str(self.cfg.ai4ns_repo / script)
        interpreter = se.python or 'python3'
        return ([interpreter, '-u', script_path, *argv_mod.build(run_cfg)],
                interpreter, se.env)

    def submit(self, run_cfg: RunConfig) -> str:
        run_id = runstore.new_run_id(run_cfg.mode.value)
        d = runstore.create(self.runs_root, run_id)
        artifacts = d / 'artifacts'

        # Redirect the optimizer's output into this run's directory so artifacts are
        # never shared between runs, whatever the form says.
        launch_cfg = _with_output_dir(run_cfg, str(artifacts))
        full_argv, interpreter, child_env = self.plan(launch_cfg)
        cwd = str(self.cfg.ai4ns_repo)

        se = env_mod.build(self.cfg)
        (d / 'command.txt').write_text(
            argv_mod.preview(interpreter, full_argv[2], full_argv[3:]) + '\n')
        (d / 'run.sh').write_text(argv_mod.run_script(
            interpreter, full_argv[2], full_argv[3:], cwd, env_mod.notable(se)))
        (d / 'config.json').write_text(_config_json(launch_cfg))

        status = RunStatus(
            run_id=run_id, state=runstore.STATE_RUNNING,
            started_at=time.time(), mode=run_cfg.mode.value,
            argv=full_argv, cwd=cwd, interpreter=interpreter,
            artifact_dir=str(artifacts), log_path=str(d / 'stdout.log'),
        )

        handle = RunHandle(run_id, self.runs_root, status,
                           tail_lines=self.cfg.runner.tail_lines)
        proc = PtyProcess(
            full_argv, cwd=cwd, env=child_env, log_path=d / 'stdout.log',
            on_line=handle._on_line, on_exit=handle._on_exit,
        )
        handle.proc = proc
        spawned = proc.start()
        status.pid, status.pgid = spawned.pid, spawned.pgid
        runstore.write_status(self.runs_root, status)
        self.registry.put(handle)
        return run_id

    # ------------------------------------------------------------------ read

    def snapshot(self, run_id: str) -> Snapshot | None:
        handle = self.registry.get(run_id)
        if handle is not None:
            return handle.snapshot()
        return self._degraded_snapshot(run_id)

    def _degraded_snapshot(self, run_id: str) -> Snapshot | None:
        """Rebuild from disk when there is no live handle.

        Happens after a hot-reload, a Streamlit restart, or a new browser session.
        Progress is replayed from the log, so the pane looks the same as the live one.
        """
        st = runstore.read_status(self.runs_root, run_id)
        if st is None:
            return None
        lines = runstore.tail(st.log_path, lines=self.cfg.runner.tail_lines)
        progress = replay(runstore.read_all_lines(st.log_path))
        state = st.state
        if state in (runstore.STATE_RUNNING, runstore.STATE_CANCELLING):
            from .ptyproc import pgid_alive
            if not (st.pgid and pgid_alive(st.pgid)):
                state = runstore.STATE_ORPHANED
        return Snapshot(run_id=run_id, state=state, progress=progress,
                        tail=lines, seq=len(lines), returncode=st.returncode,
                        degraded=True)

    def cancel(self, run_id: str) -> None:
        handle = self.registry.get(run_id)
        if handle is not None:
            handle.cancel(grace=self.cfg.runner.term_grace_seconds)
            return
        # Degraded path: no handle, but status.json still has the process group.
        st = runstore.read_status(self.runs_root, run_id)
        if st and st.pgid:
            kill_pgid(st.pgid, grace=self.cfg.runner.term_grace_seconds)
            st.state = runstore.STATE_CANCELLED
            st.ended_at = time.time()
            runstore.write_status(self.runs_root, st)


def _with_output_dir(cfg: RunConfig, path: str) -> RunConfig:
    import copy
    out = copy.deepcopy(cfg)
    out.output_dir = path
    # Same reasoning for validation output. Left alone, an explicit
    # --fenics_output_dir sends ground-truth results somewhere outside the run
    # directory, where the results panel cannot find them and two runs overwrite
    # each other. Empty is the good case: upstream then defaults it to
    # <output_dir>/fenics, which is already inside this run.
    if out.fenics.output_dir.strip():
        out.fenics.output_dir = str(Path(path) / 'fenics')
    return out


def _config_json(cfg: RunConfig) -> str:
    import dataclasses
    import json

    def default(o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        if isinstance(o, frozenset):
            return sorted(o)
        return str(o)

    return json.dumps(cfg, default=default, indent=2, sort_keys=True)

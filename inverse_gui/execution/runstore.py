"""On-disk run layout. The durable half of the streaming design.

    runs/<run_id>/
        status.json     {run_id, state, pid, pgid, argv, cwd, returncode, ...}
        config.json     the RunConfig that produced it
        command.txt     copy-pasteable command
        run.sh          standalone reproduction script
        stdout.log      raw child output, written by the PTY reader thread
        artifacts/      --output_dir points here

status.json is what makes the app survive a hot-reload, a browser refresh past the
120s session TTL, and a Streamlit restart. Given a pgid we can still Stop; given a
log we can still replay progress. Without this file a live optimizer becomes
unreachable the first time a source file is saved.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .ptyproc import pgid_alive

STATE_RUNNING = 'running'
STATE_CANCELLING = 'cancelling'
STATE_DONE = 'done'
STATE_FAILED = 'failed'
STATE_CANCELLED = 'cancelled'
STATE_ORPHANED = 'orphaned'      # status said running, but the pgid is gone

TERMINAL = frozenset({STATE_DONE, STATE_FAILED, STATE_CANCELLED, STATE_ORPHANED})


@dataclass
class RunStatus:
    run_id: str
    state: str = STATE_RUNNING
    pid: int = 0
    pgid: int = 0
    returncode: int | None = None
    started_at: float = 0.0
    ended_at: float | None = None
    mode: str = ''
    argv: list[str] = field(default_factory=list)
    cwd: str = ''
    interpreter: str = ''
    artifact_dir: str = ''
    log_path: str = ''
    note: str = ''

    @property
    def duration(self) -> float:
        return (self.ended_at or time.time()) - self.started_at

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL


def new_run_id(mode: str, *, now: float | None = None) -> str:
    stamp = time.strftime('%Y%m%d_%H%M%S', time.localtime(now or time.time()))
    tag = 'sp' if mode.startswith('single') else 'pa'
    return f'run_{stamp}_{tag}'


def run_dir(runs_root: Path, run_id: str) -> Path:
    return Path(runs_root) / run_id


def create(runs_root: Path, run_id: str) -> Path:
    d = run_dir(runs_root, run_id)
    (d / 'artifacts').mkdir(parents=True, exist_ok=True)
    return d


def status_path(runs_root: Path, run_id: str) -> Path:
    return run_dir(runs_root, run_id) / 'status.json'


def write_status(runs_root: Path, st: RunStatus) -> None:
    """Atomic write: a torn status.json read by the next app load is worse than none."""
    p = status_path(runs_root, st.run_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(asdict(st), indent=2, sort_keys=True))
    os.replace(tmp, p)


def read_status(runs_root: Path, run_id: str) -> RunStatus | None:
    try:
        data = json.loads(status_path(runs_root, run_id).read_text())
    except (OSError, json.JSONDecodeError):
        return None
    known = {f for f in RunStatus.__dataclass_fields__}
    return RunStatus(**{k: v for k, v in data.items() if k in known})


def scan(runs_root: Path, *, limit: int = 200) -> list[RunStatus]:
    """All runs, newest first."""
    root = Path(runs_root)
    if not root.is_dir():
        return []
    out: list[RunStatus] = []
    for d in sorted(root.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        st = read_status(root, d.name)
        if st is not None:
            out.append(st)
        if len(out) >= limit:
            break
    return out


def scan_live(runs_root: Path) -> list[RunStatus]:
    """Runs whose status says running and whose process group is still alive.

    Anything claiming to run whose pgid is gone is reported as ORPHANED and its
    status file corrected, so a killed Streamlit server does not leave phantom
    entries in the active-runs strip forever.
    """
    live: list[RunStatus] = []
    for st in scan(runs_root):
        if st.state not in (STATE_RUNNING, STATE_CANCELLING):
            continue
        if st.pgid and pgid_alive(st.pgid):
            live.append(st)
        else:
            st.state = STATE_ORPHANED
            st.ended_at = st.ended_at or time.time()
            st.note = 'process group no longer exists'
            write_status(runs_root, st)
    return live


def tail(path: str | Path, *, lines: int = 200, max_bytes: int = 2_000_000) -> list[str]:
    """Last `lines` lines, reading at most `max_bytes` from the end."""
    p = Path(path)
    try:
        size = p.stat().st_size
    except OSError:
        return []
    with open(p, 'rb') as fh:
        if size > max_bytes:
            fh.seek(size - max_bytes)
            fh.readline()          # discard the partial first line
        data = fh.read()
    text = data.decode('utf-8', errors='replace')
    return text.splitlines()[-lines:]


def read_all_lines(path: str | Path):
    try:
        with open(path, encoding='utf-8', errors='replace') as fh:
            yield from fh
    except OSError:
        return


def artifact_dir(runs_root: Path, run_id: str) -> Path:
    return run_dir(runs_root, run_id) / 'artifacts'

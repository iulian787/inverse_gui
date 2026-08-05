"""Process-global registry of live runs, plus the shutdown hooks.

Why the hooks matter: start_new_session=True is required for killpg to reach the
optimizer's whole process tree, and its direct consequence is that Ctrl-C in the
terminal running Streamlit is NOT delivered to the child. Without an atexit/signal
hook, every dev-server restart leaks a multi-GB torch+IPOPT process — reintroducing
through a different door the exact failure the direct-interpreter launch was chosen
to prevent.

This module does not import streamlit. The UI obtains the singleton through
st.cache_resource (see ui/state.py), which keeps it alive across the module
evictions Streamlit performs on every source-file save.
"""

from __future__ import annotations

import atexit
import signal
import threading


class Registry:
    def __init__(self, *, kill_on_exit: bool = True, grace: float = 5.0) -> None:
        self._lock = threading.Lock()
        self._handles: dict[str, object] = {}
        self.kill_on_exit = kill_on_exit
        self.grace = grace
        self._hooks_installed = False

    # ------------------------------------------------------------------ access

    def put(self, handle) -> None:
        with self._lock:
            self._handles[handle.run_id] = handle

    def get(self, run_id: str):
        with self._lock:
            return self._handles.get(run_id)

    def drop(self, run_id: str) -> None:
        with self._lock:
            self._handles.pop(run_id, None)

    def live_ids(self) -> list[str]:
        with self._lock:
            items = list(self._handles.items())
        return [rid for rid, h in items
                if getattr(h.status, 'state', '') in ('running', 'cancelling')]

    def all_handles(self) -> list:
        with self._lock:
            return list(self._handles.values())

    # ------------------------------------------------------------------ shutdown

    def install_hooks(self) -> None:
        """Idempotent. Chains to any previously-installed handler."""
        if self._hooks_installed:
            return
        self._hooks_installed = True
        atexit.register(self.shutdown)
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                previous = signal.getsignal(sig)
                signal.signal(sig, self._make_handler(sig, previous))
            except (ValueError, OSError):
                # Not on the main thread, or the platform disallows it.
                pass

    def _make_handler(self, sig, previous):
        def handler(signum, frame):
            self.shutdown()
            if callable(previous) and previous not in (
                signal.SIG_IGN, signal.SIG_DFL,
            ):
                previous(signum, frame)
            elif previous is signal.SIG_DFL:
                signal.signal(signum, signal.SIG_DFL)
                signal.raise_signal(signum)
        return handler

    def shutdown(self) -> None:
        """Cancel every live run. Never raises — it runs during interpreter teardown.

        Note atexit does not fire on SIGKILL, which is why runstore.scan_live()
        exists: a hard-killed server leaves status.json claiming 'running', and the
        next app load reconciles it against the real process group.
        """
        if not self.kill_on_exit:
            return
        for handle in self.all_handles():
            try:
                if getattr(handle.status, 'state', '') in ('running', 'cancelling'):
                    handle.cancel(grace=self.grace)
            except Exception:
                pass


def build(cfg) -> Registry:
    reg = Registry(kill_on_exit=cfg.runner.kill_on_exit,
                   grace=cfg.runner.term_grace_seconds)
    reg.install_hooks()
    return reg

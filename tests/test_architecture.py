"""The layering rule, enforced.

Nothing under domain/, execution/, parse/ or artifacts/ may import streamlit.

This is not style. The PTY reader runs on a daemon thread with no ScriptRunContext,
where st.session_state does NOT raise -- it silently returns a process-global mock
shared by every thread and every browser session. The resulting cross-run data
corruption is invisible with a single tab open and appears weeks later.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parent.parent / 'inverse_gui'
PURE = ('domain', 'execution', 'parse', 'artifacts')


def pure_modules() -> list[Path]:
    out: list[Path] = []
    for sub in PURE:
        out.extend(sorted((PKG / sub).rglob('*.py')))
    return out


def imported_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name.split('.')[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:
                names.add(node.module.split('.')[0])
    return names


def test_there_are_pure_modules_to_check():
    assert len(pure_modules()) >= 8


@pytest.mark.parametrize('path', pure_modules(), ids=lambda p: str(p.name))
def test_no_streamlit_in_pure_layers(path):
    assert 'streamlit' not in imported_names(path), (
        f'{path.relative_to(PKG.parent)} imports streamlit; it runs off the '
        'script-run thread where session_state silently corrupts across sessions'
    )


def test_domain_does_not_import_execution_or_ui():
    """Domain is the innermost layer: it may not depend on how runs are executed."""
    for path in sorted((PKG / 'domain').rglob('*.py')):
        text = path.read_text()
        assert 'from ..execution' not in text and 'from ..ui' not in text, path


def test_pure_layers_do_not_import_plotly():
    """Plotting belongs to the UI; the loader returns data, not figures."""
    for path in pure_modules():
        assert 'plotly' not in imported_names(path), path

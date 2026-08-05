"""Pure line classifier: one stdout line -> zero or one Event.

Kept free of state so it can be unit-tested against golden transcripts and replayed
over a log file to rebuild progress after a reattach.

The IPOPT row parser TOKENISES rather than matching a fixed regex, because:
  * iteration 0 prints '-' in several numeric columns,
  * restoration-phase iterations suffix the iteration number with a letter ('12r'),
  * column count varies between IPOPT versions.
A regex written against fake_optimizer.py's clean output passes every test and then
fails on the first real run.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

# Stage banners use U+2500 BOX DRAWINGS LIGHT HORIZONTAL, not ASCII '-'.
RULE_CHARS = '─=-'


class EventKind(str, Enum):
    IPOPT_ITER = 'ipopt_iter'
    IPOPT_BANNER = 'ipopt_banner'
    STAGE = 'stage'
    GRID_TOTAL = 'grid_total'
    GRID_POINT = 'grid_point'
    RESTART = 'restart'
    PAYOFF = 'payoff'
    SAVED = 'saved'
    DEVICE = 'device'
    ESTIMATE = 'estimate'
    WARNING = 'warning'
    FENICS = 'fenics'
    DONE = 'done'
    OTHER = 'other'


@dataclass(frozen=True)
class Event:
    kind: EventKind
    raw: str
    data: dict


_STAGE_RE = re.compile(r'^\s*Stage\s+(\d+):\s*(.+?)\s*$')
_GRID_TOTAL_RE = re.compile(
    r'Epsilon grid:.*?\(\s*(\d+)\s*combinations?\s*[x×]\s*(\d+)\s*restarts?\s*'
    r'=\s*(\d+)\s*solves?\s*\)'
)
_GRID_POINT_RE = re.compile(r'^\s*Grid point\s+(\d+)\s*/\s*(\d+)\s*:\s*(.*)$')
_RESTART_RE = re.compile(
    r'^\s*restart\s+(\d+)\s*:\s*(?:status=(\S+))?.*?(?:obj=([-\d.eE+]+))?\s*$'
)
_PAYOFF_RE = re.compile(r'^\s*\[Payoff\]\s+(\S+?):\s*achieved=([-\d.eE+]+)')
_SAVED_RE = re.compile(r'Saved:\s*(\S+)')
_DEVICE_RE = re.compile(r'^\s*Device:\s*(\S+)')
_ESTIMATE_RE = re.compile(r'Estimated solves:.*?=\s*(\d+)\s*total')
_FENICS_RE = re.compile(r'^\s*\[FEniCS\]\s*(.+)$')
_NUM_RE = re.compile(r'^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$')
# Real IPOPT decorates two columns with single-letter codes: the iteration number
# gains 'r' during restoration, and alpha_pr gains the step-acceptance character
# (f, F, h, k, r, R, w, s, ...). Tokens like '1.00e+00f' are numbers with a suffix.
_NUM_SUFFIXED_RE = re.compile(r'^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?[a-zA-Z]?$')
_ITER_RE = re.compile(r'^(\d+)([a-zA-Z]*)$')


def _is_number(tok: str) -> bool:
    return bool(_NUM_RE.match(tok))


def _is_numberish(tok: str) -> bool:
    """A numeric column, possibly carrying an IPOPT status letter."""
    return bool(_NUM_SUFFIXED_RE.match(tok))


def _num(tok: str) -> float | None:
    return float(tok) if _is_number(tok) else None


def classify(line: str) -> Event:
    s = line.rstrip()
    stripped = s.strip()

    if not stripped or set(stripped) <= set(RULE_CHARS):
        return Event(EventKind.OTHER, s, {})

    if stripped.startswith('This is Ipopt version'):
        return Event(EventKind.IPOPT_BANNER, s, {'text': stripped})

    if stripped.startswith('iter ') and 'objective' in stripped:
        return Event(EventKind.IPOPT_BANNER, s, {'header': True})

    m = _STAGE_RE.match(stripped)
    if m:
        return Event(EventKind.STAGE, s,
                     {'stage': int(m.group(1)), 'title': m.group(2)})

    m = _GRID_TOTAL_RE.search(s)
    if m:
        return Event(EventKind.GRID_TOTAL, s, {
            'combinations': int(m.group(1)),
            'restarts': int(m.group(2)),
            'solves': int(m.group(3)),
        })

    m = _GRID_POINT_RE.match(s)
    if m:
        return Event(EventKind.GRID_POINT, s, {
            'index': int(m.group(1)), 'total': int(m.group(2)),
            'eps': m.group(3)[:2000],     # bound it; it is a dict repr
        })

    m = _PAYOFF_RE.match(s)
    if m:
        return Event(EventKind.PAYOFF, s,
                     {'prop': m.group(1), 'achieved': _num(m.group(2))})

    m = _ESTIMATE_RE.search(s)
    if m:
        return Event(EventKind.ESTIMATE, s, {'total_solves': int(m.group(1))})

    m = _DEVICE_RE.match(s)
    if m:
        return Event(EventKind.DEVICE, s, {'device': m.group(1)})

    m = _FENICS_RE.match(s)
    if m:
        return Event(EventKind.FENICS, s, {'text': m.group(1)})

    m = _SAVED_RE.search(s)
    if m:
        return Event(EventKind.SAVED, s, {'path': m.group(1)})

    if stripped.startswith('restart ') or stripped.startswith('restart'):
        m = _RESTART_RE.match(s)
        if m:
            return Event(EventKind.RESTART, s, {
                'index': int(m.group(1)),
                'status': m.group(2),
                'obj': _num(m.group(3)) if m.group(3) else None,
            })

    if 'WARNING' in stripped or stripped.startswith('warning'):
        return Event(EventKind.WARNING, s, {'text': stripped})

    if stripped.startswith('Done.'):
        return Event(EventKind.DONE, s, {})

    it = _ipopt_row(s)
    if it is not None:
        return Event(EventKind.IPOPT_ITER, s, it)

    return Event(EventKind.OTHER, s, {})


def _ipopt_row(line: str) -> dict | None:
    """Recognise an IPOPT iteration row by shape, not by exact column layout.

    Row: <iter>[restoration letter] <objective> <inf_pr> <inf_du> <lg(mu)> <||d||>
         <lg(rg)> <alpha_du> <alpha_pr> <ls>
    Several columns are '-' on iteration 0 and in restoration phases.
    """
    toks = line.split()
    if len(toks) < 6:
        return None
    m = _ITER_RE.match(toks[0])
    if not m:
        return None
    if not _is_number(toks[1]):
        return None
    # Every remaining token must be a number (optionally suffixed with a status
    # letter) or a literal '-' for a column IPOPT did not fill this iteration.
    for tok in toks[2:]:
        if tok != '-' and not _is_numberish(tok):
            return None
    return {
        'iter': int(m.group(1)),
        'phase': m.group(2) or '',       # 'r' during restoration
        'objective': float(toks[1]),
        'inf_pr': _num(toks[2]) if len(toks) > 2 else None,
        'inf_du': _num(toks[3]) if len(toks) > 3 else None,
    }

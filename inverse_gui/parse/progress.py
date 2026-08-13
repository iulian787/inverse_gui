"""Reduce a stream of Events into a progress snapshot.

A pure reducer, so a log file can be replayed from scratch to rebuild state after
the Streamlit process restarts, a browser session is lost, or a hot-reload wipes
the in-memory registry. That replay path is not an edge case — it is the normal
one during development.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from .lines import Event, EventKind

PARETO_STAGES = ('feasibility', 'payoff', 'sweep', 'rank')


@dataclass(frozen=True)
class ProgressState:
    device: str = ''
    # Pareto
    stage: int | None = None
    stage_title: str = ''
    grid_index: int = 0
    grid_total: int = 0
    estimated_solves: int = 0
    # Sweep coverage. Aggregates rather than a per-point list because a grid is a
    # Cartesian product and routinely runs to hundreds of points; `recent` keeps
    # just enough for a sparkline. Upstream never prints the achieved property
    # vector per point, so this is the whole of what a live sweep can report.
    points_reported: int = 0
    points_feasible: int = 0
    solves_feasible: int = 0
    solves_attempted: int = 0
    recent: tuple[tuple[int, int, int], ...] = ()   # (grid index, feasible, total)
    # single-point / per-solve
    iter: int | None = None
    objective: float | None = None
    inf_pr: float | None = None
    restarts_seen: int = 0
    # bookkeeping
    lines: int = 0
    iters_total: int = 0
    saved: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    fenics: tuple[str, ...] = ()
    done: bool = False

    @property
    def fraction(self) -> float | None:
        """Best available completion estimate in [0,1], or None if unknowable."""
        if self.done:
            return 1.0
        if self.grid_total:
            return min(1.0, self.grid_index / self.grid_total)
        return None

    @property
    def sweep_summary(self) -> str:
        """Coverage so far, or '' before the sweep has reported a point.

        Feasibility is the outcome that matters mid-sweep: a grid point where every
        restart failed contributes nothing to the front, and a sweep that is finding
        none is worth cancelling early rather than at the end.
        """
        if not self.points_reported:
            return ''
        return (f'{self.points_feasible}/{self.points_reported} grid points '
                f'feasible · {self.solves_feasible}/{self.solves_attempted} solves')

    @property
    def headline(self) -> str:
        if self.done:
            return 'Done'
        if self.stage is not None:
            label = PARETO_STAGES[self.stage] if 0 <= self.stage < 4 else self.stage_title
            if self.grid_total:
                return f'Stage {self.stage} · {label} · {self.grid_index}/{self.grid_total}'
            return f'Stage {self.stage} · {label}'
        if self.iter is not None:
            obj = f' · obj={self.objective:.4g}' if self.objective is not None else ''
            return f'iteration {self.iter}{obj}'
        return 'starting…'


class ProgressReducer:
    """Fold events into a ProgressState. Not thread-safe; the reader thread owns it."""

    def __init__(self) -> None:
        self.state = ProgressState()

    def feed(self, event: Event) -> ProgressState:
        s = self.state
        k = event.kind
        d = event.data
        s = replace(s, lines=s.lines + 1)

        if k is EventKind.DEVICE:
            s = replace(s, device=d['device'])
        elif k is EventKind.STAGE:
            # A new stage invalidates the previous stage's grid counters.
            s = replace(s, stage=d['stage'], stage_title=d['title'],
                        grid_index=0, grid_total=0)
        elif k is EventKind.GRID_TOTAL:
            s = replace(s, grid_total=d['combinations'])
        elif k is EventKind.GRID_POINT:
            s = replace(s, grid_index=d['index'], grid_total=d['total'] or s.grid_total)
        elif k is EventKind.GRID_FEASIBLE:
            feasible, total = d['feasible'], d['total']
            s = replace(
                s,
                points_reported=s.points_reported + 1,
                points_feasible=s.points_feasible + (1 if feasible else 0),
                solves_feasible=s.solves_feasible + feasible,
                solves_attempted=s.solves_attempted + total,
                recent=(s.recent + ((s.grid_index, feasible, total),))[-60:],
            )
        elif k is EventKind.ESTIMATE:
            s = replace(s, estimated_solves=d['total_solves'])
        elif k is EventKind.IPOPT_ITER:
            s = replace(s, iter=d['iter'], objective=d['objective'],
                        inf_pr=d.get('inf_pr'), iters_total=s.iters_total + 1)
        elif k is EventKind.RESTART:
            s = replace(s, restarts_seen=s.restarts_seen + 1)
        elif k is EventKind.SAVED:
            s = replace(s, saved=s.saved + (d['path'],))
        elif k is EventKind.WARNING:
            s = replace(s, warnings=(s.warnings + (d['text'],))[-50:])
        elif k is EventKind.FENICS:
            s = replace(s, fenics=(s.fenics + (d['text'],))[-50:])
        elif k is EventKind.DONE:
            s = replace(s, done=True)

        self.state = s
        return s


def replay(lines) -> ProgressState:
    """Rebuild progress from a log file's lines. Used on reattach."""
    from .lines import classify
    r = ProgressReducer()
    for line in lines:
        r.feed(classify(line.rstrip('\n')))
    return r.state

"""The normalised shape both artifact formats load into."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Criterion:
    """What the user asked for, reconstructed from the artifact."""
    prop: str
    mode: str                      # 'range' upstream, covering both target and range
    target: float | None = None
    lo: float | None = None
    hi: float | None = None

    def band(self, target_tol: float = 0.0) -> tuple[float, float] | None:
        if self.lo is not None and self.hi is not None:
            return self.lo, self.hi
        if self.target is not None:
            return (self.target * (1 - target_tol), self.target * (1 + target_tol))
        return None

    def check(self, achieved: float | None, target_tol: float = 0.02
              ) -> tuple[bool | None, float | None]:
        """(met, fractional error). (None, None) when there is nothing to check."""
        if achieved is None:
            return None, None
        band = self.band(target_tol)
        if band is None:
            return None, None
        lo, hi = band
        if self.target not in (None, 0):
            err = abs(achieved - self.target) / abs(self.target)
        else:
            centre = 0.5 * (lo + hi)
            err = abs(achieved - centre) / abs(centre) if centre else None
        return (lo <= achieved <= hi), err


@dataclass
class Design:
    index: int
    label: str
    props: dict[str, float]
    mask: np.ndarray | None = None
    rank: int = 0
    feasible: bool = True
    status: str = ''
    final_loss: float | None = None
    crowding: float | None = None
    prop_history: dict[str, np.ndarray] = field(default_factory=dict)
    # Ground truth for the same design, when FEniCS validation ran. Empty is the
    # normal case -- validation is opt-in and can fail per physics.
    fenics_props: dict[str, float] = field(default_factory=dict)

    @property
    def volume_fraction(self) -> float | None:
        if self.mask is None:
            return None
        return float(np.asarray(self.mask).mean())

    def get(self, prop: str) -> float | None:
        return self.props.get(prop)

    @property
    def has_fenics(self) -> bool:
        return bool(self.fenics_props)


@dataclass
class DesignSet:
    kind: str                      # 'single_point' | 'pareto'
    prop_names: list[str]
    designs: list[Design]
    criteria: list[Criterion] = field(default_factory=list)
    loss_hist: np.ndarray | None = None
    rho_directive_mode: str = ''
    source: str = ''

    def __len__(self) -> int:
        return len(self.designs)

    @property
    def is_empty(self) -> bool:
        return not self.designs

    def axis_choices(self) -> list[str]:
        """Properties that at least one design actually has a value for."""
        present = [p for p in self.prop_names
                   if any(d.get(p) is not None for d in self.designs)]
        if self.kind == 'single_point' and self.loss_hist is not None:
            present = present + ['final_loss']
        return present

    def default_axes(self) -> tuple[str, str]:
        choices = self.axis_choices()
        if not choices:
            return '', ''
        if self.kind == 'pareto' and 'rho' in choices:
            other = next((c for c in choices if c != 'rho'), 'rho')
            return 'rho', other
        if len(choices) == 1:
            return choices[0], choices[0]
        return choices[0], choices[1]

    def values(self, prop: str) -> list[float | None]:
        if prop == 'final_loss':
            return [d.final_loss for d in self.designs]
        return [d.get(prop) for d in self.designs]

    def criterion_for(self, prop: str) -> Criterion | None:
        return next((c for c in self.criteria if c.prop == prop), None)

    def summary(self) -> dict[str, int]:
        return {
            'total': len(self.designs),
            'feasible': sum(1 for d in self.designs if d.feasible),
            'front': sum(1 for d in self.designs if d.rank == 0),
        }

    @property
    def has_fenics(self) -> bool:
        return any(d.has_fenics for d in self.designs)

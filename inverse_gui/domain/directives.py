"""The property-directive grammar.

Mirrors amit_AI4NS/utils/optimization/directives.py:58-80, which accepts:

    max | min | free | target <V> | range <L> <H>

Two upstream behaviours the GUI must respect:

* `target V` and `range L H` BOTH parse to mode 'range' internally, distinguished
  only by the presence of a 'target' key. The band for `target` is derived at
  consumption time from --target_tol, not by the parser. We keep them as distinct
  GUI modes because they are distinct user intents, and only merge on emit.
* An EMPTY value string raises IndexError upstream (parts[0] on an empty list),
  not a clean ValueError. `render()` therefore never emits a flag it cannot fill.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Mode(str, Enum):
    FREE = 'free'
    MAX = 'max'
    MIN = 'min'
    TARGET = 'target'
    RANGE = 'range'

    @property
    def is_objective(self) -> bool:
        """max/min become objective terms (single-point) or epsilon axes (Pareto)."""
        return self in (Mode.MAX, Mode.MIN)

    @property
    def is_constraint(self) -> bool:
        """target/range become hard constraints (or soft penalties if infeasible)."""
        return self in (Mode.TARGET, Mode.RANGE)

    @property
    def n_values(self) -> int:
        return {Mode.TARGET: 1, Mode.RANGE: 2}.get(self, 0)


@dataclass(frozen=True)
class Directive:
    prop: str
    mode: Mode = Mode.FREE
    value: float | None = None     # target V, or range lo
    hi: float | None = None        # range hi
    weight: float = 1.0            # --weight_<prop>, single-point only
    ref: float | None = None       # --<prop>_ref; None => leave at upstream default

    @property
    def active(self) -> bool:
        return self.mode is not Mode.FREE

    def render(self) -> str | None:
        """The CLI value string, or None when the flag should be omitted entirely.

        Returns None rather than '' for free/incomplete directives -- emitting an
        empty string would raise IndexError in the upstream parser.
        """
        if self.mode is Mode.FREE:
            return None
        if self.mode in (Mode.MAX, Mode.MIN):
            return self.mode.value
        if self.mode is Mode.TARGET:
            return None if self.value is None else f'target {_num(self.value)}'
        if self.mode is Mode.RANGE:
            if self.value is None or self.hi is None:
                return None
            return f'range {_num(self.value)} {_num(self.hi)}'
        return None

    @property
    def complete(self) -> bool:
        """False for a mode that needs values it does not have."""
        return self.mode is Mode.FREE or self.render() is not None


def _num(x: float) -> str:
    """Compact, round-trippable float rendering.

    repr() gives '1e-05' for 1e-5 and '200000.0' for 2e5 -- both parse fine with
    float(), and repr guarantees no precision is lost on the way to the CLI.
    """
    if x == int(x) and abs(x) < 1e16:
        return str(int(x))
    return repr(x)


def parse(spec: str | None) -> Directive:
    """Parse a CLI directive string. Mirrors upstream, for round-trip tests and presets.

    `prop` is filled by the caller; this returns a Directive with prop=''.
    """
    if spec is None:
        return Directive(prop='', mode=Mode.FREE)
    parts = spec.split()
    if not parts:
        raise ValueError('empty directive (upstream raises IndexError on this)')
    head = parts[0]
    if head in ('max', 'min', 'free') and len(parts) == 1:
        return Directive(prop='', mode=Mode(head))
    if head == 'target' and len(parts) == 2:
        return Directive(prop='', mode=Mode.TARGET, value=float(parts[1]))
    if head == 'range' and len(parts) == 3:
        return Directive(prop='', mode=Mode.RANGE,
                         value=float(parts[1]), hi=float(parts[2]))
    raise ValueError(f"Invalid directive: {spec!r}")


def expand_isotropic(
    directives: dict[str, Directive],
    *,
    iso_map: dict[str, tuple[str, str]],
) -> dict[str, Directive]:
    """Reproduce upstream's isotropic expansion, for previewing what the solver sees.

    The GUI does not emit expanded directives -- it emits `--E` and lets upstream
    expand. This exists so the form can SHOW the user that `--E target X` becomes
    two independent constraints at half weight, which is not what most people
    assume it means.

    Mirrors directives.py:36-45: a component is only overwritten if still free, so
    an explicit component directive wins over the isotropic one.
    """
    out = {p: d for p, d in directives.items()}
    for iso, comps in iso_map.items():
        d = out.get(iso)
        if d is None or not d.active:
            continue
        for comp in comps:
            existing = out.get(comp)
            if existing is None or not existing.active:
                out[comp] = Directive(
                    prop=comp, mode=d.mode, value=d.value, hi=d.hi,
                    weight=d.weight * 0.5, ref=None,
                )
        out.pop(iso, None)
    return out

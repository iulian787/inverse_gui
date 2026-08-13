"""Load run artifacts into one shape the design-space view can plot.

The two scripts write different things:

  single-point  inverse_result_fm_multi_ac.npz
                one design (plus one per restart), a convergence history, and
                effective_<prop>/target_<prop>/hist_<prop> keys.
  Pareto        pareto_results.npz
                n designs, props as an [n,k] NaN-padded matrix, uint8 masks,
                pareto_rank/crowding_distance/feasible/statuses. No PNG at all.

Both normalise to a DesignSet, so one plotting component serves both.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from . import fenics as fenics_mod
from .model import Design, DesignSet, Criterion

SINGLE_NAME = 'inverse_result_fm_multi_ac.npz'
PARETO_NAME = 'pareto_results.npz'
_RESTART_RE = re.compile(r'inverse_result_fm_multi_ac_restart(\d+)\.npz$')


class ArtifactError(RuntimeError):
    pass


def _load_npz(path: Path):
    """Read an artifact npz.

    allow_pickle is required: upstream stores `statuses` and `rho_directive_mode` as
    object arrays. That is safe here because these files are written by a subprocess
    this app launched into its own run directory -- do not point this at npz files
    from an untrusted source.
    """
    import pickle
    try:
        return np.load(path, allow_pickle=True)
    except (ValueError, pickle.UnpicklingError, EOFError) as exc:
        raise ArtifactError(
            f'{path.name} could not be read ({exc}). It may be truncated, or use an '
            'array type this loader does not support.'
        ) from exc
    except OSError as exc:
        raise ArtifactError(f'{path.name} could not be opened ({exc}).') from exc


def _scalar(d, key, default=None):
    if key not in d:
        return default
    v = d[key]
    try:
        return v.item()
    except (AttributeError, ValueError):
        return v


def find_artifacts(artifact_dir: str | Path) -> dict[str, Path]:
    """What exists in a run's artifact directory."""
    d = Path(artifact_dir)
    out: dict[str, Path] = {}
    if not d.is_dir():
        return out
    if (d / SINGLE_NAME).exists():
        out['single'] = d / SINGLE_NAME
    if (d / PARETO_NAME).exists():
        out['pareto'] = d / PARETO_NAME
    restarts = sorted(p for p in d.glob('inverse_result_fm_multi_ac_restart*.npz'))
    if restarts:
        out['restarts'] = restarts[0].parent
    pngs = sorted(d.glob('*.png'))
    if pngs:
        out['png'] = pngs[0]
    return out


def load_run(artifact_dir: str | Path) -> DesignSet | None:
    """Load whichever artifact shape is present. None when nothing has been written."""
    found = find_artifacts(artifact_dir)
    if 'pareto' in found:
        ds = load_pareto(found['pareto'])
    elif 'single' in found:
        ds = load_single_point(Path(artifact_dir))
    else:
        return None
    # Optional and separate: validation writes its own tree next to these files, and
    # it may be absent, partial, or from a physics that failed. Never fatal.
    fenics_mod.attach(ds, artifact_dir)
    return ds


# ------------------------------------------------------------------ single point

def load_single_point(artifact_dir: Path) -> DesignSet:
    """The best design, plus one entry per restart when they exist."""
    main = artifact_dir / SINGLE_NAME
    d = _load_npz(main)

    prop_names = sorted(
        k[len('effective_'):] for k in d.files if k.startswith('effective_')
    )
    designs = [_single_design(d, prop_names, index=0, label='best')]

    for path in sorted(artifact_dir.glob('inverse_result_fm_multi_ac_restart*.npz')):
        m = _RESTART_RE.search(path.name)
        r = int(m.group(1)) if m else len(designs)
        rd = _load_npz(path)
        designs.append(_single_design(rd, prop_names, index=len(designs),
                                      label=f'restart {r}'))

    hist = d['loss_hist'] if 'loss_hist' in d.files else None
    return DesignSet(
        kind='single_point',
        prop_names=prop_names,
        designs=designs,
        criteria=_criteria(d, prop_names),
        loss_hist=np.asarray(hist) if hist is not None else None,
        source=str(main),
    )


def _single_design(d, prop_names: list[str], *, index: int, label: str) -> Design:
    mask = d['optimized_material'] if 'optimized_material' in d.files else None
    props = {}
    for name in prop_names:
        key = f'effective_{name}'
        if key in d.files:
            props[name] = float(np.asarray(d[key]).reshape(-1)[0])
    hist = {}
    for name in prop_names:
        key = f'hist_{name}'
        if key in d.files:
            hist[name] = np.asarray(d[key])
    return Design(
        index=index,
        label=label,
        props=props,
        mask=None if mask is None else np.asarray(mask),
        rank=0,
        feasible=True,
        status='ok',
        final_loss=_scalar(d, 'final_loss'),
        prop_history=hist,
    )


def _criteria(d, prop_names: list[str]) -> list[Criterion]:
    """Reconstruct what was asked for, so the detail panel can show pass/fail."""
    out: list[Criterion] = []
    for name in prop_names:
        mode = _scalar(d, f'directive_{name}')
        if mode is None:
            continue
        mode = str(mode)
        target = _scalar(d, f'target_{name}')
        lo = _scalar(d, f'range_{name}_lo')
        hi = _scalar(d, f'range_{name}_hi')
        out.append(Criterion(prop=name, mode=mode,
                             target=None if target is None else float(target),
                             lo=None if lo is None else float(lo),
                             hi=None if hi is None else float(hi)))
    return out


# ------------------------------------------------------------------ pareto

def load_pareto(path: str | Path) -> DesignSet:
    p = Path(path)
    d = _load_npz(p)

    prop_names = [str(x) for x in np.asarray(d['prop_names']).reshape(-1)] \
        if 'prop_names' in d.files else []
    props = np.asarray(d['props']) if 'props' in d.files else np.zeros((0, 0))
    masks = np.asarray(d['microstructures']) if 'microstructures' in d.files else None
    rho = np.asarray(d['rho_cost']).reshape(-1) if 'rho_cost' in d.files else None
    rank = np.asarray(d['pareto_rank']).reshape(-1) if 'pareto_rank' in d.files else None
    crowd = (np.asarray(d['crowding_distance']).reshape(-1)
             if 'crowding_distance' in d.files else None)
    feas = np.asarray(d['feasible']).reshape(-1) if 'feasible' in d.files else None
    statuses = ([str(s) for s in np.asarray(d['statuses']).reshape(-1)]
                if 'statuses' in d.files else None)

    n = int(props.shape[0]) if props.size else (len(rho) if rho is not None else 0)
    designs: list[Design] = []
    for i in range(n):
        row = {}
        for j, name in enumerate(prop_names):
            if j < props.shape[1]:
                val = float(props[i, j])
                if not np.isnan(val):       # props is NaN-padded
                    row[name] = val
        if rho is not None and i < len(rho):
            row.setdefault('rho', float(rho[i]))
        designs.append(Design(
            index=i,
            label=f'#{i}',
            props=row,
            mask=None if masks is None else np.asarray(masks[i]),
            rank=int(rank[i]) if rank is not None and i < len(rank) else 0,
            feasible=bool(feas[i]) if feas is not None and i < len(feas) else True,
            status=statuses[i] if statuses and i < len(statuses) else '',
            crowding=float(crowd[i]) if crowd is not None and i < len(crowd) else None,
        ))

    names = list(dict.fromkeys([*prop_names, 'rho']))
    return DesignSet(
        kind='pareto',
        prop_names=names,
        designs=designs,
        criteria=[],
        rho_directive_mode=str(_scalar(d, 'rho_directive_mode', '') or ''),
        source=str(p),
    )

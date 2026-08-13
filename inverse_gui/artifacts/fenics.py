"""Ground-truth FEniCS results, loaded alongside the surrogate's own artifacts.

Where they land, and why nothing extra has to be plumbed: upstream computes
`fenics_out = args.fenics_output_dir or <output_dir>/fenics`
(run_inverse_design_fm_multi_ac.py:701), and the runner already rewrites
`--output_dir` to runs/<id>/artifacts. Three layouts result:

  single-point, restarts=1   fenics/<physics>/fenics_<physics>_results.npz
  single-point, restarts>1   fenics/restart<r>/<physics>/…   (no flat copy)
  Pareto                     fenics/rank0_<i:03d>/<physics>/… plus a summary
                             fenics_validation.npz holding a JSON blob keyed by
                             the solution's index in the full result set

The maths is a deliberate port of upstream's `_fenics_aniso_props` (:213-234),
1e-6·I regularisation and 1e-8 guards included. The panel's job is to agree with
the number the CLI prints in its own comparison table, so a tidier inverse would
be the wrong answer.
"""

from __future__ import annotations

import json
import pickle
import re
import zipfile
from pathlib import Path

import numpy as np

# What a half-written or non-npz file raises on the way through np.load. A run that
# was killed mid-validation leaves exactly this, and it must not take the whole
# results view down with it.
_UNREADABLE = (OSError, ValueError, EOFError, pickle.UnpicklingError,
               zipfile.BadZipFile)

# physics name -> the key its result npz carries
PHYSICS_KEYS = {
    'elastic': 'C_hom',
    'thermal_conductivity': 'kappa_hom',
    'thermal_expansion': 'homogenized_strain',
}
SUMMARY_NAME = 'fenics_validation.npz'
_RESTART_DIR_RE = re.compile(r'^restart(\d+)$')


def result_name(physics: str) -> str:
    return f'fenics_{physics}_results.npz'


# ------------------------------------------------------------------ conversion

def props_from_arrays(*, C_hom=None, kappa_hom=None, hom_strain=None
                      ) -> dict[str, float]:
    """Port of upstream `_fenics_aniso_props`. Same guards, same rounding path."""
    props: dict[str, float] = {}
    if C_hom is not None:
        C = np.asarray(C_hom, dtype=float)
        S = np.linalg.inv(C + 1e-6 * np.eye(3))
        props['E_xx'] = 1.0 / (S[0, 0] + 1e-8)
        props['E_yy'] = 1.0 / (S[1, 1] + 1e-8)
        props['G_xy'] = 1.0 / (S[2, 2] + 1e-8)
        props['nu_xy'] = -S[0, 1] * props['E_xx']
        props['nu_yx'] = -S[1, 0] * props['E_yy']
        props['E'] = 0.5 * (props['E_xx'] + props['E_yy'])
        props['nu'] = 0.5 * (props['nu_xy'] + props['nu_yx'])
    if kappa_hom is not None:
        k = np.asarray(kappa_hom, dtype=float)
        props['kappa_x'] = float(k[0, 0])
        props['kappa_y'] = float(k[1, 1])
        props['kappa'] = 0.5 * (props['kappa_x'] + props['kappa_y'])
    if hom_strain is not None:
        e = np.asarray(hom_strain, dtype=float).reshape(-1)
        props['alpha_xx'] = float(e[0])
        props['alpha_yy'] = float(e[1])
        props['alpha_xy'] = float(e[2])
        props['alpha'] = 0.5 * (props['alpha_xx'] + props['alpha_yy'])
    return {k: float(v) for k, v in props.items()}


def load_tree(root: str | Path) -> dict[str, float]:
    """Merge whichever of the three physics ran under one FEniCS output dir.

    A physics whose solver failed leaves no npz -- upstream prints and continues
    (:293-299) -- so a partial result is normal and must not be an error.
    """
    d = Path(root)
    out: dict[str, float] = {}
    if not d.is_dir():
        return out
    for physics, key in PHYSICS_KEYS.items():
        path = d / physics / result_name(physics)
        if not path.exists():
            continue
        try:
            with np.load(path) as data:
                if key not in data.files:
                    continue
                out.update(props_from_arrays(**{_ARG[physics]: data[key]}))
        except _UNREADABLE:
            continue                      # truncated/unreadable: skip, don't crash
    return out


_ARG = {
    'elastic': 'C_hom',
    'thermal_conductivity': 'kappa_hom',
    'thermal_expansion': 'hom_strain',
}


# ------------------------------------------------------------------ discovery

def fenics_root(artifact_dir: str | Path) -> Path:
    return Path(artifact_dir) / 'fenics'


def load_single_point(artifact_dir: str | Path) -> dict[str, dict[str, float]]:
    """{'best': props} and/or {'restart <r>': props}, keyed to match Design.label."""
    root = fenics_root(artifact_dir)
    out: dict[str, dict[str, float]] = {}
    if not root.is_dir():
        return out

    flat = load_tree(root)
    if flat:
        out['best'] = flat
    for sub in sorted(root.iterdir()):
        m = _RESTART_DIR_RE.match(sub.name) if sub.is_dir() else None
        if not m:
            continue
        props = load_tree(sub)
        if props:
            out[f'restart {int(m.group(1))}'] = props
    return out


def load_pareto(artifact_dir: str | Path) -> dict[int, dict[str, float]]:
    """{design index: props}, from the summary npz. Empty when it is absent."""
    summary = Path(artifact_dir) / SUMMARY_NAME
    return _from_summary(summary) if summary.exists() else {}


def load_pareto_dirs(artifact_dir: str | Path) -> dict[int, dict[str, float]]:
    """{rank-0 enumeration index: props} from the rank0_NNN directories.

    Deliberately a different function from `load_pareto`, and deliberately a
    different kind of key: `rank0_003` is the *fourth rank-0 solution*, not design 3.
    Merging the two dicts would silently pin ground truth to the wrong microstructure
    whenever any solution is dominated.
    """
    root = fenics_root(Path(artifact_dir))
    out: dict[int, dict[str, float]] = {}
    if not root.is_dir():
        return out
    for i, sub in enumerate(sorted(p for p in root.iterdir()
                                   if p.is_dir() and p.name.startswith('rank0_'))):
        props = load_tree(sub)
        if props:
            out[i] = props
    return out


def _from_summary(path: Path) -> dict[int, dict[str, float]]:
    """Read fenics_validation.npz: one object-array cell holding a JSON list."""
    try:
        with np.load(path, allow_pickle=True) as data:
            if 'fenics_summary' not in data.files:
                return {}
            raw = data['fenics_summary']
        blob = raw.item() if hasattr(raw, 'item') else raw
        if isinstance(blob, bytes):
            blob = blob.decode()
        entries = json.loads(blob) if isinstance(blob, str) else blob
    except (*_UNREADABLE, TypeError, json.JSONDecodeError):
        return {}

    out: dict[int, dict[str, float]] = {}
    for entry in entries or []:
        try:
            idx = int(entry['solution_idx'])
            props = {str(k): float(v) for k, v in (entry.get('fenics') or {}).items()}
        except (KeyError, TypeError, ValueError):
            continue
        if props:
            out[idx] = props
    return out


# ------------------------------------------------------------------ attachment

def attach(ds, artifact_dir: str | Path) -> int:
    """Hang FEniCS props on the designs they belong to. Returns how many matched."""
    if ds is None or not ds.designs:
        return 0
    if ds.kind == 'pareto':
        return _attach_pareto(ds, artifact_dir)
    return _attach_single(ds, artifact_dir)


def _attach_single(ds, artifact_dir) -> int:
    by_label = load_single_point(artifact_dir)
    if not by_label:
        return 0
    n = 0
    for design in ds.designs:
        props = by_label.get(design.label)
        if props:
            design.fenics_props = props
            n += 1

    # With restarts > 1 upstream writes no flat tree: the best design is whichever
    # restart matched best_loss (:713-715). Mirror that rule rather than leaving the
    # design the user clicks first with an empty column.
    best = ds.designs[0]
    if not best.fenics_props and best.final_loss is not None:
        twin = next((d for d in ds.designs[1:]
                     if d.fenics_props and d.final_loss == best.final_loss), None)
        if twin is not None:
            best.fenics_props = twin.fenics_props
            n += 1
    return n


def _attach_pareto(ds, artifact_dir) -> int:
    n = 0
    by_design = {d.index: d for d in ds.designs}
    for idx, props in load_pareto(artifact_dir).items():
        design = by_design.get(idx)
        if design is not None:
            design.fenics_props = props
            n += 1
    if n:
        return n

    # No usable summary. The directories enumerate rank-0 solutions in the order
    # upstream visited them (np.where over the ranks), so map them onto the rank-0
    # designs in index order -- never onto design[i].
    rank0 = [d for d in ds.designs if d.rank == 0]
    for i, props in sorted(load_pareto_dirs(artifact_dir).items()):
        if i < len(rank0):
            rank0[i].fenics_props = props
            n += 1
    return n

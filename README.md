# inverse_gui

A GUI between the researcher and the AI4NS inverse-design optimizer: specify property targets in a form,
launch single-point or Pareto runs, and browse the resulting space of designs.

**Status: the application does not exist yet.** What is here is the design deck, the environment
scaffolding to build both required Python environments, and a fake optimizer that lets the GUI be
developed and tested before the model checkpoints are available.

| | |
|---|---|
| `design_slides.html` | the design walkthrough — open in a browser |
| `scripts/setup.sh` | builds both environments, idempotent |
| `scripts/fake_optimizer.py` | stand-in for the optimizer; unblocks development |
| `docs/environment.md` | why the environments are split the way they are |
| `CLAUDE.md` | working notes and measured constraints |

## Prerequisites

- **conda** — miniforge, miniconda, or anaconda. Required, not optional: `cyipopt` has never published a
  wheel, so the solver cannot be pip-installed. See `docs/environment.md`.
- **uv** — <https://docs.astral.sh/uv/>. Used for the GUI environment.
- **~4 GB free** on whichever partition holds your conda envs (~7.5 GB if you also build the FEniCS env).
- **The `amit_AI4NS` optimizer repo**, checked out as a sibling directory. It is a separate repo and is
  required at runtime — the GUI shells out to its two entry-point scripts.

Expected layout, because `config.example.toml` defaults to `../amit_AI4NS`:

```
<parent>/
  inverse_gui/     <- this repo
  amit_AI4NS/      <- the optimizer
```

Any other layout works too; set `[paths].ai4ns_repo` in `config.toml` (or `INVERSE_GUI_PATHS_AI4NS_REPO`).

## Setup

```bash
git clone <this repo> inverse_gui
cd inverse_gui
./scripts/setup.sh
```

If the partition holding your conda install is tight, put the new environments elsewhere — the script
registers the location with conda so `conda run -n <name>` still resolves:

```bash
INVERSE_GUI_ENVS_DIR=/big/disk/envs ./scripts/setup.sh
```

Add `--fenics` to also build the optional FEniCS validation environment (~3.5 GB, deferred by default).

The script is idempotent: re-running it skips environments that already exist, re-verifies the solver, and
regenerates the machine-specific bits. It finishes by creating `config.toml` from `config.example.toml` if
you don't have one.

### What it builds

| environment | manager | Python | why |
|---|---|---|---|
| `.venv` | uv | 3.12 | the GUI. Pure-PyPI deps; never solves, only reads `.npz` artifacts. |
| `cenv` | conda-forge | 3.11 | the optimizer. Conda is mandatory — `cyipopt` is sdist-only on PyPI. |
| `fenics_env` | conda-forge | 3.11 | optional ground-truth PDE validation. Not built unless `--fenics`. |

Two files are generated per machine and are **not** in git: `config.toml` (your paths) and
`env/cenv.activation.json` (captured conda activation variables). Both come from `setup.sh`.

## Using the GUI environment

You generally do **not** need to activate it — a venv's `bin/python` already knows its own `sys.path`:

```bash
.venv/bin/python -c "import streamlit, plotly, numpy"
.venv/bin/streamlit run app.py          # once app.py exists
```

This is the form to use in scripts and anything spawned programmatically, since there is no shell state to
get wrong. Two alternatives:

```bash
uv run streamlit run app.py             # re-syncs against uv.lock first; best interactively
source .venv/bin/activate               # traditional; `deactivate` to leave
```

Note this is a plain venv, not conda — `conda activate` does not apply to it. You can be inside `.venv` and
still launch `cenv`'s interpreter by absolute path, which is exactly what the GUI will do.

## Smoke test

`app.py` does not exist yet, so the thing to run is the fake optimizer. It mirrors the real CLI surface,
streams a realistic IPOPT iteration table, and writes artifacts with the real key sets:

```bash
.venv/bin/python scripts/fake_optimizer.py \
    --ckpt_elastic_fm fake.pt --E "target 200000" --nu "target 0.25" \
    --output_dir /tmp/fakerun --iters 5
```

Pareto mode is selected by `--pareto_steps`, matching upstream. `--crash` and `--hang` exercise the
failure and cancel paths. Verify the solver environment separately with:

```bash
conda run -n cenv --no-capture-output python -c "
import numpy as np
from cyipopt import minimize_ipopt
r = minimize_ipopt(lambda x: (x[0]-1)**2 + (x[1]-2)**2, np.zeros(2))
print('IPOPT ok:', np.allclose(r.x, [1, 2], atol=1e-5))"
```

Importing `cyipopt` is not proof it works; that command exercises the IPOPT C++ library and its MUMPS
linear solver end to end.

## Known blocker: model checkpoints

Real runs need the three EffPropNet checkpoints at `amit_AI4NS/models/fm_multi_store/*.pt`, and the
optimizer's argparse hard-fails without at least one. **They do not exist on disk**, are excluded by
`.gitignore` upstream, and must be obtained separately (~50–200 MB each). The `.pt` files in the sibling
`AI4NS/` repo are a different generation — ResUNet, not EffPropNet — and will not load.

This blocks only real solves. Everything else — the form, the execution layer, streaming, cancel, artifact
parsing, the design-space plot, run history — can be built and tested against `fake_optimizer.py`.

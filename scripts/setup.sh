#!/usr/bin/env bash
# Idempotent environment setup for the inverse-design GUI.
#
#   ./scripts/setup.sh              # GUI venv + solver env
#   ./scripts/setup.sh --fenics     # also build the FEniCS validation env (~3.5 GB)
#
# Three environments, by design:
#   .venv     uv, Python 3.12  -- the GUI. Never solves; reads .npz artifacts.
#   cenv      conda, 3.11      -- the solver. Conda is mandatory: cyipopt has never
#                                 shipped a wheel for any platform or release.
#   fenics_env conda, 3.11     -- optional ground-truth PDE check, deferred by default.
#
# See ../CLAUDE.md and env/cenv.yml for why the upstream requirements.txt cannot be used.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

BUILD_FENICS=0
[[ "${1:-}" == "--fenics" ]] && BUILD_FENICS=1

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

# ---------------------------------------------------------------- locate conda
CONDA="${CONDA_EXE:-}"
if [[ -z "$CONDA" || ! -x "$CONDA" ]]; then
  CONDA="$(command -v conda || true)"
fi
for p in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" /opt/conda; do
  [[ -z "$CONDA" || ! -x "$CONDA" ]] && [[ -x "$p/bin/conda" ]] && CONDA="$p/bin/conda"
done
[[ -x "$CONDA" ]] || { echo "conda not found; set CONDA_EXE" >&2; exit 1; }
say "conda: $CONDA"

# --------------------------------------------------------- phase 0: disk space
# Conda envs default to <conda>/envs on the root partition. If that partition is
# tight, point new envs at a roomier disk. The envs_dir must be PREPENDED, not
# appended: conda creates named envs in the first writable entry. It must also stay
# registered, because run_inverse_design_fm_multi_ac.py:283 hardcodes
# `conda run -n fenics_env` and -n only resolves names found in envs_dirs.
if [[ -n "${INVERSE_GUI_ENVS_DIR:-}" ]]; then
  say "phase 0: registering envs dir $INVERSE_GUI_ENVS_DIR"
  mkdir -p "$INVERSE_GUI_ENVS_DIR"
  if ! "$CONDA" config --show envs_dirs | grep -qF "$INVERSE_GUI_ENVS_DIR"; then
    "$CONDA" config --prepend envs_dirs "$INVERSE_GUI_ENVS_DIR"
  fi
  "$CONDA" config --show envs_dirs
fi

avail_gb=$(df -BG --output=avail "$(dirname "$CONDA")" | tail -1 | tr -dc '0-9')
if (( avail_gb < 6 )); then
  echo "WARNING: only ${avail_gb}G free where conda lives." >&2
  echo "  Reclaim the package cache:  $CONDA clean -a -y   (envs are untouched)" >&2
  echo "  Or relocate:  INVERSE_GUI_ENVS_DIR=/big/disk/envs ./scripts/setup.sh" >&2
fi

# ------------------------------------------------------------ phase 1: GUI env
say "phase 1: GUI venv"
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || { echo "uv not found: https://docs.astral.sh/uv/" >&2; exit 1; }
uv venv --python 3.12 .venv
uv sync --extra dev
.venv/bin/python -c "import streamlit, plotly, numpy; print('GUI deps OK')"

# --------------------------------------------------------- phase 2: solver env
say "phase 2: solver env (cenv)"
if "$CONDA" env list | awk '{print $1}' | grep -qx cenv; then
  echo "cenv already exists; skipping create (delete it to rebuild)"
else
  "$CONDA" env create -f env/cenv.yml
fi

# Importing cyipopt is not proof it works -- this exercises the IPOPT C++ library and
# its MUMPS linear solver end to end.
"$CONDA" run -n cenv --no-capture-output python -c "
import numpy as np
from cyipopt import minimize_ipopt
r = minimize_ipopt(lambda x: (x[0]-1)**2 + (x[1]-2)**2, np.zeros(2))
assert np.allclose(r.x, [1, 2], atol=1e-5), r.x
print('IPOPT solve OK')" 2>&1 | grep -v '^\*\{10,\}$' | tail -2

# The Runner launches <prefix>/bin/python directly, so activate.d hooks never fire.
# Capture their effect once and let the Runner inject it -- see probe_solver_env.py.
say "phase 2b: capturing activation delta"
.venv/bin/python scripts/probe_solver_env.py --env cenv --conda-exe "$CONDA" \
  --out env/cenv.activation.json

# --------------------------------------------------- phase 5 (opt): FEniCS env
if (( BUILD_FENICS )); then
  say "phase 5: fenics_env (~3.5 GB)"
  YML="$(python3 -c "import os;print(os.path.realpath('$REPO/../amit_AI4NS/fenics_validation/environment_fenics.yml'))")"
  [[ -f "$YML" ]] || { echo "not found: $YML" >&2; exit 1; }
  # The name must stay fenics_env, or be threaded through --fenics_conda_env.
  if "$CONDA" env list | awk '{print $1}' | grep -qx fenics_env; then
    echo "fenics_env exists; installing any missing pieces into it"
  else
    "$CONDA" env create -f "$YML"
  fi
  # The yml is incomplete. fenics_validation/mesh.py imports dolfinx_mpc (the
  # periodic BCs every solver uses) and output.py imports pandas; the yml lists
  # neither, so a by-the-book env resolves under `conda run -n` and then dies at
  # import -- after the solve, before artifacts are written.
  "$CONDA" install -n fenics_env -c conda-forge -y dolfinx_mpc pandas
  # Verify what the optimizer actually imports, not just dolfinx. Same cwd as the
  # optimizer child, because nothing in the ai4ns repo is installed.
  ( cd "$(dirname "$YML")/.." && "$CONDA" run -n fenics_env --no-capture-output \
      python -c "import dolfinx, fenics_validation.validate as v; \
print('dolfinx', dolfinx.__version__, '+ fenics_validation OK')" )
else
  say "skipping fenics_env (deferred; pass --fenics to build it)"
fi

# ---------------------------------------------------------------------- config
[[ -f config.toml ]] || { cp config.example.toml config.toml; echo "created config.toml"; }

say "done"
cat <<EOF
  GUI      : $REPO/.venv/bin/python
  solver   : $(python3 -c "import json;print(json.load(open('env/cenv.activation.json'))['python'])")
  config   : $REPO/config.toml

Last step for real runs: put the EffPropNet checkpoints in
<ai4ns_repo>/models/fm_multi_store/ and record their absolute paths in the
[checkpoints] table of config.toml. The form is pre-filled from it. The Doctor
panel in the app tells you whether they resolve.

No checkpoints yet? Point both [scripts] entries at scripts/fake_optimizer.py;
everything but the physics still works. To drive it directly:
  .venv/bin/python scripts/fake_optimizer.py --ckpt_elastic_fm f.pt \\
      --E "target 200000" --output_dir /tmp/fakerun --iters 5
EOF

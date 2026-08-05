# Why the environments are split this way

Background for anyone picking this up on another machine. The short version lives in `CLAUDE.md`; this is
the reasoning, including the measurements that settled the debatable parts.

## The forcing constraint: cyipopt has no wheel

`cyipopt` is the NLP solver behind the optimizer (`amit_AI4NS/utils/optimization/solver.py:188`,
`cyipopt.Problem`). PyPI serves **an sdist and nothing else** — no wheel, for any platform, in any release
including the current 1.7.0. Installing it with pip therefore requires a compiler toolchain *and* a
pre-existing IPOPT + MUMPS/HSL installation with discoverable headers and pkg-config. The project's own
description says to use conda.

So the solver environment must be conda. That is the one non-negotiable fact; everything else follows from
deciding what *else* has to live in that environment.

## The GUI does not belong in it

With the subprocess execution model, the GUI process imports `streamlit`, `plotly`, `numpy` — that is the
whole set. It builds an argv, streams the child's stdout, and reads `.npz` artifacts. It never solves.

Three reasons to keep it out of the solver env:

1. **Streamlit's transitive closure collides with the solver's pin surface.** It pulls `protobuf`,
   `tornado`, `pyarrow`, `altair`, `gitpython`. `amit_AI4NS/pyproject.toml` pins ~292 packages to exact
   versions. Anything resolved into that environment can silently move a shared transitive dependency, and
   the failure mode is not "Streamlit breaks" — it is "the solver breaks three weeks later."
2. **There is nothing to share.** The solver env has to be built either way, so combining them saves zero
   setup steps and only adds coupling.
3. **It would forfeit the point of the subprocess design**, which exists precisely to decouple GUI
   lifecycle from solver lifecycle.

Cost of separation: ~440 MB and one `uv venv` command.

The GUI env is a **uv venv rather than conda** because its dependencies are 100% pure-PyPI universal
wheels — nothing compiled, nothing conda-only. Spawning `conda run` needs conda's *absolute path*, not
conda in the parent environment, so being a venv costs nothing there. `uv` is also the established local
convention (both `amit_AI4NS/.venv` and `compogen/.venv` are uv-authored), and `uv.lock` gives the GUI the
reproducibility the neighbouring `pyproject.toml` conspicuously lacks.

**Accepted cost:** the in-process execution backend is off the table — it would require the GUI process to
*be* the solver env. This is reversible with `conda install -n cenv streamlit plotly` plus a config change.
Independently, `cancel()` is unimplementable in-process: you cannot interrupt a thread blocked inside
C++ IPOPT.

## The upstream dependency files cannot build an environment

`amit_AI4NS/pyproject.toml` and `requirements.txt` are both ~292-line `pip freeze` dumps of an unrelated
environment. They carry `open-webui`, `langchain`, `langgraph`, `chromadb`, `faster-whisper`, `globus-cli`,
`pytube` — and **neither lists `cyipopt`**, the one package with a real platform constraint. They also
disagree with each other on numpy's version.

`env/cenv.yml` in this repo is the actual transitive closure of the two optimizer entry points: `python`,
`cyipopt`, `pytorch`, `numpy`, `scipy`, `h5py`, `matplotlib`. `mpi4py`,
`intel_extension_for_pytorch` and `oneccl_bindings_for_pytorch` are import-guarded and Aurora/Polaris-only,
so they are omitted. GPU is optional — the device probe is a try/except that falls back to CPU cleanly, and
there is no `--device` flag.

## Three measured findings

These were tested on the real environment rather than reasoned about, because each one silently breaks a
feature the design deck depends on.

### 1. `conda run` orphans the process you asked it to kill

`conda run` builds a four-deep process tree (`conda run` → `conda` → `bash` → `python`). `Popen.terminate()`
hits the outermost, and the optimizer keeps running.

| invocation | survivors after cancel |
|---|---|
| `conda run -n cenv` + `Popen.terminate()` | **1 — orphaned** |
| `<prefix>/bin/python` + `killpg` | 0 — clean |

A Runner built the obvious way would report successful cancellation while a multi-GB torch+IPOPT process
burned CPU indefinitely. Launch the environment's interpreter directly, with `start_new_session=True`, and
cancel with `os.killpg(os.getpgid(pid), SIGTERM)` then `SIGKILL`. The new session is required *even with*
the direct interpreter, because the optimizer spawns its own grandchildren.

Note buffering is **not** a reason to avoid `conda run` — `--no-capture-output` fixes that correctly.

### 2. The live IPOPT table needs a PTY

`--ipopt_print` defaults to 5, so the per-iteration table is on by default — and the design deck makes it
the centerpiece of the progress view. But that table is written by IPOPT's **C++ Journalist to C-level
stdout**. `python -u` and `PYTHONUNBUFFERED` only affect CPython's `io` layer, not glibc's `FILE*` buffering
inside `libipopt.so`. Against a pipe, glibc block-buffers at 4 KB.

Measured, emitting four rows at 1 s intervals:

| stdout target | row arrival times (s) |
|---|---|
| plain pipe | 4.14, 4.14, 4.14, 4.14 — all at once, on exit |
| `stdbuf -oL` + pipe | 0.1, 1.1, 2.1, 3.1 |
| PTY | 0.1, 1.1, 2.1, 3.1 |

There is no Python-side substitute: `utils/optimization/solver.py` prints only at stage boundaries. Build
the streaming layer around `pty.openpty()` from the start, with `stdbuf` as a fallback — `stdbuf` is
defeated if a library calls `setvbuf` on itself after startup, and retrofitting a PTY later means
rewriting the reader loop, the cancel path, and the tests.

`scripts/fake_optimizer.py` deliberately prints through `ctypes` `printf` so it reproduces this behaviour;
a Runner that streams correctly against it will stream correctly against the real optimizer.

### 3. Bypassing `conda run` skips activation hooks

The direct-interpreter approach means `<prefix>/etc/conda/activate.d/` never executes. `cenv` ships three
hooks, and one is load-bearing:

- `libblas_mkl_activate.sh` → `MKL_INTERFACE_LAYER=LP64,GNU` — **matters**; selects the MKL BLAS interface
  used by numpy/scipy/torch.
- `libglib_activate.sh` → `GSETTINGS_SCHEMA_DIR` — GTK schemas, irrelevant headless.
- `libxml2-split_activate.sh` → `XML_CATALOG_FILES` — libxml2 catalogs, irrelevant.

`scripts/probe_solver_env.py` runs the activation once and records the environment delta to
`env/cenv.activation.json` (git-ignored, machine-specific, regenerated by `setup.sh`). The Runner merges
that into the child's environment — fully activated child, flat and killable process tree. Re-run the probe
whenever the solver environment is rebuilt.

## Other required child-process settings

- **`cwd` must be the ai4ns repo root.** The entry scripts do `sys.path.insert(0, dirname(abspath(__file__)))`
  and nothing is installed — no `setup.py`, no `[build-system]`, no console scripts.
- **`MPLBACKEND=Agg`** — `run_inverse_design_fm_multi_ac.py:47` is a bare `import matplotlib.pyplot as plt`
  with no `matplotlib.use('Agg')`, which can raise at import in a DISPLAY-less child.
- **`OMP_NUM_THREADS` / `MKL_NUM_THREADS`** — IPOPT, MUMPS and torch each default to every core.
- **conda on the child's `PATH`** whenever FEniCS validation might run; see below.

## Design-deck claims that are factually wrong

Verified against the source. Do not build on these.

**"Skipped gracefully if the solver env is unavailable" (validation slide) — false, and destructive.**
`run_fenics_validation` is called at `run_inverse_design_fm_multi_ac.py:701-726` with **no try/except
anywhere in the call chain**, and shells `subprocess.run(['conda', 'run', ...])` at `:283`. With
`auto_activate_base: false` and a non-login subprocess, conda missing from the child's `PATH` is the
*default* case → uncaught `FileNotFoundError`. It fires *after* the solve completes and *before*
`plot_results` at `:728`, so a twenty-minute run dies without writing artifacts. Three defenses, all
needed: inject conda into the child `PATH`, default the toggle off, and gate it on a preflight check.

**"Built from the existing Pydantic schema" (architecture slide) — there is no such schema.** Zero hits for
`pydantic`/`BaseModel` across `amit_AI4NS`. Pydantic exists only in `compogen`, a different repo with no
coupling to ai4ns. The run configuration must be authored from scratch against the argparse surface.

**"compogen reuses staging, logbook, and cluster scale" (execution slide).** Staging and scale are real;
the logbook is not — no `read_design_logbook`, no `annotate_last_run`, and no inverse-design or Pareto tool
in its 20-tool registry, whose workflow is strictly forward simulation. compogen depends on `compgen`
(a MOOSE/Sculpt toolkit), not on ai4ns. Treat it as a later integration project, not a backend you
configure. The barrier to merging the environments is ai4ns's ~292 exact pins versus compogen's
251-package `uv.lock` — dependency resolution, not interpreter version.

**"Three interchangeable backends" (execution slide).** In-process silently makes the GUI environment *be*
the solver environment, and leaves `cancel()` unimplementable. The three are not freely interchangeable,
and the reasons are environmental.

**"Tail stdout" (progress slide).** True only with a PTY — see finding 2.

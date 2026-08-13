# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Streamlit GUI over the AI4NS inverse-design optimizer: a form that generates command lines for the two
optimizer CLIs, launches them, streams their output live, and plots the resulting design space.

```bash
./scripts/setup.sh                       # builds .venv and cenv, idempotent
.venv/bin/streamlit run app.py
.venv/bin/python -m pytest tests/ -q     # ~221 tests, ~47s
```

Working, **against the real optimizer** — the checkpoints are on disk now and a single-point solve has run
end to end from the GUI: form (sections A–F), physics gating, validation, cost estimate,
launch/stream/cancel, reattach, design-space scatter with click-to-inspect, run history list-and-reload,
cross-run compare (up to 4 pinned designs), FEniCS ground-truth column in the design detail, live sweep
coverage, preflight doctor.
Not built: a chat surface, and live *design points* during a run — see the gotcha about why that one
needs an upstream change. Every preflight check passes on this machine, FEniCS included.

## Layering — the one rule that matters

```
domain/ execution/ parse/ artifacts/     pure; MUST NOT import streamlit
ui/  app.py                              the only places that may
```

`tests/test_architecture.py` enforces this. The reason is not tidiness: the PTY reader runs on a daemon
thread, and `st.session_state` accessed without a script-run context **does not raise** — it silently
returns a process-global mock shared by every thread and every browser session
(`runtime/state/session_state_proxy.py`). That corruption is invisible with one tab open.

Related: the run registry lives in `st.cache_resource`, not a module global, because Streamlit evicts
every watched local module from `sys.modules` on any source save. A module global would be emptied
mid-run, leaving a live optimizer with no Stop button.

## Environments — three of them, deliberately

```bash
./scripts/setup.sh            # idempotent; add --fenics for the optional validation env
```

| env | manager | Python | where | role |
|---|---|---|---|---|
| `.venv` | uv | 3.12 | `inverse_gui/.venv` | the GUI. Never solves. |
| `cenv` | conda-forge | 3.11 | `../envs/cenv` (big disk) | the optimizer |
| `fenics_env` | conda-forge | 3.11 | `../envs/fenics_env` (big disk) | optional PDE validation; built and working |

**Why the solver env must be conda:** `cyipopt` has never published a wheel — PyPI serves an sdist only,
for every release and platform. Building it needs a compiler plus a pre-existing IPOPT + MUMPS/HSL.

**Why the GUI is separate and is *not* conda:** with the subprocess Runner the GUI imports only
`streamlit`, `plotly`, `numpy` and reads `.npz` artifacts — no torch, no cyipopt. Keeping Streamlit's
dependency tree out of the solver env avoids perturbing it. Spawning conda needs conda's absolute path,
not conda in the parent env. This does foreclose the deck's in-process backend; reversible with
`conda install -n cenv streamlit plotly` plus a config change.

`env/cenv.yml` replaces the upstream dependency files: `amit_AI4NS/pyproject.toml` and `requirements.txt`
are both ~292-line `pip freeze` dumps (open-webui, langchain, chromadb, pytube) and **neither lists
cyipopt**. Never build an environment from them.

**`environment_fenics.yml` is incomplete the same way.** It builds a working dolfinx 0.11 environment in
which the upstream validator will not import: `fenics_validation/mesh.py:7` needs `dolfinx_mpc` (periodic
BCs, all four solvers) at package level and `output.py:7` needs `pandas`, and the yml lists neither.
`fenics_env` here has both now, so `import fenics_validation.validate` succeeds and the panel is live;
`setup.sh --fenics` installs them and verifies by importing the validator, not `dolfinx`.

**Installing into `fenics_env` needs an exactly-pinned build.** A plain
`conda install -n fenics_env -c conda-forge dolfinx_mpc pandas` never finished here — three attempts, 40+
minutes each, libmamba still in "Solving environment". What worked in seconds was pinning the build that
matches the env's own variant and freezing the rest:

```bash
conda install -n fenics_env --override-channels -c conda-forge --freeze-installed \
    dolfinx_mpc=0.11.0=py311hbef1974_0 pandas
```

The build string is not arbitrary: of the four py311 builds, `py311hbef1974_0` is the mpich + real-PETSc
one, matching `mpich 5.0.1` and `petsc 3.25.4=real_*` in this env. Pick the matching variant with
`conda search -c conda-forge --info dolfinx_mpc=0.11.0` before installing.

Because of that, the `FEniCS env` preflight check **probes the import** (`conda run -n <env> python -c
'import fenics_validation.validate'`, cwd = the ai4ns root) instead of grepping `conda env list` for the
name. A name-only check is a false green here, and the cost of a false green is one completed solve —
validation runs after the solve and before artifacts are written, with no try/except upstream.
`tests/test_fenics_env.py` covers name resolution, importability, and conda's reachability from the
*child's* PATH separately; the real-env tests skip when `fenics_env` is absent.

## Launching the optimizer — three non-obvious constraints

All three are measured, not assumed. Full reasoning and the raw numbers are in `docs/environment.md`.

1. **Never `conda run`.** It builds a 4-deep process tree and `Popen.terminate()` kills only the
   outermost — measured here: `conda run` + `terminate()` leaves the optimizer running, direct
   interpreter + `killpg` is clean. Spawn `<prefix>/bin/python` directly with `start_new_session=True`,
   cancel with `os.killpg(os.getpgid(pid), SIGTERM)`. (Buffering is *not* the reason —
   `--no-capture-output` fixes that fine.)
2. **Read stdout through a PTY.** The live IPOPT table is written by IPOPT's C++ Journalist to C-level
   stdout; `python -u` and `PYTHONUNBUFFERED` only touch CPython's io layer. Measured with 1 s iterations:
   plain pipe delivered all four rows at once on exit; PTY and `stdbuf -oL` both delivered at 1 s
   intervals. PTY is the more robust of the two (`stdbuf` loses to a library calling `setvbuf`).
3. **Inject the activation delta.** Bypassing `conda run` means `activate.d` hooks never fire, and `cenv`
   ships three. One matters: `MKL_INTERFACE_LAYER=LP64,GNU`. `scripts/probe_solver_env.py` captures the
   delta to `env/cenv.activation.json` (git-ignored, machine-specific) for the Runner to merge in.

Also required in the child: `cwd` = the ai4ns repo root (the scripts `sys.path.insert` and nothing is
installed), `MPLBACKEND=Agg` (bare `import matplotlib.pyplot` at line 47), and `OMP_NUM_THREADS` /
`MKL_NUM_THREADS` (IPOPT, MUMPS and torch each grab every core by default).

## Gotchas found the hard way

Each of these cost real debugging time; the comment in the code names the failure.

- **Validate after the form renders, not before.** Widgets write into `RunConfig` during
  `sections.render()`, so validating first makes the Launch gate lag one interaction. `app.py` validates
  twice: once for inline field hints (needed on the very first render) and once after, for the gate.
- **To change a widget programmatically, assign its session_state key — do not delete it.** Deleting
  works under `AppTest` and fails in a browser, where the frontend re-sends the old value and it wins
  over `value=`. This bit the mode toggle's `target_tol`/`restarts` defaults.
- **Do not set `dragmode='select'` on the scatter.** A plain click then draws a zero-area selection box
  and selects nothing.
- **Selection must read `customdata`, never `point_index`.** Designs are split across traces so the
  legend can filter them, and `point_index` is trace-relative — clicking a dominated point would open a
  different design's microstructure.
- **The AppTest suite reads the real `config.toml`.** Once `[checkpoints]` was populated, section A came
  up pre-filled and every test asserting the unconfigured state failed on a machine-local file. The
  autouse `no_local_checkpoints` fixture in `tests/conftest.py` blanks them via env vars; any new config
  key the form seeds from needs the same treatment.
- **Streamlit's file watcher does not fire reliably on this mount.** After editing, restart the server;
  otherwise you are testing stale code and will chase phantom bugs.
- **`--pareto_steps` is always emitted**, even at its default, because it is the defining parameter of
  the sweep (and it is how a single stand-in script tells the two modes apart).
- **Designs cannot appear in the scatter mid-run, and this is not a missing feature.** Both scripts write
  their npz *after* the solve — single-point's per-restart files in a post-solve loop
  (`run_inverse_design_fm_multi_ac.py:769-787`), Pareto's once at `:709` — and stdout never carries an
  achieved property vector per grid point: the sweep prints the ε *targets*, per-restart `status=/obj=`,
  and `N/M restarts feasible` (`:562`, `:441`, `:577`). So the live pane shows sweep **coverage**
  (`ProgressState.points_reported/points_feasible`, rendered by `run_pane._sweep_coverage`). Real
  incremental points would need an upstream change; don't re-litigate it from the deck's wording.

## Checkpoints — present, and configured in `config.toml`

The three EffPropNet checkpoints now exist at `../amit_AI4NS/models/fm_multi_store/` (~137 MB each):

```
elastic_effpropnet_silu_f64_128_256_6554_fmmulti_epoch860.pt
thermal_conductivity_effpropnet_silu_f64_128_256_6554_fmmulti_epoch1000.pt
thermal_expansion_effpropnet_silu_f64_128_256_6554_fmmulti_epoch1000.pt
```

Their **absolute** paths live in the `[checkpoints]` table of `config.toml` (git-ignored and
machine-specific — `config.example.toml` ships the keys empty). `ui/state.py:get_run_config` seeds section A
of the form from them on first render, so the app opens ready to launch. `[paths].ckpt_dir` is *not* a
resolution root — it only supplies the placeholder text in section A; the values themselves must be
complete paths.

A fourth file, `plasticity_yield_..._epoch1000.pt`, sits in the same directory and is unusable: neither
entry point has a plasticity checkpoint flag, directive, or property. Section A says so in its caption.
The `.pt` files in the sibling `AI4NS/` repo are also unusable — a different generation (ResUNet).

Real solves work end to end from the GUI. Verified 2026-08-12, single-point, `--E 'target 200000'`:
`EXIT: Optimal Solution Found`, 17 s in IPOPT, both `inverse_result_fm_multi_ac.npz` and the result png
written to `runs/<id>/artifacts/`. Argparse still hard-fails with no checkpoint at all, so the doctor's
Checkpoints check stays.

`scripts/fake_optimizer.py` remains the fallback for machines without the `.pt` files (and is what the
tests use): it mirrors the CLI surface of both entry points, emits a realistic IPOPT table **through ctypes
printf** so it reproduces the real C-level buffering, writes synthetic `.npz` artifacts with the real key
sets, and has `--crash` / `--hang` for exercising the Runner's failure and cancel paths. Point `[scripts]`
at it to switch. Pareto mode is selected by `--pareto_steps`, matching upstream.

## Reading FEniCS ground truth

`artifacts/fenics.py` finds the validator's own `.npz` files and hangs the converted properties on the
designs they belong to; `design_space.criteria_rows` then renders an NN-vs-FEniCS column. Four things
about it are load-bearing:

- **The maths is a deliberate port**, `_fenics_aniso_props` at `run_inverse_design_fm_multi_ac.py:213-234`,
  including `inv(C + 1e-6·I)` and the `1/(S + 1e-8)` guards. That guard is *not* negligible: at
  E ≈ 210 GPa it reads 0.2% low. Keeping it is the point — the panel has to agree with the number the CLI
  prints in its own comparison table. `test_the_upstream_compliance_guard_is_preserved` pins it.
- **Three different layouts**, one per upstream branch: `fenics/<physics>/` (restarts = 1),
  `fenics/restart<r>/<physics>/` (restarts > 1, and then there is *no* flat copy — the best design
  inherits from the restart whose `final_loss` matched), and `fenics/rank0_<i>/<physics>/` plus a
  `fenics_validation.npz` summary for Pareto.
- **`rank0_<i>` is the i-th rank-0 solution, not design i.** The summary npz is keyed by the real design
  index; the directory fallback is not. They are separate functions (`load_pareto`, `load_pareto_dirs`)
  for exactly that reason — merging them pins ground truth to the wrong microstructure the moment any
  solution is dominated.
- **The runner rewrites `--fenics_output_dir`** the way it rewrites `--output_dir`. Upstream defaults it
  to `<output_dir>/fenics`, which is already inside the run; an explicit value in section F would put
  ground truth somewhere shared, where the panel cannot find it and two runs overwrite each other.

## Comparing designs across runs

`ui/compare.py`. Pins are `(run_id, design index)` in `session_state`, capped at 4, resolved against disk
on every render — so a pin cannot go stale against a re-run, and a deleted run degrades to a caption
rather than an exception. The spread column is relative to the mean so E (1e5) and alpha (1e-5) are
comparable, and is blank when a property is missing from any pin. Convergence curves are NaN-padded, not
zero- or last-value-padded: a run that converged in 40 iterations must stop being drawn at 40 rather than
flatline across the width of the longest run.

## The design deck

`design_slides.html` is self-contained — CSS, SVG figures, and navigation JS all inline. Keep it that way;
it is meant to be shared as a single file.

```bash
xdg-open design_slides.html
```

## Deck structure and conventions

Twelve `<section class="slide">` elements inside `<main class="deck">`. Only the one with `.active` is
displayed. The typical slide is:

```html
<section class="slide">
  <div class="kicker">Component N</div>       <!-- uppercase accent label -->
  <h2>Title</h2>
  <p class="sub">One-line framing.</p>
  <div class="cols">                          <!-- 2-col grid; add .wide for full bleed -->
    <div class="figwrap"><svg viewBox="0 0 W H" role="img" aria-label="…">…</svg></div>
    <ul class="notes"><li><b>Lead-in:</b> detail.</li></ul>
  </div>
</section>
```

**Adding or removing a slide requires no other edits.** The script at the bottom derives the total count,
builds the dot navigation, and wires prev/next from `document.querySelectorAll('.slide')`. The hardcoded
`11` in `<span id="tot">` is overwritten on load. Keyboard nav (arrows, PageUp/Down, Home/End) is global.

**Colour inside a figure must come from a class, never a `fill=` attribute.** `.txt` already sets `fill`,
and a CSS declaration beats a presentation attribute — so `class="txt" fill="var(--ok)"` renders as plain
ink. Every ✓/✗ in the deck was silently monochrome until `.txt.ok` / `.txt.bad` / `.txt.hl` were added.

## Theming

All colors come from CSS custom properties on `:root`, with a full override block under
`@media (prefers-color-scheme: dark)`. **Any new color variable must be defined in both blocks** or the
deck breaks in one theme. `--a`/`--b`/`--c`/`--d` are the categorical series colors used for data points.

## SVG figures

Figures are hand-authored inline SVG, not generated. They theme correctly only because they use the
shared utility classes rather than literal colors:

- `.txt` (with `.mut`, `.sm`, `.b` modifiers) for all text
- `.box` (with `.acc` for accent-filled) for panels and nodes
- `.chip` for muted background fills
- `.flow` for arrowed connector paths
- anything else: `fill="var(--…)"` / `stroke="var(--…)"`, never a hex literal

**Cross-slide dependency:** the `<marker id="arrow">` referenced by every `.flow` path in the deck is
defined once, in the `<defs>` of the slide-1 SVG. Deleting or restructuring that first figure will strip
the arrowheads from every other diagram.

Sizing: `svg` is width-100% with `max-height:460px`, so pick a `viewBox` whose aspect ratio fits that box
(existing figures are roughly 380×300 to 900×340). Always set `role="img"` and `aria-label`.

## Domain context

The deck describes a GUI layer over an existing optimizer stack; understanding the slides requires knowing
the upstream repos, which are siblings of this directory:

- `../amit_AI4NS/run_inverse_design_fm_multi_ac.py` — single-point optimizer CLI (IPOPT + Allen–Cahn)
- `../amit_AI4NS/run_pareto_epsilon_fm_multi_ac.py` — Pareto ε-constraint CLI
- `../amit_AI4NS/utils/optimization/` — the optimizer package an in-process backend would import
- `../compogen/` — the chat/MCP/HPC plugin proposed as a third execution backend

The deck's central design claim is that a `Runner` interface (`submit`/`stream`/`result`/`cancel`) is the
single seam between the GUI and the optimizer, so the three backends (subprocess, in-process, compogen)
are interchangeable — a claim the deck now retracts on its own Component 5 slide. Decided choices carry a
`<span class="tag done">` badge, unsettled ones `<span class="tag open">`; both are currently `done`
(subprocess, Streamlit). Keep them in sync with what has actually been decided — they are the deck's
status markers — and keep the final "Where it landed" slide honest about built / deviated / not built.

### Upstream facts the deck used to get wrong

The deck has been reconciled with the implementation, so these are no longer claims you will read there —
but they are the underlying facts, and each was verified against the source. Don't reintroduce them:

- **FEniCS validation is not "skipped gracefully" when the env is unavailable.** `run_fenics_validation`
  is called at `:701-726` with no try/except in the chain and shells `conda run` at `:283`. With
  `auto_activate_base: false` and a non-login subprocess, `conda` missing from the child's PATH is the
  *default* case → uncaught `FileNotFoundError`, fired *after* the solve completes but *before*
  `plot_results` at `:728`. A twenty-minute run dies without writing artifacts.
- **There is no upstream Pydantic schema to reuse** — zero hits for `pydantic`/`BaseModel` across
  `amit_AI4NS`. `RunConfig` was written from scratch against the argparse surface.
- **compogen has no logbook and no coupling to ai4ns.** Staging and cluster scale are real; there is no
  `read_design_logbook`, no `annotate_last_run`, and no inverse-design or Pareto tool in its 20-tool
  registry (strictly forward simulation). It depends on `compgen` (MOOSE/Sculpt).
- **The three backends are not interchangeable.** In-process forces the GUI process to *be* `cenv`, and
  leaves `cancel()` unimplementable — you cannot interrupt a thread inside C++ IPOPT.
- **"Tail stdout" only works with a PTY**; see the launching constraints above.
- **Neither script has any plasticity flag, directive, or checkpoint**, despite the `plasticity_yield_*.pt`
  sitting in `models/fm_multi_store/`.

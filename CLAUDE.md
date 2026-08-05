# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A GUI over the AI4NS inverse-design optimizer. **The application itself does not exist yet** — so far the
directory holds the design deck (`design_slides.html`) and the environment scaffolding needed to build it.

Not a git repository (yet). No tests.

## Environments — three of them, deliberately

```bash
./scripts/setup.sh            # idempotent; add --fenics for the optional validation env
```

| env | manager | Python | where | role |
|---|---|---|---|---|
| `.venv` | uv | 3.12 | `inverse_gui/.venv` | the GUI. Never solves. |
| `cenv` | conda-forge | 3.11 | `../envs/cenv` (big disk) | the optimizer |
| `fenics_env` | conda-forge | 3.11 | not built | optional PDE validation |

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

## Developing without checkpoints

The three `models/fm_multi_store/*.pt` EffPropNet checkpoints **do not exist anywhere on disk**, and
argparse hard-fails without at least one. The `.pt` files in the sibling `AI4NS/` repo are a different
generation (ResUNet) and will not load.

`scripts/fake_optimizer.py` unblocks everything else: it mirrors the CLI surface of both entry points,
emits a realistic IPOPT table **through ctypes printf** so it reproduces the real C-level buffering, writes
synthetic `.npz` artifacts with the real key sets, and has `--crash` / `--hang` for exercising the Runner's
failure and cancel paths. Pareto mode is selected by `--pareto_steps`, matching upstream.

## The design deck

`design_slides.html` is self-contained — CSS, SVG figures, and navigation JS all inline. Keep it that way;
it is meant to be shared as a single file.

```bash
xdg-open design_slides.html
```

## Deck structure and conventions

Eleven `<section class="slide">` elements inside `<main class="deck">`. Only the one with `.active` is
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
are interchangeable. Slides that describe unsettled choices carry a `<span class="tag open">` badge; the
three open decisions are enumerated on the final slide. When editing, keep those badges in sync with
whatever has actually been decided — they are the deck's status markers.

### Deck claims that are factually wrong

Verified against the source. Don't build on these:

- **Slide 9, "skipped gracefully if the solver env is unavailable."** False, and the failure is
  destructive. `run_fenics_validation` is called at `:701-726` with no try/except in the chain and shells
  `conda run` at `:283`. With `auto_activate_base: false` and a non-login subprocess, `conda` missing from
  the child's PATH is the *default* case → uncaught `FileNotFoundError`, fired *after* the solve completes
  but *before* `plot_results` at `:728`. A twenty-minute run dies without writing artifacts.
- **Slide 2, "the existing Pydantic schema."** There is none — zero hits for `pydantic`/`BaseModel` across
  `amit_AI4NS`. `RunConfig` must be written from scratch against the argparse surface.
- **Slide 6, compogen "reuses staging, logbook, and cluster scale."** Staging and scale are real; the
  logbook is not — no `read_design_logbook`, no `annotate_last_run`, and no inverse-design or Pareto tool
  in its 20-tool registry (the workflow is strictly forward simulation). compogen depends on `compgen`
  (MOOSE/Sculpt), not on ai4ns, and has no coupling to it at all.
- **Slide 6, the three backends as freely interchangeable.** In-process forces the GUI process to *be*
  `cenv`, and leaves `cancel()` unimplementable — you cannot interrupt a thread inside C++ IPOPT.
- **Slide 8, "tail stdout."** True only with a PTY; see the launching constraints above.

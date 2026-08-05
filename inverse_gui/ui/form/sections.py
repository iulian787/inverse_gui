"""Form sections A-F.

Deliberately not st.form: batching submission would kill live validation, the live
cost estimate and the command preview, which are the three things that make this
better than typing the CLI by hand.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from ...domain import properties as props
from ...domain.schema import (ALPHA_FOR_Y_CHOICES, MU_STRATEGY_CHOICES,
                              SCALING_CHOICES, RunConfig, RunMode)
from . import directive_rows


def render(rc: RunConfig, cfg, issues_by_field: dict, *, fenics_available: bool) -> None:
    _section_a(rc, cfg, issues_by_field)
    with st.expander('B · Property targets', expanded=True):
        directive_rows.render(rc, issues_by_field)
    with st.expander('C · Phase properties (A / B / interface)', expanded=False):
        _section_c(rc, issues_by_field)
    with st.expander('D · Tuning', expanded=False):
        _section_d(rc, issues_by_field)
    if rc.mode is RunMode.PARETO:
        with st.expander('E · Pareto sweep', expanded=True):
            _section_e(rc, issues_by_field)
    with st.expander('F · Validation', expanded=False):
        _section_f(rc, cfg, fenics_available)


# ---------------------------------------------------------------- A. checkpoints

def _section_a(rc: RunConfig, cfg, issues_by_field: dict) -> None:
    with st.expander('A · Physics / checkpoints', expanded=True):
        st.caption('Which checkpoints you load decides which property targets exist. '
                   'Plasticity is not available: neither script has any plasticity '
                   'checkpoint, directive or flag.')
        ckpt_dir = cfg.paths.ckpt_dir or str(cfg.ai4ns_repo / 'models' / 'fm_multi_store')
        for fam in props.FAMILIES:
            current = rc.checkpoints.path_for(fam.physics)
            val = st.text_input(
                fam.label, value=current, key=f'ckpt.{fam.physics}',
                placeholder=f'{ckpt_dir}/…{fam.physics}…_fmmulti_epoch*.pt',
                help=f'--{fam.ckpt_flag}. Enables: '
                     f'{", ".join(fam.all_props)}',
            )
            setattr(rc.checkpoints, fam.physics, val)
            for issue in issues_by_field.get(f'ckpt.{fam.physics}', []):
                st.caption(f'🔴 {issue.message}')
        for issue in issues_by_field.get('checkpoints', []):
            st.error(f'{issue.message}\n\n{issue.remedy}', icon='🔴')

        rc.act = st.text_input(
            'Activation override (--act)', value=rc.act, key='act',
            placeholder='leave empty — the checkpoint carries its own',
            help='An unknown name silently falls back to ReLU upstream, producing '
                 'wrong predictions if the checkpoint was trained with silu.',
        )
        for issue in issues_by_field.get('act', []):
            st.error(issue.message, icon='🔴')


# ---------------------------------------------------------------- C. phase props

def _section_c(rc: RunConfig, issues_by_field: dict) -> None:
    loaded = rc.checkpoints.loaded_physics()
    required = props.required_phase_flags(loaded)
    st.caption('MOOSE-native keys. Left blank, upstream resolves them from '
               '`--phase_props_json`, then the first `consolidated_*.h5` in '
               '`--data_dir`, then built-in defaults — first match wins, not a merge. '
               'These per-property overrides are applied last.')

    head = st.columns([2, 1.6, 1.6, 1.4])
    head[1].caption('**Phase A** (mask = 1)')
    head[2].caption('**Phase B**')
    head[3].caption('required by')

    for pf in props.PHASE_FLAGS:
        needed = pf.moose_key in required
        cols = st.columns([2, 1.6, 1.6, 1.4])
        label = f'{pf.label}' + (' *(unused)*' if pf.dead else '')
        cols[0].markdown(label, help=f'`{pf.moose_key}`')
        for i, (stem, store) in enumerate(((pf.flag_a, rc.phase.a),
                                           (pf.flag_b, rc.phase.b)), start=1):
            default = props.default_phase_value(pf.moose_key, 'A' if i == 1 else 'B')
            cur = store.get(stem)
            txt = cols[i].text_input(
                stem, value='' if cur is None else repr(cur), key=f'phase.{stem}',
                placeholder='' if default is None else f'{default:g}',
                label_visibility='collapsed', disabled=pf.dead,
            )
            store[stem] = _maybe_float(txt)
        if pf.dead:
            cols[3].caption('no physics')
        else:
            cols[3].caption(', '.join(sorted(
                p for p in loaded
                if pf.moose_key in props.REQUIRED_BULK_KEYS.get(p, ())
            )) or '—')

    st.divider()
    txt = st.text_input(
        'Interface conductivity (AB)', key='phase.interface_kappa_AB',
        value='' if rc.phase.interface_kappa_AB is None
        else repr(rc.phase.interface_kappa_AB),
        placeholder='1e7',
        help='--interface_kappa_AB → AB.interfacial_conductivity. Required by '
             'thermal_conductivity.',
    )
    rc.phase.interface_kappa_AB = _maybe_float(txt)

    rc.phase.phase_props_json = st.text_input(
        'phase_props_json', value=rc.phase.phase_props_json, key='phase.json',
        placeholder='optional JSON overriding the per-phase defaults')
    rc.phase.data_dir = st.text_input(
        'data_dir', value=rc.phase.data_dir, key='phase.data_dir',
        placeholder='optional directory containing consolidated_*.h5')

    for key in ('phase.data_dir', 'phase.interface_kappa_AB', 'phase.density'):
        for issue in issues_by_field.get(key, []):
            st.caption(f'🟡 {issue.message} — {issue.remedy}')
    for key, items in issues_by_field.items():
        if key.startswith('phase.'):
            for issue in items:
                if issue.severity.value == 'error':
                    st.error(f'{issue.message}\n\n{issue.remedy}', icon='🔴')


def _maybe_float(text: str) -> float | None:
    text = (text or '').strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


# ---------------------------------------------------------------- D. tuning

def _section_d(rc: RunConfig, issues_by_field: dict) -> None:
    c1, c2, c3 = st.columns(3)
    rc.restarts = c1.number_input('restarts', min_value=1, value=rc.restarts,
                                  key='restarts')
    rc.target_tol = c2.number_input(
        'target_tol', min_value=0.0, value=rc.target_tol, format='%.4g',
        key='target_tol',
        help='Half-width of the band around a `target` directive. Upstream warns '
             'that 0 makes it a point constraint and may be infeasible.')
    rc.seed = c3.number_input('seed', value=rc.seed, key='seed')

    rc.enforce_isotropy = st.checkbox(
        'enforce_isotropy', value=rc.enforce_isotropy, key='enforce_isotropy',
        help='Adds a derived G_xy = E/(2(1+nu)) band. Fires only when E and nu are '
             'both *target* directives and G_xy is free.')
    for key in ('restarts', 'target_tol', 'enforce_isotropy'):
        for issue in issues_by_field.get(key, []):
            st.caption(f'🟡 {issue.message} — {issue.remedy}')

    st.markdown('**Volume fraction**')
    if rc.mode is RunMode.SINGLE:
        c1, c2 = st.columns(2)
        rc.vf_min = c1.number_input('vf_min', 0.0, 1.0, value=rc.vf_min, key='vf_min')
        rc.vf_max = c2.number_input('vf_max', 0.0, 1.0, value=rc.vf_max, key='vf_max')
        st.caption('Bounds the **raw design variable**, which is not the same '
                   'quantity as a `rho` directive (that bounds the Allen–Cahn '
                   'filtered field).')
    else:
        lo, hi = rc.derived_vf_bounds()
        st.caption(f'Derived from the `rho` directive and `target_tol`: '
                   f'**[{lo:.4g}, {hi:.4g}]**. The Pareto script has no vf flags.')
    for issue in issues_by_field.get('vf', []):
        st.caption(f'{"🔴" if issue.severity.value == "error" else "ℹ️"} '
                   f'{issue.message} {issue.remedy}')

    st.markdown('**Allen–Cahn filter**')
    a = st.columns(5)
    rc.ac.epsi = a[0].number_input('ε', value=rc.ac.epsi, key='ac.epsi', format='%.4g')
    rc.ac.lam = a[1].number_input('λ', value=rc.ac.lam, key='ac.lam', format='%.4g')
    rc.ac.dt = a[2].number_input('Δτ', value=rc.ac.dt, key='ac.dt', format='%.4g')
    rc.ac.steps = a[3].number_input('steps', value=rc.ac.steps, key='ac.steps')
    rc.ac.sharpen_beta = a[4].number_input('sharpen β', value=rc.ac.sharpen_beta,
                                           key='ac.sharpen', format='%.4g')

    st.markdown('**Beam search**')
    b = st.columns(3)
    rc.beam.width = b[0].number_input('width (0 = off)', min_value=0,
                                      value=rc.beam.width, key='beam.width')
    rc.beam.depth = b[1].number_input('depth', min_value=1, value=rc.beam.depth,
                                      key='beam.depth')
    rc.beam.c = b[2].number_input('c', value=rc.beam.c, key='beam.c', format='%.4g')

    with st.expander('Advanced · IPOPT', expanded=False):
        i1 = st.columns(3)
        rc.ipopt.max_iter = i1[0].number_input('max_iter', min_value=1,
                                               value=rc.ipopt.max_iter, key='ip.max')
        rc.ipopt.tol = i1[1].number_input('tol', value=rc.ipopt.tol, format='%.3g',
                                          key='ip.tol')
        rc.ipopt.constr_viol_tol = i1[2].number_input(
            'constr_viol_tol', value=rc.ipopt.constr_viol_tol, format='%.3g',
            key='ip.cvt')
        i2 = st.columns(3)
        rc.ipopt.print_level = i2[0].number_input(
            'print_level', 0, 12, value=rc.ipopt.print_level, key='ip.print',
            help='5 (the default) produces the per-iteration table the live log '
                 'shows. 0 makes the run look frozen.')
        rc.ipopt.acceptable_tol = i2[1].number_input(
            'acceptable_tol', value=rc.ipopt.acceptable_tol, format='%.3g',
            key='ip.atol')
        rc.ipopt.acceptable_iter = i2[2].number_input(
            'acceptable_iter', min_value=0, value=rc.ipopt.acceptable_iter,
            key='ip.aiter')
        i3 = st.columns(3)
        rc.ipopt.scaling = i3[0].selectbox(
            'scaling', SCALING_CHOICES,
            index=SCALING_CHOICES.index(rc.ipopt.scaling), key='ip.scaling')
        rc.ipopt.mu_strategy = i3[1].selectbox(
            'mu_strategy', MU_STRATEGY_CHOICES,
            index=MU_STRATEGY_CHOICES.index(rc.ipopt.mu_strategy), key='ip.mu')
        rc.ipopt.lbfgs_history = i3[2].number_input(
            'lbfgs_history', min_value=1, value=rc.ipopt.lbfgs_history, key='ip.lbfgs')
        rc.ipopt.alpha_for_y = st.selectbox(
            'alpha_for_y', ALPHA_FOR_Y_CHOICES,
            index=ALPHA_FOR_Y_CHOICES.index(rc.ipopt.alpha_for_y), key='ip.alpha')
        st.caption('`--ipopt_recalc_y` is `store_true` with `default=True` upstream, '
                   'so it is always on and cannot be turned off from the CLI.')


# ---------------------------------------------------------------- E. pareto

def _section_e(rc: RunConfig, issues_by_field: dict) -> None:
    c = st.columns(3)
    rc.pareto_steps = c[0].number_input(
        'pareto_steps', min_value=0, value=rc.pareto_steps, key='pareto_steps',
        help='Grid resolution per epsilon axis. The grid is a Cartesian product, so '
             'the solve count grows as (2N-1) ** (number of max/min objectives).')
    rc.bin_viol_tol = c[1].number_input('bin_viol_tol', value=rc.bin_viol_tol,
                                        format='%.3g', key='bin_viol_tol')
    rc.feasibility_threshold = c[2].number_input(
        'feasibility_threshold', value=rc.feasibility_threshold, format='%.3g',
        key='feas_thresh',
        help='Above this, stage 0 converts target/range constraints into soft '
             'objective penalties instead of hard constraints.')

    objectives = rc.objective_props()
    constraints = rc.constraint_props()
    st.caption(
        f'ε-swept axes (non-ρ max/min): **{", ".join(objectives) or "none"}** · '
        f'hard constraints (target/range): **{", ".join(constraints) or "none"}**. '
        'ρ is always the objective and is never swept.')
    for key in ('pareto_steps', 'dry_run'):
        for issue in issues_by_field.get(key, []):
            st.caption(f'🟡 {issue.message} — {issue.remedy}')


# ---------------------------------------------------------------- F. validation

def _section_f(rc: RunConfig, cfg, fenics_available: bool) -> None:
    if not fenics_available:
        st.info(
            f'FEniCS validation is unavailable: the conda env '
            f'`{rc.fenics.conda_env}` was not found.\n\n'
            'This toggle stays disabled deliberately. Upstream calls the validator '
            'with no try/except, after the solve finishes but **before** artifacts '
            'are written — so a missing env destroys a completed run.',
            icon='🟡')
        st.code('conda env create -f '
                '<ai4ns>/fenics_validation/environment_fenics.yml', language='bash')
        rc.fenics.validate = False
        return

    rc.fenics.validate = st.checkbox(
        'Run FEniCS ground-truth validation', value=rc.fenics.validate,
        key='fenics.validate')
    rc.fenics.conda_env = st.text_input('conda env', value=rc.fenics.conda_env,
                                        key='fenics.env')
    st.caption('Per-phase FEniCS properties are a separate set from section C and '
               'are not kept in sync upstream.')

"""Section B: property targets.

Generated from the property tables rather than hand-written per property, so the
form stays in step with `domain.properties`.

Two invalid states are made structurally impossible rather than merely flagged:

* A family's rows exist only when its checkpoint is set. The single-point script
  does no gating of its own -- a target on an unpredictable property becomes a
  constraint with g=0 and zero gradient, i.e. permanent infeasibility reported as
  an Ipopt convergence failure.
* Isotropic and per-component rows are alternatives, never both. Upstream would
  silently keep the component value for one axis and apply the isotropic one to
  the other.
"""

from __future__ import annotations

import streamlit as st

from ...domain import properties as props
from ...domain.directives import Directive, Mode
from ...domain.schema import RunConfig, RunMode

MODES = [m.value for m in (Mode.FREE, Mode.MAX, Mode.MIN, Mode.TARGET, Mode.RANGE)]


def render(rc: RunConfig, issues_by_field: dict) -> None:
    loaded = rc.checkpoints.loaded_physics()

    if not loaded:
        st.info('Set a checkpoint in section A to choose property targets. '
                'Only properties the loaded models can predict are shown.')
    for fam in props.FAMILIES:
        if fam.physics in loaded:
            _family(rc, fam, issues_by_field)

    st.divider()
    _row(rc, 'rho', issues_by_field,
         help_text='Volume fraction of phase A (mask value 1). Always available — '
                   'it is the mask mean, computed without a model.')


def _family(rc: RunConfig, fam: props.Family, issues_by_field: dict) -> None:
    st.markdown(f'**{fam.label}**')
    key = f'aniso.{fam.physics}'
    default = 'Per-component' if fam.physics in rc.anisotropic else 'Isotropic'
    choice = st.segmented_control(
        'representation', ['Isotropic', 'Per-component'],
        default=default, key=key, label_visibility='collapsed',
    ) or default

    aniso = set(rc.anisotropic)
    if choice == 'Per-component':
        aniso.add(fam.physics)
    else:
        aniso.discard(fam.physics)
    rc.anisotropic = frozenset(aniso)

    isotropic = choice == 'Isotropic'
    if isotropic:
        st.caption(
            f'`--{fam.isotropic[0]}` is expanded upstream into '
            f'`{"`, `".join(props.ISOTROPIC_EXPAND.get(fam.isotropic[0], ()))}` — two '
            'independent constraints at half weight each, not a constraint on their mean.'
        )
    # Directives for rows that are not currently visible are cleared, so a hidden
    # widget cannot smuggle a value into the command line.
    visible = set(fam.props_for(isotropic))
    for prop in fam.all_props:
        if prop not in visible:
            rc.directives.pop(prop, None)

    for prop in fam.props_for(isotropic):
        _row(rc, prop, issues_by_field)
    st.write('')


def _row(rc: RunConfig, prop: str, issues_by_field: dict, help_text: str = '') -> None:
    d = rc.directives.get(prop) or Directive(prop=prop)
    unit = props.PROP_UNIT.get(prop, '')
    fmt = props.PROP_FORMAT.get(prop, '%.4g')
    single = rc.mode is RunMode.SINGLE

    cols = st.columns([1.1, 1.7, 1.6, 1.6, 0.9, 0.6] if single
                      else [1.1, 1.7, 1.6, 1.6, 0.6])
    label = f'{prop}' + (f'  ({unit})' if unit else '')
    cols[0].markdown(f'`{prop}`', help=help_text or (unit or None))

    # A selectbox rather than a segmented control: five options wrap onto four
    # lines in a pane this narrow, which pushes every row apart.
    mode_val = cols[1].selectbox(
        label, MODES, index=MODES.index(d.mode.value), key=f'dir.{prop}.mode',
        label_visibility='collapsed',
    )
    mode = Mode(mode_val)

    value, hi = d.value, d.hi
    if mode in (Mode.TARGET, Mode.RANGE):
        value = cols[2].number_input(
            'lo' if mode is Mode.RANGE else 'value',
            value=float(value) if value is not None
            else float(props.PROP_DEFAULTS.get(prop, 1.0)),
            format=fmt, key=f'dir.{prop}.value', label_visibility='collapsed',
        )
    else:
        cols[2].write('')
    if mode is Mode.RANGE:
        hi = cols[3].number_input(
            'hi',
            value=float(hi) if hi is not None
            else float(props.PROP_DEFAULTS.get(prop, 1.0)) * 1.1,
            format=fmt, key=f'dir.{prop}.hi', label_visibility='collapsed',
        )
    else:
        cols[3].write('')

    weight = d.weight
    if single:
        # Weights affect only max/min in the objective; for target/range they matter
        # solely through beam-search rescoring. Showing that as a disabled control
        # teaches the rule instead of warning about it afterwards.
        applies = mode.is_objective or rc.beam.width > 0
        weight = cols[4].number_input(
            'w', value=float(d.weight), step=0.5, min_value=0.0,
            key=f'dir.{prop}.w', label_visibility='collapsed',
            disabled=not applies,
            help=('Objective weight.' if applies else
                  'Weights apply to max/min. For target/range they only matter when '
                  'beam search is on (section D).'),
        )

    ref = d.ref
    with cols[-1].popover('⚙', width='stretch'):
        default_ref = props.PROP_DEFAULTS.get(prop, 1.0)
        st.caption(f'`--{prop}_ref` normalises this property so Ipopt sees O(1) '
                   'values. It does not change the feasible set, but it does change '
                   'how `constr_viol_tol` is interpreted.')
        ref = st.number_input(
            f'{prop}_ref', value=float(ref) if ref is not None else float(default_ref),
            format=fmt, key=f'dir.{prop}.ref', min_value=0.0,
        )
        if mode in (Mode.TARGET, Mode.RANGE) and value:
            # Widget callbacks run before Streamlit starts the next script render,
            # which is the only safe time to update another widget's state. Doing
            # this in ``if st.button(...)`` runs after the number_input above has
            # already been instantiated and raises StreamlitAPIException.
            st.button(
                f'Set to target ({value:{fmt[1:]}})',
                key=f'dir.{prop}.reffix',
                on_click=_set_widget_value,
                args=(f'dir.{prop}.ref', float(value)),
            )
        if ref == default_ref:
            ref = None

    rc.directives[prop] = Directive(prop=prop, mode=mode, value=value, hi=hi,
                                    weight=weight, ref=ref)

    for issue in issues_by_field.get(f'directive.{prop}', []):
        st.caption(f'🔴 {issue.message}' if issue.severity.value == 'error'
                   else f'🟡 {issue.message}')
    for issue in issues_by_field.get(f'ref.{prop}', []):
        st.caption(f'🟡 {issue.message} — {issue.remedy}')


def _set_widget_value(key: str, value: float) -> None:
    """Update widget state from Streamlit's pre-render callback phase."""
    st.session_state[key] = value

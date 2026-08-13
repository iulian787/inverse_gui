"""Small shared UI pieces."""

from __future__ import annotations

import numpy as np
import streamlit as st

from ..domain.validate import Issue, Severity

_ICON = {Severity.ERROR: '🔴', Severity.WARNING: '🟡', Severity.INFO: 'ℹ️'}


def issue_list(issues: list[Issue], *, show_info: bool = False) -> None:
    errors = [i for i in issues if i.severity is Severity.ERROR]
    warnings = [i for i in issues if i.severity is Severity.WARNING]
    infos = [i for i in issues if i.severity is Severity.INFO]

    for issue in errors:
        with st.container(border=True):
            st.markdown(f'{_ICON[issue.severity]} **{issue.message}**')
            if issue.remedy:
                st.caption(issue.remedy)
    for issue in warnings:
        st.warning(f'{issue.message}\n\n{issue.remedy}' if issue.remedy
                   else issue.message, icon='🟡')
    if infos and show_info:
        with st.expander(f'{len(infos)} note(s)', expanded=False):
            for issue in infos:
                st.caption(f'**{issue.message}** {issue.remedy}')


def inline_issues(issues: list[Issue]) -> None:
    """Compact per-field rendering, used inside form rows."""
    for issue in issues:
        if issue.severity is Severity.ERROR:
            st.error(issue.message, icon='🔴')
        elif issue.severity is Severity.WARNING:
            st.caption(f'🟡 {issue.message}')


def mask_image(mask, *, caption: str = '', width: int | str = 260):
    """Render a 128x128 binary field.

    st.image on a uint8 array rather than a Plotly heatmap: for a binary field the
    heatmap costs far more to serialise and looks no better. Flipped vertically to
    match the optimizer's own origin='lower' plots.

    `width` is pixels or one of Streamlit's keywords ('stretch', 'content'); None is
    rejected by st.image, so callers that want the column width pass 'stretch'.
    """
    if mask is None:
        st.caption('no microstructure in this artifact')
        return
    arr = np.asarray(mask)
    img = (arr[::-1] > 0.5).astype(np.uint8) * 255
    st.image(img, caption=caption, width=width, clamp=True)


def check_row(check) -> None:
    icon = '✅' if check.ok else ('🔴' if check.critical else '🟡')
    cols = st.columns([3, 6])
    cols[0].markdown(f'{icon} **{check.name}**')
    cols[1].caption(check.detail or '')
    if not check.ok and check.remedy:
        st.code(check.remedy, language='bash')


def kv_row(label: str, value: str, help: str = '') -> None:
    a, b = st.columns([2, 3])
    a.caption(label)
    b.markdown(f'`{value}`' if value else '—', help=help or None)

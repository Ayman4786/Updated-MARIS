from typing import Any

import streamlit as st


def render_sources(sources: list[dict[str, Any]]) -> int | None:
    st.subheader("Grounded Sources")
    if not sources:
        st.caption("No source pages were returned for this answer.")
        return None

    page_options: list[tuple[str, int]] = []
    for index, source in enumerate(sources, start=1):
        filename = source.get("filename") or "Unknown document"
        page = source.get("page")
        score = source.get("score")
        page_label = str(page) if page is not None else "unknown"
        score_label = f"{float(score):.3f}" if isinstance(score, (int, float)) else "unavailable"
        st.markdown(
            f"**{index}. {filename}**  \n"
            f"Source: Page {page_label}  \n"
            f"Relevance: {score_label}"
        )
        if isinstance(page, int) and page > 0:
            page_options.append((f"Page {page} - {filename}", page))

    if not page_options:
        return None

    selected_label = st.selectbox(
        "View a source page",
        [label for label, _ in page_options],
        key="source_page_selector",
    )
    return dict(page_options)[selected_label]

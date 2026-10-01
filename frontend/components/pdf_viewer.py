import base64

import streamlit as st
import streamlit.components.v1 as components


def _embedded_pdf(pdf_bytes: bytes, selected_page: int | None, height: int) -> None:
    encoded_pdf = base64.b64encode(pdf_bytes).decode("ascii")
    page_fragment = f"#page={selected_page}" if selected_page else ""
    components.html(
        f"""
        <iframe
            src="data:application/pdf;base64,{encoded_pdf}{page_fragment}"
            width="100%"
            height="{height}px"
            style="border: 1px solid #d9dde5; border-radius: 6px;"
            title="Uploaded PDF"
        ></iframe>
        """,
        height=height,
        scrolling=False,
    )


def display_pdf(
    pdf_bytes: bytes,
    selected_page: int | None = None,
    height: int = 720,
) -> None:
    if not pdf_bytes:
        st.info("Upload a PDF to open the reader.")
        return

    native_pdf = getattr(st, "pdf", None)
    if callable(native_pdf) and selected_page is None:
        native_pdf(pdf_bytes, height=height)
        return

    _embedded_pdf(pdf_bytes, selected_page, height)

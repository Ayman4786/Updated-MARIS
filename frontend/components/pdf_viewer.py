import streamlit as st


def display_pdf(
    pdf_bytes: bytes,
    selected_page: int | None = None,
    height: int = 720,
    pdf_url: str | None = None,
) -> None:
    if not pdf_bytes:
        st.info("Upload a PDF to open the reader.")
        return

    native_pdf = getattr(st, "pdf", None)

    if callable(native_pdf):
        native_pdf(
            pdf_bytes,
            height=height,
        )
        return

    st.error(
        "st.pdf is unavailable. Install the PDF support with: "
        "pip install 'streamlit[pdf]'"
    )
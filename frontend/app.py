import hashlib
from typing import Any

import streamlit as st

from api.client import BackendError, MarisClient
from components.chat_panel import render_chat_panel
from components.pdf_viewer import display_pdf
from components.source_panel import render_sources


st.set_page_config(
    page_title="MARIS",
    page_icon="M",
    layout="wide",
)


def initialize_state() -> None:
    defaults: dict[str, Any] = {
        "pdf_bytes": None,
        "filename": None,
        "document_id": None,
        "page_count": None,
        "chat_history": [],
        "sources": [],
        "selected_source_page": None,
        "latest_answer": None,
        "upload_signature": None,
        "upload_error_signature": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def reset_document_state(pdf_bytes: bytes, filename: str, signature: str) -> None:
    st.session_state.pdf_bytes = pdf_bytes
    st.session_state.filename = filename
    st.session_state.document_id = None
    st.session_state.page_count = None
    st.session_state.chat_history = []
    st.session_state.sources = []
    st.session_state.selected_source_page = None
    st.session_state.latest_answer = None
    st.session_state.upload_signature = signature
    st.session_state.upload_error_signature = None


def process_upload(uploaded_file: Any, client: MarisClient) -> None:
    filename = uploaded_file.name or "uploaded.pdf"
    pdf_bytes = uploaded_file.getvalue()
    signature = hashlib.sha256(pdf_bytes).hexdigest()

    if signature == st.session_state.upload_signature:
        return

    reset_document_state(pdf_bytes, filename, signature)

    if not filename.lower().endswith(".pdf") or not pdf_bytes.startswith(b"%PDF"):
        st.session_state.upload_error_signature = signature
        return

    try:
        with st.spinner("Processing document..."):
            upload_response = client.upload_pdf(pdf_bytes, filename)
    except BackendError as exc:
        st.session_state.upload_error_signature = signature
        st.error(str(exc))
        return

    document_id = upload_response.get("document_id")
    if upload_response.get("status") != "success" or not document_id:
        st.session_state.upload_error_signature = signature
        st.error("The backend did not finish processing this document.")
        return

    st.session_state.document_id = str(document_id)
    st.session_state.page_count = upload_response.get("pages")
    st.success(
        f"{filename} is ready. "
        f"{st.session_state.page_count or 'The document'} page(s) indexed."
    )


def main() -> None:
    initialize_state()
    client = MarisClient()

    st.title("MARIS")
    st.caption("Multimodal Document Intelligence")

    st.subheader("Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF",
        type=["pdf"],
        accept_multiple_files=False,
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        process_upload(uploaded_file, client)
        if st.session_state.upload_error_signature == st.session_state.upload_signature:
            st.error("Please choose a valid PDF and make sure the backend is running.")

    question, clear_chat = render_chat_panel(
        st.session_state.chat_history,
        enabled=bool(st.session_state.document_id),
    )

    if clear_chat:
        st.session_state.chat_history = []
        st.session_state.sources = []
        st.session_state.latest_answer = None
        st.session_state.selected_source_page = None
        st.rerun()

    if question is not None:
        if not question:
            st.sidebar.warning("Enter a question before asking MARIS.")
        else:
            try:
                with st.spinner("MARIS is thinking..."):
                    chat_response = client.ask(
                        question,
                        st.session_state.document_id,
                    )
            except BackendError as exc:
                st.sidebar.error(str(exc))
            else:
                answer = chat_response.get("answer")
                if not isinstance(answer, str):
                    st.sidebar.error("The backend returned no usable answer.")
                else:
                    sources = chat_response.get("sources")
                    if not isinstance(sources, list):
                        sources = []
                    st.session_state.chat_history.append(
                        {"question": question, "answer": answer}
                    )
                    st.session_state.latest_answer = answer
                    st.session_state.sources = sources
                    st.session_state.selected_source_page = None
                    st.rerun()

    if st.session_state.latest_answer:
        st.subheader("MARIS")
        st.write(st.session_state.latest_answer)

    left_column, right_column = st.columns([2, 1])
    with right_column:
        if st.session_state.latest_answer:
            selected_page = render_sources(st.session_state.sources)
            if selected_page is not None:
                st.session_state.selected_source_page = selected_page

    with left_column:
        st.subheader("PDF Reader")
        if st.session_state.pdf_bytes:
            page_label = ""
            if st.session_state.selected_source_page:
                page_label = f"Source page: {st.session_state.selected_source_page}"
            elif st.session_state.page_count:
                page_label = f"Pages: {st.session_state.page_count}"
            if page_label:
                st.caption(page_label)
            display_pdf(
                st.session_state.pdf_bytes,
                selected_page=st.session_state.selected_source_page,
            )
        else:
            st.info("Upload a PDF to open the reader and talk to MARIS.")


if __name__ == "__main__":
    main()

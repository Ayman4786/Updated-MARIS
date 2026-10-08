import hashlib
from typing import Any

import streamlit as st

from api.client import BackendError, MarisClient
from components.chat_panel import render_question_form
from components.pdf_viewer import display_pdf
from components.source_panel import render_sources


st.set_page_config(page_title="MARIS", page_icon="M", layout="wide")


def initialize_state() -> None:
    defaults: dict[str, Any] = {
        "documents": [],
        "conversations": [],
        "document_id": None,
        "conversation_id": None,
        "pdf_bytes": None,
        "filename": None,
        "page_count": None,
        "messages": [],
        "sources": [],
        "selected_source_page": None,
        "upload_signature": None,
        "upload_error_signature": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def refresh_documents(client: MarisClient) -> None:
    try:
        st.session_state.documents = client.list_documents()
    except BackendError as exc:
        st.sidebar.error(str(exc))


def select_document(client: MarisClient, document: dict[str, Any]) -> None:
    document_id = str(document["document_id"])
    if document_id == st.session_state.document_id:
        return
    try:
        pdf_bytes = client.get_document_pdf(document_id)
    except BackendError as exc:
        st.error(str(exc))
        return
    st.session_state.document_id = document_id
    st.session_state.filename = document.get("filename")
    st.session_state.page_count = document.get("pages")
    st.session_state.pdf_bytes = pdf_bytes
    st.session_state.conversation_id = None
    st.session_state.messages = []
    st.session_state.sources = []
    st.session_state.selected_source_page = None


def process_upload(uploaded_file: Any, client: MarisClient) -> None:
    filename = uploaded_file.name or "uploaded.pdf"
    pdf_bytes = uploaded_file.getvalue()
    signature = hashlib.sha256(pdf_bytes).hexdigest()
    if signature == st.session_state.upload_signature:
        return
    st.session_state.upload_signature = signature
    if not filename.lower().endswith(".pdf") or not pdf_bytes.startswith(b"%PDF"):
        st.session_state.upload_error_signature = signature
        return
    try:
        with st.spinner("Processing document..."):
            response = client.upload_pdf(pdf_bytes, filename)
    except BackendError as exc:
        st.session_state.upload_error_signature = signature
        st.error(str(exc))
        return
    if response.get("status") != "success" or not response.get("document_id"):
        st.session_state.upload_error_signature = signature
        st.error("The backend did not finish processing this document.")
        return
    st.session_state.upload_error_signature = None
    st.session_state.document_id = str(response["document_id"])
    st.session_state.filename = filename
    st.session_state.pdf_bytes = pdf_bytes
    st.session_state.page_count = response.get("pages")
    st.session_state.conversation_id = None
    st.session_state.messages = []
    st.session_state.sources = []
    refresh_documents(client)
    st.success(f"{filename} is ready.")


def load_conversation(client: MarisClient, conversation_id: str) -> None:
    try:
        conversation = client.get_conversation(conversation_id)
        document_id = conversation["document_id"]
        document = next(
            item for item in st.session_state.documents
            if item["document_id"] == document_id
        )
        select_document(client, document)
        st.session_state.conversation_id = conversation_id
        st.session_state.messages = conversation.get("messages", [])
        st.session_state.sources = []
    except (BackendError, StopIteration, KeyError) as exc:
        st.error(str(exc) or "Could not restore conversation.")


def create_chat(client: MarisClient) -> None:
    if not st.session_state.document_id:
        return
    try:
        conversation = client.create_conversation(st.session_state.document_id)
    except BackendError as exc:
        st.error(str(exc))
        return
    st.session_state.conversation_id = conversation["conversation_id"]
    st.session_state.messages = []
    st.session_state.sources = []


def render_document_library(client: MarisClient) -> None:
    st.subheader("Documents")
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"], key="pdf_upload")
    if uploaded_file is not None:
        process_upload(uploaded_file, client)
    for document in st.session_state.documents:
        document_id = document["document_id"]
        status = document.get("status", "unknown")
        marker = "● " if document_id == st.session_state.document_id else ""
        label = f"{marker}{document.get('filename', document_id)} ({status})"
        if st.button(label, key=f"document_{document_id}", use_container_width=True):
            select_document(client, document)
    if st.session_state.document_id and st.button(
        "＋ New chat for document", use_container_width=True
    ):
        create_chat(client)
        st.rerun()


def render_chat(client: MarisClient) -> None:
    st.subheader("MARIS")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["text"])
    question = render_question_form(bool(st.session_state.document_id))
    if question is None:
        return
    try:
        with st.spinner("MARIS is thinking..."):
            response = client.ask_in_conversation(
                question,
                st.session_state.document_id,
                st.session_state.conversation_id,
            )
    except BackendError as exc:
        st.error(str(exc))
        return
    answer = response.get("answer")
    if not isinstance(answer, str):
        st.error("The backend returned no usable answer.")
        return
    st.session_state.conversation_id = response.get(
        "conversation_id", st.session_state.conversation_id
    )
    st.session_state.messages.extend(
        [
            {"role": "user", "text": question},
            {
                "role": "assistant",
                "text": answer,
                "metadata": {
                    "sources": response.get("sources", []),
                    "images_used": response.get("images_used", []),
                },
            },
        ]
    )
    st.session_state.sources = response.get("sources", [])
    st.rerun()


def main() -> None:
    initialize_state()
    client = MarisClient()
    refresh_documents(client)

    st.title("MARIS")
    st.caption("Multimodal Document Intelligence")
    left, center, right = st.columns([1.1, 2.5, 1.1], gap="large")
    with left:
        render_document_library(client)
    with right:
        st.subheader("Chat History")
        if st.session_state.document_id:
            try:
                st.session_state.conversations = client.list_conversations(
                    st.session_state.document_id
                )
            except BackendError as exc:
                st.error(str(exc))
        for conversation in st.session_state.conversations:
            cid = conversation["conversation_id"]
            marker = "● " if cid == st.session_state.conversation_id else ""
            if st.button(
                marker + conversation.get("title", "New chat"),
                key=f"conversation_{cid}",
                use_container_width=True,
            ):
                load_conversation(client, cid)
                st.rerun()
        if st.session_state.document_id and st.button(
            "＋ New chat", use_container_width=True
        ):
            create_chat(client)
            st.rerun()
        if st.session_state.conversation_id and st.button(
            "Delete active conversation", use_container_width=True
        ):
            if st.session_state.get("confirm_delete"):
                client.delete_conversation(st.session_state.conversation_id)
                st.session_state.conversation_id = None
                st.session_state.messages = []
                st.session_state.confirm_delete = False
                st.rerun()
            st.session_state.confirm_delete = True
            st.warning("Click delete again to confirm.")
    with center:
        render_chat(client)
        if st.session_state.sources:
            page = render_sources(st.session_state.sources)
            if page is not None:
                st.session_state.selected_source_page = page
        st.subheader("PDF Reader")
        display_pdf(
            st.session_state.pdf_bytes,
            selected_page=st.session_state.selected_source_page,
        )


if __name__ == "__main__":
    main()

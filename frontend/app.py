from typing import Any

import hashlib

import streamlit as st

from api.client import BackendError, MarisClient
from components.chat_panel import render_question_form
from components.pdf_viewer import display_pdf
from components.source_panel import render_sources


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="MARIS",
    page_icon="M",
    layout="wide",
)


# ============================================================
# GLOBAL CSS
# ============================================================

st.markdown(
    """
    <style>

    :root {
        color-scheme: dark;
    }

    /* ========================================================
       PAGE
       ======================================================== */

    html,
    body {
        margin: 0;
        padding: 0;
        height: 100%;
        overflow: hidden;
    }

    [data-testid="stAppViewContainer"] {
        height: 100%;
        overflow: hidden;
        background: #0b1020;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    .block-container {
        box-sizing: border-box;

        width: 100%;
        max-width: 100%;

        height: 100dvh;

        padding:
            clamp(.65rem, 1.5dvw, 1.5rem)
            clamp(.65rem, 2dvw, 1.5rem);

        overflow: hidden;
    }

    /* ========================================================
       MAIN VERTICAL STREAMLIT BLOCK
       ======================================================== */

    [data-testid="stMainBlockContainer"] {
        height: 100%;
        min-height: 0;

        display: flex;
        flex-direction: column;
    }

    [data-testid="stMainBlockContainer"] > div {
        min-height: 0;
    }

    [data-testid="stMainBlockContainer"]
    > [data-testid="stVerticalBlock"] {
        height: 100%;
        min-height: 0;

        display: flex;
        flex-direction: column;
    }

    /* ========================================================
       THREE COLUMN ROW
       ======================================================== */

    [data-testid="stHorizontalBlock"] {
        width: 100%;

        min-height: 0;

        flex: 1 1 auto;

        align-items: stretch;

        gap: clamp(.5rem, 1.2dvw, 1.25rem);
    }

    /* ========================================================
       COLUMNS
       ======================================================== */

    [data-testid="stHorizontalBlock"]
    > [data-testid="column"] {
        box-sizing: border-box;

        min-width: 0;
        min-height: 0;

        display: flex;
        flex-direction: column;

        overflow: hidden;
    }

    [data-testid="stHorizontalBlock"]
    > [data-testid="column"]
    > [data-testid="stVerticalBlock"] {
        width: 100%;
        min-height: 0;

        display: flex;
        flex-direction: column;
    }

    /* ========================================================
       PANEL BORDER
       ======================================================== */

    [data-testid="stVerticalBlockBorderWrapper"] {
        box-sizing: border-box;

        min-height: 0;

        border: 1px solid #26324a;
        border-radius: 12px;

        background: #11182a;
    }

    [data-testid="stVerticalBlockBorderWrapper"] > div {
        box-sizing: border-box;

        min-height: 0;

        padding: clamp(.55rem, 1dvw, .8rem);
    }

    /* ========================================================
       TITLES
       ======================================================== */

    .maris-eyebrow {
        color: #8fa7ca;

        font-size: .72rem;
        letter-spacing: .16em;
        text-transform: uppercase;
    }

    .maris-panel-title {
        color: #f1f5ff;

        font-size: .86rem;
        font-weight: 700;
        letter-spacing: .12em;
    }

    /* ========================================================
       BUTTONS
       ======================================================== */

    [data-testid="stButton"] button {
        border-color: #2e3c59;
    }

    [data-testid="stButton"] button:hover {
        border-color: #69a5ff;
        color: #ffffff;
    }

    /* ========================================================
       SCROLLABLE STREAMLIT CONTAINERS
       ======================================================== */

    /*
       These selectors target Streamlit containers that have
       an explicit height. The actual scrolling is provided by
       Streamlit, not by fake HTML div wrappers.
    */

    [data-testid="stVerticalBlockBorderWrapper"] {
        overflow: hidden;
    }

    /* ========================================================
       MOBILE
       ======================================================== */

    @media (max-width: 900px) {

        html,
        body,
        [data-testid="stAppViewContainer"] {
            height: auto;
            overflow: visible;
        }

        .block-container {
            height: auto;
            min-height: 0;
            overflow: visible;

            padding: .8rem;
        }

        [data-testid="stMainBlockContainer"] {
            height: auto;
        }

        [data-testid="stHorizontalBlock"] {
            height: auto;

            flex: 0 0 auto;

            flex-wrap: wrap;
        }

        [data-testid="stHorizontalBlock"]
        > [data-testid="column"] {
            width: 100% !important;
            max-width: 100% !important;

            flex: 1 1 100% !important;

            overflow: visible;
        }

        [data-testid="stVerticalBlockBorderWrapper"] {
            max-height: none;
        }
    }

    /* ========================================================
       SHORT DESKTOP SCREENS
       ======================================================== */

    @media (min-width: 901px) and (max-height: 760px) {

        .maris-panel-title {
            font-size: .78rem;
        }

        .block-container {
            padding-top: .5rem;
            padding-bottom: .5rem;
        }
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

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


# ============================================================
# DOCUMENT FUNCTIONS
# ============================================================

def refresh_documents(client: MarisClient) -> None:
    try:
        st.session_state.documents = client.list_documents()
    except BackendError as exc:
        st.error(str(exc))


def select_document(
    client: MarisClient,
    document: dict[str, Any],
) -> None:
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


def process_upload(
    uploaded_file: Any,
    client: MarisClient,
) -> None:
    filename = uploaded_file.name or "uploaded.pdf"
    pdf_bytes = uploaded_file.getvalue()

    signature = hashlib.sha256(pdf_bytes).hexdigest()

    if signature == st.session_state.upload_signature:
        return

    st.session_state.upload_signature = signature

    if (
        not filename.lower().endswith(".pdf")
        or not pdf_bytes.startswith(b"%PDF")
    ):
        st.error("Please select a valid PDF file.")
        return

    try:
        with st.spinner("Processing document..."):
            response = client.upload_pdf(
                pdf_bytes,
                filename,
            )
    except BackendError as exc:
        st.error(str(exc))
        return

    if (
        response.get("status") != "success"
        or not response.get("document_id")
    ):
        st.error(
            "The backend did not finish processing this document."
        )
        return

    st.session_state.document_id = str(
        response["document_id"]
    )
    st.session_state.filename = filename
    st.session_state.pdf_bytes = pdf_bytes
    st.session_state.page_count = response.get("pages")

    st.session_state.conversation_id = None
    st.session_state.messages = []
    st.session_state.sources = []

    refresh_documents(client)


# ============================================================
# CONVERSATION FUNCTIONS
# ============================================================

def load_conversation(
    client: MarisClient,
    conversation_id: str,
) -> None:
    try:
        conversation = client.get_conversation(
            conversation_id
        )

        document = next(
            item
            for item in st.session_state.documents
            if item["document_id"]
            == conversation["document_id"]
        )

        select_document(client, document)

        st.session_state.conversation_id = conversation_id
        st.session_state.messages = conversation.get(
            "messages",
            [],
        )
        st.session_state.sources = []

    except (
        BackendError,
        StopIteration,
        KeyError,
    ) as exc:
        st.error(
            str(exc) or "Could not restore conversation."
        )


def create_chat(client: MarisClient) -> None:
    if not st.session_state.document_id:
        return

    try:
        conversation = client.create_conversation(
            st.session_state.document_id
        )
    except BackendError as exc:
        st.error(str(exc))
        return

    st.session_state.conversation_id = (
        conversation["conversation_id"]
    )
    st.session_state.messages = []
    st.session_state.sources = []


# ============================================================
# LEFT — DOCUMENT LIBRARY
# ============================================================

def render_document_library(
    client: MarisClient,
) -> None:

    st.markdown(
        '<div class="maris-panel-title">DOCUMENTS</div>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        key="pdf_upload",
    )

    if uploaded_file is not None:
        process_upload(
            uploaded_file,
            client,
        )

    for document in st.session_state.documents:

        document_id = document["document_id"]

        selected = (
            document_id
            == st.session_state.document_id
        )

        marker = "● " if selected else ""

        status = document.get(
            "status",
            "unknown",
        )

        label = (
            f"{marker}"
            f"{document.get('filename', document_id)} "
            f"({status})"
        )

        if st.button(
            label,
            key=f"document_{document_id}",
            use_container_width=True,
            type=(
                "primary"
                if selected
                else "secondary"
            ),
        ):
            select_document(
                client,
                document,
            )

            st.rerun()


# ============================================================
# LEFT — CHAT HISTORY
# ============================================================

def render_history(
    client: MarisClient,
) -> None:

    st.markdown(
        '<div class="maris-panel-title">CHAT HISTORY</div>',
        unsafe_allow_html=True,
    )

    if not st.session_state.document_id:

        st.caption(
            "Select a processed document to view its chats."
        )

        return

    if st.button(
        "＋ New Chat",
        use_container_width=True,
        type="primary",
    ):
        create_chat(client)
        st.rerun()

    for conversation in st.session_state.conversations:

        cid = conversation["conversation_id"]

        title = (
            conversation.get("title")
            or "New chat"
        )

        active = (
            cid
            == st.session_state.conversation_id
        )

        row = st.columns(
            [5, 1],
            gap="small",
        )

        with row[0]:

            if st.button(
                f"{'● ' if active else ''}{title}",
                key=f"conversation_{cid}",
                use_container_width=True,
                type=(
                    "primary"
                    if active
                    else "secondary"
                ),
            ):

                load_conversation(
                    client,
                    cid,
                )

                st.rerun()

        with row[1]:

            with st.popover(
                "⋮",
                use_container_width=True,
            ):

                if st.button(
                    "Rename",
                    key=f"rename_{cid}",
                    use_container_width=True,
                ):

                    st.session_state[
                        f"editing_{cid}"
                    ] = True

                if st.button(
                    "Delete",
                    key=f"delete_{cid}",
                    use_container_width=True,
                ):

                    try:
                        client.delete_conversation(
                            cid
                        )

                    except BackendError as exc:
                        st.error(str(exc))

                    else:

                        if (
                            cid
                            == st.session_state.conversation_id
                        ):
                            st.session_state.conversation_id = (
                                None
                            )
                            st.session_state.messages = []

                        st.rerun()

        if st.session_state.get(
            f"editing_{cid}"
        ):

            with st.form(
                f"rename_form_{cid}"
            ):

                new_title = st.text_input(
                    "Conversation title",
                    value=title,
                    max_chars=200,
                )

                if st.form_submit_button(
                    "Save name"
                ):

                    try:
                        client.rename_conversation(
                            cid,
                            new_title,
                        )

                    except BackendError as exc:
                        st.error(str(exc))

                    else:
                        st.session_state[
                            f"editing_{cid}"
                        ] = False

                        st.rerun()


# ============================================================
# RIGHT — CHAT MESSAGES
# ============================================================

def render_chat_messages() -> None:
    """
    Render only the messages.

    This container has a fixed height, so Streamlit itself
    provides the scrollbar.
    """

    message_height = 430

    with st.container(
        height=message_height,
        border=False,
    ):

        if not st.session_state.messages:

            st.caption(
                "Ask a question about the selected document."
            )

        else:

            for message in st.session_state.messages:

                with st.chat_message(
                    message["role"]
                ):

                    st.write(
                        message["text"]
                    )


# ============================================================
# RIGHT — QUESTION / CHAT PROCESSING
# ============================================================

def render_question_and_messages(
    client: MarisClient,
) -> None:

    st.markdown(
        '<div class="maris-panel-title">MARIS CHAT</div>',
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # SCROLLABLE CHAT MESSAGES
    # --------------------------------------------------------

    render_chat_messages()

    # --------------------------------------------------------
    # FIXED QUESTION COMPOSER
    # --------------------------------------------------------

    st.markdown(
        '<div class="maris-panel-title" '
        'style="margin-top:.6rem;">QUESTION</div>',
        unsafe_allow_html=True,
    )

    question = render_question_form(
        bool(st.session_state.document_id)
    )

    if question is None:
        return

    try:

        with st.spinner(
            "MARIS is thinking..."
        ):

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

        st.error(
            "The backend returned no usable answer."
        )

        return

    st.session_state.conversation_id = (
        response.get(
            "conversation_id",
            st.session_state.conversation_id,
        )
    )

    st.session_state.messages.extend(
        [
            {
                "role": "user",
                "text": question,
            },
            {
                "role": "assistant",
                "text": answer,
                "metadata": {
                    "sources": response.get(
                        "sources",
                        [],
                    ),
                    "images_used": response.get(
                        "images_used",
                        [],
                    ),
                },
            },
        ]
    )

    st.session_state.sources = response.get(
        "sources",
        [],
    )

    st.rerun()


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    initialize_state()

    client = MarisClient()

    refresh_documents(client)

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    st.title("MARIS")

    st.caption(
        "Multimodal Document Intelligence"
    )

    # --------------------------------------------------------
    # THREE PANELS
    # --------------------------------------------------------

    left, center, right = st.columns(
        [1.05, 2.8, 1.2],
        gap="medium",
    )

    # ========================================================
    # LEFT PANEL
    # ========================================================

    with left:

        with st.container(
            height=600,
            border=True,
        ):

            render_document_library(client)

            st.divider()

            # Load conversations for selected document.
            if st.session_state.document_id:

                try:

                    st.session_state.conversations = (
                        client.list_conversations(
                            st.session_state.document_id
                        )
                    )

                except BackendError as exc:

                    st.error(str(exc))

            render_history(client)

    # ========================================================
    # CENTER PANEL
    # ========================================================

    with center:

        with st.container(
            height=600,
            border=True,
        ):

            st.markdown(
                '<div class="maris-eyebrow">'
                "FULL PDF WORKSPACE"
                "</div>",
                unsafe_allow_html=True,
            )

            st.markdown(
                (
                    f"<h3>"
                    f"{st.session_state.filename or 'Select a document'}"
                    f"</h3>"
                ),
                unsafe_allow_html=True,
            )

            # ------------------------------------------------
            # PDF
            # ------------------------------------------------

            display_pdf(
                st.session_state.pdf_bytes,
                selected_page=(
                    st.session_state.selected_source_page
                ),
                height=500,
                pdf_url=(
                    client.get_document_pdf_url(
                        st.session_state.document_id
                    )
                    if st.session_state.document_id
                    else None
                ),
            )

            # ------------------------------------------------
            # GROUNDED SOURCES
            # ------------------------------------------------

            st.markdown(
                '<div class="maris-eyebrow" '
                'style="margin-top:1rem;">'
                "GROUNDED SOURCES"
                "</div>",
                unsafe_allow_html=True,
            )

            page = render_sources(
                st.session_state.sources
            )

            if page is not None:

                st.session_state.selected_source_page = (
                    page
                )

    # ========================================================
    # RIGHT PANEL
    # ========================================================

    with right:

        with st.container(
            height=600,
            border=True,
        ):

            render_question_and_messages(
                client
            )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
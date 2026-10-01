from typing import Any

import streamlit as st


def render_chat_panel(
    chat_history: list[dict[str, Any]],
    enabled: bool,
) -> tuple[str | None, bool]:
    with st.sidebar:
        st.subheader("Talk to MARIS")
        st.caption("Ask questions about this document.")

        question = None
        if enabled:
            with st.form("ask_maris_form", clear_on_submit=True):
                question_input = st.text_input("Question", placeholder="Ask about the document")
                submitted = st.form_submit_button("Ask MARIS", use_container_width=True)
            if submitted:
                question = question_input.strip()
        else:
            st.info("Upload and process a PDF to start asking questions.")

        clear_chat = st.button(
            "Clear chat",
            use_container_width=True,
            disabled=not chat_history,
        )

        if chat_history:
            st.divider()
            st.caption("Chat history")
            for entry in chat_history:
                with st.chat_message("user"):
                    st.write(entry["question"])
                with st.chat_message("assistant"):
                    st.write(entry["answer"])

    return question, clear_chat

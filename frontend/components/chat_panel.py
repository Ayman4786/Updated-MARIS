from typing import Any

import streamlit as st


def render_question_form(enabled: bool) -> str | None:
    if not enabled:
        st.info("Select a processed PDF to start asking questions.")
        return None
    with st.form("ask_maris_form", clear_on_submit=True):
        question = st.text_input("Question", placeholder="Ask about the document")
        submitted = st.form_submit_button("Ask MARIS", use_container_width=True)
    return question.strip() if submitted else None


def render_history_panel(
    conversations: list[dict[str, Any]],
    active_id: str | None,
) -> tuple[str | None, bool]:
    st.subheader("Chat History")
    new_chat = st.button("＋ New chat", use_container_width=True)
    selected = None
    for conversation in conversations:
        label = conversation.get("title") or "New chat"
        if st.button(
            f"{'● ' if conversation.get('conversation_id') == active_id else ''}{label}",
            key=f"conversation_{conversation.get('conversation_id')}",
            use_container_width=True,
        ):
            selected = conversation.get("conversation_id")
    delete_id = None
    if active_id and st.button("Delete active conversation", use_container_width=True):
        delete_id = active_id
    return selected, new_chat or bool(delete_id)


def render_chat_panel(
    chat_history: list[dict[str, Any]], enabled: bool
) -> tuple[str | None, bool]:
    """Compatibility wrapper retained for callers of the original component."""
    with st.sidebar:
        st.subheader("Talk to MARIS")
        question = render_question_form(enabled)
        clear_chat = st.button("Clear chat", disabled=not chat_history)
    return question, clear_chat

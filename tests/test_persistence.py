import hashlib

from backend.persistence import MarisStore
from ai.llm.prompt_builder import build_prompt


def test_document_hash_reuses_successful_registration(tmp_path):
    store = MarisStore(tmp_path / "maris.db")
    payload = b"%PDF-one"
    digest = hashlib.sha256(payload).hexdigest()

    action, _ = store.begin_document(digest, "same.pdf", "same_a1")
    assert action == "process"
    store.complete_document("same_a1", 2)

    action, document = store.begin_document(digest, "renamed.pdf", "same_b2")
    assert action == "reuse"
    assert document["document_id"] == "same_a1"


def test_same_filename_with_different_bytes_gets_separate_documents(tmp_path):
    store = MarisStore(tmp_path / "maris.db")
    first = hashlib.sha256(b"%PDF-first").hexdigest()
    second = hashlib.sha256(b"%PDF-second").hexdigest()
    store.begin_document(first, "report.pdf", "report_a1")
    store.complete_document("report_a1", 1)
    store.begin_document(second, "report.pdf", "report_b2")
    store.complete_document("report_b2", 1)

    document_ids = {item["document_id"] for item in store.list_documents()}
    assert {"report_a1", "report_b2"} <= document_ids


def test_failed_registration_can_be_retried(tmp_path):
    store = MarisStore(tmp_path / "maris.db")
    digest = hashlib.sha256(b"%PDF-retry").hexdigest()
    store.begin_document(digest, "retry.pdf", "retry_a1")
    store.fail_document("retry_a1", "extractor failed")

    action, document = store.begin_document(digest, "retry.pdf", "retry_b2")
    assert action == "process"
    assert document["document_id"] == "retry_a1"
    assert document["status"] == "processing"


def test_conversation_and_messages_survive_reopening(tmp_path):
    database = tmp_path / "maris.db"
    store = MarisStore(database)
    digest = hashlib.sha256(b"%PDF-chat").hexdigest()
    store.begin_document(digest, "chat.pdf", "chat_a1")
    store.complete_document("chat_a1", 1)
    conversation = store.create_conversation("chat_a1")
    store.add_message(conversation["conversation_id"], "user", "What is this?")
    store.add_message(
        conversation["conversation_id"],
        "assistant",
        "It is a document.",
        {"sources": [{"page": 1}]},
    )

    reopened = MarisStore(database)
    restored = reopened.get_conversation(conversation["conversation_id"])
    assert restored["document_id"] == "chat_a1"
    assert [message["role"] for message in reopened.list_messages(
        conversation["conversation_id"]
    )] == ["user", "assistant"]
    assert reopened.delete_conversation(conversation["conversation_id"])
    assert reopened.get_document("chat_a1")["status"] == "success"


def test_prompt_contains_only_bounded_follow_up_context():
    prompt = build_prompt(
        "The document context.",
        "What does that mean?",
        conversation_history=[
            {"role": "user", "text": "Define the term."},
            {"role": "assistant", "text": "It means a concept."},
        ],
    )
    assert "Define the term." in prompt
    assert "What does that mean?" in prompt

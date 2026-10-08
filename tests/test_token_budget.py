from ai.rag.context_builder import ContextBuilder
from backend.routes.chat import bound_image_paths
from ai.llm.llm_service import (
    is_daily_token_quota_error,
    is_request_size_error,
)


def _chunk(number, text, score):
    return {
        "score": score,
        "text": text,
        "metadata": {
            "filename": "diagram.pdf",
            "page_number": number,
            "heading": "Architecture",
            "document_id": "doc-a",
        },
    }


def test_context_budget_keeps_highest_ranked_chunks_first():
    context = ContextBuilder.build_context(
        [
            _chunk(2, "AI-Powered Skill Intelligence Platform: Role -> Evidence", 0.9),
            _chunk(6, "Turbo C logo and references", 0.4),
        ],
        max_chars=1000,
    )

    assert "Page 2" in context
    assert context.index("Page 2") < context.index("Page 6")


def test_context_deduplicates_repeated_chunk_text():
    context = ContextBuilder.build_context(
        [
            _chunk(2, "Role -> Evidence -> Capability", 0.9),
            _chunk(2, "Role -> Evidence -> Capability", 0.8),
        ],
        max_chars=1000,
    )

    assert context.count("Role -> Evidence -> Capability") == 1


def test_final_image_list_is_bounded_and_unique():
    assert bound_image_paths(["a.png", "a.png", "b.png"], max_images=1) == [
        "a.png"
    ]


def test_empty_image_list_preserves_text_only_behavior():
    assert bound_image_paths([], max_images=1) == []


def test_groq_size_errors_are_detected_for_graceful_response():
    assert is_request_size_error(
        RuntimeError("maximum context length exceeded")
    )


def test_groq_daily_quota_errors_are_distinguished_from_request_size_errors():
    error = RuntimeError(
        "Rate limit reached on tokens per day (TPD): Limit 200000"
    )
    assert is_daily_token_quota_error(error)
    assert not is_request_size_error(error)

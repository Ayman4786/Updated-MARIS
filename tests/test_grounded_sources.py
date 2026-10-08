from backend.routes.chat import deduplicate_grounded_sources
from frontend.components.source_panel import deduplicate_display_sources


def test_duplicate_chunks_same_page_keep_highest_score():
    chunks = [
        {
            "metadata": {
                "document_id": "doc-a",
                "filename": "report.pdf",
                "page_number": 2,
            },
            "score": 0.42,
        },
        {
            "metadata": {
                "document_id": "doc-a",
                "filename": "report.pdf",
                "page_number": 2,
            },
            "score": 0.91,
        },
    ]

    assert deduplicate_grounded_sources(chunks) == [
        {
            "document_id": "doc-a",
            "filename": "report.pdf",
            "page": 2,
            "score": 0.91,
        }
    ]


def test_different_pages_same_document_remain_separate():
    chunks = [
        {
            "metadata": {"document_id": "doc-a", "page_number": 1},
            "score": 0.5,
        },
        {
            "metadata": {"document_id": "doc-a", "page_number": 2},
            "score": 0.6,
        },
    ]

    sources = deduplicate_grounded_sources(chunks)

    assert [(source["document_id"], source["page"]) for source in sources] == [
        ("doc-a", 1),
        ("doc-a", 2),
    ]


def test_same_page_different_documents_remain_separate():
    sources = deduplicate_display_sources(
        [
            {"document_id": "doc-a", "filename": "a.pdf", "page": 1, "score": 0.7},
            {"document_id": "doc-b", "filename": "b.pdf", "page": 1, "score": 0.8},
        ]
    )

    assert {(source["document_id"], source["page"]) for source in sources} == {
        ("doc-a", 1),
        ("doc-b", 1),
    }

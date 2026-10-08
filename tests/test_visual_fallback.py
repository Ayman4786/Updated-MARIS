import asyncio
import types
from PIL import Image
from pathlib import Path
from unittest.mock import MagicMock, patch

from backend.routes.chat import (
    chat,
    identify_most_relevant_pages,
    render_relevant_full_pages,
)


class QuestionRequest:
    def __init__(self, question, document_id=None):
        self.question = question
        self.document_id = document_id


def _common_mocks(mock_load_docs, mock_retriever):
    mock_load_docs.return_value = ([], [MagicMock(name="doc1")])
    mock_retriever.return_value.retrieve.return_value = [
        {
            "metadata": {
                "document_id": "doc1",
                "page_number": 4,
                "image_paths": ["crop.png"],
            },
            "text": "process flow diagram",
            "score": 1.0,
        }
    ]


@patch("backend.routes.chat.generate_answer")
@patch("backend.routes.chat.render_relevant_full_pages")
@patch("backend.routes.chat.rank_visual_candidates")
@patch("backend.routes.chat.load_all_documents")
@patch("backend.routes.chat.HybridRetriever")
@patch("backend.routes.chat.VectorStoreManager")
@patch("backend.routes.chat.EmbeddingsService")
def test_rejected_crops_use_verified_full_page(
    mock_emb,
    mock_vs,
    mock_retriever,
    mock_load_docs,
    mock_rank,
    mock_render_pages,
    mock_generate,
):
    crop = Path("crop.png")
    fallback = Path("full-page.png")
    crop.touch()
    fallback.touch()
    try:
        _common_mocks(mock_load_docs, mock_retriever)
        mock_rank.return_value = (
            [str(crop)],
            [{
                "image_path": str(crop),
                "score": 0.9,
                "type": "picture",
                "document_id": "doc1",
                "page_number": 4,
            }],
        )
        mock_render_pages.return_value = [{
            "image_path": str(fallback),
            "score": 0.0,
            "type": "full_page_fallback",
            "document_id": "doc1",
            "page_number": 4,
        }]
        mock_generate.side_effect = [
            '{"relevant": false, "confidence": 0.9, "reason": "logo"}',
            '{"relevant": true, "confidence": 0.9, "reason": "process flow"}',
            "answer",
        ]

        result = asyncio.run(chat(QuestionRequest(question="Explain the process flow")))

        assert result["images_used"] == [str(fallback)]
        assert result["answer"] == "answer"
    finally:
        crop.unlink(missing_ok=True)
        fallback.unlink(missing_ok=True)


@patch("backend.routes.chat.generate_answer")
@patch("backend.routes.chat.render_relevant_full_pages")
@patch("backend.routes.chat.rank_visual_candidates")
@patch("backend.routes.chat.load_all_documents")
@patch("backend.routes.chat.HybridRetriever")
@patch("backend.routes.chat.VectorStoreManager")
@patch("backend.routes.chat.EmbeddingsService")
def test_rejected_full_page_sends_no_image(
    mock_emb,
    mock_vs,
    mock_retriever,
    mock_load_docs,
    mock_rank,
    mock_render_pages,
    mock_generate,
):
    crop = Path("crop-rejected.png")
    fallback = Path("full-page-rejected.png")
    crop.touch()
    fallback.touch()
    try:
        _common_mocks(mock_load_docs, mock_retriever)
        mock_rank.return_value = (
            [str(crop)],
            [{"image_path": str(crop), "score": 0.9, "type": "picture"}],
        )
        mock_render_pages.return_value = [{
            "image_path": str(fallback),
            "score": 0.0,
            "type": "full_page_fallback",
            "document_id": "doc1",
            "page_number": 4,
        }]
        mock_generate.side_effect = [
            '{"relevant": false, "confidence": 1.0, "reason": "logo"}',
            '{"relevant": false, "confidence": 1.0, "reason": "heading"}',
            "text answer",
        ]

        result = asyncio.run(chat(QuestionRequest(question="Explain the process flow")))

        assert result["images_used"] == []
        assert mock_generate.call_args_list[-1].kwargs["image_paths"] == []
    finally:
        crop.unlink(missing_ok=True)
        fallback.unlink(missing_ok=True)


def test_full_page_render_preserves_document_and_page_metadata(tmp_path):
    document_dir = tmp_path / "doc-42"
    document_dir.mkdir()
    (document_dir / "source.pdf").write_bytes(b"%PDF")

    class Bitmap:
        def to_pil(self):
            return Image.new("RGB", (20, 20), "white")

    class Page:
        def render(self, scale):
            assert scale == 2.0
            return Bitmap()

    class Pdf:
        def __getitem__(self, index):
            assert index == 6
            return Page()

    fake_pdfium = types.SimpleNamespace(PdfDocument=lambda path: Pdf())
    chunks = [{
        "text": "process flow diagram",
        "score": 1.0,
        "metadata": {"document_id": "doc-42", "page_number": 7},
    }]

    with patch("pypdfium2.PdfDocument", fake_pdfium.PdfDocument):
        candidates = render_relevant_full_pages(chunks, tmp_path, max_pages=3)

    assert candidates[0]["document_id"] == "doc-42"
    assert candidates[0]["page_number"] == 7
    assert candidates[0]["type"] == "full_page_fallback"
    assert Path(candidates[0]["image_path"]).exists()


def test_specific_page_text_beats_broad_visual_similarity():
    chunks = [
        {
            "text": "Research and References. Turbo C comparison table.",
            "score": 1.4,
            "metadata": {
                "document_id": "sih",
                "page_number": 6,
                "image_paths": ["logo.png"],
            },
        },
        {
            "text": (
                "AI-Powered Skill Intelligence Platform: "
                "Role -> Evidence -> Capability -> Gap -> Action"
            ),
            "score": 0.8,
            "metadata": {
                "document_id": "sih",
                "page_number": 2,
                "image_paths": ["diagram.png"],
            },
        },
    ]

    pages = identify_most_relevant_pages(
        chunks,
        "What is that AI-Powered Skill Intelligence Platform?",
    )

    assert pages[0] == ("sih", 2)

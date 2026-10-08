import asyncio
from unittest.mock import Mock, patch

import requests
import pytest

from ai.llm.prompt_builder import build_prompt
from backend.routes.chat import QuestionRequest, chat
from backend.services.serpapi import WebSearchError, search_web


def test_search_web_normalizes_organic_results():
    response = Mock()
    response.json.return_value = {
        "organic_results": [
            {
                "title": "MARIS",
                "link": "https://www.example.com/maris",
                "snippet": "A document assistant.",
            },
            {"title": "Missing URL"},
            "malformed",
        ]
    }
    response.raise_for_status.return_value = None

    with patch("backend.services.serpapi.load_dotenv"), patch(
        "backend.services.serpapi.os.getenv", return_value="configured"
    ), patch("backend.services.serpapi.requests.get", return_value=response):
        results = search_web("MARIS", num_results=5)

    assert results == [{
        "title": "MARIS",
        "url": "https://www.example.com/maris",
        "snippet": "A document assistant.",
        "source": "example.com",
    }]


def test_search_web_requires_api_key():
    with patch("backend.services.serpapi.load_dotenv"), patch(
        "backend.services.serpapi.os.getenv", return_value=None
    ):
        with pytest.raises(WebSearchError, match="SERPAPI_API_KEY"):
            search_web("MARIS")


def test_search_web_api_failure_is_normalized():
    response = Mock()
    response.raise_for_status.side_effect = requests.Timeout("unexpected")

    with patch("backend.services.serpapi.load_dotenv"), patch(
        "backend.services.serpapi.os.getenv", return_value="configured"
    ), patch("backend.services.serpapi.requests.get", return_value=response):
        with pytest.raises(WebSearchError, match="temporarily unavailable"):
            search_web("MARIS")


def test_question_request_web_search_defaults_to_disabled():
    assert QuestionRequest(question="What is MARIS").web_search is False
    assert QuestionRequest(question="What is current?", web_search=True).web_search


def test_prompt_keeps_web_results_separate_from_document_context():
    prompt = build_prompt(
        document_text="Uploaded PDF evidence.",
        user_question="What is current?",
        web_results=[{
            "title": "External result",
            "url": "https://example.com",
            "snippet": "External evidence.",
        }],
    )

    assert "DOCUMENT CONTEXT:\nUploaded PDF evidence." in prompt
    assert "WEB SEARCH RESULTS" in prompt
    assert "External result" in prompt
    assert "not extracted from the uploaded document" in prompt
    assert "current," in prompt
    assert "up-to-date information not contained in the document" in prompt
    assert "provided web results when they are relevant" in prompt
    assert "Never present web information as if it came from the uploaded document" in prompt


def test_enabled_web_search_is_returned_separately(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "backend.routes.chat.DOCUMENTS_ROOT",
        tmp_path,
    )
    monkeypatch.setattr(
        "backend.routes.chat.load_all_documents",
        lambda _: ([], [tmp_path / "doc1"]),
    )
    retriever = Mock()
    retriever.retrieve.return_value = []
    monkeypatch.setattr("backend.routes.chat.HybridRetriever", Mock(return_value=retriever))
    monkeypatch.setattr("backend.routes.chat.EmbeddingsService", Mock())
    monkeypatch.setattr("backend.routes.chat.VectorStoreManager", Mock())
    captured_prompts: list[str] = []

    def capture_answer(prompt, image_paths=None):
        captured_prompts.append(prompt)
        return "answer"

    monkeypatch.setattr(
        "backend.routes.chat.search_web",
        Mock(return_value=[{
            "title": "UNIQUE_WEB_TEST_TITLE_12345",
            "url": "https://example.com/web-test",
            "snippet": "UNIQUE_WEB_TEST_FACT_98765",
            "source": "example.com",
        }]),
    )
    monkeypatch.setattr("backend.routes.chat.generate_answer", capture_answer)

    result = asyncio.run(chat(QuestionRequest(question="What is current?", web_search=True)))

    assert result["answer"] == "answer"
    assert result["sources"] == []
    assert result["web_sources"][0]["title"] == "UNIQUE_WEB_TEST_TITLE_12345"
    assert len(captured_prompts) == 1
    assert "UNIQUE_WEB_TEST_TITLE_12345" in captured_prompts[0]
    assert "UNIQUE_WEB_TEST_FACT_98765" in captured_prompts[0]

"""Small SerpAPI client for optional external web search."""

from __future__ import annotations

import os

from urllib.parse import urlparse

import requests
from dotenv import load_dotenv


SERPAPI_URL = "https://serpapi.com/search.json"
DEFAULT_RESULT_COUNT = 5


class WebSearchError(RuntimeError):
    """Raised when an optional web search cannot be completed."""


def _source_from_url(url: str) -> str | None:
    hostname = urlparse(url).hostname
    return hostname.removeprefix("www.") if hostname else None


def search_web(
    query: str,
    num_results: int = DEFAULT_RESULT_COUNT,
) -> list[dict[str, str]]:
    """Return normalized Google web results from SerpAPI."""
    load_dotenv()
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        raise WebSearchError(
            "Web search is unavailable because SERPAPI_API_KEY is not configured."
        )

    try:
        response = requests.get(
            SERPAPI_URL,
            params={
                "engine": "google",
                "q": query,
                "api_key": api_key,
                "num": max(1, min(num_results, 10)),
            },
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        raise WebSearchError("Web search is temporarily unavailable.") from exc
    except ValueError as exc:
        raise WebSearchError("Web search returned an invalid response.") from exc

    if not isinstance(payload, dict):
        raise WebSearchError("Web search returned an invalid response.")
    if payload.get("error"):
        raise WebSearchError("Web search returned an API error.")

    organic_results = payload.get("organic_results", [])
    if not isinstance(organic_results, list):
        return []

    normalized: list[dict[str, str]] = []
    for result in organic_results:
        if not isinstance(result, dict):
            continue
        title = result.get("title")
        url = result.get("link")
        snippet = result.get("snippet")
        if not isinstance(title, str) or not isinstance(url, str):
            continue
        item: dict[str, str] = {
            "title": title,
            "url": url,
        }
        if isinstance(snippet, str) and snippet.strip():
            item["snippet"] = snippet
        source = _source_from_url(url)
        if source:
            item["source"] = source
        normalized.append(item)
        if len(normalized) >= max(1, min(num_results, 10)):
            break
    return normalized

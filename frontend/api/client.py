from typing import Any

import requests

from utils.config import get_backend_url


class BackendError(RuntimeError):
    pass


class MarisClient:
    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = (base_url or get_backend_url()).rstrip("/")

    def upload_pdf(self, pdf_bytes: bytes, filename: str) -> dict[str, Any]:
        try:
            response = requests.post(
                f"{self.base_url}/upload",
                files={"file": (filename, pdf_bytes, "application/pdf")},
                timeout=900,
            )
        except requests.RequestException as exc:
            raise BackendError(
                "MARIS backend is unavailable. Start FastAPI and try again."
            ) from exc

        return self._parse_response(response, "Document upload failed")

    def ask(self, question: str, document_id: str) -> dict[str, Any]:
        try:
            response = requests.post(
                f"{self.base_url}/chat",
                json={"question": question, "document_id": document_id},
                timeout=300,
            )
        except requests.RequestException as exc:
            raise BackendError(
                "MARIS backend is unavailable. Check the FastAPI server and try again."
            ) from exc

        return self._parse_response(response, "MARIS could not answer the question")

    @staticmethod
    def _parse_response(response: requests.Response, failure_message: str) -> dict[str, Any]:
        try:
            payload = response.json()
        except ValueError as exc:
            raise BackendError(f"{failure_message}: the backend returned invalid data.") from exc

        if not response.ok:
            detail = payload.get("detail") if isinstance(payload, dict) else None
            message = f"{failure_message}."
            if detail:
                message = f"{message} {detail}"
            raise BackendError(message)

        if not isinstance(payload, dict):
            raise BackendError(f"{failure_message}: the backend returned an unexpected response.")

        return payload

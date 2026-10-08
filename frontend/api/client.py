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
        return self.ask_in_conversation(question, document_id, None)

    def ask_in_conversation(
        self,
        question: str,
        document_id: str,
        conversation_id: str | None,
    ) -> dict[str, Any]:
        try:
            response = requests.post(
                f"{self.base_url}/chat",
                json={
                    "question": question,
                    "document_id": document_id,
                    "conversation_id": conversation_id,
                },
                timeout=300,
            )
        except requests.RequestException as exc:
            raise BackendError(
                "MARIS backend is unavailable. Check the FastAPI server and try again."
            ) from exc

        return self._parse_response(response, "MARIS could not answer the question")

    def list_documents(self) -> list[dict[str, Any]]:
        return self._get_json("/documents", "Could not load document library")

    def get_document_pdf(self, document_id: str) -> bytes:
        try:
            response = requests.get(
                f"{self.base_url}/documents/{document_id}/pdf", timeout=60
            )
        except requests.RequestException as exc:
            raise BackendError("Could not load the stored PDF.") from exc
        if not response.ok:
            raise BackendError("Could not load the stored PDF.")
        return response.content

    def get_document_pdf_url(self, document_id: str) -> str:
        return f"{self.base_url}/documents/{document_id}/pdf"

    def list_conversations(self, document_id: str | None = None) -> list[dict[str, Any]]:
        params = {"document_id": document_id} if document_id else {}
        return self._get_json("/conversations", "Could not load chat history", params)

    def get_conversation(self, conversation_id: str) -> dict[str, Any]:
        return self._get_json(
            f"/conversations/{conversation_id}", "Could not load conversation"
        )

    def create_conversation(self, document_id: str) -> dict[str, Any]:
        return self._post_json(
            "/conversations",
            {"document_id": document_id},
            "Could not create conversation",
        )

    def delete_conversation(self, conversation_id: str) -> dict[str, Any]:
        try:
            response = requests.delete(
                f"{self.base_url}/conversations/{conversation_id}", timeout=30
            )
        except requests.RequestException as exc:
            raise BackendError("Could not delete conversation.") from exc
        return self._parse_response(response, "Could not delete conversation")

    def _get_json(
        self, path: str, failure_message: str, params: dict[str, Any] | None = None
    ) -> Any:
        try:
            response = requests.get(
                f"{self.base_url}{path}", params=params or {}, timeout=60
            )
        except requests.RequestException as exc:
            raise BackendError(failure_message) from exc
        return self._parse_response(response, failure_message)

    def _post_json(self, path: str, payload: dict[str, Any], failure_message: str) -> Any:
        try:
            response = requests.post(
                f"{self.base_url}{path}", json=payload, timeout=60
            )
        except requests.RequestException as exc:
            raise BackendError(failure_message) from exc
        return self._parse_response(response, failure_message)

    @staticmethod
    def _parse_response(
        response: requests.Response, failure_message: str
    ) -> Any:
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

        if not isinstance(payload, (dict, list)):
            raise BackendError(f"{failure_message}: the backend returned an unexpected response.")

        return payload

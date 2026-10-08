from pathlib import Path

from fastapi.testclient import TestClient

import backend.routes.session as session_route
from backend.main import app


def test_document_pdf_is_inline_pdf_response(tmp_path, monkeypatch):
    document_id = "pdf_test"
    pdf_path = tmp_path / "SIH.pdf"
    pdf_path.write_bytes(b"%PDF-test-payload")

    class FakeStore:
        def get_document(self, requested_id):
            if requested_id != document_id:
                return None
            return {
                "document_folder": str(tmp_path),
                "filename": pdf_path.name,
            }

    monkeypatch.setattr(session_route, "store", FakeStore())

    response = TestClient(app).get(f"/documents/{document_id}/pdf")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.headers["content-disposition"].startswith("inline;")
    assert response.content.startswith(b"%PDF")


def test_document_pdf_supports_requested_byte_range(tmp_path, monkeypatch):
    document_id = "pdf_range_test"
    pdf_path = tmp_path / "SIH.pdf"
    pdf_path.write_bytes(b"%PDF-" + bytes(range(256)) * 2048)

    class FakeStore:
        def get_document(self, requested_id):
            if requested_id != document_id:
                return None
            return {
                "document_folder": str(tmp_path),
                "filename": pdf_path.name,
            }

    monkeypatch.setattr(session_route, "store", FakeStore())

    response = TestClient(app).get(
        f"/documents/{document_id}/pdf",
        headers={"Range": "bytes=131072-262143"},
    )

    assert response.status_code == 206
    assert response.headers["content-type"] == "application/pdf"
    assert response.headers["content-disposition"].startswith("inline;")
    assert response.headers["content-range"].startswith("bytes 131072-262143/")
    assert response.headers["content-length"] == "131072"
    assert len(response.content) == 131072
    assert response.content == pdf_path.read_bytes()[131072:262144]


def test_document_pdf_without_range_returns_complete_pdf(tmp_path, monkeypatch):
    document_id = "pdf_full_test"
    pdf_path = tmp_path / "SIH.pdf"
    pdf_path.write_bytes(b"%PDF-complete-payload")

    class FakeStore:
        def get_document(self, requested_id):
            if requested_id != document_id:
                return None
            return {
                "document_folder": str(tmp_path),
                "filename": pdf_path.name,
            }

    monkeypatch.setattr(session_route, "store", FakeStore())

    response = TestClient(app).get(f"/documents/{document_id}/pdf")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.headers["content-disposition"].startswith("inline;")
    assert response.content == pdf_path.read_bytes()

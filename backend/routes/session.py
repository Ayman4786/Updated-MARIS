from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.persistence import MarisStore


router = APIRouter()
store = MarisStore()


class ConversationRequest(BaseModel):
    document_id: str
    title: str = "New chat"


@router.get("/documents")
def documents() -> list[dict[str, Any]]:
    return store.list_documents()


@router.get("/documents/{document_id}/pdf")
def document_pdf(document_id: str) -> FileResponse:
    document = store.get_document(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    folder = Path(document["document_folder"])
    pdf_path = folder / document["filename"]
    if not pdf_path.exists():
        pdfs = list(folder.glob("*.pdf"))
        if pdfs:
            pdf_path = pdfs[0]
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="Stored PDF not found")
    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        headers={
            "Content-Disposition": (
                f'inline; filename="{document["filename"]}"'
            )
        },
    )


@router.get("/conversations")
def conversations(document_id: str | None = None) -> list[dict[str, Any]]:
    return store.list_conversations(document_id)


@router.post("/conversations")
def create_conversation(request: ConversationRequest) -> dict[str, Any]:
    document = store.get_document(request.document_id)
    if not document or document["status"] != "success":
        raise HTTPException(status_code=404, detail="Processed document not found")
    return store.create_conversation(request.document_id, request.title)


@router.get("/conversations/{conversation_id}")
def conversation(conversation_id: str) -> dict[str, Any]:
    item = store.get_conversation(conversation_id)
    if not item:
        raise HTTPException(status_code=404, detail="Conversation not found")
    item["messages"] = store.list_messages(conversation_id)
    return item


@router.delete("/conversations/{conversation_id}")
def delete_conversation(conversation_id: str) -> dict[str, bool]:
    if not store.delete_conversation(conversation_id):
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"deleted": True}

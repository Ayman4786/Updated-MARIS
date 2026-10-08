from fastapi import APIRouter
from fastapi import UploadFile
from fastapi import File

from pathlib import Path
import uuid
import json
import hashlib
from functools import wraps

from ai.extraction.docling_extractor import extract_text
from ai.extraction.diagram_describer import append_page_descriptions

from ai.rag.chunker import RecursiveChunker
from ai.rag.chunk_storage import load_chunks, save_chunks
from ai.rag.embeddings_service import EmbeddingsService
from ai.rag.vector_store import VectorStoreManager
from backend.persistence import (
    CURRENT_UPLOAD_DOCUMENT,
    DIAGRAM_ENRICHMENT_VERSION,
    DOCUMENTS_ROOT,
    MarisStore,
)


router = APIRouter()


def _mark_failed_on_error(function):
    @wraps(function)
    async def wrapped(*args, **kwargs):
        try:
            return await function(*args, **kwargs)
        except Exception as exc:
            document_id = CURRENT_UPLOAD_DOCUMENT.get()
            if document_id:
                MarisStore().fail_document(document_id, str(exc))
                CURRENT_UPLOAD_DOCUMENT.set(None)
            raise

    return wrapped


# --------------------------------------------------
# Main document storage
# --------------------------------------------------

DOCUMENT_DIR = DOCUMENTS_ROOT

DOCUMENT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


@router.post("/upload")
@_mark_failed_on_error
async def upload_pdf(
    file: UploadFile = File(...)
):

    content = await file.read()
    filename = Path(file.filename or "uploaded.pdf").name
    content_hash = hashlib.sha256(content).hexdigest()
    store = MarisStore()

    document_id = (
        f"{Path(filename).stem}_"
        f"{uuid.uuid4().hex[:8]}"
    )
    action, existing = store.begin_document(content_hash, filename, document_id)
    if action == "reuse" and existing:
        chunks_path = Path(existing["document_folder"]) / "chunks.json"
        enriched = False
        if chunks_path.exists():
            try:
                cached_chunks = load_chunks(str(chunks_path))
                enriched = bool(
                    cached_chunks
                    and cached_chunks[0].get("metadata", {}).get(
                        "diagram_enrichment_version"
                    )
                    == DIAGRAM_ENRICHMENT_VERSION
                )
            except (OSError, ValueError, TypeError):
                enriched = False
        if enriched:
            return {
                "filename": existing["filename"],
                "document_id": existing["document_id"],
                "status": "success",
                "reused": True,
                "pages": existing["pages"],
                "document_folder": existing["document_folder"],
            }
        store.mark_for_reprocessing(str(existing["document_id"]))
        existing = store.get_document(str(existing["document_id"]))
    if existing:
        document_id = str(existing["document_id"])
    CURRENT_UPLOAD_DOCUMENT.set(document_id)

    # ==================================================
    # 1. CREATE UNIQUE DOCUMENT ID
    # ==================================================

    # ==================================================
    # 2. CREATE DOCUMENT DIRECTORY
    # ==================================================

    document_dir = (
        DOCUMENT_DIR / document_id
    )

    document_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print("----------------------------------------")
    print(
        f"Document ID: {document_id}"
    )
    print(
        f"Document folder: {document_dir}"
    )
    print("----------------------------------------")

    # ==================================================
    # 3. SAVE PDF
    # ==================================================

    pdf_path = document_dir / filename

    with open(
        pdf_path,
        "wb"
    ) as pdf_file:

        pdf_file.write(content)

    print(
        f"PDF saved: {pdf_path}"
    )

    # ==================================================
    # 4. EXTRACT DOCUMENT
    # ==================================================

    extracted_data = extract_text(
        str(pdf_path),
        output_dir=str(document_dir)
    )

    # --------------------------------------------------
    # Complete Markdown
    # --------------------------------------------------

    markdown = extracted_data[
        "markdown"
    ]

    # --------------------------------------------------
    # Page-specific Markdown
    # --------------------------------------------------

    pages = extracted_data.get(
        "pages",
        []
    )

    # --------------------------------------------------
    # Visual information
    # --------------------------------------------------

    image_count = extracted_data.get(
        "image_count",
        0
    )

    visual_elements = extracted_data.get(
        "visual_elements",
        []
    )

    page_images = extracted_data.get(
        "page_images",
        {}
    )

    page_descriptions = extracted_data.get(
        "page_descriptions",
        {}
    )

    print(
        f"Pages extracted: {len(pages)}"
    )

    print(
        f"Visual elements extracted: "
        f"{len(visual_elements)}"
    )

    # ==================================================
    # 5. SAVE COMPLETE MARKDOWN
    # ==================================================

    markdown_path = (
        document_dir / "document.md"
    )

    with open(
        markdown_path,
        "w",
        encoding="utf-8"
    ) as md_file:

        md_file.write(
            markdown
        )

    print(
        f"Markdown saved: "
        f"{markdown_path}"
    )

    # ==================================================
    # 6. CHUNK PAGE-BY-PAGE
    # ==================================================

    chunker = RecursiveChunker()

    all_chunks = []

    global_chunk_id = 0

    print("----------------------------------------")
    print("CREATING PAGE-AWARE CHUNKS")
    print("----------------------------------------")

    for page_data in pages:

        page_number = page_data[
            "page_number"
        ]

        page_markdown = page_data[
            "markdown"
        ]

        # ------------------------------------------
        # Skip completely empty pages
        # ------------------------------------------

        if not page_markdown.strip():

            print(
                f"Page {page_number}: "
                "empty - skipped"
            )

            continue

        # ------------------------------------------
        # Chunk this page only
        # ------------------------------------------

        page_chunks = (
            chunker.split_text(
                page_markdown
            )
        )

        page_visual_descriptions = [
            dict(item, document_id=document_id)
            for item in page_descriptions.get(page_number, [])
        ]
        append_page_descriptions(page_chunks, page_visual_descriptions)

        print(
            f"Page {page_number}: "
            f"{len(page_chunks)} chunks"
        )

        # ------------------------------------------
        # Images belonging to this page
        # ------------------------------------------

        page_image_paths = (
            page_images.get(
                page_number,
                []
            )
        )

        # ------------------------------------------
        # Track which image belongs to
        # which image marker
        # ------------------------------------------

        page_image_index = 0
        total_page_images = len(page_image_paths)
        num_chunks_on_page = len(page_chunks)

        # ------------------------------------------
        # Create metadata
        # ------------------------------------------

        for chunk_idx, page_chunk in enumerate(page_chunks):

            # --------------------------------------
            # Count image markers in this chunk
            # --------------------------------------

            image_marker_count = (
                page_chunk.count(
                    "<!-- image -->"
                )
            )

            chunk_image_paths = []

            # --------------------------------------
            # Assign the next k images to this chunk
            # --------------------------------------

            if (
                image_marker_count > 0
                and page_image_index < total_page_images
            ):

                end_index = min(
                    page_image_index + image_marker_count,
                    total_page_images
                )

                chunk_image_paths = [
                    p
                    for p in page_image_paths[page_image_index:end_index]
                    if Path(p).exists()
                ]

                page_image_index = end_index

            # --------------------------------------
            # If this is the last chunk on this page
            # and any page images remain unassigned,
            # attach them so no images are lost
            # --------------------------------------

            if (
                chunk_idx == num_chunks_on_page - 1
                and page_image_index < total_page_images
            ):

                remaining_images = [
                    p
                    for p in page_image_paths[page_image_index:total_page_images]
                    if Path(p).exists() and p not in chunk_image_paths
                ]

                chunk_image_paths.extend(remaining_images)
                page_image_index = total_page_images

            primary_image = (
                chunk_image_paths[0]
                if chunk_image_paths
                else ""
            )

            metadata = {

                # Document identity
                "document_id":
                    document_id,

                # Original PDF name
                "filename":
                    filename,

                # Chunk identity
                "chunk_id":
                    global_chunk_id,

                # IMPORTANT
                # Actual PDF page
                "page_number":
                    page_number,

                # Multi-image list for chunks.json
                "image_paths":
                    chunk_image_paths,

                # Backward compatibility
                "image_path":
                    primary_image,

                "image_count":
                    len(chunk_image_paths),

                # JSON string for ChromaDB primitive compatibility
                "image_paths_json":
                    json.dumps(chunk_image_paths),

                "visual_descriptions":
                    page_visual_descriptions,

                "visual_descriptions_json":
                    json.dumps(page_visual_descriptions),

                "diagram_enrichment_version":
                    DIAGRAM_ENRICHMENT_VERSION,
            }

            if chunk_image_paths:
                print(
                    f"Images attached to Chunk {global_chunk_id} (Page {page_number}): "
                    f"{len(chunk_image_paths)} image(s) -> {[Path(p).name for p in chunk_image_paths]}"
                )

            # --------------------------------------
            # Create final chunk
            # --------------------------------------

            all_chunks.append(
                {
                    "text": page_chunk,

                    "metadata": metadata
                }
            )

            global_chunk_id += 1

    # ==================================================
    # 7. EXTRACT CHUNK TEXT
    # ==================================================

    chunks = [

        item["text"]

        for item in all_chunks

    ]

    print("----------------------------------------")

    print(
        f"Total chunks created: "
        f"{len(chunks)}"
    )

    # ==================================================
    # 8. SAVE CHUNKS
    # ==================================================

    chunk_path = (
        document_dir / "chunks.json"
    )

    save_chunks(
        str(chunk_path),
        all_chunks
    )

    print(
        f"Chunks saved: "
        f"{chunk_path}"
    )

    # ==================================================
    # 9. SHOW PAGE DISTRIBUTION
    # ==================================================

    print("----------------------------------------")
    print("CHUNK PAGE DISTRIBUTION")
    print("----------------------------------------")

    for item in all_chunks:

        metadata = item[
            "metadata"
        ]

        print(
            f"Chunk "
            f"{metadata['chunk_id']} "
            f"-> Page "
            f"{metadata['page_number']}"
        )

    # ==================================================
    # 10. GENERATE EMBEDDINGS
    # ==================================================

    embedder = EmbeddingsService()

    embeddings = []

    for index, chunk in enumerate(
        chunks
    ):

        print(
            f"Embedding chunk "
            f"{index + 1}/"
            f"{len(chunks)}"
        )

        embedding = (
            embedder.get_embedding(
                chunk
            )
        )

        embeddings.append(
            embedding
        )

    print(
        f"Embeddings created: "
        f"{len(embeddings)}"
    )

    # ==================================================
    # 11. EXTRACT CHROMA METADATA
    # ==================================================

    # Ensure ChromaDB metadata only contains primitive types (no lists)
    chroma_metadata_list = []
    for item in all_chunks:
        meta = dict(item["metadata"])
        meta.pop("image_paths", None)
        if meta.get("image_path") is None:
            meta["image_path"] = ""
        chroma_metadata_list.append(meta)

    vector_store = (
        VectorStoreManager()
    )

    vector_store.add_chunks(

        chunks=chunks,

        metadata_list=chroma_metadata_list,

        embeddings=embeddings

    )

    print(
        "Chunks stored in ChromaDB"
    )

    # ==================================================
    # 13. RETURN RESULT
    # ==================================================

    store.complete_document(document_id, len(pages))
    CURRENT_UPLOAD_DOCUMENT.set(None)
    return {

        "filename":
            filename,

        "document_id":
            document_id,

        "status":
            "success",

        "pages":
            len(pages),

        "chunks_created":
            len(chunks),

        "embeddings_created":
            len(embeddings),

        "images_created":
            image_count,

        "visual_elements":
            len(visual_elements),

        "document_folder":
            str(document_dir),

        "chunks_file":
            str(chunk_path),

        "preview":
            markdown[:1000]
    }
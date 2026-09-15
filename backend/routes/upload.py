from fastapi import APIRouter
from fastapi import UploadFile
from fastapi import File

from pathlib import Path
import uuid

from ai.extraction.docling_extractor import extract_text

from ai.rag.chunker import RecursiveChunker
from ai.rag.chunk_storage import save_chunks
from ai.rag.embeddings_service import EmbeddingsService
from ai.rag.vector_store import VectorStoreManager


router = APIRouter()


# --------------------------------------------------
# Main document storage
# --------------------------------------------------

DOCUMENT_DIR = Path(
    "storage/documents"
)

DOCUMENT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


@router.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...)
):

    # ==================================================
    # 1. CREATE UNIQUE DOCUMENT ID
    # ==================================================

    document_id = (
        f"{Path(file.filename).stem}_"
        f"{uuid.uuid4().hex[:8]}"
    )

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

    pdf_path = (
        document_dir / file.filename
    )

    content = await file.read()

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

        # ------------------------------------------
        # Create metadata
        # ------------------------------------------

        for page_chunk in page_chunks:

            metadata = {

                # Document identity
                "document_id":
                    document_id,

                # Original PDF name
                "filename":
                    file.filename,

                # Chunk identity
                "chunk_id":
                    global_chunk_id,

                # IMPORTANT
                # Actual PDF page
                "page_number":
                    page_number
            }

            # --------------------------------------
            # Attach image to chunk
            # --------------------------------------

            image_marker_count = (
                page_chunk.count(
                    "<!-- image -->"
                )
            )

            if (
                image_marker_count > 0
                and page_image_index
                < len(page_image_paths)
            ):

                image_path = (
                    page_image_paths[
                        page_image_index
                    ]
                )

                if Path(
                    image_path
                ).exists():

                    metadata[
                        "image_path"
                    ] = image_path

                    print(
                        f"Image attached:"
                        f" Page {page_number}"
                        f" -> {image_path}"
                    )

                page_image_index += 1

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

    metadata_list = [

        item["metadata"]

        for item in all_chunks

    ]

    # ==================================================
    # 12. STORE IN CHROMADB
    # ==================================================

    vector_store = (
        VectorStoreManager()
    )

    vector_store.add_chunks(

        chunks=chunks,

        metadata_list=metadata_list,

        embeddings=embeddings

    )

    print(
        "Chunks stored in ChromaDB"
    )

    # ==================================================
    # 13. RETURN RESULT
    # ==================================================

    return {

        "filename":
            file.filename,

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
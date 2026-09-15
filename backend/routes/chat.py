from fastapi import APIRouter
from pydantic import BaseModel

from pathlib import Path

from ai.rag.chunk_storage import load_chunks
from ai.rag.embeddings_service import EmbeddingsService
from ai.rag.vector_store import VectorStoreManager
from ai.rag.retriever import HybridRetriever
from ai.rag.context_builder import ContextBuilder
from ai.rag.visual_rag import VisualRAG

from ai.llm.prompt_builder import build_prompt
from ai.llm.llm_service import generate_answer


router = APIRouter()


# ==================================================
# REQUEST SCHEMA
# ==================================================

class QuestionRequest(BaseModel):

    question: str

    # --------------------------------------------------
    # OPTIONAL
    #
    # Normally user does NOT need to provide this.
    #
    # If provided, search only that document.
    # If omitted, search ALL uploaded documents.
    # --------------------------------------------------

    document_id: str | None = None


# ==================================================
# DETECT EXPLICIT VISUAL QUESTIONS
# ==================================================

def requires_explicit_vision(
    question: str
) -> bool:

    question = (
        question
        .lower()
        .strip()
    )

    visual_phrases = [

        "explain the figure",
        "explain the diagram",
        "explain the image",
        "explain the picture",
        "explain the chart",
        "explain the graph",
        "explain the table",

        "describe the figure",
        "describe the diagram",
        "describe the image",
        "describe the picture",
        "describe the chart",
        "describe the graph",
        "describe the table",

        "what is shown in the figure",
        "what is shown in the diagram",
        "what is shown in the image",
        "what is shown in the picture",
        "what is shown in the chart",
        "what is shown in the graph",

        "what does the figure show",
        "what does the diagram show",
        "what does the image show",
        "what does the chart show",
        "what does the graph show",

        "explain this figure",
        "explain this diagram",
        "explain this image",
        "explain this chart",
        "explain this graph",

        "look at the figure",
        "look at the diagram",
        "look at the image",
        "look at the chart",
        "look at the graph",

        "visual representation",
        "what does it look like"

    ]

    for phrase in visual_phrases:

        if phrase in question:

            return True

    visual_objects = [

        "figure",
        "diagram",
        "image",
        "picture",
        "chart",
        "graph",
        "table",
        "illustration"

    ]

    visual_actions = [

        "explain",
        "describe",
        "show",
        "identify",
        "analyze",
        "interpret",
        "contain",
        "represent"

    ]

    has_visual_object = any(
        word in question
        for word in visual_objects
    )

    has_visual_action = any(
        word in question
        for word in visual_actions
    )

    return (
        has_visual_object
        and has_visual_action
    )


# ==================================================
# GET IMAGES FROM RETRIEVED CHUNKS
# ==================================================

def get_retrieved_images(
    retrieved_chunks: list[dict]
) -> list[str]:

    image_paths = []

    for chunk in retrieved_chunks:

        metadata = (
            chunk.get(
                "metadata",
                {}
            )
        )

        image_path = (
            metadata.get(
                "image_path"
            )
        )

        if not image_path:

            continue

        image_path = str(
            Path(image_path)
        )

        if not Path(
            image_path
        ).exists():

            print(
                "❌ Retrieved image "
                f"does not exist: {image_path}"
            )

            continue

        if image_path not in image_paths:

            image_paths.append(
                image_path
            )

    return image_paths


# ==================================================
# LOAD ALL DOCUMENTS
# ==================================================

def load_all_documents(
    documents_root: Path
) -> tuple[list[dict], list[Path]]:

    all_chunks = []

    document_dirs = []

    if not documents_root.exists():

        return (
            all_chunks,
            document_dirs
        )

    # --------------------------------------------------
    # Find every uploaded document directory
    # --------------------------------------------------

    for path in documents_root.iterdir():

        if not path.is_dir():

            continue

        chunks_path = (
            path / "chunks.json"
        )

        if not chunks_path.exists():

            print(
                f"⚠️ Skipping {path.name} "
                "because chunks.json is missing."
            )

            continue

        try:

            chunks = load_chunks(
                str(chunks_path)
            )

            print(
                f"✅ Loaded "
                f"{len(chunks)} chunks "
                f"from {path.name}"
            )

            # --------------------------------------------------
            # Safety: make sure each chunk has document_id
            # --------------------------------------------------

            for chunk in chunks:

                metadata = (
                    chunk
                    .get(
                        "metadata",
                        {}
                    )
                    .copy()
                )

                if not metadata.get(
                    "document_id"
                ):

                    metadata[
                        "document_id"
                    ] = path.name

                if not metadata.get(
                    "filename"
                ):

                    metadata[
                        "filename"
                    ] = path.name

                chunk["metadata"] = metadata

                all_chunks.append(
                    chunk
                )

            document_dirs.append(
                path
            )

        except Exception as e:

            print(
                f"❌ Failed loading "
                f"{path.name}: {e}"
            )

    return (
        all_chunks,
        document_dirs
    )


# ==================================================
# CHAT ROUTE
# ==================================================

@router.post("/chat")
async def chat(
    request: QuestionRequest
):

    print("\n")
    print("=" * 60)
    print("CHAT REQUEST")
    print("=" * 60)

    print(
        f"Question: {request.question}"
    )

    print(
        f"Requested document: "
        f"{request.document_id or 'ALL DOCUMENTS'}"
    )

    # ==================================================
    # DOCUMENT ROOT
    # ==================================================

    documents_root = Path(
        "storage/documents"
    )

    # ==================================================
    # LOAD ALL DOCUMENTS
    # ==================================================

    (
        all_chunks,
        document_dirs
    ) = load_all_documents(
        documents_root
    )

    if not document_dirs:

        return {

            "question":
                request.question,

            "answer":
                "No uploaded documents were found.",

            "document_id":
                None,

            "sources":
                [],

            "images_used":
                []

        }

    print("\n")
    print("=" * 60)
    print("DOCUMENT SEARCH")
    print("=" * 60)

    print(
        f"Documents available: "
        f"{len(document_dirs)}"
    )

    for directory in document_dirs:

        print(
            f"📄 {directory.name}"
        )

    print(
        f"Total chunks available: "
        f"{len(all_chunks)}"
    )

    # ==================================================
    # VALIDATE OPTIONAL DOCUMENT ID
    # ==================================================

    selected_document_id = (
        request.document_id
    )

    if selected_document_id:

        matching_document = any(

            directory.name
            == selected_document_id

            for directory in document_dirs

        )

        if not matching_document:

            return {

                "question":
                    request.question,

                "answer":
                    (
                        "The requested document "
                        "was not found."
                    ),

                "document_id":
                    selected_document_id,

                "sources":
                    [],

                "images_used":
                    []

            }

    # ==================================================
    # INITIALIZE RAG
    # ==================================================

    embedder = EmbeddingsService()

    vector_store = VectorStoreManager()

    retriever = HybridRetriever(

        vector_store=
            vector_store,

        embeddings_service=
            embedder

    )

    # ==================================================
    # HYBRID RETRIEVAL
    # ==================================================

    retrieved_chunks = (
        retriever.retrieve(

            query=
                request.question,

            all_chunks=
                all_chunks,

            # --------------------------------------------------
            # None = search ALL PDFs
            # --------------------------------------------------

            document_id=
                selected_document_id,

            top_k=5

        )
    )

    print("\n")
    print("=" * 60)
    print("TEXT RAG RESULTS")
    print("=" * 60)

    if not retrieved_chunks:

        print(
            "❌ No relevant chunks found."
        )

    for index, chunk in enumerate(
        retrieved_chunks
    ):

        metadata = (
            chunk.get(
                "metadata",
                {}
            )
        )

        print(
            f"\nChunk {index + 1}"
        )

        print(
            f"Document: "
            f"{metadata.get('document_id')}"
        )

        print(
            f"Filename: "
            f"{metadata.get('filename')}"
        )

        print(
            f"Score: "
            f"{chunk.get('score')}"
        )

        print(
            chunk.get(
                "text",
                ""
            )[:700]
        )

        if metadata.get(
            "image_path"
        ):

            print(
                "Associated image: "
                f"{metadata['image_path']}"
            )

        print(
            "-" * 50
        )

    # ==================================================
    # BUILD CONTEXT
    # ==================================================

    context = (
        ContextBuilder.build_context(
            retrieved_chunks
        )
    )

    # ==================================================
    # FIND RETRIEVED IMAGES
    # ==================================================

    retrieved_images = (
        get_retrieved_images(
            retrieved_chunks
        )
    )

    # ==================================================
    # DETERMINE VISION MODE
    # ==================================================

    explicit_vision = (
        requires_explicit_vision(
            request.question
        )
    )

    image_paths = []
    visual_results = []

    # --------------------------------------------------
    # If relevant retrieved chunks have images,
    # use those images automatically.
    #
    # This is NOT dependent on document ID.
    # --------------------------------------------------

    if retrieved_images:

        image_paths = retrieved_images

    # --------------------------------------------------
    # If user explicitly asks about a visual and
    # retrieval didn't find one, use VisualRAG.
    # --------------------------------------------------

    elif explicit_vision:

        print("\n")
        print("=" * 60)
        print("VISUAL RAG FALLBACK")
        print("=" * 60)

        visual_rag = VisualRAG()

        visual_results = (
            visual_rag.retrieve(

                query=
                    request.question,

                document_id=
                    selected_document_id,

                top_k=2

            )
        )

        for visual in visual_results:

            image_path = (
                visual.get(
                    "image_path"
                )
            )

            if not image_path:

                continue

            image_path = str(
                Path(image_path)
            )

            if not Path(
                image_path
            ).exists():

                print(
                    f"❌ Missing image: "
                    f"{image_path}"
                )

                continue

            if image_path not in image_paths:

                image_paths.append(
                    image_path
                )

    # ==================================================
    # VISION LOG
    # ==================================================

    print("\n")
    print("=" * 60)
    print("VISION MODE")
    print("=" * 60)

    print(
        f"Explicit visual question: "
        f"{explicit_vision}"
    )

    print(
        f"Relevant images found: "
        f"{len(image_paths)}"
    )

    # ==================================================
    # IMAGE VERIFICATION
    # ==================================================

    print("\n")
    print("=" * 60)
    print("IMAGES BEING SENT TO QWEN")
    print("=" * 60)

    if image_paths:

        for image_path in image_paths:

            path = Path(
                image_path
            )

            if path.exists():

                print(
                    f"✅ EXACT IMAGE: "
                    f"{path}"
                )

            else:

                print(
                    f"❌ IMAGE MISSING: "
                    f"{path}"
                )

    else:

        print(
            "No images will be sent."
        )

    # ==================================================
    # PRINT TEXT CONTEXT
    # ==================================================

    print("\n")
    print("=" * 60)
    print("TEXT CONTEXT")
    print("=" * 60)

    print(
        context[:5000]
    )

    # ==================================================
    # BUILD PROMPT
    # ==================================================

    prompt = build_prompt(

        document_text=
            context,

        user_question=
            request.question,

        has_images=
            bool(image_paths)

    )

    # ==================================================
    # SEND TO QWEN
    # ==================================================

    print("\n")
    print("=" * 60)
    print("SENDING TO QWEN")
    print("=" * 60)

    if image_paths:

        print(
            "Sending retrieved TEXT + "
            "retrieved IMAGE(S) to Qwen."
        )

    else:

        print(
            "Sending retrieved TEXT ONLY to Qwen."
        )

    answer = generate_answer(

        prompt,

        image_paths=
            image_paths

    )

    # ==================================================
    # BUILD SOURCE INFORMATION
    # ==================================================

    sources = []

    for chunk in retrieved_chunks:

        metadata = (
            chunk.get(
                "metadata",
                {}
            )
        )

        source = {

            "document_id":
                metadata.get(
                    "document_id"
                ),

            "filename":
                metadata.get(
                    "filename"
                ),

            "page":
                metadata.get(
                    "page_number"
                ),

            "score":
                chunk.get(
                    "score"
                )

        }

        # --------------------------------------------------
        # Avoid duplicate sources
        # --------------------------------------------------

        if source not in sources:

            sources.append(
                source
            )

    # ==================================================
    # FINAL RESPONSE
    # ==================================================

    return {

        "question":
            request.question,

        # --------------------------------------------------
        # If one document was explicitly selected,
        # return it.
        #
        # Otherwise return "auto".
        # --------------------------------------------------

        "document_id":
            selected_document_id
            or "auto",

        "answer":
            answer,

        "sources":
            sources,

        "images_used":
            image_paths

    }
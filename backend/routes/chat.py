from fastapi import APIRouter
from pydantic import BaseModel

from pathlib import Path
import json
import re

from ai.rag.chunk_storage import load_chunks
from ai.rag.embeddings_service import EmbeddingsService
from ai.rag.vector_store import VectorStoreManager
from ai.rag.retriever import HybridRetriever
from ai.rag.context_builder import ContextBuilder
from ai.rag.visual_rag import VisualRAG

from ai.llm.prompt_builder import build_prompt
from ai.llm.llm_service import generate_answer
from backend.persistence import DOCUMENTS_ROOT, MarisStore


router = APIRouter()


def bound_image_paths(image_paths: list[str], max_images: int = 1) -> list[str]:
    """Keep the final multimodal request bounded and deterministic."""
    result = []
    seen = set()
    for path in image_paths:
        normalized = str(path)
        if normalized and normalized not in seen:
            result.append(normalized)
            seen.add(normalized)
        if len(result) >= max_images:
            break
    return result


# ==================================================
# REQUEST SCHEMA
# ==================================================

class QuestionRequest(BaseModel):

    question: str

    # --------------------------------------------------
    # Optional document ID
    #
    # If provided:
    # search only that document.
    #
    # If omitted:
    # search all uploaded documents.
    # --------------------------------------------------

    document_id: str | None = None
    conversation_id: str | None = None


# ==================================================
# DETECT VISUAL / DIAGRAM QUESTIONS
# ==================================================

def requires_explicit_vision(
    question: str
) -> bool:

    question = (
        question
        .lower()
        .strip()
    )

    # ==================================================
    # 1. DIRECT VISUAL PHRASES
    # ==================================================

    visual_phrases = [

        # --------------------------------------------------
        # Figures / diagrams / images
        # --------------------------------------------------

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

        # --------------------------------------------------
        # Architecture
        # --------------------------------------------------

        "system architecture",
        "system architecture diagram",
        "architecture diagram",
        "architecture of the system",
        "explain the architecture",
        "explain system architecture",
        "explain the system architecture",
        "describe the architecture",
        "describe system architecture",
        "describe the system architecture",

        # --------------------------------------------------
        # Workflow / pipeline
        # --------------------------------------------------

        "workflow diagram",
        "system workflow",
        "system workflow diagram",
        "explain the workflow",
        "describe the workflow",

        "pipeline diagram",
        "system pipeline",
        "explain the pipeline",
        "describe the pipeline",

        # --------------------------------------------------
        # Block / flow diagrams
        # --------------------------------------------------

        "block diagram",
        "block diagram of the system",
        "explain the block diagram",
        "describe the block diagram",

        "flow diagram",
        "flow chart",
        "flowchart",
        "explain the flow chart",
        "explain the flowchart",
        "describe the flow chart",
        "describe the flowchart",

        # --------------------------------------------------
        # Visual representation
        # --------------------------------------------------

        "visual representation",
        "visual representation of the system",
        "visual representation of the architecture",

        "what does it look like",
        "what is represented in the diagram",
        "what is represented in the figure",
        "what is represented in the architecture",

        # --------------------------------------------------
        # Components shown visually
        # --------------------------------------------------

        "components in the diagram",
        "components shown in the diagram",
        "components of the architecture",
        "components shown in the architecture",

        "explain the components in the diagram",
        "explain the components of the architecture",

        # --------------------------------------------------
        # Connections / relationships
        # --------------------------------------------------

        "how are the components connected",
        "how are the components connected in the diagram",
        "how does the architecture work",
        "how does the system architecture work",

    ]

    # ==================================================
    # CHECK DIRECT PHRASES
    # ==================================================

    for phrase in visual_phrases:

        if phrase in question:

            print(
                f"[OK] Visual intent detected "
                f"through phrase: '{phrase}'"
            )

            return True

    # ==================================================
    # 2. VISUAL OBJECT KEYWORDS
    # ==================================================

    visual_objects = [

        "figure",
        "diagram",
        "image",
        "picture",
        "chart",
        "graph",
        "table",
        "illustration",

        # Architecture-related
        "architecture",

        # Workflow-related
        "workflow",
        "pipeline",

        # Diagram-related
        "flowchart",
        "flow",
        "block diagram"

    ]

    # ==================================================
    # 3. VISUAL ACTION KEYWORDS
    # ==================================================

    visual_actions = [

        "explain",
        "describe",
        "show",
        "identify",
        "analyze",
        "analyse",
        "interpret",
        "contain",
        "represent",
        "illustrate"

    ]

    # ==================================================
    # CHECK OBJECT + ACTION
    # ==================================================

    has_visual_object = any(
        word in question
        for word in visual_objects
    )

    has_visual_action = any(
        word in question
        for word in visual_actions
    )

    if (
        has_visual_object
        and has_visual_action
    ):

        print(
            "[OK] Visual intent detected "
            "through object + action keywords."
        )

        return True

    # ==================================================
    # NO VISUAL INTENT
    # ==================================================

    print(
        "[INFO] No explicit visual intent detected."
    )

    return False


# ==================================================
# ==================================================
# CENTRALIZED VISUAL-NOISE FILTER THRESHOLDS
# ==================================================

MIN_DIAGRAM_WIDTH = 80.0    # points / pixels
MIN_DIAGRAM_HEIGHT = 80.0   # points / pixels
MIN_DIAGRAM_AREA = 10000.0  # width * height points / pixels


# ==================================================
# IDENTIFY MOST RELEVANT PAGES FOR VISUAL QUESTIONS
# ==================================================

def identify_most_relevant_pages(
    retrieved_chunks: list[dict],
    question: str = ""
) -> list[tuple[str, int]]:
    """
    Ranks (document_id, page_number) pairs from retrieved chunks by combining:
    - Text retrieval score
    - Visual query keyword overlap in chunk text (e.g. process, flow, diagram)
    - Direct presence of attached images in the chunk
    """
    page_scores = {}
    stop_words = {
        "what", "that", "this", "does", "show", "explain", "describe",
        "tell", "about", "from", "with", "the", "for", "into", "which",
        "where", "when", "are", "how", "can", "could", "would",
    }
    q_words = [
        word for word in re.findall(r"[a-z0-9]+", question.lower().replace("-", " "))
        if len(word) > 2 and word not in stop_words
    ] if question else []
    query_phrase = " ".join(q_words)

    for chunk in retrieved_chunks:
        metadata = chunk.get("metadata", {}) or {}
        page_num = metadata.get("page_number") or metadata.get("page")
        doc_id = metadata.get("document_id")

        if not page_num or not doc_id:
            continue

        pair = (str(doc_id), int(page_num))
        chunk_score = float(chunk.get("score", 0.0) or 0.0)
        chunk_text = re.sub(
            r"\s+", " ", str(chunk.get("text", "")).lower().replace("-", " ")
        ).strip()

        matched_words = {word for word in q_words if word in chunk_text}
        coverage = len(matched_words) / len(set(q_words)) if q_words else 0.0
        phrase_match = bool(query_phrase and query_phrase in chunk_text)

        # Exact concept/title matches must outweigh broad document similarity.
        # Image presence is only a small tie-breaker, never the page authority.
        has_images = bool(metadata.get("image_paths") or metadata.get("image_path"))
        lexical_score = (3.0 * coverage) + (2.0 if phrase_match else 0.0)
        visual_tiebreaker = 0.1 if has_images else 0.0
        page_scores[pair] = page_scores.get(pair, 0.0) + (
            chunk_score + lexical_score + visual_tiebreaker
        )

    # Sort descending by score
    sorted_pages = sorted(
        page_scores.keys(),
        key=lambda p: page_scores[p],
        reverse=True
    )
    return sorted_pages


# ==================================================
# GET IMAGE CANDIDATES FROM RETRIEVED CHUNKS & PAGES
# ==================================================

def get_retrieved_images(
    retrieved_chunks: list[dict],
    question: str = "",
    documents_root: Path = Path("storage/documents"),
    include_page_images: bool = True,
    max_page_coverage: int = 1
) -> list[str]:
    """
    Collects candidate image paths from retrieved chunks and their parent pages.
    Supports in priority order:
    1. metadata['image_paths'] (native list)
    2. metadata['image_paths_json'] (json string)
    3. legacy metadata['image_path'] (string)

    If include_page_images is True, also discovers all images belonging to the
    most relevant parent pages (top max_page_coverage pages) via pages.json.
    """
    image_paths = []
    seen_paths = set()

    for chunk in retrieved_chunks:

        metadata = (
            chunk.get(
                "metadata",
                {}
            )
            or {}
        )

        chunk_images = []

        # 1. image_paths (native list)
        if "image_paths" in metadata and isinstance(metadata["image_paths"], list):
            chunk_images = metadata["image_paths"]

        # 2. image_paths_json (json serialized string)
        elif "image_paths_json" in metadata and isinstance(metadata["image_paths_json"], str):
            try:
                parsed = json.loads(metadata["image_paths_json"])
                if isinstance(parsed, list):
                    chunk_images = parsed
            except Exception:
                pass

        # 3. legacy image_path (string)
        elif "image_path" in metadata and metadata["image_path"]:
            chunk_images = [metadata["image_path"]]

        for img in chunk_images:
            if not img:
                continue
            norm_path = str(Path(img))
            if norm_path not in seen_paths and Path(norm_path).exists():
                seen_paths.add(norm_path)
                image_paths.append(norm_path)

    # Also discover images belonging to the top relevant parent pages
    if include_page_images and retrieved_chunks:
        relevant_pages = identify_most_relevant_pages(
            retrieved_chunks=retrieved_chunks,
            question=question
        )
        for doc_id, page_num in relevant_pages[:max_page_coverage]:
            pages_file = documents_root / doc_id / "pages.json"
            if pages_file.exists():
                try:
                    with open(pages_file, "r", encoding="utf-8") as f:
                        pages_data = json.load(f)
                    for page_entry in pages_data:
                        if page_entry.get("page_number") == page_num:
                            for p_img in page_entry.get("images", []):
                                norm_p = str(Path(p_img))
                                if norm_p not in seen_paths and Path(norm_p).exists():
                                    seen_paths.add(norm_p)
                                    image_paths.append(norm_p)
                except Exception as err:
                    print(
                        f"Error loading page images for {doc_id} page {page_num}: {err}"
                    )

    return image_paths


# ==================================================
# FILTER VISUAL NOISE (SMALL ICONS / LOGOS)
# ==================================================

def filter_visual_noise(
    candidate_paths: list[str],
    document_id: str | None,
    documents_root: Path = Path("storage/documents")
) -> list[str]:
    """
    Conservative visual-noise filter.
    Prefers sufficiently large diagram/figure candidates.
    Penalizes or excludes clearly tiny icon/logo candidates when larger valid visual
    candidates exist.
    If no larger candidates exist, all candidates are preserved.
    """
    if not candidate_paths or not document_id:
        return candidate_paths

    manifest_path = documents_root / document_id / "visual_manifest.json"
    manifest_by_path = {}

    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
            for v in manifest_data:
                ip = v.get("image_path")
                if ip:
                    manifest_by_path[Path(ip).as_posix().lower()] = v
                    manifest_by_path[Path(ip).name.lower()] = v
        except Exception as e:
            print(f"Warning: could not load visual_manifest for noise filter: {e}")

    large_candidates = []
    small_candidates = []

    for path_str in candidate_paths:
        p = Path(path_str)
        visual = manifest_by_path.get(p.as_posix().lower()) or manifest_by_path.get(p.name.lower())
        w, h, area = 0.0, 0.0, 0.0

        if visual and "bbox" in visual:
            bbox = visual["bbox"]
            w = abs(bbox.get("r", 0) - bbox.get("l", 0))
            h = abs(bbox.get("t", 0) - bbox.get("b", 0))
            area = w * h
        elif p.exists():
            try:
                from PIL import Image
                with Image.open(p) as img:
                    w, h = img.size
                    area = w * h
            except Exception:
                pass

        if (w >= MIN_DIAGRAM_WIDTH and h >= MIN_DIAGRAM_HEIGHT) or area >= MIN_DIAGRAM_AREA:
            large_candidates.append(path_str)
        else:
            small_candidates.append(path_str)

    # Prefer larger diagram/figure candidates if available
    if large_candidates:
        print(
            f"Visual noise filter: selected {len(large_candidates)} diagrams, "
            f"filtered {len(small_candidates)} small icons"
        )
        return large_candidates
    else:
        # If no large candidates exist, retain all candidates
        return candidate_paths


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
                f"[WARN] Skipping {path.name} "
                "because chunks.json is missing."
            )

            continue

        try:

            chunks = load_chunks(
                str(chunks_path)
            )

            print(
                f"[OK] Loaded "
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
                f"[ERROR] Failed loading "
                f"{path.name}: {e}"
            )

    return (
        all_chunks,
        document_dirs
    )


# ==================================================
# RANK IMAGE CANDIDATES USING VISUAL RAG
# ==================================================

def rank_visual_candidates(
    question: str,
    document_id: str | None,
    candidate_images: list[str],
    top_k: int = 1
) -> tuple[list[str], list[dict]]:

    if not candidate_images:

        print(
            "No image candidates available "
            "for visual ranking."
        )

        return [], []

    if not document_id:

        print(
            "[WARN] Visual ranking requires a "
            "specific document_id."
        )

        return [], []

    print("\n")
    print("=" * 60)
    print("VISUAL RAG RANKING (CANDIDATES ONLY)")
    print("=" * 60)

    print(
        f"Candidate images to score: "
        f"{len(candidate_images)}"
    )

    print(
        f"Visual top_k: {top_k}"
    )

    try:

        visual_rag = VisualRAG()

        # Score ONLY the supplied candidate images directly
        scored_candidates = (
            visual_rag.score_candidates(
                query=question,
                candidate_paths=candidate_images,
                document_id=document_id
            )
        )

    except Exception as error:

        print(
            "[ERROR] Visual RAG candidate scoring failed:"
        )

        print(error)

        return [], []

    selected_paths = []
    selected_results = []

    for visual in scored_candidates:

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

        if (
            image_path not in selected_paths
            and Path(image_path).exists()
        ):
            selected_paths.append(
                image_path
            )

            selected_results.append(
                visual
            )

            if len(selected_paths) >= top_k:
                break

    # --------------------------------------------------
    # Debug
    # --------------------------------------------------

    print("\n")
    print(
        f"Final selected visual(s): "
        f"{len(selected_paths)}"
    )

    for visual in selected_results:

        print(
            f"[OK] Selected image: "
            f"{visual.get('image_path')}"
        )

        print(
            f"   Page: "
            f"{visual.get('page_number')}"
        )

        print(
            f"   Score: "
            f"{visual.get('score'):.4f}"
        )

    return (
        selected_paths,
        selected_results
    )


def deduplicate_grounded_sources(
    retrieved_chunks: list[dict],
) -> list[dict]:
    """Return one highest-scoring grounded source per document page."""
    sources_by_page: dict[tuple[object, object], dict] = {}
    for chunk in retrieved_chunks:
        metadata = chunk.get("metadata", {}) or {}
        document_id = metadata.get("document_id")
        page = metadata.get("page_number")
        source = {
            "document_id": document_id,
            "filename": metadata.get("filename"),
            "page": page,
            "score": chunk.get("score"),
        }
        key = (document_id, page)
        existing = sources_by_page.get(key)
        if existing is None:
            sources_by_page[key] = source
            continue
        current_score = existing.get("score")
        new_score = source.get("score")
        if (
            isinstance(new_score, (int, float))
            and not isinstance(new_score, bool)
            and (
                not isinstance(current_score, (int, float))
                or isinstance(current_score, bool)
                or new_score > current_score
            )
        ):
            sources_by_page[key] = source
    return list(sources_by_page.values())


# ==================================================
# FULL-PAGE VISUAL FALLBACK
# ==================================================

def render_relevant_full_pages(
    retrieved_chunks: list[dict],
    documents_root: Path,
    document_id: str | None = None,
    question: str = "",
    max_pages: int = 3,
) -> list[dict]:
    """Render only text-retrieved source pages for visual verification."""
    try:
        import pypdfium2 as pdfium
    except ImportError as error:
        print(f"[WARN] Full-page fallback unavailable: {error}")
        return []

    rendered = []
    seen = set()
    ranked_pages = identify_most_relevant_pages(
        retrieved_chunks,
        question=question,
    )
    for page_document_id, page_number in ranked_pages:
        if document_id and page_document_id != document_id:
            continue
        if len(rendered) >= max_pages or (page_document_id, page_number) in seen:
            continue
        seen.add((page_document_id, page_number))
        document_dir = documents_root / page_document_id
        pdfs = list(document_dir.glob("*.pdf"))
        if not pdfs:
            print(
                f"[WARN] No source PDF for full-page fallback: "
                f"{page_document_id} page {page_number}"
            )
            continue
        output_dir = document_dir / "images"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"full_page_{page_number}.png"
        try:
            if not output_path.exists():
                pdf = pdfium.PdfDocument(str(pdfs[0]))
                page = pdf[page_number - 1]
                bitmap = page.render(scale=2.0)
                bitmap.to_pil().save(output_path)
            if output_path.exists():
                rendered.append(
                    {
                        "image_path": str(output_path),
                        "document_id": page_document_id,
                        "page_number": page_number,
                        "type": "full_page_fallback",
                        "score": 0.0,
                    }
                )
                print(
                    f"[FULL-PAGE] document={page_document_id} "
                    f"page={page_number} path={output_path}"
                )
        except Exception as error:
            print(
                f"[WARN] Could not render full-page candidate "
                f"{page_document_id} page {page_number}: {error}"
            )
    return rendered


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

    documents_root = DOCUMENTS_ROOT
    store = MarisStore()

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
            f"[DOC] {directory.name}"
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

    conversation = None
    requested_conversation_id = getattr(request, "conversation_id", None)
    if requested_conversation_id:
        conversation = store.get_conversation(requested_conversation_id)
        if not conversation or conversation["document_id"] != selected_document_id:
            return {
                "question": request.question,
                "answer": "The requested conversation was not found for this document.",
                "document_id": selected_document_id,
                "conversation_id": requested_conversation_id,
                "sources": [],
                "images_used": [],
            }
    elif selected_document_id:
        conversation = store.create_conversation(selected_document_id)
    conversation_id = conversation["conversation_id"] if conversation else None
    prior_messages = (
        store.recent_messages(conversation_id)
        if conversation_id
        else []
    )
    if conversation_id:
        store.add_message(conversation_id, "user", request.question)

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
    # TEXT RETRIEVAL
    # ==================================================

    TEXT_TOP_K = 5

    retrieved_chunks = (
        retriever.retrieve(

            query=
                request.question,

            all_chunks=
                all_chunks,

            document_id=
                selected_document_id,

            top_k=
                TEXT_TOP_K

        )
    )

    print("\n")
    print("=" * 60)
    print("TEXT RAG RESULTS")
    print("=" * 60)

    if not retrieved_chunks:

        print(
            "[WARN] No relevant chunks found."
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

        try:
            print(
                chunk.get(
                    "text",
                    ""
                )[:700]
            )
        except UnicodeEncodeError:
            print(
                chunk.get(
                    "text",
                    ""
                )[:700].encode("ascii", errors="replace").decode("ascii")
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
    # BUILD TEXT CONTEXT
    # ==================================================

    context = (
        ContextBuilder.build_context(
            retrieved_chunks,
        )
    )

    # ==================================================
    # DETERMINE VISUAL INTENT
    # ==================================================

    explicit_vision = (
        requires_explicit_vision(
            request.question
        )
    )

    print("\n")
    print("=" * 60)
    print("VISUAL INTENT")
    print("=" * 60)

    print(
        f"Visual intent detected: "
        f"{explicit_vision}"
    )

    # ==================================================
    # VISUAL RETRIEVAL & IMAGE GATING
    # ==================================================

    VISUAL_TOP_K = 3
    image_paths = []
    visual_results = []
    candidate_images = []

    # --------------------------------------------------
    # MANDATORY: Text-only questions MUST NEVER send images
    # --------------------------------------------------

    if not explicit_vision:

        print("\n")
        print("=" * 60)
        print("VISUAL RAG NOT TRIGGERED")
        print("=" * 60)
        print("[INFO] No explicit visual intent detected in question.")
        print("[INFO] Text-only query: ZERO images will be sent to LLM.")

    else:

        print("\n")
        print("=" * 60)
        print("VISUAL INTENT DETECTED")
        print("=" * 60)

        # Identify effective document ID from request or top retrieved chunk
        effective_document_id = selected_document_id
        if not effective_document_id and retrieved_chunks:
            effective_document_id = (
                retrieved_chunks[0]
                .get("metadata", {})
                .get("document_id")
            )

        print(
            f"Effective document ID for visual search: {effective_document_id}"
        )

        # --------------------------------------------------
        # Collect candidate images from retrieved chunks & parent pages
        # --------------------------------------------------

        raw_candidates = get_retrieved_images(
            retrieved_chunks=retrieved_chunks,
            question=request.question,
            documents_root=documents_root,
            include_page_images=True,
            max_page_coverage=1
        )

        print(
            f"Raw candidate image count: {len(raw_candidates)}"
        )

        # --------------------------------------------------
        # Apply conservative visual-noise filter
        # --------------------------------------------------

        candidate_images = filter_visual_noise(
            candidate_paths=raw_candidates,
            document_id=effective_document_id,
            documents_root=documents_root
        )

        print("\n" + "=" * 60)
        print("FINAL FILTERED IMAGE CANDIDATES")
        print("=" * 60)
        print(f"Candidates to score: {len(candidate_images)}")
        for img in candidate_images:
            print(f"Candidate: {img}")

        # --------------------------------------------------
        # Score candidates with VisualRAG
        # --------------------------------------------------

        if effective_document_id:

            if candidate_images:

                print(
                    "Scoring page/chunk candidates using Visual RAG..."
                )

                (
                    image_paths,
                    visual_results
                ) = rank_visual_candidates(
                    question=request.question,
                    document_id=effective_document_id,
                    candidate_images=candidate_images,
                    top_k=VISUAL_TOP_K
                )

            else:
                print(
                    "No crop candidates found; full-page fallback will use "
                    "retrieved page text rather than the complete visual manifest."
                )

    # ==================================================
    # VISION LOG
    # ==================================================

    print("\n")
    print("=" * 60)
    print("FINAL VISION SELECTION")
    print("=" * 60)

    print(
        f"Explicit visual question: "
        f"{explicit_vision}"
    )

    print(
        f"Candidate images: "
        f"{len(candidate_images)}"
    )

    # ==================================================
    # LAYER 3: VERIFIED VISUAL RANKING
    # ==================================================

    if explicit_vision:
        print("\n")
        print("=" * 60)
        print("LAYER 3: VERIFIED VISUAL RANKING")
        print("=" * 60)

        # Helper function for verification
        def verify_visual_candidate(q: str, ip: str) -> dict:
            prompt = f"""You are a strict visual verification system.
The user's question is: "{q}"

Evaluate whether the attached image itself contains the specific visual information
requested by the user. A document logo, product logo, section heading, decorative
icon, title page, or merely related page is NOT relevant unless it contains the
requested diagram, chart, table, workflow, or process structure. Mark relevant
false when the requested visual cannot be identified in the image.

Respond STRICTLY with valid JSON only, no markdown, no backticks.
{{
    "relevant": true or false,
    "confidence": 0.0 to 1.0,
    "reason": "short explanation"
}}"""
            try:
                ans = generate_answer(prompt, image_paths=[ip])
                ans = str(ans).strip()
                if ans.startswith("```json"): ans = ans[7:]
                if ans.startswith("```"): ans = ans[3:]
                if ans.endswith("```"): ans = ans[:-3]
                
                verdict = json.loads(ans.strip())
                if not isinstance(verdict, dict):
                    return None
                verdict["verification_status"] = "success"
                return verdict
            except Exception as e:
                print(f"[ERROR] Verification failed for {ip}: {e}")
                return None

        def verify_and_rank_visuals(visuals: list[dict]) -> list[dict]:
            verified = []
            for visual in visuals:
                ip = visual.get("image_path")
                if not ip or not Path(ip).exists():
                    continue
                visual_document_id = visual.get("document_id", effective_document_id)
                if (
                    effective_document_id
                    and visual_document_id
                    and str(visual_document_id) != str(effective_document_id)
                ):
                    print(
                        f"[WARN] Rejecting cross-document visual candidate: {ip}"
                    )
                    continue
                document_root = documents_root / str(effective_document_id)
                if document_root.exists():
                    try:
                        Path(ip).resolve().relative_to(document_root.resolve())
                    except ValueError:
                        print(
                            f"[WARN] Rejecting visual outside document storage: {ip}"
                        )
                        continue
                print(
                    f"\nVISUAL VERIFICATION document="
                    f"{visual.get('document_id', effective_document_id)} "
                    f"page={visual.get('page_number', 'unknown')} "
                    f"type={visual.get('type', '')} path={ip}"
                )
                verdict = verify_visual_candidate(request.question, str(ip))
                relevant = bool(
                    verdict
                    and verdict.get("verification_status") == "success"
                    and verdict.get("relevant") is True
                )
                print(
                    f"Verification: relevant={relevant} "
                    f"reason={verdict.get('reason', '') if verdict else 'invalid'}"
                )
                if relevant:
                    verified.append(
                        {
                            "image_path": str(ip),
                            "visual": visual,
                            "final_score": float(visual.get("score", 0.0) or 0.0),
                            "verdict": verdict,
                        }
                    )
            return sorted(
                verified,
                key=lambda item: item["final_score"],
                reverse=True,
            )

        verified_candidates = verify_and_rank_visuals(visual_results)
        if not verified_candidates:
            print("[INFO] All crop candidates rejected; rendering relevant pages.")
            fallback_visuals = render_relevant_full_pages(
                retrieved_chunks=retrieved_chunks,
                documents_root=documents_root,
                document_id=effective_document_id,
                question=request.question,
                max_pages=3,
            )
            verified_candidates = verify_and_rank_visuals(fallback_visuals)

        if verified_candidates:
            best_candidate = verified_candidates[0]
            selected_visual = best_candidate["visual"]
            print(
                f"[VISUAL SELECTED] document="
                f"{selected_visual.get('document_id', effective_document_id)} "
                f"page={selected_visual.get('page_number', 'unknown')} "
                f"type={selected_visual.get('type', '')} "
                f"path={best_candidate['image_path']}"
            )
            image_paths = bound_image_paths(
                [best_candidate["image_path"]]
            )
            visual_results = [selected_visual]
        else:
            print(
                "[INFO] No visual candidate passed verification; "
                "sending no image to final Qwen call."
            )
            image_paths = []
            visual_results = []
            
    print(
        f"Images selected for Qwen: "
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
                    f"[OK] EXACT IMAGE: "
                    f"{path}"
                )

            else:

                print(
                    f"[MISSING] IMAGE MISSING: "
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

    try:
        print(
            context[:5000]
        )
    except UnicodeEncodeError:
        print(
            context[:5000].encode("ascii", errors="replace").decode("ascii")
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
            bool(image_paths),

        conversation_history=
            prior_messages

    )

    print(
        f"[LLM REQUEST BUDGET] model=qwen/qwen3.8-27b "
        f"context_chars={len(context)} chunks={len(retrieved_chunks)} "
        f"images={len(image_paths)} prompt_chars={len(prompt)}"
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
            "selected relevant IMAGE(S) to Qwen."
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

    sources = deduplicate_grounded_sources(retrieved_chunks)

    # ==================================================
    # FINAL RESPONSE
    # ==================================================

    llm_failure_prefixes = (
        "Qwen did not return",
        "Qwen returned",
        "LLM API key",
        "The request is too large",
        "The LLM service is currently unavailable",
    )
    if (
        conversation_id
        and isinstance(answer, str)
        and answer.strip()
        and not answer.startswith(llm_failure_prefixes)
    ):
        store.add_message(
            conversation_id,
            "assistant",
            answer,
            {"sources": sources, "images_used": image_paths},
        )

    return {

        "question":
            request.question,

        "document_id":
            selected_document_id
            or "auto",

        "conversation_id":
            conversation_id,

        "answer":
            answer,

        "sources":
            sources,

        "images_used":
            image_paths

    }
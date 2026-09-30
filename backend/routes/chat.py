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


router = APIRouter()


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
    q_words = [
        w for w in re.findall(r'\w+', question.lower())
        if len(w) > 3
    ] if question else []

    for chunk in retrieved_chunks:
        metadata = chunk.get("metadata", {}) or {}
        page_num = metadata.get("page_number") or metadata.get("page")
        doc_id = metadata.get("document_id")

        if not page_num or not doc_id:
            continue

        pair = (str(doc_id), int(page_num))
        chunk_score = float(chunk.get("score", 0.0) or 0.0)
        chunk_text = str(chunk.get("text", "")).lower()

        # Keyword overlap bonus
        overlap = sum(1 for w in q_words if w in chunk_text) if q_words else 0

        # Direct visual presence bonus
        has_images = bool(metadata.get("image_paths") or metadata.get("image_path"))
        visual_factor = 2.0 if has_images else 0.5

        page_scores[pair] = (
            page_scores.get(pair, 0.0)
            + (chunk_score + (0.25 * overlap)) * visual_factor
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
            retrieved_chunks
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
                    "No candidate images found in retrieved text context. "
                    "Falling back to complete document visual manifest."
                )

                try:
                    visual_rag = VisualRAG()
                    visual_results = visual_rag.retrieve(
                        query=request.question,
                        document_id=effective_document_id,
                        top_k=VISUAL_TOP_K
                    )
                    for visual in visual_results:
                        ip = visual.get("image_path")
                        if ip and Path(ip).exists():
                            image_paths.append(str(Path(ip)))
                except Exception as error:
                    print(f"[ERROR] Visual RAG fallback failed: {error}")

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

    if explicit_vision and image_paths:
        print("\n")
        print("=" * 60)
        print("LAYER 3: VERIFIED VISUAL RANKING")
        print("=" * 60)

        # Helper function for verification
        def verify_visual_candidate(q: str, ip: str) -> dict:
            prompt = f"""You are a strict visual verification system.
The user's question is: "{q}"

Evaluate if the attached image is semantically relevant to answering the question.
Does it contain the specific diagram, chart, or architecture requested?
If it's just a tiny decorative icon, a logo, or an unrelated screenshot, mark relevant as false.

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
                
                return json.loads(ans.strip())
            except Exception as e:
                print(f"[ERROR] Verification failed for {ip}: {e}")
                return None

        final_scores = []
        for visual in visual_results:
            ip = visual.get("image_path")
            if not ip or not Path(ip).exists():
                continue
                
            clip_score = visual.get("score", 0.0)
            v_type = visual.get("type", "")
            
            print(f"\nVISUAL VERIFICATION")
            print(f"Question: {request.question}")
            print(f"Candidate: {ip}")
            print(f"Type: {v_type}")
            print(f"CLIP score: {clip_score:.4f}")
            
            verdict = verify_visual_candidate(request.question, ip)
            
            if not verdict:
                print("[WARN] Verification fallback.")
                verdict = {"verification_status": "failed"}
            else:
                verdict["verification_status"] = "success"
                
            status = verdict.get("verification_status", "failed")
            relevant = verdict.get("relevant", False)
            conf = verdict.get("confidence", 0.0)
            reason = verdict.get("reason", "")
            
            print(f"Verification status: {status}")
            print(f"Verification: relevant={relevant}")
            print(f"Verification confidence: {conf}")
            print(f"Reason: {reason}")
            
            final_score = clip_score
            
            if status == "success":
                if relevant:
                    final_score += 1.0 # Semantic relevance boost
                    final_score += conf * 0.5 
                    
                    if v_type == "reconstructed_group":
                        final_score += 0.1 # Modest structural preference / tie-breaker
                else:
                    final_score -= 1.0 # Penalty
            # If failed, final_score == clip_score (no modifications)
                
            print(f"Final visual score: {final_score:.4f}")
            
            final_scores.append({
                "image_path": ip,
                "visual": visual,
                "final_score": final_score,
                "verdict": verdict
            })
            
        if final_scores:
            final_scores.sort(key=lambda x: x["final_score"], reverse=True)
            best_candidate = final_scores[0]
            
            print("\nFINAL VISUAL SELECTION")
            print(f"Selected: {best_candidate['image_path']}")
            print(f"Type: {best_candidate['visual'].get('type', '')}")
            print(f"Score: {best_candidate['final_score']:.4f}")
            print(f"Verified: {best_candidate['verdict']}")
            
            image_paths = [best_candidate["image_path"]]
            visual_results = [best_candidate["visual"]]
            
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
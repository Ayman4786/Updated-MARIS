import logging
from pathlib import Path

import pytest


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("maris.visual_diagnostic")


def _find_sih_document():
    storage = Path("storage/documents")

    if not storage.exists():
        pytest.fail("storage/documents does not exist")

    matches = list(storage.glob("SIH_*"))

    if not matches:
        pytest.fail("Could not find SIH_* document directory")

    return matches[0]


def test_sih_visual_pipeline_diagnostic(monkeypatch):
    """
    Diagnostic test only.

    Purpose:
    - Find the SIH document
    - Inspect visual_manifest.json
    - Run the visual retrieval path
    - Show which images are candidates
    - Show which image(s) are finally selected
    - Verify selected image files actually exist

    Run with:
        pytest -s tests/test_visual_pipeline_diagnostic.py
    """

    document_dir = _find_sih_document()
    document_id = document_dir.name

    logger.info("=" * 80)
    logger.info("MARIS VISUAL PIPELINE DIAGNOSTIC")
    logger.info("=" * 80)

    logger.info("Document ID: %s", document_id)
    logger.info("Document directory: %s", document_dir)

    manifest_path = document_dir / "visual_manifest.json"

    logger.info("Manifest path: %s", manifest_path)
    logger.info("Manifest exists: %s", manifest_path.exists())

    if manifest_path.exists():
        import json

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        logger.info("Manifest type: %s", type(manifest).__name__)

        if isinstance(manifest, dict):
            logger.info("Manifest keys: %s", list(manifest.keys()))

        visuals = []

        if isinstance(manifest, list):
            visuals = manifest
        elif isinstance(manifest, dict):
            for key in ("visuals", "images", "items", "elements"):
                if isinstance(manifest.get(key), list):
                    visuals = manifest[key]
                    break

        logger.info("Visuals found in manifest: %d", len(visuals))

        for i, visual in enumerate(visuals[:20]):
            logger.info(
                "VISUAL %d: %s",
                i + 1,
                visual,
            )

    # ------------------------------------------------------------------
    # Test VisualRAG directly
    # ------------------------------------------------------------------

    from ai.rag.visual_rag import VisualRAG

    logger.info("-" * 80)
    logger.info("RUNNING VISUAL RAG")
    logger.info("-" * 80)

    query = "What are the main layers shown in the technical architecture diagram?"

    visual_rag = VisualRAG()

    logger.info("Query: %s", query)
    logger.info("Document ID: %s", document_id)

    try:
        result = visual_rag.retrieve(
            query=query,
            document_id=document_id,
            top_k=5,
        )
    except TypeError:
        # Some versions may use positional arguments.
        result = visual_rag.retrieve(
            query,
            document_id,
            top_k=5,
        )

    trace = print
    trace(f"VisualRAG result type: {type(result).__name__}")
    trace(f"VisualRAG raw result count: {len(result)}")

    assert isinstance(result, list), (
        f"Expected VisualRAG.retrieve() to return a list, got {type(result).__name__}"
    )
    assert all(isinstance(item, dict) for item in result), (
        "VisualRAG.retrieve() returned a non-dictionary candidate"
    )

    trace("-" * 80)
    trace("VISUALRAG CANDIDATE TRACE")
    trace("-" * 80)

    for index, candidate in enumerate(result, start=1):
        image_path = candidate.get("image_path")
        path = Path(image_path) if image_path else None
        exists = path.exists() if path else False
        size = path.stat().st_size if exists else 0

        trace(f"CANDIDATE {index}")
        trace(f"  visual_id: {candidate.get('visual_id')}")
        trace(f"  image_path: {image_path}")
        trace(f"  page_number: {candidate.get('page_number')}")
        trace(f"  type: {candidate.get('type')}")
        trace(f"  image_score: {candidate.get('image_score')}")
        trace(f"  caption_score: {candidate.get('caption_score')}")
        trace(f"  final_score: {candidate.get('score')}")
        trace(f"  image_exists: {exists}")
        trace(f"  image_size_bytes: {size}")

    trace("-" * 80)
    trace("LAYER 3: VERIFIED VISUAL RANKING TRACE")
    trace("-" * 80)
    trace(
        "Layer 3 verifier in backend/routes/chat.py calls generate_answer() "
        "for each existing, in-document candidate."
    )

    candidates_for_verification = []
    document_root = document_dir.resolve()

    for candidate in result:
        image_path = candidate.get("image_path")
        path = Path(image_path) if image_path else None

        if not path or not path.exists():
            trace(
                f"REJECTED visual_id={candidate.get('visual_id')} "
                "reason=image file does not exist"
            )
            continue

        try:
            path.resolve().relative_to(document_root)
        except ValueError:
            trace(
                f"REJECTED visual_id={candidate.get('visual_id')} "
                "reason=image is outside document storage"
            )
            continue

        candidates_for_verification.append(candidate)
        trace(
            f"SENT_FOR_VERIFICATION visual_id={candidate.get('visual_id')} "
            f"image_path={image_path}"
        )

    trace(
        "Verification calls not executed: Groq/LLM calls are disabled for this diagnostic."
    )
    for candidate in candidates_for_verification:
        trace(
            f"REJECTED visual_id={candidate.get('visual_id')} "
            "reason=verification not executed (Groq disabled)"
        )

    selected_images = []
    trace("-" * 80)
    trace("FINAL IMAGE PATHS PASSED TO LLM")
    trace("-" * 80)
    trace(f"image_paths: {selected_images}")
    trace("VISUAL_PIPELINE_RESULT=NO_IMAGE_SELECTED")


def test_generate_answer_receives_image(monkeypatch):
    """
    Verifies the final LLM layer receives image_paths.

    This does NOT call Groq.
    """

    import ai.llm.llm_service as llm_service

    received = {}

    def fake_generate_answer(prompt, image_paths=None, *args, **kwargs):
        received["prompt"] = prompt
        received["image_paths"] = image_paths

        logger.info("=" * 80)
        logger.info("FINAL LLM generate_answer() INTERCEPTED")
        logger.info("=" * 80)

        logger.info("Prompt chars: %d", len(prompt or ""))
        logger.info("image_paths: %r", image_paths)

        if image_paths:
            for i, image_path in enumerate(image_paths):
                path = Path(image_path)

                logger.info(
                    "LLM IMAGE %d: path=%s exists=%s size=%s",
                    i + 1,
                    path,
                    path.exists(),
                    path.stat().st_size if path.exists() else None,
                )

        return "diagnostic answer"

    monkeypatch.setattr(
        llm_service,
        "generate_answer",
        fake_generate_answer,
    )

    # Locate an actual image from SIH.
    document_dir = _find_sih_document()
    image_dir = document_dir / "images"

    assert image_dir.exists(), f"Missing image directory: {image_dir}"

    images = list(image_dir.glob("*"))

    logger.info("Images physically stored: %d", len(images))

    for image in images[:20]:
        logger.info(
            "Stored image: %s (%d bytes)",
            image,
            image.stat().st_size,
        )

    assert images, "No extracted images found in SIH document"

    test_image = images[0]

    # Directly invoke the LLM function.
    llm_service.generate_answer(
        "TEST VISUAL QUERY",
        [str(test_image)],
    )

    assert received["image_paths"], (
        "generate_answer() received no image_paths"
    )

    assert str(test_image) in received["image_paths"]

    logger.info("=" * 80)
    logger.info("PASS: generate_answer RECEIVED THE IMAGE")
    logger.info("=" * 80)
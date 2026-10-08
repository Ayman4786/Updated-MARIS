"""Optional, failure-tolerant descriptions for meaningful document visuals."""

from __future__ import annotations

from pathlib import Path


MIN_DESCRIPTION_WIDTH = 160
MIN_DESCRIPTION_HEIGHT = 120
MIN_DESCRIPTION_AREA = 30000


def _is_meaningful_visual(visual: dict) -> bool:
    bbox = visual.get("bbox") or {}
    width = abs(float(bbox.get("r", 0) or 0) - float(bbox.get("l", 0) or 0))
    height = abs(float(bbox.get("t", 0) or 0) - float(bbox.get("b", 0) or 0))
    if width >= MIN_DESCRIPTION_WIDTH and height >= MIN_DESCRIPTION_HEIGHT:
        return True
    if width * height >= MIN_DESCRIPTION_AREA:
        return True
    try:
        from PIL import Image

        with Image.open(visual["image_path"]) as image:
            image_width, image_height = image.size
        return (
            image_width >= MIN_DESCRIPTION_WIDTH
            and image_height >= MIN_DESCRIPTION_HEIGHT
        ) or image_width * image_height >= MIN_DESCRIPTION_AREA
    except (KeyError, OSError, ValueError):
        return False


def describe_visual(
    image_path: str,
    page_number: int | None,
    caption: str = "",
) -> str:
    """Describe one visual using its Docling caption or the existing Qwen client."""
    if caption and caption.strip():
        return caption.strip()
    try:
        from ai.llm.llm_service import generate_answer

        prompt = f"""Describe this document visual for semantic search.
Page: {page_number}
Write one concise factual description. Include a readable title, labels,
components, arrows, and process relationships only when visibly present.
Do not guess, infer hidden content, or describe decorative logos as diagrams.
If it is not a meaningful diagram, chart, table, or figure, respond exactly:
NOT_MEANINGFUL."""
        description = str(generate_answer(prompt, image_paths=[image_path])).strip()
        if not description or description.upper() == "NOT_MEANINGFUL":
            return ""
        return description
    except Exception as error:
        print(f"[WARN] Visual description failed for {image_path}: {error}")
        return ""


def build_page_descriptions(
    visual_elements: list[dict],
) -> dict[int, list[dict]]:
    """Build descriptions once per canonical visual and group them by page."""
    descriptions: dict[int, list[dict]] = {}
    for visual in visual_elements:
        if not _is_meaningful_visual(visual):
            continue
        description = describe_visual(
            image_path=str(visual.get("image_path", "")),
            page_number=visual.get("page_number"),
            caption=str(visual.get("caption", "") or ""),
        )
        if not description:
            continue
        page_number = visual.get("page_number")
        if page_number is None:
            continue
        entry = {
            "document_id": None,
            "page_number": page_number,
            "image_path": visual.get("image_path"),
            "visual_id": visual.get("visual_id"),
            "type": visual.get("type"),
            "description": description,
        }
        descriptions.setdefault(int(page_number), []).append(entry)
    return descriptions


def append_page_descriptions(
    page_chunks: list[str],
    descriptions: list[dict],
) -> list[str]:
    """Attach page-local visual text to the first chunk only."""
    description_text = "\n".join(
        f"Visual description: {item.get('description', '')}"
        for item in descriptions
        if item.get("description")
    )
    if description_text and page_chunks:
        page_chunks[0] = f"{page_chunks[0]}\n\n{description_text}"
    return page_chunks

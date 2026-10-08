from pathlib import Path

from PIL import Image

from ai.extraction import diagram_describer


def test_diagram_description_is_grouped_by_page_and_keeps_visual_metadata(
    tmp_path, monkeypatch
):
    image_path = Path(tmp_path) / "diagram.png"
    Image.new("RGB", (400, 250), "white").save(image_path)
    monkeypatch.setattr(
        diagram_describer,
        "describe_visual",
        lambda image_path, page_number, caption="": (
            "Role -> Evidence -> Capability -> Gap -> Action"
        ),
    )

    descriptions = diagram_describer.build_page_descriptions(
        [
            {
                "visual_id": "image_1",
                "image_path": str(image_path),
                "page_number": 2,
                "bbox": {"l": 0, "r": 400, "t": 250, "b": 0},
                "caption": "",
                "type": "picture",
            }
        ]
    )

    assert descriptions[2][0]["page_number"] == 2
    assert descriptions[2][0]["image_path"] == str(image_path)
    assert "Capability" in descriptions[2][0]["description"]


def test_description_is_embedded_only_in_the_matching_page_chunk():
    chunks = diagram_describer.append_page_descriptions(
        ["page two text", "page two continuation"],
        [
            {
                "document_id": "doc-a",
                "page_number": 2,
                "image_path": "doc-a/diagram.png",
                "description": "Role -> Evidence -> Action",
            }
        ],
    )

    assert "Role -> Evidence -> Action" in chunks[0]
    assert "Role -> Evidence -> Action" not in chunks[1]


def test_captioning_failure_does_not_break_description_indexing(
    tmp_path, monkeypatch
):
    image_path = Path(tmp_path) / "diagram.png"
    Image.new("RGB", (400, 250), "white").save(image_path)
    monkeypatch.setattr(
        diagram_describer,
        "describe_visual",
        lambda image_path, page_number, caption="": "",
    )

    assert diagram_describer.build_page_descriptions(
        [
            {
                "visual_id": "image_1",
                "image_path": str(image_path),
                "page_number": 4,
                "bbox": {"l": 0, "r": 400, "t": 250, "b": 0},
                "caption": "",
            }
        ]
    ) == {}

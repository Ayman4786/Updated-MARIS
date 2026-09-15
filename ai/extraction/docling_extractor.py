# ai/extraction/docling_extractor.py
#
# Converts PDF to Markdown page-by-page and extracts
# visual elements with page and bounding-box metadata.

from pathlib import Path
import json

from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption
)

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.types.doc import PictureItem


def extract_text(pdf_path, output_dir):

    # --------------------------------------------------
    # 1. Configure PDF pipeline
    # --------------------------------------------------

    pipeline_options = PdfPipelineOptions(
        generate_picture_images=True,
        images_scale=2.0
    )

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options
            )
        }
    )

    # --------------------------------------------------
    # 2. Convert PDF
    # --------------------------------------------------

    result = converter.convert(pdf_path)

    doc = result.document

    # --------------------------------------------------
    # 3. Create document-specific directories
    # --------------------------------------------------

    document_dir = Path(output_dir)

    image_dir = document_dir / "images"

    image_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------
    # 4. Export page-by-page Markdown
    # --------------------------------------------------
    #
    # IMPORTANT:
    # Do NOT use only:
    #
    #     doc.export_to_markdown()
    #
    # because that loses the page association
    # once we send the resulting text to the chunker.
    #
    # Docling allows exporting a specific page using
    # page_no=...
    # --------------------------------------------------

    pages = []

    print("----------------------------------------")
    print("EXTRACTING PDF PAGE-BY-PAGE")
    print("----------------------------------------")

    for page_no in sorted(doc.pages.keys()):

        page_markdown = doc.export_to_markdown(
            page_no=page_no,
            image_placeholder="<!-- image -->"
        )

        pages.append(
            {
                "page_number": page_no,
                "markdown": page_markdown
            }
        )

        print(
            f"Page {page_no}: "
            f"{len(page_markdown)} characters"
        )

    print(
        f"Total pages extracted: {len(pages)}"
    )

    # --------------------------------------------------
    # 5. Extract pictures
    # --------------------------------------------------

    image_count = 0
    saved_count = 0

    visual_elements = []

    # Map:
    #
    # page number -> image paths
    #
    page_images = {}

    for item, _level in doc.iterate_items():

        # --------------------------------------------------
        # Only actual pictures
        # --------------------------------------------------

        if not isinstance(item, PictureItem):

            continue

        image_count += 1

        print(
            f"PictureItem found: "
            f"{image_count}"
        )

        image_path = (
            image_dir
            / f"image_{image_count}.png"
        )

        try:

            # --------------------------------------------------
            # Get actual image
            # --------------------------------------------------

            image = item.get_image(doc)

            if image is None:

                print(
                    f"Image {image_count}: "
                    "get_image() returned None"
                )

                continue

            # --------------------------------------------------
            # Save image
            # --------------------------------------------------

            image.save(
                image_path
            )

            saved_count += 1

            print(
                f"Image saved: "
                f"{image_path}"
            )

            # --------------------------------------------------
            # Extract page + bounding box
            # --------------------------------------------------

            page_number = None
            bbox_data = None

            if item.prov:

                prov = item.prov[0]

                page_number = prov.page_no

                bbox = prov.bbox

                bbox_data = {
                    "l": bbox.l,
                    "t": bbox.t,
                    "r": bbox.r,
                    "b": bbox.b,
                    "coord_origin": str(
                        bbox.coord_origin
                    )
                }

            # --------------------------------------------------
            # Store image by page
            # --------------------------------------------------

            if page_number is not None:

                if page_number not in page_images:

                    page_images[
                        page_number
                    ] = []

                page_images[
                    page_number
                ].append(
                    str(image_path)
                )

            # --------------------------------------------------
            # Extract caption
            # --------------------------------------------------

            caption = ""

            try:

                caption = item.caption_text(
                    doc
                )

            except Exception:

                caption = ""

            # --------------------------------------------------
            # Visual ID
            # --------------------------------------------------

            visual_id = (
                f"image_{image_count}"
            )

            # --------------------------------------------------
            # Visual metadata
            # --------------------------------------------------

            visual_element = {

                "visual_id": visual_id,

                "image_path": str(
                    image_path
                ),

                "page_number": page_number,

                "bbox": bbox_data,

                "caption": caption,

                "type": str(
                    item.label
                ),

                "self_ref": item.self_ref
            }

            visual_elements.append(
                visual_element
            )

            print(
                f"Visual metadata created: "
                f"{visual_id} "
                f"(Page {page_number})"
            )

        except Exception as e:

            print(
                f"Could not save image "
                f"{image_count}: {e}"
            )

    # --------------------------------------------------
    # 6. Save visual manifest
    # --------------------------------------------------

    visual_manifest_path = (
        document_dir
        / "visual_manifest.json"
    )

    with open(
        visual_manifest_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            visual_elements,
            file,
            ensure_ascii=False,
            indent=4
        )

    # --------------------------------------------------
    # 7. Save page metadata
    # --------------------------------------------------

    pages_manifest_path = (
        document_dir
        / "pages.json"
    )

    pages_manifest = []

    for page in pages:

        page_number = page[
            "page_number"
        ]

        pages_manifest.append(
            {
                "page_number": page_number,

                "character_count": len(
                    page["markdown"]
                ),

                "images": page_images.get(
                    page_number,
                    []
                )
            }
        )

    with open(
        pages_manifest_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            pages_manifest,
            file,
            ensure_ascii=False,
            indent=4
        )

    # --------------------------------------------------
    # 8. Combine Markdown for document.md
    # --------------------------------------------------

    markdown_parts = []

    for page in pages:

        markdown_parts.append(

            f"\n\n"
            f"<!-- PAGE {page['page_number']} -->\n\n"
            f"{page['markdown']}"
        )

    markdown = "".join(
        markdown_parts
    )

    # --------------------------------------------------
    # 9. Extraction summary
    # --------------------------------------------------

    print("----------------------------------------")

    print(
        f"Total pages: "
        f"{len(pages)}"
    )

    print(
        f"Total PictureItems found: "
        f"{image_count}"
    )

    print(
        f"Total images actually saved: "
        f"{saved_count}"
    )

    print(
        f"Visual elements with metadata: "
        f"{len(visual_elements)}"
    )

    print(
        f"Image directory: "
        f"{image_dir}"
    )

    print(
        f"Visual manifest: "
        f"{visual_manifest_path}"
    )

    print(
        f"Pages manifest: "
        f"{pages_manifest_path}"
    )

    print("----------------------------------------")

    # --------------------------------------------------
    # 10. Return extracted data
    # --------------------------------------------------

    return {

        "document": doc,

        # Complete Markdown
        "markdown": markdown,

        # IMPORTANT:
        # Page-specific Markdown
        "pages": pages,

        # Image directory
        "image_dir": str(
            image_dir
        ),

        # Image count
        "image_count": saved_count,

        # Visual metadata
        "visual_elements": visual_elements,

        # Images grouped by page
        "page_images": page_images,

        # Manifest
        "visual_manifest": str(
            visual_manifest_path
        ),

        "pages_manifest": str(
            pages_manifest_path
        )
    }
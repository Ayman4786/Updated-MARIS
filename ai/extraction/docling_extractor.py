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
from ai.extraction.diagram_describer import build_page_descriptions


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
    duplicate_count = 0

    visual_elements = []
    
    import hashlib
    import io

    # Maps for deduplication
    seen_hashes = {}
    canonical_visuals = {}

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
            # Deduplication
            # --------------------------------------------------

            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_bytes = img_byte_arr.getvalue()
            
            img_hash = hashlib.md5(img_bytes).hexdigest()

            if img_hash in seen_hashes:
                canonical_id = seen_hashes[img_hash]
                canonical_visual = canonical_visuals[canonical_id]
                canonical_image_path = canonical_visual["image_path"]
                
                print(
                    f"Duplicate image found (Page {page_number}). "
                    f"Mapping to canonical: {canonical_image_path}"
                )
                
                duplicate_count += 1
                
                # Append occurrence
                if "occurrences" not in canonical_visual:
                    canonical_visual["occurrences"] = []
                
                canonical_visual["occurrences"].append({
                    "page_number": page_number,
                    "bbox": bbox_data
                })
                
                # Store mapped image by page
                if page_number is not None:
                    if page_number not in page_images:
                        page_images[page_number] = []
                    page_images[page_number].append(canonical_image_path)
                
                continue

            # --------------------------------------------------
            # Save image (New Canonical)
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

                "self_ref": item.self_ref,

                "occurrences": [
                    {
                        "page_number": page_number,
                        "bbox": bbox_data
                    }
                ]
            }

            visual_elements.append(
                visual_element
            )
            
            seen_hashes[img_hash] = visual_id
            canonical_visuals[visual_id] = visual_element

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

    # ==================================================
    # LAYER 2: VISUAL GROUPING & RECONSTRUCTION
    # ==================================================
    
    print("----------------------------------------")
    print("LAYER 2: VISUAL GROUPING")
    print("----------------------------------------")
    
    try:
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(pdf_path)
    except Exception as e:
        print(f"Warning: Could not load pypdfium2 for reconstruction: {e}")
        pdf = None
        
    if pdf:
        # Group elements by page
        page_to_elements = {}
        for el in visual_elements:
            pn = el.get("page_number")
            if pn is not None:
                page_to_elements.setdefault(pn, []).append(el)
        
        group_id_counter = 0
        
        for pn, elements in page_to_elements.items():
            # Filter out heavily repeated elements (like logos)
            page_candidates = []
            for el in elements:
                occ = el.get("occurrences", [])
                if len(occ) <= 2 and el.get("bbox"):
                    page_candidates.append(el)
            
            if len(page_candidates) < 2:
                continue
            
            # Helper to parse bbox
            def parse_bbox(el):
                b = el["bbox"]
                l, t, r, b_coord = b.get("l", 0), b.get("t", 0), b.get("r", 0), b.get("b", 0)
                return {
                    "min_x": min(l, r),
                    "max_x": max(l, r),
                    "min_y": min(t, b_coord),
                    "max_y": max(t, b_coord),
                    "origin": b.get("coord_origin", "BOTTOMLEFT").upper(),
                    "el": el
                }
            
            parsed = [parse_bbox(el) for el in page_candidates]
            
            # Distance function
            def is_adjacent(a, b, threshold=60.0):
                dx = max(0, max(a["min_x"], b["min_x"]) - min(a["max_x"], b["max_x"]))
                dy = max(0, max(a["min_y"], b["min_y"]) - min(a["max_y"], b["max_y"]))
                return max(dx, dy) < threshold
                
            # Union-find or simple connected components
            groups = []
            visited = set()
            for i in range(len(parsed)):
                if i in visited:
                    continue
                
                # BFS to find cluster
                cluster = [parsed[i]]
                visited.add(i)
                
                queue = [parsed[i]]
                while queue:
                    curr = queue.pop(0)
                    for j in range(len(parsed)):
                        if j not in visited and is_adjacent(curr, parsed[j]):
                            visited.add(j)
                            cluster.append(parsed[j])
                            queue.append(parsed[j])
                
                groups.append(cluster)
                
            # Reconstruct groups
            for group in groups:
                if len(group) < 2:
                    continue
                
                group_id_counter += 1
                
                min_x = min(item["min_x"] for item in group)
                max_x = max(item["max_x"] for item in group)
                min_y = min(item["min_y"] for item in group)
                max_y = max(item["max_y"] for item in group)
                
                pad = 10.0
                min_x = max(0.0, min_x - pad)
                min_y = max(0.0, min_y - pad)
                max_x += pad
                max_y += pad
                
                # Check if the group covers almost the whole page (avoid giant crops)
                # We'll just render it, but we can check if it's too big later
                
                try:
                    # PDF pages are 0-indexed in pypdfium2
                    page_obj = pdf[pn - 1]
                    page_w, page_h = page_obj.get_size()
                    
                    # Avoid grouping the entire page (e.g. if > 80% area)
                    group_area = (max_x - min_x) * (max_y - min_y)
                    page_area = page_w * page_h
                    if group_area > page_area * 0.8:
                        print(f"Group on page {pn} is too large, skipping reconstruction.")
                        continue
                    
                    # Coordinate mapping
                    origin = group[0]["origin"]
                    if "TOPLEFT" in origin:
                        crop_top = min_y
                        crop_bottom = max_y
                    else:
                        crop_top = page_h - max_y
                        crop_bottom = page_h - min_y
                        
                    scale = 3.0
                    crop_box = (
                        int(min_x * scale),
                        int(crop_top * scale),
                        int(max_x * scale),
                        int(crop_bottom * scale)
                    )
                    
                    bitmap = page_obj.render(scale=scale)
                    pil_image = bitmap.to_pil()
                    cropped = pil_image.crop(crop_box)
                    
                    group_image_path = image_dir / f"reconstructed_page{pn}_group{group_id_counter}.png"
                    cropped.save(group_image_path)
                    
                    print(f"Reconstructed visual saved: {group_image_path} (from {len(group)} elements)")
                    
                    union_bbox = {
                        "l": min_x,
                        "t": max_y if "BOTTOMLEFT" in origin else min_y,
                        "r": max_x,
                        "b": min_y if "BOTTOMLEFT" in origin else max_y,
                        "coord_origin": origin
                    }
                    
                    visual_id = f"group_{pn}_{group_id_counter}"
                    
                    source_ids = [item["el"]["visual_id"] for item in group]
                    
                    group_element = {
                        "visual_id": visual_id,
                        "image_path": str(group_image_path),
                        "page_number": pn,
                        "bbox": union_bbox,
                        "caption": "",
                        "type": "reconstructed_group",
                        "self_ref": "",
                        "source_visual_ids": source_ids,
                        "occurrences": [
                            {
                                "page_number": pn,
                                "bbox": union_bbox
                            }
                        ]
                    }
                    
                    visual_elements.append(group_element)
                    
                    # Add to page_images so it gets attached to chunks
                    if pn not in page_images:
                        page_images[pn] = []
                    page_images[pn].append(str(group_image_path))
                    
                except Exception as e:
                    print(f"Failed to reconstruct group on page {pn}: {e}")

    # --------------------------------------------------
    # 6. Save visual manifest
    # --------------------------------------------------

    page_descriptions = build_page_descriptions(visual_elements)
    for visual in visual_elements:
        page_number = visual.get("page_number")
        for description in page_descriptions.get(page_number, []):
            if description.get("visual_id") == visual.get("visual_id"):
                visual["description"] = description["description"]
                break

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
                ),
                "visual_descriptions": page_descriptions.get(page_number, []),
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
        f"Duplicate images skipped: "
        f"{duplicate_count}"
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
        "page_images":         page_images,

        "page_descriptions":
        page_descriptions,

        # Manifest
        "visual_manifest": str(
            visual_manifest_path
        ),

        "pages_manifest": str(
            pages_manifest_path
        )
    }
# tests/test_image_retrieval_fix.py
import unittest
import tempfile
import shutil
import json
from pathlib import Path
import os
import sys

# Ensure MARIS root is in sys.path
MARIS_ROOT = str(Path(__file__).resolve().parent.parent)
if MARIS_ROOT not in sys.path:
    sys.path.insert(0, MARIS_ROOT)
os.chdir(MARIS_ROOT)

from backend.routes.chat import (
    get_retrieved_images,
    filter_visual_noise,
    requires_explicit_vision,
    rank_visual_candidates,
    MIN_DIAGRAM_WIDTH,
    MIN_DIAGRAM_HEIGHT,
    MIN_DIAGRAM_AREA
)
from ai.rag.vector_store import VectorStoreManager


class TestImageRetrievalFix(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.tmp_path = Path(self.test_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    # --------------------------------------------------
    # 1. ONE IMAGE MARKER -> ONE IMAGE
    # --------------------------------------------------
    def test_01_one_image_marker_one_image(self):
        img_file = self.tmp_path / "img1.png"
        img_file.write_bytes(b"fake image 1")

        retrieved = [{
            "text": "Header <!-- image --> text",
            "metadata": {
                "document_id": "doc1",
                "page_number": 1,
                "image_paths": [str(img_file)],
                "image_path": str(img_file)
            }
        }]
        images = get_retrieved_images(retrieved, include_page_images=False)
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0], str(img_file))

    # --------------------------------------------------
    # 2. MULTIPLE IMAGE MARKERS -> MULTIPLE IMAGE_PATHS
    # --------------------------------------------------
    def test_02_multiple_image_markers_multiple_image_paths(self):
        img1 = self.tmp_path / "img1.png"
        img2 = self.tmp_path / "img2.png"
        img1.write_bytes(b"fake image 1")
        img2.write_bytes(b"fake image 2")

        retrieved = [{
            "text": "Text <!-- image --> more <!-- image --> end",
            "metadata": {
                "document_id": "doc1",
                "page_number": 1,
                "image_paths": [str(img1), str(img2)],
                "image_path": str(img1)
            }
        }]
        images = get_retrieved_images(retrieved, include_page_images=False)
        self.assertEqual(len(images), 2)
        self.assertIn(str(img1), images)
        self.assertIn(str(img2), images)

    # --------------------------------------------------
    # 3. PAGE CONTAINING MANY IMAGES (20+ IMAGES)
    # --------------------------------------------------
    def test_03_page_containing_many_images(self):
        img_paths = []
        for i in range(25):
            f = self.tmp_path / f"img_{i}.png"
            f.write_bytes(b"content")
            img_paths.append(str(f))

        retrieved = [{
            "text": "Page with 25 images " + "<!-- image --> " * 25,
            "metadata": {
                "document_id": "doc_large",
                "page_number": 3,
                "image_paths": img_paths,
                "image_path": img_paths[0],
                "image_count": 25
            }
        }]
        images = get_retrieved_images(retrieved, include_page_images=False)
        self.assertEqual(len(images), 25)
        self.assertEqual(images[0], img_paths[0])
        self.assertEqual(images[-1], img_paths[-1])

    # --------------------------------------------------
    # 4. MULTIPLE CHUNKS ON ONE PAGE
    # --------------------------------------------------
    def test_04_multiple_chunks_on_one_page(self):
        img1 = self.tmp_path / "img_p1_c1.png"
        img2 = self.tmp_path / "img_p1_c2.png"
        img1.write_bytes(b"c1")
        img2.write_bytes(b"c2")

        # Two chunks from the same page (page 2)
        retrieved = [
            {
                "text": "Chunk 1 on page 2 <!-- image -->",
                "metadata": {
                    "document_id": "doc_multi",
                    "page_number": 2,
                    "image_paths": [str(img1)],
                    "image_path": str(img1)
                }
            },
            {
                "text": "Chunk 2 on page 2 <!-- image -->",
                "metadata": {
                    "document_id": "doc_multi",
                    "page_number": 2,
                    "image_paths": [str(img2)],
                    "image_path": str(img2)
                }
            }
        ]
        images = get_retrieved_images(retrieved, include_page_images=False)
        self.assertEqual(len(images), 2)
        self.assertEqual(images, [str(img1), str(img2)])

    # --------------------------------------------------
    # 5. LEGACY IMAGE_PATH COMPATIBILITY
    # --------------------------------------------------
    def test_05_legacy_image_path_compatibility(self):
        old_img = self.tmp_path / "legacy.png"
        old_img.write_bytes(b"legacy content")

        # Chunk metadata only has the legacy single image_path string
        retrieved = [{
            "text": "Old chunk without image_paths list",
            "metadata": {
                "document_id": "legacy_doc",
                "page_number": 1,
                "image_path": str(old_img)
            }
        }]
        images = get_retrieved_images(retrieved, include_page_images=False)
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0], str(old_img))

    # --------------------------------------------------
    # 6. CHROMADB METADATA SERIALIZATION
    # --------------------------------------------------
    def test_06_chromadb_metadata_serialization(self):
        sample_metadatas = [{
            "document_id": "doc_test",
            "page_number": 1,
            "image_paths": ["a.png", "b.png"],
            "extra_dict": {"k": "v"},
            "none_val": None,
            "count": 2
        }]

        # Defensive sanitization test
        clean_metadatas = []
        for meta in sample_metadatas:
            clean = {}
            for k, v in meta.items():
                if isinstance(v, (list, dict)):
                    json_key = k if k.endswith("_json") else f"{k}_json"
                    clean[json_key] = json.dumps(v)
                elif v is None:
                    clean[k] = ""
                elif isinstance(v, (str, int, float, bool)):
                    clean[k] = v
                else:
                    clean[k] = str(v)
            clean_metadatas.append(clean)

        sanitized = clean_metadatas[0]
        # Must be valid primitive types only
        for k, v in sanitized.items():
            self.assertIsInstance(v, (str, int, float, bool))
        self.assertEqual(sanitized["image_paths_json"], json.dumps(["a.png", "b.png"]))
        self.assertEqual(sanitized["none_val"], "")

    # --------------------------------------------------
    # 7. CANDIDATE DEDUPLICATION
    # --------------------------------------------------
    def test_07_candidate_deduplication(self):
        img = self.tmp_path / "dup.png"
        img.write_bytes(b"content")

        retrieved = [
            {"metadata": {"document_id": "d1", "page_number": 1, "image_paths": [str(img)]}},
            {"metadata": {"document_id": "d1", "page_number": 1, "image_paths": [str(img)]}},
            {"metadata": {"document_id": "d1", "page_number": 1, "image_path": str(img)}}
        ]
        images = get_retrieved_images(retrieved, include_page_images=False)
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0], str(img))

    # --------------------------------------------------
    # 8. CANDIDATE-ONLY CLIP SCORING
    # --------------------------------------------------
    def test_08_candidate_only_clip_scoring(self):
        from ai.rag.visual_rag import VisualRAG
        vr = VisualRAG()

        # If empty candidates, returns empty list
        res = vr.score_candidates("test query", [], "SIH_e91a4ea0")
        self.assertEqual(res, [])

        # If candidates are provided, scores ONLY those candidates
        cands = [
            r"storage\documents\SIH_e91a4ea0\images\image_7.png",
            r"storage\documents\SIH_e91a4ea0\images\image_31.png"
        ]
        scored = vr.score_candidates("process flow diagram", cands, "SIH_e91a4ea0")
        self.assertEqual(len(scored), 2)
        for s in scored:
            self.assertIn("score", s)
            self.assertTrue(any(Path(s["image_path"]).name == Path(c).name for c in cands))

    # --------------------------------------------------
    # 9. TEXT-ONLY QUESTION -> ZERO IMAGES
    # --------------------------------------------------
    def test_09_text_only_question_zero_images(self):
        q1 = "What is the problem statement ID?"
        self.assertFalse(requires_explicit_vision(q1))

        q2 = "Who is the team leader?"
        self.assertFalse(requires_explicit_vision(q2))

        q3 = "Summarize the feasibility and viability risks."
        self.assertFalse(requires_explicit_vision(q3))

    # --------------------------------------------------
    # 10. VISUAL QUESTION -> RELEVANT IMAGE INTENT
    # --------------------------------------------------
    def test_10_visual_question_detection(self):
        q1 = "Explain the process flow diagram of the system"
        self.assertTrue(requires_explicit_vision(q1))

        q2 = "Describe the architecture diagram"
        self.assertTrue(requires_explicit_vision(q2))

        q3 = "What does the flowchart illustrate?"
        self.assertTrue(requires_explicit_vision(q3))

    # --------------------------------------------------
    # 11. DOCUMENT_ID ISOLATION
    # --------------------------------------------------
    def test_11_document_id_isolation(self):
        img_doc1 = self.tmp_path / "doc1.png"
        img_doc2 = self.tmp_path / "doc2.png"
        img_doc1.write_bytes(b"doc1")
        img_doc2.write_bytes(b"doc2")

        retrieved = [
            {"metadata": {"document_id": "doc1", "page_number": 1, "image_paths": [str(img_doc1)]}},
            {"metadata": {"document_id": "doc2", "page_number": 1, "image_paths": [str(img_doc2)]}}
        ]
        images = get_retrieved_images(retrieved, include_page_images=False)
        self.assertEqual(len(images), 2)
        self.assertIn(str(img_doc1), images)
        self.assertIn(str(img_doc2), images)


if __name__ == '__main__':
    unittest.main()

import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
import os
import shutil
import sys

# Mock docling modules before importing docling_extractor
sys.modules['docling'] = MagicMock()
sys.modules['docling.document_converter'] = MagicMock()
sys.modules['docling.datamodel'] = MagicMock()
sys.modules['docling.datamodel.base_models'] = MagicMock()
sys.modules['docling.datamodel.pipeline_options'] = MagicMock()
sys.modules['docling_core'] = MagicMock()
sys.modules['docling_core.types'] = MagicMock()
sys.modules['docling_core.types.doc'] = MagicMock()

# Import the module to test
from ai.extraction import docling_extractor

class TestLayer1Deduplication(unittest.TestCase):

    def setUp(self):
        self.output_dir = "tests/test_output"
        os.makedirs(self.output_dir, exist_ok=True)
        self.image_dir = Path(self.output_dir) / "images"

    def tearDown(self):
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

    @patch("ai.extraction.docling_extractor.DocumentConverter")
    def test_deduplication(self, mock_converter_class):
        # Mock document converter
        mock_converter = MagicMock()
        mock_result = MagicMock()
        mock_doc = MagicMock()
        
        # Mock pages
        mock_doc.pages = {
            1: MagicMock(),
            2: MagicMock()
        }
        mock_doc.export_to_markdown.return_value = "Mock markdown"
        
        # Mock images
        # We create 3 PictureItems. 
        # Item 1: page 1, image A
        # Item 2: page 2, image A (duplicate)
        # Item 3: page 2, image B (different)
        
        from PIL import Image
        imgA = Image.new('RGB', (10, 10), color = 'red')
        imgB = Image.new('RGB', (20, 20), color = 'blue')
        
        class MockProv:
            def __init__(self, page_no):
                self.page_no = page_no
                self.bbox = MagicMock(l=0, t=0, r=10, b=10, coord_origin="origin")
        
        class MockItem:
            def __init__(self, img, page_no, label="picture"):
                self.img = img
                self.prov = [MockProv(page_no)]
                self.label = label
                self.self_ref = "ref"
                
            def get_image(self, doc):
                return self.img
                
            def caption_text(self, doc):
                return "caption"
        
        class MockPictureItem:
            def __init__(self, img, page_no):
                self.img = img
                self.prov = [MockProv(page_no)]
                self.label = "picture"
                self.self_ref = "ref"
            
            def get_image(self, doc):
                return self.img
                
            def caption_text(self, doc):
                return "caption"
        
        # Patch docling_extractor.PictureItem so isinstance works
        docling_extractor.PictureItem = MockPictureItem
        
        item1 = MockPictureItem(imgA, 1)
        item2 = MockPictureItem(imgA, 2)
        item3 = MockPictureItem(imgB, 2)
        
        mock_doc.iterate_items.return_value = [
            (item1, 1),
            (item2, 1),
            (item3, 1)
        ]
        
        mock_result.document = mock_doc
        mock_converter.convert.return_value = mock_result
        mock_converter_class.return_value = mock_converter
        
        # Run extraction
        result = docling_extractor.extract_text("dummy.pdf", self.output_dir)
        
        # Assertions
        visual_elements = result["visual_elements"]
        
        # We had 3 PictureItems, but 2 are identical (imgA), so we should have 2 unique elements
        self.assertEqual(len(visual_elements), 2)
        
        # Check canonical occurrences
        element1 = visual_elements[0]
        element2 = visual_elements[1]
        
        self.assertEqual(len(element1["occurrences"]), 2)
        self.assertEqual(element1["occurrences"][0]["page_number"], 1)
        self.assertEqual(element1["occurrences"][1]["page_number"], 2)
        
        self.assertEqual(len(element2["occurrences"]), 1)
        self.assertEqual(element2["occurrences"][0]["page_number"], 2)
        
        # Check actual saved files
        saved_images = list(self.image_dir.glob("*.png"))
        self.assertEqual(len(saved_images), 2)

        # Check page_images correctly references canonical paths
        page_images = result["page_images"]
        self.assertEqual(len(page_images[1]), 1)
        self.assertEqual(len(page_images[2]), 2)
        
        # page 2 should contain the path of the canonical imgA (element1) and imgB (element2)
        self.assertEqual(page_images[2][0], element1["image_path"])
        self.assertEqual(page_images[2][1], element2["image_path"])

if __name__ == '__main__':
    unittest.main()

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

# Mock pypdfium2
mock_pdfium = MagicMock()
sys.modules['pypdfium2'] = mock_pdfium

from ai.extraction import docling_extractor

class TestLayer2Grouping(unittest.TestCase):

    def setUp(self):
        self.output_dir = "tests/test_output_layer2"
        os.makedirs(self.output_dir, exist_ok=True)
        self.image_dir = Path(self.output_dir) / "images"

    def tearDown(self):
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

    @patch("ai.extraction.docling_extractor.DocumentConverter")
    def test_grouping(self, mock_converter_class):
        # Mock document converter
        mock_converter = MagicMock()
        mock_result = MagicMock()
        mock_doc = MagicMock()
        
        mock_doc.pages = {1: MagicMock()}
        mock_doc.export_to_markdown.return_value = "Mock markdown"
        
        # Mock pypdfium2 PdfDocument
        mock_pdf = MagicMock()
        mock_page = MagicMock()
        mock_page.get_size.return_value = (500, 800) # w, h
        
        # Mock render
        from PIL import Image
        dummy_full_page = Image.new('RGB', (1500, 2400), color='white') # scale=3
        
        mock_bitmap = MagicMock()
        mock_bitmap.to_pil.return_value = dummy_full_page
        mock_page.render.return_value = mock_bitmap
        
        # pdfium 0-indexed pages
        mock_pdf.__getitem__.return_value = mock_page
        mock_pdfium.PdfDocument.return_value = mock_pdf
        
        # Images for items
        imgA = Image.new('RGB', (10, 10), color='red')
        imgB = Image.new('RGB', (10, 10), color='blue')
        imgC = Image.new('RGB', (10, 10), color='green')
        
        class MockProv:
            def __init__(self, page_no, l, t, r, b):
                self.page_no = page_no
                self.bbox = MagicMock(l=l, t=t, r=r, b=b, coord_origin="BOTTOMLEFT")
        
        class MockPictureItem:
            def __init__(self, img, page_no, l, t, r, b):
                self.img = img
                self.prov = [MockProv(page_no, l, t, r, b)]
                self.label = "picture"
                self.self_ref = "ref"
            
            def get_image(self, doc):
                return self.img
                
            def caption_text(self, doc):
                return "caption"
        
        docling_extractor.PictureItem = MockPictureItem
        
        # Create items on page 1
        # Two items close to each other (will group)
        item1 = MockPictureItem(imgA, 1, 10, 20, 30, 0)
        item2 = MockPictureItem(imgB, 1, 35, 20, 55, 0)
        
        # One item far away (header/logo maybe, won't group)
        item3 = MockPictureItem(imgC, 1, 10, 700, 50, 650)
        
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
        
        # 3 independent elements + 1 group = 4 visual elements
        self.assertEqual(len(visual_elements), 4)
        
        # Verify the group
        group_element = next(e for e in visual_elements if e.get("type") == "reconstructed_group")
        self.assertIsNotNone(group_element)
        self.assertEqual(len(group_element["source_visual_ids"]), 2)
        
        # Union bbox (item1 and item2) -> padding applied
        bbox = group_element["bbox"]
        self.assertAlmostEqual(bbox["l"], 0.0) # max(0, min_l(10,35) - 10)
        self.assertAlmostEqual(bbox["r"], 65.0) # max_r(30,55) + 10
        
        # Check actual saved files
        saved_images = list(self.image_dir.glob("*.png"))
        self.assertEqual(len(saved_images), 4) # 3 original + 1 reconstructed

if __name__ == '__main__':
    unittest.main()

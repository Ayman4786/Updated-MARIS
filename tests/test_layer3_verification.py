import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import json
import sys
from unittest.mock import patch, MagicMock

sys.modules['fastapi'] = MagicMock()
def mock_post(*args, **kwargs):
    def decorator(func):
        return func
    return decorator
sys.modules['fastapi'].APIRouter.return_value.post = mock_post
sys.modules['pydantic'] = MagicMock()
sys.modules['torch'] = MagicMock()
sys.modules['transformers'] = MagicMock()
sys.modules['chromadb'] = MagicMock()
sys.modules['groq'] = MagicMock()
sys.modules['rank_bm25'] = MagicMock()

class QuestionRequest:
    def __init__(self, question, document_id=None):
        self.question = question
        self.document_id = document_id

# We still need to mock backend.routes.chat imports if needed, but fastapi/pydantic are the immediate failures.
from backend.routes.chat import chat

class TestLayer3Verification(unittest.IsolatedAsyncioTestCase):
    
    def setUp(self):
        self.mock_image = Path("dummy.png")
        self.mock_image.touch()
        
    def tearDown(self):
        if self.mock_image.exists():
            self.mock_image.unlink()

    @patch("backend.routes.chat.generate_answer")
    @patch("backend.routes.chat.rank_visual_candidates")
    @patch("backend.routes.chat.load_all_documents")
    @patch("backend.routes.chat.HybridRetriever")
    @patch("backend.routes.chat.VectorStoreManager")
    @patch("backend.routes.chat.EmbeddingsService")
    async def test_layer3_reconstructed_beats_clip(
        self, mock_emb, mock_vs, mock_hr, mock_load_docs, mock_rank, mock_gen_answer
    ):
        # Setup mocks
        mock_load_docs.return_value = ([], [MagicMock(name="doc1")])
        mock_hr.return_value.retrieve.return_value = [
            {"metadata": {"document_id": "doc1", "page_number": 1, "image_paths": [str(self.mock_image)]}, "text": "test", "score": 1.0}
        ]
        
        mock_c1_img = Path("dummy_c1.png")
        mock_c1_img.touch()
        mock_c2_img = Path("dummy_c2.png")
        mock_c2_img.touch()
        
        # Mock rank_visual_candidates to return 2 candidates
        c1 = {"image_path": str(mock_c1_img), "score": 0.9, "type": "picture"} # High CLIP score
        c2 = {"image_path": str(mock_c2_img), "score": 0.8, "type": "reconstructed_group"} # Lower CLIP score
        mock_rank.return_value = ([str(mock_c1_img), str(mock_c2_img)], [c1, c2])
        
        # Mock generate_answer for verification and final answer
        def mock_generate_answer(prompt, image_paths=None):
            if "visual verification system" in prompt:
                # If evaluating the first candidate (which had higher CLIP score)
                if "visual_results" not in prompt: # simplified check
                    # We will use side_effect to return specific json
                    pass
            return "final answer"
            
        mock_gen_answer.side_effect = [
            '{"relevant": false, "confidence": 0.9, "reason": "unrelated"}', # For c1
            '{"relevant": true, "confidence": 0.9, "reason": "diagram"}',    # For c2
            'Final Answer' # For actual chat
        ]

        req = QuestionRequest(question="Explain the diagram")
        result = await chat(req)

        # Ensure c2 won because it was relevant
        self.assertEqual(len(result["images_used"]), 1)
        self.assertEqual(result["images_used"][0], str(mock_c2_img))
        self.assertEqual(result["answer"], "Final Answer")
        
        mock_c1_img.unlink(missing_ok=True)
        mock_c2_img.unlink(missing_ok=True)

    @patch("backend.routes.chat.generate_answer")
    @patch("backend.routes.chat.rank_visual_candidates")
    @patch("backend.routes.chat.load_all_documents")
    @patch("backend.routes.chat.HybridRetriever")
    @patch("backend.routes.chat.VectorStoreManager")
    @patch("backend.routes.chat.EmbeddingsService")
    async def test_layer3_verification_fallback(
        self, mock_emb, mock_vs, mock_hr, mock_load_docs, mock_rank, mock_gen_answer
    ):
        # Setup mocks
        mock_load_docs.return_value = ([], [MagicMock(name="doc1")])
        mock_hr.return_value.retrieve.return_value = [
            {"metadata": {"document_id": "doc1", "page_number": 1, "image_paths": [str(self.mock_image)]}, "text": "test", "score": 1.0}
        ]
        
        c1 = {"image_path": str(self.mock_image), "score": 0.9, "type": "picture", "id": "winner"}
        c2 = {"image_path": str(self.mock_image), "score": 0.8, "type": "picture", "id": "loser"}
        mock_rank.return_value = ([str(self.mock_image), str(self.mock_image)], [c1, c2])
        
        # Simulate verification throwing error for both
        mock_gen_answer.side_effect = [
            "invalid json format 1",
            "invalid json format 2",
            "Final Answer"
        ]

        req = QuestionRequest(question="Explain the diagram")
        result = await chat(req)

        # Should fallback and use the image with the highest original CLIP score (c1)
        self.assertEqual(len(result["images_used"]), 1)
        self.assertEqual(result["images_used"][0], str(self.mock_image))
        
    @patch("backend.routes.chat.generate_answer")
    @patch("backend.routes.chat.rank_visual_candidates")
    @patch("backend.routes.chat.load_all_documents")
    @patch("backend.routes.chat.HybridRetriever")
    @patch("backend.routes.chat.VectorStoreManager")
    @patch("backend.routes.chat.EmbeddingsService")
    async def test_layer3_highly_relevant_original_beats_weak_reconstructed(
        self, mock_emb, mock_vs, mock_hr, mock_load_docs, mock_rank, mock_gen_answer
    ):
        mock_load_docs.return_value = ([], [MagicMock(name="doc1")])
        mock_hr.return_value.retrieve.return_value = [
            {"metadata": {"document_id": "doc1", "page_number": 1, "image_paths": [str(self.mock_image)]}, "text": "test", "score": 1.0}
        ]
        
        mock_orig_image = Path("dummy_orig.png")
        mock_orig_image.touch()
        mock_recon_image = Path("dummy_recon.png")
        mock_recon_image.touch()
        
        # c1: strong original candidate (CLIP 0.9)
        c1 = {"image_path": str(mock_orig_image), "score": 0.9, "type": "picture"}
        # c2: weak reconstructed candidate (CLIP 0.5)
        c2 = {"image_path": str(mock_recon_image), "score": 0.5, "type": "reconstructed_group"}
        mock_rank.return_value = ([str(mock_orig_image), str(mock_recon_image)], [c1, c2])
        
        # Both verified as relevant, but c1 retains higher overall score because of high CLIP
        mock_gen_answer.side_effect = [
            '{"relevant": true, "confidence": 0.9, "reason": "strong original"}',
            '{"relevant": true, "confidence": 0.8, "reason": "weak reconstructed"}',
            'Final Answer'
        ]

        req = QuestionRequest(question="Explain the diagram")
        result = await chat(req)

        self.assertEqual(len(result["images_used"]), 1)
        self.assertEqual(result["images_used"][0], str(mock_orig_image))
        
        mock_orig_image.unlink(missing_ok=True)
        mock_recon_image.unlink(missing_ok=True)
        
    @patch("backend.routes.chat.generate_answer")
    @patch("backend.routes.chat.rank_visual_candidates")
    @patch("backend.routes.chat.load_all_documents")
    @patch("backend.routes.chat.HybridRetriever")
    @patch("backend.routes.chat.VectorStoreManager")
    @patch("backend.routes.chat.EmbeddingsService")
    async def test_text_only_question_no_verification(
        self, mock_emb, mock_vs, mock_hr, mock_load_docs, mock_rank, mock_gen_answer
    ):
        mock_load_docs.return_value = ([], [MagicMock(name="doc1")])
        mock_hr.return_value.retrieve.return_value = []
        mock_gen_answer.return_value = "Final Answer"

        req = QuestionRequest(question="What is the system name?")
        result = await chat(req)

        # No rank_visual_candidates called
        mock_rank.assert_not_called()
        self.assertEqual(len(result["images_used"]), 0)
        self.assertEqual(result["answer"], "Final Answer")

if __name__ == '__main__':
    unittest.main()

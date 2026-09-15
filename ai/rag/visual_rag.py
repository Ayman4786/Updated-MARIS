# ai/rag/visual_rag.py

from pathlib import Path
import json

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


class VisualRAG:

    def __init__(
        self,
        model_name="openai/clip-vit-base-patch32"
    ):

        self.model_name = model_name

        print("Loading Vision RAG model...")

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.model = CLIPModel.from_pretrained(
            self.model_name
        ).to(self.device)

        self.processor = CLIPProcessor.from_pretrained(
            self.model_name
        )

        self.model.eval()

        print(
            f"Vision RAG model loaded on: "
            f"{self.device}"
        )

    # --------------------------------------------------
    # Normalize embedding
    # --------------------------------------------------

    def _normalize(self, embedding):

        return torch.nn.functional.normalize(
            embedding,
            p=2,
            dim=-1
        )

    # --------------------------------------------------
    # Create text/query embedding
    # --------------------------------------------------

    def _get_text_embedding(
        self,
        text: str
    ):

        inputs = self.processor(
            text=[text],
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        input_ids = inputs["input_ids"].to(
            self.device
        )

        attention_mask = inputs[
            "attention_mask"
        ].to(self.device)

        with torch.no_grad():

            # ------------------------------------------
            # Run CLIP text encoder directly
            # ------------------------------------------

            text_outputs = self.model.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            # ------------------------------------------
            # Get pooled text representation
            # ------------------------------------------

            pooled_output = (
                text_outputs.pooler_output
            )

            # ------------------------------------------
            # Project into CLIP embedding space
            # ------------------------------------------

            embedding = self.model.text_projection(
                pooled_output
            )

        return self._normalize(
            embedding
        )

    # --------------------------------------------------
    # Create image embedding
    # --------------------------------------------------

    def _get_image_embedding(
        self,
        image_path: str
    ):

        image = Image.open(
            image_path
        ).convert("RGB")

        inputs = self.processor(
            images=image,
            return_tensors="pt"
        )

        pixel_values = (
            inputs["pixel_values"]
            .to(self.device)
        )

        with torch.no_grad():

            # ------------------------------------------
            # Run CLIP vision encoder directly
            # ------------------------------------------

            vision_outputs = (
                self.model.vision_model(
                    pixel_values=pixel_values
                )
            )

            # ------------------------------------------
            # Get pooled image representation
            # ------------------------------------------

            pooled_output = (
                vision_outputs.pooler_output
            )

            # ------------------------------------------
            # Project into CLIP embedding space
            # ------------------------------------------

            embedding = self.model.visual_projection(
                pooled_output
            )

        return self._normalize(
            embedding
        )

    # --------------------------------------------------
    # Load visual manifest
    # --------------------------------------------------

    def _load_manifest(
        self,
        document_dir: Path
    ):

        manifest_path = (
            document_dir
            / "visual_manifest.json"
        )

        if not manifest_path.exists():

            print(
                "Visual manifest not found: "
                f"{manifest_path}"
            )

            return []

        try:

            with open(
                manifest_path,
                "r",
                encoding="utf-8"
            ) as file:

                return json.load(file)

        except Exception as error:

            print(
                f"Failed to load visual manifest: "
                f"{error}"
            )

            return []

    # --------------------------------------------------
    # Cache path
    # --------------------------------------------------

    def _get_cache_path(
        self,
        document_dir: Path
    ):

        return (
            document_dir
            / "visual_embeddings.json"
        )

    # --------------------------------------------------
    # Load cached image embeddings
    # --------------------------------------------------

    def _load_embedding_cache(
        self,
        document_dir: Path
    ):

        cache_path = self._get_cache_path(
            document_dir
        )

        if not cache_path.exists():

            return {}

        try:

            with open(
                cache_path,
                "r",
                encoding="utf-8"
            ) as file:

                return json.load(file)

        except Exception as error:

            print(
                f"Failed to load embedding cache: "
                f"{error}"
            )

            return {}

    # --------------------------------------------------
    # Save cached image embeddings
    # --------------------------------------------------

    def _save_embedding_cache(
        self,
        document_dir: Path,
        cache: dict
    ):

        cache_path = self._get_cache_path(
            document_dir
        )

        with open(
            cache_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                cache,
                file
            )

    # --------------------------------------------------
    # Get or create image embedding
    # --------------------------------------------------

    def _get_cached_image_embedding(
        self,
        visual: dict,
        document_dir: Path,
        cache: dict
    ):

        visual_id = visual.get(
            "visual_id"
        )

        image_path = visual.get(
            "image_path"
        )

        if not visual_id or not image_path:

            return None

        # ----------------------------------------------
        # Already cached
        # ----------------------------------------------

        if visual_id in cache:

            try:

                embedding = torch.tensor(
                    cache[visual_id],
                    dtype=torch.float32,
                    device=self.device
                )

                return embedding.unsqueeze(0)

            except Exception:

                print(
                    f"Invalid cached embedding "
                    f"for visual: {visual_id}"
                )

        # ----------------------------------------------
        # Image must exist
        # ----------------------------------------------

        if not Path(image_path).exists():

            print(
                f"Visual image not found: "
                f"{image_path}"
            )

            return None

        # ----------------------------------------------
        # Generate embedding
        # ----------------------------------------------

        try:

            embedding = self._get_image_embedding(
                image_path
            )

        except Exception as error:

            print(
                f"Failed to create embedding "
                f"for {image_path}: {error}"
            )

            return None

        # ----------------------------------------------
        # Save embedding to cache
        # ----------------------------------------------

        cache[visual_id] = (
            embedding
            .squeeze(0)
            .cpu()
            .tolist()
        )

        return embedding

    # --------------------------------------------------
    # Retrieve most relevant visual
    # --------------------------------------------------

    def retrieve(
        self,
        query: str,
        document_id: str,
        top_k=1
    ) -> list[dict]:

        document_dir = (
            Path("storage/documents")
            / document_id
        )

        # ----------------------------------------------
        # Check document
        # ----------------------------------------------

        if not document_dir.exists():

            print(
                "Visual RAG document not found: "
                f"{document_dir}"
            )

            return []

        # ----------------------------------------------
        # Load visuals
        # ----------------------------------------------

        visuals = self._load_manifest(
            document_dir
        )

        if not visuals:

            print(
                "No visual elements available "
                "for this document."
            )

            return []

        print(
            f"Vision RAG searching "
            f"{len(visuals)} visual(s)"
        )

        # ----------------------------------------------
        # Query embedding
        # ----------------------------------------------

        try:

            query_embedding = (
                self._get_text_embedding(
                    query
                )
            )

        except Exception as error:

            print(
                "Failed to create query embedding:"
            )

            print(error)

            return []

        # ----------------------------------------------
        # Load image embedding cache
        # ----------------------------------------------

        cache = self._load_embedding_cache(
            document_dir
        )

        results = []

        # ----------------------------------------------
        # Compare query with every visual
        # ----------------------------------------------

        for visual in visuals:

            image_embedding = (
                self._get_cached_image_embedding(
                    visual,
                    document_dir,
                    cache
                )
            )

            if image_embedding is None:

                continue

            # ------------------------------------------
            # Image similarity
            # ------------------------------------------

            image_score = (
                torch.matmul(
                    query_embedding,
                    image_embedding.T
                )
                .item()
            )

            # ------------------------------------------
            # Caption similarity
            # ------------------------------------------

            caption = (
                visual
                .get("caption", "")
                .strip()
            )

            caption_score = 0.0

            if caption:

                try:

                    caption_embedding = (
                        self._get_text_embedding(
                            caption
                        )
                    )

                    caption_score = (
                        torch.matmul(
                            query_embedding,
                            caption_embedding.T
                        )
                        .item()
                    )

                except Exception as error:

                    print(
                        f"Caption embedding failed: "
                        f"{error}"
                    )

                    caption_score = 0.0

            # ------------------------------------------
            # Hybrid visual + caption score
            # ------------------------------------------

            if caption:

                final_score = (
                    0.70 * image_score
                    +
                    0.30 * caption_score
                )

            else:

                final_score = image_score

            result = {

                "visual_id": visual.get(
                    "visual_id"
                ),

                "image_path": visual.get(
                    "image_path"
                ),

                "page_number": visual.get(
                    "page_number"
                ),

                "bbox": visual.get(
                    "bbox"
                ),

                "caption": caption,

                "type": visual.get(
                    "type"
                ),

                "image_score": float(
                    image_score
                ),

                "caption_score": float(
                    caption_score
                ),

                "score": float(
                    final_score
                )
            }

            results.append(
                result
            )

        # ----------------------------------------------
        # Save newly created embeddings
        # ----------------------------------------------

        self._save_embedding_cache(
            document_dir,
            cache
        )

        # ----------------------------------------------
        # Rank visuals
        # ----------------------------------------------

        results.sort(
            key=lambda item: item["score"],
            reverse=True
        )

        selected = results[:top_k]

        # ----------------------------------------------
        # Debug output
        # ----------------------------------------------

        print(
            "\nVISION RAG RESULTS:\n"
        )

        for visual in selected:

            print(
                f"Visual: "
                f"{visual['visual_id']}"
            )

            print(
                f"Path: "
                f"{visual['image_path']}"
            )

            print(
                f"Page: "
                f"{visual['page_number']}"
            )

            print(
                f"Type: "
                f"{visual['type']}"
            )

            print(
                f"Caption: "
                f"{visual['caption']}"
            )

            print(
                f"Image Score: "
                f"{visual['image_score']:.4f}"
            )

            print(
                f"Caption Score: "
                f"{visual['caption_score']:.4f}"
            )

            print(
                f"Final Score: "
                f"{visual['score']:.4f}"
            )

            print(
                "-" * 50
            )

        return selected
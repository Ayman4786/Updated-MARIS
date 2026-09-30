# ai/rag/retriever.py

from rank_bm25 import BM25Okapi

from ai.rag.embeddings_service import EmbeddingsService
from ai.rag.vector_store import VectorStoreManager


class HybridRetriever:

    def __init__(
        self,
        vector_store: VectorStoreManager,
        embeddings_service: EmbeddingsService
    ):

        self.vector_store = vector_store
        self.embeddings_service = embeddings_service

    # ==================================================
    # HYBRID RETRIEVAL
    # ==================================================

    def retrieve(
        self,
        query: str,
        all_chunks: list[dict],
        document_id: str | None = None,
        top_k=5
    ) -> list[dict]:

        if not all_chunks:

            print(
                "❌ No chunks available for retrieval."
            )

            return []

        # ==================================================
        # OPTIONAL DOCUMENT FILTER
        # ==================================================

        if document_id:

            searchable_chunks = [

                chunk

                for chunk in all_chunks

                if chunk.get(
                    "metadata",
                    {}
                ).get(
                    "document_id"
                ) == document_id

            ]

        else:

            searchable_chunks = all_chunks

        print(
            f"Chunks available for retrieval: "
            f"{len(searchable_chunks)}"
        )

        if not searchable_chunks:

            print(
                "❌ No chunks available for "
                "the selected document."
            )

            return []

        # ==================================================
        # RETRIEVAL CANDIDATE COUNT
        #
        # We retrieve more candidates first.
        #
        # Final top_k is selected only AFTER
        # combining semantic + lexical scores.
        #
        # This allows the RAG system to examine
        # the available chunks before deciding
        # what is actually relevant.
        # ==================================================

        candidate_k = min(
            max(top_k * 3, 10),
            len(searchable_chunks)
        )

        print(
            f"Retrieval candidate count: "
            f"{candidate_k}"
        )

        # ==================================================
        # 1. VECTOR SEMANTIC SEARCH
        # ==================================================

        query_vector = (
            self.embeddings_service.get_embedding(
                query
            )
        )

        vector_results = (
            self.vector_store.search_vectors(
                query_embedding=query_vector,
                top_k=candidate_k,
                document_id=document_id
            )
        )

        # ==================================================
        # 2. BM25 LEXICAL SEARCH
        # ==================================================

        tokenized_corpus = [

            chunk.get(
                "text",
                ""
            ).lower().split()

            for chunk in searchable_chunks

        ]

        bm25 = BM25Okapi(
            tokenized_corpus
        )

        tokenized_query = (
            query.lower().split()
        )

        bm25_scores = bm25.get_scores(
            tokenized_query
        )

        top_bm25_indices = sorted(

            range(
                len(bm25_scores)
            ),

            key=lambda i:
                bm25_scores[i],

            reverse=True

        )[:candidate_k]

        # ==================================================
        # 3. COLLECT VECTOR RESULTS
        # ==================================================

        candidates = {}

        if (
            vector_results
            and vector_results.get("documents")
            and vector_results.get("metadatas")
        ):

            documents = (
                vector_results["documents"][0]
            )

            metadatas = (
                vector_results["metadatas"][0]
            )

            distances = (
                vector_results.get(
                    "distances",
                    [[]]
                )[0]
            )

            for index, (doc, meta) in enumerate(
                zip(
                    documents,
                    metadatas
                )
            ):

                metadata = (
                    meta or {}
                )

                result_document_id = (
                    metadata.get(
                        "document_id"
                    )
                )

                # --------------------------------------------------
                # Never allow another document into the result
                # --------------------------------------------------

                if (
                    document_id
                    and result_document_id
                    != document_id
                ):

                    continue

                key = (
                    result_document_id,
                    doc
                )

                distance = (

                    distances[index]

                    if index < len(distances)

                    else 0

                )

                candidates[key] = {

                    "text":
                        doc,

                    "metadata":
                        metadata,

                    "vector_score":
                        1 / (1 + distance),

                    "bm25_score":
                        0.0

                }

        # ==================================================
        # 4. ADD BM25 RESULTS
        # ==================================================

        for index in top_bm25_indices:

            chunk = searchable_chunks[index]

            text = chunk.get(
                "text",
                ""
            )

            metadata = (
                chunk.get(
                    "metadata",
                    {}
                ).copy()
            )

            result_document_id = (
                metadata.get(
                    "document_id"
                )
            )

            key = (
                result_document_id,
                text
            )

            if key not in candidates:

                candidates[key] = {

                    "text":
                        text,

                    "metadata":
                        metadata,

                    "vector_score":
                        0.0,

                    "bm25_score":
                        float(
                            bm25_scores[index]
                        )

                }

            else:

                candidates[key][
                    "bm25_score"
                ] = float(
                    bm25_scores[index]
                )

        # ==================================================
        # 5. NORMALIZE BM25
        # ==================================================

        if candidates:

            max_bm25 = max(

                candidate[
                    "bm25_score"
                ]

                for candidate
                in candidates.values()

            )

            if max_bm25 > 0:

                for candidate in candidates.values():

                    candidate[
                        "bm25_normalized"
                    ] = (

                        candidate[
                            "bm25_score"
                        ]

                        / max_bm25

                    )

            else:

                for candidate in candidates.values():

                    candidate[
                        "bm25_normalized"
                    ] = 0.0

        # ==================================================
        # 6. COMBINE SCORES
        # ==================================================

        for candidate in candidates.values():

            vector_score = (
                candidate.get(
                    "vector_score",
                    0.0
                )
            )

            bm25_score = (
                candidate.get(
                    "bm25_normalized",
                    0.0
                )
            )

            candidate[
                "final_score"
            ] = (

                0.60 * vector_score

                +

                0.40 * bm25_score

            )

        # ==================================================
        # 7. SORT ALL CANDIDATES
        # ==================================================

        ranked_candidates = sorted(

            candidates.values(),

            key=lambda item:
                item["final_score"],

            reverse=True

        )

        print("\n")
        print("=" * 60)
        print("RANKED RETRIEVAL CANDIDATES")
        print("=" * 60)

        for index, candidate in enumerate(
            ranked_candidates
        ):

            metadata = (
                candidate.get(
                    "metadata",
                    {}
                )
            )

            print(
                f"{index + 1}. "
                f"Page={metadata.get('page_number')} | "
                f"Score={candidate['final_score']:.4f}"
            )

        # ==================================================
        # 8. FINAL TOP-K TEXT CHUNKS
        #
        # IMPORTANT:
        #
        # Only these chunks are passed forward
        # to ContextBuilder / LLM.
        # ==================================================

        final_candidates = (
            ranked_candidates[:top_k]
        )

        # ==================================================
        # 9. ATTACH ONLY DIRECTLY ASSOCIATED IMAGES
        #
        # DO NOT search nearby chunks here.
        #
        # Searching nearby chunks was the reason a single
        # retrieved text result could acquire multiple
        # unrelated images.
        #
        # An image is now attached only when the retrieved
        # chunk itself contains image_path.
        # ==================================================

        enhanced_hits = []

        for hit in final_candidates:

            metadata = (
                hit["metadata"].copy()
            )

            image_path = (
                metadata.get(
                    "image_path"
                )
            )

            # --------------------------------------------------
            # Validate image path
            # --------------------------------------------------

            if image_path:

                from pathlib import Path

                image_file = Path(
                    str(image_path)
                )

                if image_file.exists():

                    metadata[
                        "image_path"
                    ] = str(
                        image_file
                    )

                    print(
                        "Direct image associated "
                        "with retrieved chunk: "
                        f"{image_file}"
                    )

                else:

                    print(
                        "⚠️ Image path stored in "
                        "chunk does not exist: "
                        f"{image_path}"
                    )

                    metadata.pop(
                        "image_path",
                        None
                    )

            # --------------------------------------------------
            # Add final hit
            # --------------------------------------------------

            enhanced_hits.append({

                "text":
                    hit["text"],

                "metadata":
                    metadata,

                "score":
                    hit["final_score"]

            })

        # ==================================================
        # 10. FINAL RESULTS
        # ==================================================

        print("\n")
        print("=" * 60)
        print("FINAL RETRIEVED CHUNKS")
        print("=" * 60)

        for index, hit in enumerate(
            enhanced_hits
        ):

            metadata = (
                hit.get(
                    "metadata",
                    {}
                )
            )

            print(
                f"\nChunk {index + 1}"
            )

            print(
                f"Document: "
                f"{metadata.get('document_id')}"
            )

            print(
                f"Page: "
                f"{metadata.get('page_number')}"
            )

            print(
                f"Score: "
                f"{hit.get('score')}"
            )

            if metadata.get(
                "image_path"
            ):

                print(
                    f"Image: "
                    f"{metadata['image_path']}"
                )

            else:

                print(
                    "Image: None"
                )

            print(
                "-" * 50
            )

        return enhanced_hits
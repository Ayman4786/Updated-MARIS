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

        searchable_chunks = all_chunks

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

        print(
            f"Chunks available for retrieval: "
            f"{len(searchable_chunks)}"
        )

        if not searchable_chunks:

            return []

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
                top_k=top_k,
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

        )[:top_k]

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

                metadata = meta or {}

                result_document_id = (
                    metadata.get(
                        "document_id"
                    )
                )

                # --------------------------------------------------
                # Safety filter
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

                    "text": doc,

                    "metadata": metadata,

                    "vector_score":
                        1 / (1 + distance),

                    "bm25_score": 0.0

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

                    "text": text,

                    "metadata": metadata,

                    "vector_score": 0.0,

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
                candidate["bm25_score"]
                for candidate in candidates.values()
            )

            if max_bm25 > 0:

                for candidate in candidates.values():

                    candidate[
                        "bm25_normalized"
                    ] = (
                        candidate["bm25_score"]
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

            # Semantic search gets slightly higher weight.
            candidate["final_score"] = (
                0.60 * vector_score
                +
                0.40 * bm25_score
            )

        # ==================================================
        # 7. SORT FINAL RESULTS
        # ==================================================

        ranked_candidates = sorted(

            candidates.values(),

            key=lambda item:
                item["final_score"],

            reverse=True

        )

        # ==================================================
        # 8. ATTACH IMAGES FROM NEARBY CHUNKS
        # ==================================================

        enhanced_hits = []

        for hit in ranked_candidates:

            metadata = (
                hit["metadata"].copy()
            )

            image_path = metadata.get(
                "image_path"
            )

            # --------------------------------------------------
            # Already has image
            # --------------------------------------------------

            if image_path:

                enhanced_hits.append({

                    "text":
                        hit["text"],

                    "metadata":
                        metadata,

                    "score":
                        hit["final_score"]

                })

                continue

            # --------------------------------------------------
            # Find corresponding chunk
            # --------------------------------------------------

            matching_index = None

            for index, chunk in enumerate(
                searchable_chunks
            ):

                if (
                    chunk.get("text")
                    == hit["text"]
                ):

                    chunk_document_id = (
                        chunk.get(
                            "metadata",
                            {}
                        ).get(
                            "document_id"
                        )
                    )

                    hit_document_id = (
                        metadata.get(
                            "document_id"
                        )
                    )

                    if (
                        chunk_document_id
                        == hit_document_id
                    ):

                        matching_index = index

                        break

            # --------------------------------------------------
            # Search nearby chunks
            # --------------------------------------------------

            if matching_index is not None:

                start = max(
                    0,
                    matching_index - 1
                )

                end = min(
                    len(searchable_chunks),
                    matching_index + 2
                )

                for nearby_chunk in searchable_chunks[
                    start:end
                ]:

                    nearby_metadata = (
                        nearby_chunk.get(
                            "metadata",
                            {}
                        )
                    )

                    nearby_document_id = (
                        nearby_metadata.get(
                            "document_id"
                        )
                    )

                    current_document_id = (
                        metadata.get(
                            "document_id"
                        )
                    )

                    # --------------------------------------------------
                    # NEVER take image from another PDF
                    # --------------------------------------------------

                    if (
                        nearby_document_id
                        != current_document_id
                    ):

                        continue

                    nearby_image = (
                        nearby_metadata.get(
                            "image_path"
                        )
                    )

                    if nearby_image:

                        metadata[
                            "image_path"
                        ] = nearby_image

                        print(
                            "Image associated "
                            "with retrieved chunk: "
                            f"{nearby_image}"
                        )

                        break

            enhanced_hits.append({

                "text":
                    hit["text"],

                "metadata":
                    metadata,

                "score":
                    hit["final_score"]

            })

            # --------------------------------------------------
            # Stop after top_k
            # --------------------------------------------------

            if len(enhanced_hits) >= top_k:

                break

        return enhanced_hits
import chromadb
import hashlib


class VectorStoreManager:

    def __init__(
        self,
        persist_path="./storage/chroma_db"
    ):

        self.client = chromadb.PersistentClient(
            path=persist_path
        )

        self.collection = (
            self.client.get_or_create_collection(
                name="pdf_chunks"
            )
        )

    # ==================================================
    # ADD CHUNKS TO CHROMADB
    # ==================================================

    def add_chunks(
        self,
        chunks: list[str],
        metadata_list: list[dict],
        embeddings: list[list[float]]
    ):

        ids = []

        for index, chunk in enumerate(chunks):

            metadata = metadata_list[index]

            filename = metadata.get(
                "filename",
                "unknown"
            )

            document_id = metadata.get(
                "document_id",
                "unknown"
            )

            # --------------------------------------------------
            # Stable document-specific chunk ID
            # --------------------------------------------------

            unique_string = (
                f"{document_id}_"
                f"{filename}_"
                f"{index}_"
                f"{chunk}"
            )

            chunk_id = hashlib.md5(
                unique_string.encode("utf-8")
            ).hexdigest()

            ids.append(chunk_id)

        # --------------------------------------------------
        # Store in ChromaDB
        # --------------------------------------------------

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadata_list,
            documents=chunks
        )

        print(
            f"Stored {len(chunks)} chunks in ChromaDB"
        )

    # ==================================================
    # SEARCH VECTORS
    # ==================================================

    def search_vectors(
        self,
        query_embedding: list[float],
        top_k=5,
        document_id=None
    ):

        query_kwargs = {

            "query_embeddings": [
                query_embedding
            ],

            "n_results": top_k
        }

        # --------------------------------------------------
        # IMPORTANT
        #
        # If document_id is provided:
        # search only that document.
        #
        # If document_id is None:
        # search ALL documents.
        # --------------------------------------------------

        if document_id:

            query_kwargs["where"] = {
                "document_id": document_id
            }

        results = self.collection.query(
            **query_kwargs
        )

        return results
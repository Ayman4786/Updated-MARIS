from ai.rag.embeddings_service import EmbeddingsService
from ai.rag.vector_store import VectorStoreManager


embedder = EmbeddingsService()
vector_store = VectorStoreManager()


chunks = [
    "Inheritance allows one class to acquire properties of another class.",
    "Polymorphism allows one interface to take many forms.",
    "Encapsulation hides implementation details."
]


metadata = [
    {
        "filename": "python.pdf",
        "page_number": 1,
        "heading": "Inheritance"
    },
    {
        "filename": "python.pdf",
        "page_number": 2,
        "heading": "Polymorphism"
    },
    {
        "filename": "python.pdf",
        "page_number": 3,
        "heading": "Encapsulation"
    }
]


embeddings = [
    embedder.get_embedding(chunk)
    for chunk in chunks
]


vector_store.add_chunks(
    chunks=chunks,
    metadata_list=metadata,
    embeddings=embeddings
)


query_embedding = embedder.get_embedding(
    "What is inheritance?"
)


results = vector_store.search_vectors(
    query_embedding=query_embedding,
    top_k=3
)


print(results)
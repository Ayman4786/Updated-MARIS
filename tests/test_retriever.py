from ai.rag.embeddings_service import EmbeddingsService
from ai.rag.vector_store import VectorStoreManager
from ai.rag.retriever import HybridRetriever


embedder = EmbeddingsService()
vector_store = VectorStoreManager()

retriever = HybridRetriever(
    vector_store=vector_store,
    embeddings_service=embedder
)


all_chunks = [
    {
        "text": "Inheritance allows one class to acquire properties of another class.",
        "metadata": {
            "filename": "python.pdf",
            "page_number": 1,
            "heading": "Inheritance"
        }
    },
    {
        "text": "Polymorphism allows one interface to take many forms.",
        "metadata": {
            "filename": "python.pdf",
            "page_number": 2,
            "heading": "Polymorphism"
        }
    },
    {
        "text": "Encapsulation hides implementation details.",
        "metadata": {
            "filename": "python.pdf",
            "page_number": 3,
            "heading": "Encapsulation"
        }
    }
]


results = retriever.retrieve(
    query="What is inheritance?",
    all_chunks=all_chunks,
    top_k=3
)


print("\nRetrieved Chunks:\n")

for idx, result in enumerate(results, start=1):
    print(f"Result {idx}")
    print("Text:", result["text"])
    print("Metadata:", result["metadata"])
    print("-" * 50)
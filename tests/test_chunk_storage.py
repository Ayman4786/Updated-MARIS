from ai.rag.chunk_storage import save_chunks
from ai.rag.chunk_storage import load_chunks


chunks = [
    {
        "text": "Hello World",
        "metadata": {
            "filename": "test.md"
        }
    }
]


save_chunks(
    "storage/chunked_docs/test.json",
    chunks
)

loaded = load_chunks(
    "storage/chunked_docs/test.json"
)

print(loaded)
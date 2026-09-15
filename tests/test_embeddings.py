from ai.rag.embeddings_service import EmbeddingsService

embedder = EmbeddingsService()

embedding = embedder.get_embedding(
    "What is inheritance?"
)

print(type(embedding))
print(len(embedding))
print(embedding[:10])


# use this command for testing:
    # python -m tests.test_embeddings
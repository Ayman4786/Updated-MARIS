from ai.rag.chunker import RecursiveChunker
from ai.rag.embeddings_service import EmbeddingsService
from ai.rag.vector_store import VectorStoreManager
from ai.rag.retriever import HybridRetriever
from ai.rag.context_builder import ContextBuilder

from ai.llm.prompt_builder import build_prompt
from ai.llm.llm_service import generate_answer


# STEP 1: Read markdown

with open(
    "storage/extracted_docs/CC_test.md",
    "r",
    encoding="utf-8"
) as file:
    markdown = file.read()


# STEP 2: Chunk document

chunker = RecursiveChunker()

chunks = chunker.split_text(markdown)

print(f"\nTotal Chunks: {len(chunks)}")


# STEP 3: Create embeddings

embedder = EmbeddingsService()

embeddings = [
    embedder.get_embedding(chunk)
    for chunk in chunks
]


# STEP 4: Create metadata

metadata_list = []

for i in range(len(chunks)):
    metadata_list.append(
        {
            "filename": "CC_test.md",
            "page_number": 0,
            "heading": f"chunk_{i}"
        }
    )


# STEP 5: Store in ChromaDB

vector_store = VectorStoreManager()

vector_store.add_chunks(
    chunks=chunks,
    metadata_list=metadata_list,
    embeddings=embeddings
)


# STEP 6: Prepare retriever

retriever = HybridRetriever(
    vector_store=vector_store,
    embeddings_service=embedder
)


# STEP 7: Ask question

question = "What is scalable computing?"


all_chunks = [
    {
        "text": chunk,
        "metadata": metadata_list[i]
    }
    for i, chunk in enumerate(chunks)
]


# STEP 8: Retrieve relevant chunks

retrieved_chunks = retriever.retrieve(
    query=question,
    all_chunks=all_chunks,
    top_k=3
)


print("\n===== RETRIEVED CHUNKS =====\n")

for chunk in retrieved_chunks:
    print(chunk["text"][:300])
    print("\n" + "=" * 50 + "\n")


# STEP 9: Build context

context = ContextBuilder.build_context(
    retrieved_chunks
)

print("\n===== CONTEXT =====\n")
print(context[:1000])


# STEP 10: Build prompt

prompt = build_prompt(
    document_text=context,
    user_question=question
)


# STEP 11: Ask LLM

answer = generate_answer(
    prompt
)


print("\n===== ANSWER =====\n")
print(answer)
from ai.rag.chunker import RecursiveChunker

chunker = RecursiveChunker()

markdown = open("storage/extracted_docs/CC_test.md").read()

chunks = chunker.split_text(markdown)

print(f"Chunks: {len(chunks)}")

for i, chunk in enumerate(chunks):
    print(f"\nChunk {i+1}")
    print(chunk[:200])  


# use this command for testing:
    # python -m tests.chunker_test
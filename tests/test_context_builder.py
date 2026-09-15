from ai.rag.context_builder import ContextBuilder


retrieved_chunks = [
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
    }
]


context = ContextBuilder.build_context(
    retrieved_chunks
)

print(context)
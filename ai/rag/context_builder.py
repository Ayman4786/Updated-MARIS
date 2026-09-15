class ContextBuilder:

    @staticmethod
    def build_context(
        retrieved_chunks: list[dict]
    ) -> str:

        context_blocks = []

        for idx, chunk in enumerate(
            retrieved_chunks
        ):

            text = chunk.get(
                "text",
                ""
            )

            meta = chunk.get(
                "metadata",
                {}
            ) or {}

            filename = (
                meta.get("filename")
                or "Unknown Source"
            )

            page = (
                meta.get("page_number")
                or meta.get("page")
                or "N/A"
            )

            heading = (
                meta.get("heading")
                or "General"
            )

            block = (

                f"--- Chunk {idx + 1} ---\n"

                f"Source: {filename} "
                f"(Page {page}) | "
                f"Section: {heading}\n"

                f"Content:\n"
                f"{text}\n"
            )

            context_blocks.append(
                block
            )

        return "\n".join(
            context_blocks
        )
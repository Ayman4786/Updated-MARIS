import os


DEFAULT_CONTEXT_CHAR_BUDGET = 24000


class ContextBuilder:

    @staticmethod
    def build_context(
        retrieved_chunks: list[dict],
        max_chars: int | None = None,
    ) -> str:

        context_blocks = []
        seen_text = set()
        budget = max_chars or int(
            os.getenv(
                "MARIS_CONTEXT_CHAR_BUDGET",
                DEFAULT_CONTEXT_CHAR_BUDGET,
            )
        )
        used_chars = 0

        for idx, chunk in enumerate(
            retrieved_chunks
        ):

            text = chunk.get(
                "text",
                ""
            )
            normalized_text = " ".join(str(text).split())
            if not normalized_text or normalized_text in seen_text:
                continue
            seen_text.add(normalized_text)

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

            remaining = budget - used_chars
            if remaining <= 0:
                break
            if len(block) > remaining:
                # Keep the beginning of the highest-ranked chunk, where the
                # page-local diagram description is appended by ingestion.
                block = block[:remaining].rstrip() + "\n"
            context_blocks.append(
                block
            )
            used_chars += len(block)

            if used_chars >= budget:
                break

        return "\n".join(
            context_blocks
        )
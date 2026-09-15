import re

class RecursiveChunker:
    def __init__(self, chunk_size=500, chunk_overlap=100):
        self.chunk_size = chunk_size  # In tokens
        self.chunk_overlap = chunk_overlap
        # Roughly 1 token = 4 characters in English
        self.chars_per_token = 4
        self.max_chars = chunk_size * self.chars_per_token
        self.overlap_chars = chunk_overlap * self.chars_per_token

    def split_text(self, text: str) -> list[str]:
        # Recursive separators: paragraphs, then sentences, then spaces
        separators = ["\n\n", "\n", " "]
        return self._recursive_split(text, separators)

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        if len(text) <= self.max_chars:
            return [text]
        
        if not separators:
            # Fallback if no separators work: hard cut
            return [text[i:i+self.max_chars] for i in range(0, len(text), self.max_chars - self.overlap_chars)]
        
        separator = separators[0]
        splits = text.split(separator)
        
        chunks = []
        current_chunk = ""
        
        for split in splits:
            if len(current_chunk) + len(separator) + len(split) <= self.max_chars:
                current_chunk += (separator if current_chunk else "") + split
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                # Keep overlap from the end of the previous chunk if possible
                current_chunk = split
                
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks
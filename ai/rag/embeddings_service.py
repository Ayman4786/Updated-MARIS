import torch
from transformers import AutoTokenizer, AutoModel

class EmbeddingsService:
    def __init__(self):
        # Using BGE-small-en-v1.5 as planned
        self.model_name = "BAAI/bge-small-en-v1.5"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.eval()  # Put model in evaluation mode

    def get_embedding(self, text: str) -> list[float]:
        encoded_input = self.tokenizer([text], padding=True, truncation=True, return_tensors='pt', max_length=512)
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            # Perform pooling (BGE uses the CLS token embedding)
            sentence_embeddings = model_output[0][:, 0]
        
        # Normalize embeddings
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings.tolist()[0]
    
    
"""Windows-compatible embeddings without sentence-transformers."""

import numpy as np
from typing import List
from langchain_core.embeddings import Embeddings
import torch
from transformers import AutoTokenizer, AutoModel


class TransformerEmbeddings(Embeddings):
    """Simple embeddings using transformers directly, without sentence-transformers."""

    def __init__(self, model_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct"):
        """Initialize with a transformer model.

        Args:
            model_name: HuggingFace model ID. The model URL will work with or without
                       the sentence-transformers prefix.
        """
        self.model_name = model_name

        # Handle sentence-transformers style names
        if model_name.startswith("sentence-transformers/"):
            # Convert to huggingface model format
            model_path = model_name.replace("sentence-transformers/", "")
            # Try exact name first (e.g., all-MiniLM-L6-v2)
            # If that doesn't work, it will fail gracefully
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    f"sentence-transformers/{model_path}", trust_remote_code=True
                )
                self.model = AutoModel.from_pretrained(
                    f"sentence-transformers/{model_path}", trust_remote_code=True
                )
            except Exception:
                # Fallback to direct huggingface model
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path, trust_remote_code=True
                )
                self.model = AutoModel.from_pretrained(
                    model_path, trust_remote_code=True
                )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

        # Move model to appropriate device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        self.model.eval()

    def _mean_pooling(self, model_output, attention_mask):
        """Mean pooling for embeddings."""
        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        embeddings = []

        with torch.no_grad():
            for text in texts:
                # Tokenize
                encoded = self.tokenizer(
                    text,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )

                # Move to device
                encoded = {k: v.to(self.device) for k, v in encoded.items()}

                # Get embeddings
                model_output = self.model(**encoded)
                embeddings_tensor = self._mean_pooling(
                    model_output, encoded["attention_mask"]
                )

                # Normalize
                embeddings_tensor = torch.nn.functional.normalize(
                    embeddings_tensor, p=2, dim=1
                )

                embeddings.append(embeddings_tensor[0].cpu().numpy().tolist())

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        return self.embed_documents([text])[0]


def get_embeddings(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> Embeddings:
    """Get embeddings using transformers."""
    print(f"✓ Loading embeddings ({model_name})")
    return TransformerEmbeddings(model_name)

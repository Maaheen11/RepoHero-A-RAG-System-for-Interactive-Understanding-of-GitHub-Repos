"""
Two-stage retriever:
  Stage 1 — Bi-encoder (nomic-ai/nomic-embed-code)
             encodes query + all chunks independently,
             retrieves top-K by cosine similarity
  Stage 2 — Cross-encoder (cross-encoder/ms-marco-MiniLM-L-12-v2)
             reranks the top-K candidates jointly,
             returns top-N to the LLM
"""

import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np
from typing import List, Dict, Any


class TwoStageRetriever:
    def __init__(
        self,
        bi_encoder_model: str = "nomic-ai/nomic-embed-code",
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
        bi_encoder_top_k: int = 20,
        cross_encoder_top_n: int = 5,
    ):
        self.bi_encoder_top_k = bi_encoder_top_k
        self.cross_encoder_top_n = cross_encoder_top_n

        # Detect device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        if self.device == "cuda":
            free, total = torch.cuda.mem_get_info(0)
            print(f"Free VRAM: {free/1024**3:.1f} GB / {total/1024**3:.1f} GB")

        # Load both models on CPU — we move to GPU only during encode/predict
        # so Ollama can share the GPU for LLM generation
        print(f"Loading bi-encoder: {bi_encoder_model}")
        self.bi_encoder = SentenceTransformer(
            bi_encoder_model,
            trust_remote_code=True,
            device="cpu",
            model_kwargs={"torch_dtype": torch.float16},  # fp16 halves VRAM
        )

        print(f"Loading cross-encoder: {cross_encoder_model}")
        self.cross_encoder = CrossEncoder(cross_encoder_model, device="cpu")

        # in-memory chunk store
        self._chunks: List[Dict[str, Any]] = []
        self._embeddings: np.ndarray = np.empty((0,), dtype=np.float32)

    # ------------------------------------------------------------------ #
    # Internal: borrow GPU, encode, return GPU                            #
    # ------------------------------------------------------------------ #

    def _encode(self, texts, batch_size=8, normalize=True) -> np.ndarray:
        """Move bi-encoder to GPU, encode, move back to CPU, free VRAM."""
        if self.device == "cuda":
            self.bi_encoder = self.bi_encoder.to("cuda")

        embeddings = self.bi_encoder.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
        )

        if self.device == "cuda":
            self.bi_encoder = self.bi_encoder.to("cpu")
            torch.cuda.empty_cache()

        return embeddings.astype(np.float32)

    def _rerank(self, pairs) -> np.ndarray:
        """Move cross-encoder to GPU, predict, move back to CPU, free VRAM."""
        if self.device == "cuda":
            self.cross_encoder.model = self.cross_encoder.model.to("cuda")
            self.cross_encoder._target_device = torch.device("cuda")

        scores = self.cross_encoder.predict(pairs)

        if self.device == "cuda":
            self.cross_encoder.model = self.cross_encoder.model.to("cpu")
            self.cross_encoder._target_device = torch.device("cpu")
            torch.cuda.empty_cache()

        return scores

    # ------------------------------------------------------------------ #
    # Indexing                                                             #
    # ------------------------------------------------------------------ #

    def index(self, chunks: List[str], metadatas: List[Dict[str, Any]]):
        assert len(chunks) == len(metadatas), "chunks and metadatas must be the same length"
        self._chunks = [{"chunk": c, "metadata": m} for c, m in zip(chunks, metadatas)]

        print(f"Encoding {len(chunks)} chunks with bi-encoder ...")
        self._embeddings = self._encode(chunks, batch_size=2, normalize=True)
        print("Indexing complete. GPU freed for Ollama.")

    def is_indexed(self) -> bool:
        return len(self._chunks) > 0

    # ------------------------------------------------------------------ #
    # Retrieval                                                            #
    # ------------------------------------------------------------------ #

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        Stage 1: bi-encoder retrieves top-K candidates (GPU -> CPU -> free).
        Stage 2: cross-encoder reranks to top-N (GPU -> CPU -> free).
        """
        if not self.is_indexed():
            raise RuntimeError("Retriever has not been indexed yet. Call index() first.")

        # ---- Stage 1: bi-encoder ----------------------------------------
        query_embedding = self._encode(query, batch_size=1, normalize=True)
        if query_embedding.ndim == 2:
            query_embedding = query_embedding[0]

        scores = self._embeddings @ query_embedding
        k = min(self.bi_encoder_top_k, len(self._chunks))
        top_k_indices = np.argpartition(scores, -k)[-k:]
        top_k_indices = top_k_indices[np.argsort(scores[top_k_indices])[::-1]]

        candidates = [self._chunks[i] for i in top_k_indices]
        candidate_scores = scores[top_k_indices].tolist()

        # ---- Stage 2: cross-encoder reranking ----------------------------
        pairs = [[query, c["chunk"]] for c in candidates]
        cross_scores = self._rerank(pairs)

        ranked = sorted(
            zip(cross_scores, candidate_scores, candidates),
            key=lambda x: x[0],
            reverse=True,
        )

        results = []
        for cross_score, bi_score, item in ranked[:self.cross_encoder_top_n]:
            results.append({
                "similarity": float(bi_score),
                "cross_score": float(cross_score),
                "chunk": item["chunk"],
                "metadata": item["metadata"],
            })

        return results
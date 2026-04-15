import argparse
import ollama
from typing import List, Dict, Any
import chromadb
import hashlib
from pathlib import Path
import pathspec

## self-defined modules
from chunker import PythonASTChunker

# Only imported when --biencoder is used
_TwoStageRetriever = None
def _get_two_stage_retriever():
    global _TwoStageRetriever
    if _TwoStageRetriever is None:
        from retriever import TwoStageRetriever
        _TwoStageRetriever = TwoStageRetriever
    return _TwoStageRetriever


# bge-base-en-v1.5 has a 512-token context window.
# Code tokens are often sub-word, so we use a conservative 800-char cap.
MAX_EMBED_CHARS = 800

def _split_oversized(chunks: List[str], max_chars: int = MAX_EMBED_CHARS) -> List[str]:
    """Split any chunk that exceeds max_chars into smaller pieces."""
    result = []
    for chunk in chunks:
        if len(chunk) <= max_chars:
            result.append(chunk)
        else:
            for start in range(0, len(chunk), max_chars):
                result.append(chunk[start:start + max_chars])
    return result


def _embed_one(model: str, text: str) -> List[float]:
    """Embed a single text, truncating hard at MAX_EMBED_CHARS as a last resort."""
    text = text[:MAX_EMBED_CHARS]
    return ollama.embed(model=model, input=text)["embeddings"][0]


class RepoHero:
    def __init__(
        self,
        llm_model: str = "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF",
        db_path: str = './chroma_db',
        use_biencoder: bool = False,
        # Baseline-only
        embedding_model: str = "hf.co/CompendiumLabs/bge-base-en-v1.5-gguf",
        # Biencoder-only
        bi_encoder_model: str = "nomic-ai/nomic-embed-code",
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
        bi_encoder_top_k: int = 20,
        cross_encoder_top_n: int = 5,
    ):
        self.llm_model = llm_model
        self.use_biencoder = use_biencoder
        self.embedding_model = embedding_model

        self.chromadb_client = chromadb.PersistentClient(path=db_path)

        if use_biencoder:
            # Two-stage retriever: bi-encoder + cross-encoder
            # ChromaDB is used only as a persistent text/metadata store.
            self.chroma_db = self.chromadb_client.get_or_create_collection(
                name="chroma_db_repohero",
                metadata={"hnsw:space": "cosine"},
            )
            TwoStageRetriever = _get_two_stage_retriever()
            self.retriever = TwoStageRetriever(
                bi_encoder_model=bi_encoder_model,
                cross_encoder_model=cross_encoder_model,
                bi_encoder_top_k=bi_encoder_top_k,
                cross_encoder_top_n=cross_encoder_top_n,
            )
        else:
            # Baseline: Ollama embeddings stored directly in ChromaDB
            model_name = embedding_model.split('/')[-1].replace('.', '_').replace('-', '_')
            self.chroma_db = self.chromadb_client.get_or_create_collection(
                name=f"chroma_db_{model_name}",
                metadata={"hnsw:space": "cosine"},
            )

        self.ast_chunker = PythonASTChunker(max_nws_chars=100)

    # ------------------------------------------------------------------ #
    # Fingerprinting                                                       #
    # ------------------------------------------------------------------ #

    def get_file_hash(self, path: str) -> str:
        hasher = hashlib.md5()
        with open(path, "rb") as f:
            hasher.update(f.read())
        return hasher.hexdigest()

    # ------------------------------------------------------------------ #
    # Ingestion                                                            #
    # ------------------------------------------------------------------ #

    def list_files(self, path: str):
        repo = Path(path)
        gitignore = repo / ".gitignore"
        if gitignore.exists():
            with open(gitignore) as f:
                spec = pathspec.PathSpec.from_lines("gitwildmatch", f)
        else:
            spec = pathspec.PathSpec([])

        files = []
        for file in repo.rglob("*.py"):
            if not file.is_file():
                continue
            relative_path = file.relative_to(repo)
            if spec.match_file(str(relative_path)):
                continue
            elif "test" in file.name:
                continue
            files.append(file)
        return files

    def read_file(self, file: Path) -> str:
        with open(file, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    # ------------------------------------------------------------------ #
    # Chunking                                                             #
    # ------------------------------------------------------------------ #

    def chunk(self, file_path: str) -> List[str]:
        ast_chunks = self.ast_chunker.chunk_file(file_path)
        return [chunk.contextualized_text for chunk in ast_chunks]

    # ------------------------------------------------------------------ #
    # Embedding / Indexing                                                 #
    # ------------------------------------------------------------------ #

    def embed_repo(self, path: str):
        files = self.list_files(path)

        ids, documents, metadatas = [], [], []
        embeddings = []  # only populated in baseline mode

        for file in files:
            file_hash = self.get_file_hash(file)

            existing = self.chroma_db.get(
                where={
                    "$and": [
                        {"file_path": str(file)},
                        {"file_hash": file_hash},
                    ]
                }
            )
            if existing["ids"]:
                continue  # unchanged — skip

            self.chroma_db.delete(where={"file_path": str(file)})
            chunks = self.chunk(str(file))
            chunks = _split_oversized(chunks)  # guard against context-length errors

            for i, chunk_text in enumerate(chunks):
                ids.append(f"{file}_{i}")
                documents.append(chunk_text)
                metadatas.append({"file_path": str(file), "file_hash": file_hash})
                if not self.use_biencoder:
                    # Embed one chunk at a time — avoids batch context issues
                    emb = _embed_one(self.embedding_model, chunk_text)
                    embeddings.append(emb)

        if ids:
            if self.use_biencoder:
                # Placeholder embeddings — ChromaDB is text store only
                self.chroma_db.add(
                    ids=ids,
                    embeddings=[[0.0]] * len(ids),
                    documents=documents,
                    metadatas=metadatas,
                )
            else:
                self.chroma_db.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas,
                )
        else:
            print("ChromaDB already has all files from this repo — loading from cache.")

        if self.use_biencoder:
            # Build in-memory bi-encoder index from all stored chunks
            all_docs = self.chroma_db.get(include=["documents", "metadatas"])
            self.retriever.index(all_docs["documents"], all_docs["metadatas"])

    # ------------------------------------------------------------------ #
    # Retrieval                                                            #
    # ------------------------------------------------------------------ #

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.use_biencoder:
            # top_k is controlled by cross_encoder_top_n set at init time
            return self.retriever.retrieve(query)
        else:
            # Baseline: embed query with Ollama, search ChromaDB directly
            query_embedding = _embed_one(self.embedding_model, query)

            results = self.chroma_db.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
            )

            retrieved_items = []
            for doc, dist, metadata in zip(
                results["documents"][0],
                results["distances"][0],
                results["metadatas"][0],
            ):
                retrieved_items.append({
                    "similarity": 1 - dist,
                    "cross_score": None,
                    "chunk": doc,
                    "metadata": metadata,
                })
            return retrieved_items

    # ------------------------------------------------------------------ #
    # Chat                                                                 #
    # ------------------------------------------------------------------ #

    def rewrite_query(self, input_query: str) -> str:
        system_prompt = """
        You are a query rewriter for repository-level code search.

        Convert the user's question into a concise retrieval query for semantic search
        over a Python codebase.

        Your rewritten query should:
        - preserve the user's original intent
        - include likely code symbols if relevant
        - include likely library or framework names if relevant
        - include technical synonyms when useful
        - be short and keyword-dense

        Do not answer the question.
        Do not explain your reasoning.
        Output only one rewritten query.
        """
        response = ollama.chat(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": input_query},
            ],
            stream=False,
        )
        return response["message"]["content"].strip()

    def chat(self):
        input_query = input("Ask me a question: ")

        retrieved_knowledge = self.retrieve(input_query)

        context_text = "\n".join(
            f"[source:{item['metadata'].get('file_path', 'unknown')}]\n{item['chunk']}"
            for item in retrieved_knowledge
        )

        instruction_prompt = f"""
        You are a helpful chatbot for answering questions about a code repository.
        Use only the provided context to answer the question.
        When possible, mention the relevant file path as evidence.
        If the context does not contain enough information, say so clearly.
        Do not make up any new information.

        context:
        {context_text}
        """

        stream = ollama.chat(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": instruction_prompt},
                {"role": "user", "content": input_query},
            ],
            stream=True,
        )

        print("\nChatbot response:\n")
        for chunk in stream:
            print(chunk["message"]["content"], end="", flush=True)

    # ------------------------------------------------------------------ #
    # Full Pipeline                                                        #
    # ------------------------------------------------------------------ #

    def run(self, path: str):
        self.embed_repo(path)
        print("\nRepository indexing complete.")
        mode = "biencoder + cross-encoder" if self.use_biencoder else "baseline (Ollama embeddings)"
        print(f"Retrieval mode: {mode}")
        print("You can now ask questions about the code.\n")
        while True:
            try:
                self.chat()
                cont = input("\nAsk another question? (y/n): ").strip().lower()
                if cont != "y":
                    print("Exiting RepoHero.")
                    break
            except KeyboardInterrupt:
                print("\nExiting RepoHero.")
                break


# ------------------------------------------------------------------ #
# Entry point                                                          #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RepoHero — RAG over a Python repo")
    parser.add_argument("repo_path", nargs="?", help="Path to the repository to analyse")
    parser.add_argument(
        "--biencoder",
        action="store_true",
        help="Use two-stage retrieval (bi-encoder + cross-encoder) instead of the baseline Ollama embeddings",
    )
    args = parser.parse_args()

    repo_path = args.repo_path or input("Enter the path of the repository to analyse: ")

    rag = RepoHero(use_biencoder=args.biencoder)
    rag.run(repo_path)
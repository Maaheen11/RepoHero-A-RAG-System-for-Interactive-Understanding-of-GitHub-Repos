import ollama
import heapq
from typing import List, Tuple


class RepoHero:
    def __init__(
        self,
        embedding_model: str = "hf.co/CompendiumLabs/bge-base-en-v1.5-gguf",
        llm_model: str = "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF",
    ):
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.vector_db: List[Tuple[str, List[float]]] = []

    # ------------------------
    # Ingestion
    # ------------------------
    def ingest(self, path: str) -> str:
        with open(path, "r") as f:
            data = f.read()
        print(f"Loaded {len(data)} characters from {path}")
        return data

    # ------------------------
    # Chunking
    # ------------------------
    def chunk(self, data: str) -> List[str]:
        lines = data.splitlines()
        print(f"Processed {len(lines)} lines of code")
        return lines

    # ------------------------
    # Embedding
    # ------------------------
    def embed(self, chunks: List[str]) -> None:
        self.vector_db = []

        for chunk in chunks:
            if not chunk.strip():
                continue

            embedding = ollama.embed(
                model=self.embedding_model,
                input=chunk
            )["embeddings"][0]

            self.vector_db.append((chunk, embedding))

        print(f"Embedded {len(self.vector_db)} chunks")

    # ------------------------
    # Cosine Similarity
    # ------------------------
    @staticmethod
    def cosine_similarity(a: List[float], b: List[float]) -> float:
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x ** 2 for x in a) ** 0.5
        norm_b = sum(x ** 2 for x in b) ** 0.5
        return dot_product / (norm_a * norm_b)

    # ------------------------
    # Retrieval
    # ------------------------
    def retrieve(self, query: str, top_k: int = 5):
        query_embedding = ollama.embed(
            model=self.embedding_model,
            input=query
        )["embeddings"][0]

        similarities = []

        for chunk, embedding in self.vector_db:
            score = self.cosine_similarity(query_embedding, embedding)

            heapq.heappush(similarities, (score, chunk))
            if len(similarities) > top_k:
                heapq.heappop(similarities)

        return similarities

    # ------------------------
    # Chat
    # ------------------------
    def chat(self):
        input_query = input("Ask me a question: ")

        retrieved_knowledge = self.retrieve(input_query, top_k=5)

        for score, chunk in retrieved_knowledge:
            print(f"Chunk: {chunk}\nScore: {score}\n")

        context_text = "\n".join(
            f"- {chunk}" for score, chunk in retrieved_knowledge
        )

        instruction_prompt = f"""
        You are a helpful chatbot.
        Use only the following pieces of context to answer the question.
        Don't make up any new information:

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

    # ------------------------
    # Full Pipeline
    # ------------------------
    def run(self, path: str):
        data = self.ingest(path)
        chunks = self.chunk(data)
        self.embed(chunks)
        self.chat()


# ------------------------
# Usage
# ------------------------
if __name__ == "__main__":
    rag = RepoHero()
    rag.run("main.py")
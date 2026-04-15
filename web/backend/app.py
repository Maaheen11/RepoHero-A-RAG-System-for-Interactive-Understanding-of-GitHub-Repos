from __future__ import annotations

from pathlib import Path
from typing import Any
import sys
import os
import hashlib

from flask import Flask, jsonify, request
import ollama

_here = Path(__file__).resolve()
_candidates = [_here.parents[2], _here.parents[1], _here.parents[0], Path.cwd()]
REPOHERO_ROOT = next((p for p in _candidates if (p / "main.py").exists()), _here.parents[2])
if str(REPOHERO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOHERO_ROOT))

from main import RepoHero


def _chat_answer(agent: RepoHero, query: str, top_k: int = 5) -> tuple[str, list[tuple[float, str]]]:
    """
    Same retrieval + LLM steps as RepoHero.chat(), but non-interactive (chat() uses input/print/stream).
    """
    retrieved_knowledge = agent.retrieve(query, top_k=top_k)

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

    response = ollama.chat(
        model=agent.llm_model,
        messages=[
            {"role": "system", "content": instruction_prompt},
            {"role": "user", "content": query},
        ],
        stream=False,
    )

    retrieved_pairs: list[tuple[float, str]] = []
    for item in retrieved_knowledge:
        if item.get("cross_score") is not None:
            score = float(item["cross_score"])
        else:
            score = float(item.get("similarity", 0.0))
        retrieved_pairs.append((score, item["chunk"]))

    return response["message"]["content"], retrieved_pairs


app = Flask(__name__)

CHROMA_ROOT = REPOHERO_ROOT / "web" / "backend" / "chroma_db"


def _agent_for_repo(repo_path: str) -> RepoHero:
    # Use one persistent Chroma directory per repo path to avoid cross-repo contamination.
    repo_key = hashlib.md5(repo_path.encode("utf-8")).hexdigest()
    db_path = CHROMA_ROOT / repo_key
    db_path.mkdir(parents=True, exist_ok=True)
    return RepoHero(db_path=str(db_path))


_state: dict[str, Any] = {
    "repo_path": None,
    "agent": None,
    "indexed": False,
}


def _cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response


@app.after_request
def add_cors_headers(response):
    return _cors(response)


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"ok": True})


@app.route("/api/select-repo", methods=["POST", "OPTIONS"])
def select_repo():
    if request.method == "OPTIONS":
        return _cors(app.make_default_options_response())

    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askdirectory(title="Select repository folder") or None
        root.destroy()
    except Exception:
        return jsonify(
            {
                "repoPath": None,
                "indexed": False,
                "message": "Native folder picker is unavailable in this environment. Paste path manually.",
            }
        )

    if not path:
        return jsonify({"repoPath": None, "indexed": False, "message": "No folder selected."})

    _state["repo_path"] = path
    _state["indexed"] = False

    return jsonify({"repoPath": path, "indexed": False})


@app.route("/api/index", methods=["POST", "OPTIONS"])
def index_repo():
    if request.method == "OPTIONS":
        return _cors(app.make_default_options_response())

    data = request.get_json(silent=True) or {}
    repo_path = data.get("repoPath") or _state["repo_path"]

    if not repo_path:
        return jsonify({"error": "Repository path is required."}), 400

    repo = Path(repo_path)
    if not repo.exists() or not repo.is_dir():
        return jsonify(
            {
                "error": (
                    f"Invalid repository path: {repo_path}. "
                    "When running in Docker Compose, use container paths "
                    "like /workspace/repo (or a subfolder under it), not host paths like /Users/..."
                )
            }
        ), 400

    _state["agent"] = _agent_for_repo(str(repo.resolve()))
    _state["agent"].embed_repo(str(repo))
    _state["repo_path"] = str(repo)
    _state["indexed"] = True

    return jsonify({"ok": True, "repoPath": str(repo)})


@app.route("/api/chat", methods=["POST", "OPTIONS"])
def chat():
    if request.method == "OPTIONS":
        return _cors(app.make_default_options_response())

    data = request.get_json(silent=True) or {}
    query = (data.get("message") or "").strip()

    if not query:
        return jsonify({"error": "Message cannot be empty."}), 400

    if not _state["indexed"]:
        return jsonify({"error": "Please index a repository first."}), 400
    if _state["agent"] is None:
        return jsonify({"error": "Agent is not initialized."}), 500

    answer, retrieved = _chat_answer(_state["agent"], query, top_k=5)

    return jsonify(
        {
            "answer": answer,
            "sources": [{"score": score, "chunk": chunk} for score, chunk in retrieved],
        }
    )


if __name__ == "__main__":
    host = os.getenv("REPOHERO_BACKEND_HOST", "127.0.0.1")
    port = int(os.getenv("REPOHERO_BACKEND_PORT", "5001"))
    debug = os.getenv("REPOHERO_BACKEND_DEBUG", "1") == "1"
    app.run(host=host, port=port, debug=debug)

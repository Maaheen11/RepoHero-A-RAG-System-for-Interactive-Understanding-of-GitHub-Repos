import { useEffect, useRef, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:5001/api";

export default function App() {
  const [repoPath, setRepoPath] = useState("");
  const [isIndexed, setIsIndexed] = useState(false);
  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState("Enter a repository path to begin.");
  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  async function indexRepo() {
    if (!repoPath.trim()) {
      setStatus("Please provide a repository path.");
      return;
    }

    setLoading(true);
    setStatus("Indexing repository...");

    try {
      const res = await fetch(`${API_BASE}/index`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ repoPath })
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.error || "Indexing failed.");
      }

      setIsIndexed(true);
      setStatus(`Indexed: ${data.repoPath}`);
    } catch (error) {
      setStatus(error.message);
      setIsIndexed(false);
    } finally {
      setLoading(false);
    }
  }

  async function sendQuestion(event) {
    event.preventDefault();

    if (!question.trim() || !isIndexed || loading) {
      return;
    }

    const userText = question.trim();
    setQuestion("");
    setMessages((prev) => [
      ...prev,
      { id: crypto.randomUUID(), role: "user", text: userText }
    ]);
    setLoading(true);

    try {
      const res = await fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: userText })
      });
      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.error || "Chat request failed.");
      }

      const normalizedSources = Array.isArray(data.sources)
        ? data.sources
            .filter((source) => source && typeof source.chunk === "string" && source.chunk.trim())
            .slice(0, 5)
        : [];

      setMessages((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          text: data.answer || "I could not generate an answer.",
          sources: normalizedSources
        }
      ]);
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        { id: crypto.randomUUID(), role: "assistant", text: `Error: ${error.message}` }
      ]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app">
      <header>
        <h1>Repo Hero</h1>
      </header>

      <section className="panel">
        <label htmlFor="repo-path">Repository Path</label>
        <div className="row">
          <input
            id="repo-path"
            type="text"
            value={repoPath}
            onChange={(e) => setRepoPath(e.target.value)}
            placeholder="/path/to/your/repository"
          />
          <button onClick={indexRepo} disabled={loading}>Index Repository</button>
        </div>
        <p className="status">{status}</p>
      </section>

      <section className="chat">
        <div className="messages">
          {messages.length === 0 ? (
            <p className="placeholder">Ask questions about your selected repository.</p>
          ) : (
            messages.map((msg) => (
              <div key={msg.id} className={`message ${msg.role}`}>
                <strong>{msg.role === "user" ? "You" : "Repo Hero"}:</strong> {msg.text}
                {msg.role === "assistant" && msg.sources?.length > 0 && (
                  <div className="context">
                    <div className="context-title">Relevant code context ({msg.sources.length})</div>
                    {msg.sources.map((source, sourceIdx) => (
                      <details key={`${msg.id}-ctx-${sourceIdx}`} className="context-item">
                        <summary>
                          Context {sourceIdx + 1}
                          {Number.isFinite(Number(source.score))
                            ? ` (score: ${Number(source.score).toFixed(3)})`
                            : ""}
                        </summary>
                        <pre>{source.chunk}</pre>
                      </details>
                    ))}
                  </div>
                )}
              </div>
            ))
          )}
          {loading && (
            <div className="message assistant loading-message">
              <strong>Repo Hero:</strong> Agent is thinking
              <span className="dot-loader" aria-label="Loading">
                <span>.</span><span>.</span><span>.</span>
              </span>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <form onSubmit={sendQuestion} className="row">
          <input
            type="text"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder={isIndexed ? "Ask a question..." : "Index a repository first"}
            disabled={!isIndexed || loading}
          />
          <button type="submit" disabled={!isIndexed || loading || !question.trim()}>
            Send
          </button>
        </form>
      </section>
    </div>
  );
}

import { useState } from "react";
import { Send } from "lucide-react";

const SUGGESTIONS = [
  "protocol for a collapsed structure with trapped survivors",
  "how should we prioritize flood victims",
  "supply allocation when medkits run low",
];

export default function KnowledgePanel({ askKnowledge }) {
  const [question, setQuestion] = useState("");
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);

  async function submit(q) {
    const text = (q ?? question).trim();
    if (!text || loading) return;
    setLoading(true);
    setQuestion("");
    const response = await askKnowledge(text);
    setHistory((h) => [...h, { question: text, response }]);
    setLoading(false);
  }

  return (
    <div className="panel knowledge-panel">
      <div className="panel-title">
        FIELD MANUAL
        <span className="count">LLM + RAG</span>
      </div>

      <div className="scroll knowledge-history">
        {history.length === 0 && (
          <div className="knowledge-empty">
            <p className="mono-label">Ask the doctrine knowledge base a question about the current situation.</p>
            <div className="suggestions">
              {SUGGESTIONS.map((s) => (
                <button key={s} className="suggestion-chip" onClick={() => submit(s)}>
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}
        {history.map((h, i) => (
          <div key={i} className="knowledge-entry">
            <div className="knowledge-q">{h.question}</div>
            <div className="knowledge-a">{h.response.answer}</div>
            {h.response.suggested_task_decomposition?.length > 0 && (
              <ul className="knowledge-tasks">
                {h.response.suggested_task_decomposition.map((t, j) => (
                  <li key={j}>{t}</li>
                ))}
              </ul>
            )}
            {h.response.sources?.length > 0 && <div className="knowledge-sources mono-label">source: {h.response.sources.join(", ")}</div>}
          </div>
        ))}
        {loading && <div className="knowledge-loading mono-label">retrieving doctrine…</div>}
      </div>

      <form
        className="knowledge-input-row"
        onSubmit={(e) => {
          e.preventDefault();
          submit();
        }}
      >
        <input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask about protocol, triage, supply priority…"
          aria-label="Ask the field manual"
        />
        <button type="submit" aria-label="Ask" disabled={loading}>
          <Send size={14} />
        </button>
      </form>
    </div>
  );
}

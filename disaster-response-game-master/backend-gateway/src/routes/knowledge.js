// POST /knowledge_query
//
// Real behavior (once Saatwik's agent-intelligence service is up): proxy straight through to
// AGENT_SERVICE_URL + "/knowledge_query" — same request/response shape, see docs/api-contract.md.
// Until then: canned-but-plausible answers so the frontend's knowledge panel has something real
// to render and the request/response contract is exercised end-to-end.

const CANNED = [
  {
    match: /collaps|structur|rubble|trapped/i,
    answer:
      "Stabilize the structure before entry. Require a two-person minimum with a dedicated spotter watching for secondary collapse. Mark cleared areas and confirm a victim's exact position before committing heavier equipment.",
    sources: ["FEMA US&R Field Operations Guide, ch. 4"],
    suggested_task_decomposition: [
      "Hold vehicle dispatch until structural stabilization is confirmed",
      "Assign a drone for a secondary confirmation pass before rescue-team entry",
    ],
  },
  {
    match: /flood|water|drown|submerg/i,
    answer:
      "Prioritize victims in rising-water zones over static ones even at lower severity scores — water hazard is time-dilating risk, not fixed damage. Confirm route depth before routing ground vehicles; prefer boats or drones for delivery once depth exceeds ~0.3m equivalent.",
    sources: ["NDMA Flood Response Guidelines"],
    suggested_task_decomposition: [
      "Reroute ground vehicles around cells with forecast intensity above 0.6",
      "Escalate any victim in a flood cell to top-3 mission priority regardless of severity score",
    ],
  },
  {
    match: /suppl|medkit|resourc|alloc/i,
    answer:
      "Allocate medical supplies by triage category, not by request order: immediate (red) before delayed (yellow) before minor (green). Hold a reserve of at least 15% of medkits for newly detected high-severity victims rather than depleting on the current queue.",
    sources: ["START Triage Protocol"],
    suggested_task_decomposition: [
      "Reserve 3 medkits until victim detection rate drops below 1/min",
      "Reassign lowest-priority in-progress mission's supply allocation to any new red-tier victim",
    ],
  },
];

const DEFAULT_ANSWER = {
  answer:
    "No specific doctrine match found for that query yet — this is a placeholder response from the gateway's canned knowledge base. Once the agent-intelligence service (Saatwik) is live, this will be a real RAG lookup over emergency-response documents.",
  sources: [],
  suggested_task_decomposition: [],
};

export async function knowledgeQuery(req, res) {
  const { question } = req.body || {};
  if (!question || typeof question !== "string") {
    return res.status(400).json({ error: "expected { question: string }" });
  }

  const agentServiceUrl = process.env.AGENT_SERVICE_URL;
  if (agentServiceUrl) {
    try {
      const upstream = await fetch(`${agentServiceUrl}/knowledge_query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
      const data = await upstream.json();
      return res.json(data);
    } catch (err) {
      console.error("agent-intelligence service unreachable, falling back to canned response:", err.message);
    }
  }

  const hit = CANNED.find((c) => c.match.test(question));
  return res.json(hit ? { answer: hit.answer, sources: hit.sources, suggested_task_decomposition: hit.suggested_task_decomposition } : DEFAULT_ANSWER);
}

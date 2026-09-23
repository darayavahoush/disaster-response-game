from __future__ import annotations

import os
from typing import List

from agents_rl_llm.llm_rag.task_decomposition import decompose_task


class EmergencyKnowledgeBase:
    def __init__(self, corpus_path: str | None = None):
        self.docs = [
            {
                "title": "Flood rescue safety",
                "content": "Flood rescue prioritizes rapid assessment of rising water, exclusion of unstable routes, protection of rescuers, and rescue of the highest-risk victims first. Use life jackets, maintain clear evacuation routes, and avoid entering fast-moving water. Prefer drone reconnaissance and mark hazardous zones.",
            },
            {
                "title": "Cyclone response",
                "content": "Cyclone response requires early shelter evacuation, staging of supplies away from floodplains, and routing vehicles around debris and blocked roads. Prioritize medical triage, safe sheltering, and re-check hazard forecasts before dispatching teams.",
            },
            {
                "title": "Earthquake search and rescue",
                "content": "Following an earthquake, stabilize structures before entry, maintain a two-person minimum with a spotter, and verify trapped survivor positions before heavy equipment enters. Focus on critical victims in unstable buildings and blocked routes.",
            },
            {
                "title": "Medical supplies and triage",
                "content": "Distribute medical supplies using triage categories: immediate red cases first, then delayed yellow, then minor green. Maintain reserve stock for newly detected critical victims and avoid rapidly depleting resources on low-severity incidents.",
            },
            {
                "title": "Hazardous area protocol",
                "content": "Hazardous areas including unstable structures, flooded segments, and blocked routes must be surveyed before committing rescue teams. Use remote detection first, assign suitable agents, and create a clear return/evacuation route.",
            },
        ]
        self.corpus_path = corpus_path or os.getenv("KNOWLEDGE_CORPUS_PATH", os.path.join(os.getcwd(), "agents-rl-llm", "llm-rag", "corpus"))

    def retrieve(self, query: str, top_k: int = 3) -> List[dict]:
        q = query.lower()
        scored = []
        for doc in self.docs:
            score = 0
            for keyword in ["flood", "cyclone", "earthquake", "safety", "rescue", "victim", "route", "hazard", "supply", "collapse", "medical"]:
                if keyword in q and keyword in doc["content"].lower():
                    score += 2
            score += sum(1 for token in q.split() if token in doc["content"].lower())
            if score > 0:
                scored.append({"title": doc["title"], "content": doc["content"], "score": score})
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:top_k]

    def answer(self, question: str) -> dict:
        hits = self.retrieve(question)
        if not hits:
            hits = [{
                "title": "General emergency doctrine",
                "content": "Follow triage, check hazards, route teams around danger, and evacuate victims to safe locations with priority to high-risk cases.",
                "score": 1,
            }]
        answer = "\n\n".join(item["content"] for item in hits)
        return {
            "answer": answer,
            "sources": [item["title"] for item in hits],
            "suggested_task_decomposition": decompose_task(question),
        }

# Agent Intelligence — owner: Saatwik

MAPPO/MADDPG coordination policies + LLM/RAG emergency-knowledge and task decomposition.
See `../docs/TASKS.md` for the detailed breakdown.

## Layout
- `mappo-maddpg/` — multi-agent RL training loop, reward shaping, policy nets
- `llm-rag/` — retrieval corpus + RAG pipeline + task-decomposition prompting

"""
rag_agent.py
------------
Step 4 & 5: the agent loop.

This is what separates a "RAG agent" from a plain RAG pipeline:

Plain RAG: query -> always retrieve top-k -> stuff into prompt -> generate.
Fixed, one-shot, no judgment.

Agentic RAG: Claude is given a `search_knowledge_base` TOOL. It decides:
  - whether it even needs to search (a greeting doesn't need retrieval)
  - what to search for (it can rephrase the user's question into a better
    search query, e.g. expanding an acronym)
  - whether one search was enough, or it needs to search again with a
    different query to fill a gap (multi-hop reasoning)

We cap the number of search steps (MAX_AGENT_SEARCH_STEPS) to prevent
infinite loops and control cost.
"""

import json
import anthropic

from config import ANTHROPIC_API_KEY, GENERATION_MODEL, MAX_TOKENS, MAX_AGENT_SEARCH_STEPS
from vectorstore import VectorStore

SYSTEM_PROMPT = """You are an enterprise knowledge assistant. You answer questions
strictly using information retrieved from the company knowledge base via the
search_knowledge_base tool.

Rules:
- Always search before answering a factual question about the company/data.
- You may search more than once if the first results don't fully answer the question
  (e.g. search again with a rephrased or narrower query).
- If the knowledge base doesn't contain the answer, say so plainly. Never invent facts.
- When you answer, cite sources by filename, e.g. (source: policy.pdf).
- Preserve every condition, exception, or qualifier stated in the source verbatim in
  meaning — do not drop a conditional just because it seems like the common case.
  If a rule only applies "if X" or "unless Y", state that condition explicitly rather
  than presenting the rule as unconditional. If the user's question doesn't specify
  whether a condition holds (e.g. whether they have a dedicated account manager),
  say so and give the answer for both cases rather than assuming one.
- Be concise and direct.
"""

SEARCH_TOOL = {
    "name": "search_knowledge_base",
    "description": (
        "Search the enterprise knowledge base for relevant document chunks. "
        "Use specific, targeted queries — rephrase the user's question into "
        "keywords/concepts likely to appear in the source documents. "
        "Only search again if the previous results were genuinely insufficient "
        "(missing a specific fact you still need). A second query should target "
        "a DIFFERENT concept than the first, not reword the same one — e.g. "
        "if the first search covered 'contract terms', a follow-up should target "
        "something like 'exceptions' or 'account manager process', not a synonym "
        "of the same phrase."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query."}
        },
        "required": ["query"],
    },
}


class RAGAgent:
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        self.store = VectorStore()

    def _run_search(self, query: str) -> str:
        """Execute a search and format results as a string for Claude to read."""
        hits = self.store.search(query)
        if not hits:
            return "No relevant results found."

        formatted = []
        for h in hits:
            formatted.append(f"[source: {h['source']}]\n{h['text']}")
        return "\n\n---\n\n".join(formatted)

    def ask(self, question: str, verbose: bool = True) -> str:
        """
        Run the agent loop:
        1. Send the question + search tool to Claude.
        2. If Claude requests a search, run it and feed results back.
        3. Repeat until Claude gives a final text answer (or step cap hit).
        """
        messages = [{"role": "user", "content": question}]

        for step in range(MAX_AGENT_SEARCH_STEPS):
            response = self.client.messages.create(
                model=GENERATION_MODEL,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                tools=[SEARCH_TOOL],
                messages=messages,
            )

            # Did Claude ask to use the tool, or did it give a final answer?
            tool_calls = [b for b in response.content if b.type == "tool_use"]

            if not tool_calls:
                # Final answer — concatenate any text blocks.
                return "".join(b.text for b in response.content if b.type == "text")

            # Append assistant's tool-use turn to the conversation
            messages.append({"role": "assistant", "content": response.content})

            # Execute each requested search and return results as tool_result blocks
            tool_results = []
            for call in tool_calls:
                query = call.input["query"]
                if verbose:
                    print(f"  🔎 agent searches: \"{query}\"")
                result_text = self._run_search(query)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": call.id,
                        "content": result_text,
                    }
                )

            messages.append({"role": "user", "content": tool_results})

        return "I searched multiple times but couldn't find a confident answer. Please rephrase your question."
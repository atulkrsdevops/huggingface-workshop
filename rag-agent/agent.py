"""
Level 3 Agentic RAG pipeline for MLOps/DevOps Q&A.

Pipeline flow:
  1. Query Rewriting  — flan-t5 reformulates the user query for better retrieval
  2. Initial Retrieval — top-K from ChromaDB via LlamaIndex
  3. Relevance Filtering — self-reflection to drop irrelevant chunks
  4. Answer Generation — flan-t5 grounded on retrieved context
  5. Answer Reflection — self-reflection to judge completeness
  6. Iterative Retrieval — if answer is inadequate, retry with rewritten sub-query
  7. Fallback — if no relevant context exists, generate a grounded general answer
  8. Citation Assembly — return sources used with similarity scores
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from generator import FlanT5Generator
from rag_engine import MLOpsRAGEngine

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 2


@dataclass
class AgentResponse:
    """Structured response from the agentic RAG pipeline."""
    answer: str
    original_query: str
    rewritten_query: str
    citations: list[dict] = field(default_factory=list)
    reflection_notes: str = ""
    used_fallback: bool = False
    iterations: int = 1
    relevant_chunks_found: int = 0


class MLOpsRAGAgent:
    """
    Level 3 Agentic RAG agent with query rewriting, self-reflection,
    iterative retrieval, source citation, and fallback handling.
    """

    def __init__(self, rag_engine: MLOpsRAGEngine, generator: FlanT5Generator):
        self.rag = rag_engine
        self.gen = generator
        logger.info("MLOpsRAGAgent initialized")

    def run(self, user_query: str) -> AgentResponse:
        """Execute the full Level 3 agentic RAG pipeline."""
        logger.info(f"Agent received query: '{user_query[:80]}'")

        # ── Step 1: Query Rewriting ─────────────────────────────────────────
        rewritten_query = self._rewrite_query(user_query)
        logger.info(f"Rewritten query: '{rewritten_query[:80]}'")

        # ── Step 2 & 3: Retrieve + Filter relevant chunks ──────────────────
        nodes, relevant_nodes = self._retrieve_and_filter(rewritten_query)

        # ── Step 4-6: Iterative generation with reflection ─────────────────
        answer, reflection_notes, citations, iterations, used_fallback = \
            self._generate_with_reflection(user_query, rewritten_query, relevant_nodes)

        return AgentResponse(
            answer=answer,
            original_query=user_query,
            rewritten_query=rewritten_query,
            citations=citations,
            reflection_notes=reflection_notes,
            used_fallback=used_fallback,
            iterations=iterations,
            relevant_chunks_found=len(relevant_nodes),
        )

    # ── Private helpers ─────────────────────────────────────────────────────

    def _rewrite_query(self, query: str) -> str:
        """Step 1: Rewrite the user query for optimal retrieval."""
        try:
            rewritten = self.gen.rewrite_query(query)
            # If the rewrite is substantially different and non-empty, use it
            if rewritten and rewritten.lower() != query.lower() and len(rewritten) >= 10:
                return rewritten
        except Exception as e:
            logger.warning(f"Query rewriting failed: {e}")
        return query

    def _retrieve_and_filter(self, query: str) -> tuple:
        """Steps 2 & 3: Retrieve chunks and filter by relevance."""
        try:
            nodes = self.rag.retrieve(query, top_k=6)
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return [], []

        scores = [self.rag.get_node_score(n) for n in nodes]
        logger.info(f"Raw node distances: {[round(s, 3) for s in scores]}")

        # Always keep the top 3 nodes (retriever returns them sorted by distance
        # ascending, so these are the closest matches).  Run the LLM relevance
        # check only on nodes 4-6 to optionally widen the context.
        relevant = list(nodes[:3])
        for node in nodes[3:]:
            text = self.rag.get_node_text(node)
            try:
                if self.gen.check_relevance(query, text):
                    relevant.append(node)
            except Exception:
                relevant.append(node)

        logger.info(f"Nodes after relevance filtering: {len(relevant)}/{len(nodes)}")
        return nodes, relevant

    def _generate_with_reflection(
        self,
        original_query: str,
        rewritten_query: str,
        relevant_nodes: list,
    ) -> tuple[str, str, list, int, bool]:
        """Steps 4-6: Generate answer with self-reflection and iterative retrieval."""

        iterations = 1
        used_fallback = False
        reflection_notes = ""

        # ── Fallback: no relevant context found ────────────────────────────
        if not relevant_nodes:
            logger.info("No relevant context found — using fallback generation")
            answer = self.gen.generate_fallback(original_query)
            reflection_notes = (
                "No relevant documents were found in the knowledge base. "
                "Answer generated from model's general knowledge."
            )
            return answer, reflection_notes, [], 1, True

        # ── First generation attempt ────────────────────────────────────────
        context, citations = self.rag.format_context(relevant_nodes[:4])
        answer = self.gen.generate_answer(original_query, context)

        # ── Step 5: Self-reflection on answer quality ───────────────────────
        is_adequate, reflection = self.gen.reflect_on_answer(original_query, answer)
        reflection_notes = reflection
        logger.info(f"Self-reflection (iter 1): adequate={is_adequate}")

        # ── Step 6: Iterative retrieval if answer is inadequate ─────────────
        if not is_adequate and iterations < MAX_ITERATIONS:
            iterations += 1
            logger.info("Answer inadequate — performing additional retrieval with sub-query")

            # Generate a follow-up query targeting the gap
            follow_up_query = f"{original_query} detailed explanation technical steps"
            _, extra_nodes = self._retrieve_and_filter(follow_up_query)

            if extra_nodes:
                # Combine with original context, deduplicate by text
                seen_texts = {self.rag.get_node_text(n) for n in relevant_nodes}
                new_nodes = [n for n in extra_nodes if self.rag.get_node_text(n) not in seen_texts]
                combined_nodes = relevant_nodes + new_nodes

                context, citations = self.rag.format_context(combined_nodes[:5])
                answer = self.gen.generate_answer(original_query, context)

                # Final reflection
                is_adequate, reflection = self.gen.reflect_on_answer(original_query, answer)
                reflection_notes = f"[After additional retrieval] {reflection}"
                logger.info(f"Self-reflection (iter 2): adequate={is_adequate}")
            else:
                reflection_notes = f"[No additional context found] {reflection}"

        # ── Fallback if still no good answer ───────────────────────────────
        if not answer or len(answer.strip()) < 10:
            answer = self.gen.generate_fallback(original_query)
            used_fallback = True
            reflection_notes += " | Switched to fallback generation due to empty output."

        return answer, reflection_notes, citations, iterations, used_fallback

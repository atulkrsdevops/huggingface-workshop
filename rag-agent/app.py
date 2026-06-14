import logging
import sys

import gradio as gr

logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Lazily initialised — populated on first query so the UI starts fast
_agent = None
_init_error = None


def _init_agent():
    global _agent, _init_error
    if _agent is not None:
        return True
    if _init_error is not None:
        return False
    try:
        from rag_engine import MLOpsRAGEngine
        from generator import FlanT5Generator
        from agent import MLOpsRAGAgent

        logger.info("Initialising RAG engine …")
        rag = MLOpsRAGEngine()
        rag.build_index()

        logger.info("Loading Flan-T5 generator …")
        gen = FlanT5Generator()

        _agent = MLOpsRAGAgent(rag_engine=rag, generator=gen)
        logger.info("Agent ready")
        return True
    except Exception as exc:
        _init_error = str(exc)
        logger.error(f"Agent init failed: {exc}")
        return False


def answer_query(question):
    if not question or not question.strip():
        return "Please enter a question.", "", "", ""

    if not _init_agent():
        return f"Initialisation error: {_init_error}", "", "", ""

    try:
        resp = _agent.run(question.strip())

        if resp.citations:
            citations_text = "\n\n".join(
                f"[{c['index']}] {c['source']}  (similarity: {c['score']})\n{c['snippet']}"
                for c in resp.citations
            )
        else:
            citations_text = "No citations — fallback answer used."

        query_info = (
            f"Original:  {resp.original_query}\n"
            f"Rewritten: {resp.rewritten_query}\n"
            f"Chunks found: {resp.relevant_chunks_found} | "
            f"Iterations: {resp.iterations} | "
            f"Fallback: {resp.used_fallback}"
        )

        return resp.answer, query_info, resp.reflection_notes, citations_text

    except Exception as exc:
        logger.error(f"Query error: {exc}")
        return f"Error: {exc}", "", "", ""


# ── UI ──────────────────────────────────────────────────────────────────────
with gr.Blocks(title="MLOps RAG Agent") as demo:
    gr.Markdown(
        "# 🤖 MLOps / DevOps Agentic RAG Agent\n"
        "**Level 3 Agentic RAG** — query rewriting → retrieval → "
        "relevance filtering → generation → self-reflection → citation"
    )

    question = gr.Textbox(
        label="Your MLOps/DevOps Question",
        placeholder=(
            "e.g. How do I configure auto-scaling for a SageMaker endpoint?\n"
            "     What is the difference between data drift and concept drift?\n"
            "     How do I deploy a model with canary rollout on Kubernetes?"
        ),
        lines=3,
    )
    submit_btn = gr.Button("Ask", variant="primary")
    answer = gr.Textbox(label="Answer", lines=8, interactive=False)

    with gr.Accordion("Query rewriting & pipeline info", open=False):
        query_info = gr.Textbox(label="", lines=4, interactive=False)

    with gr.Accordion("Self-reflection notes", open=False):
        reflection = gr.Textbox(label="", lines=3, interactive=False)

    with gr.Accordion("Source citations", open=False):
        citations = gr.Textbox(label="", lines=10, interactive=False)

    outputs = [answer, query_info, reflection, citations]
    submit_btn.click(fn=answer_query, inputs=[question], outputs=outputs)
    question.submit(fn=answer_query, inputs=[question], outputs=outputs)

demo.launch(server_name="0.0.0.0", server_port=7860)

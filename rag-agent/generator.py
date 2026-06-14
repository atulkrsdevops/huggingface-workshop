"""Flan-T5-base generator for query rewriting, answer generation, and self-reflection."""

import os
import logging
from typing import Optional
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, pipeline

logger = logging.getLogger(__name__)

MAX_INPUT_TOKENS = 480
MAX_NEW_TOKENS_ANSWER = 256
MAX_NEW_TOKENS_SHORT = 64


class FlanT5Generator:
    """Wrapper around flan-t5-base for all generation tasks in the agentic RAG pipeline."""

    def __init__(self, model_name: str = "google/flan-t5-base"):
        logger.info(f"Loading {model_name}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        self.model.eval()
        logger.info(f"Model loaded on {self.device}")

    def _generate(self, prompt: str, max_new_tokens: int = MAX_NEW_TOKENS_ANSWER) -> str:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=MAX_INPUT_TOKENS,
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    def rewrite_query(self, original_query: str) -> str:
        """Rewrite a user query to be more specific for MLOps knowledge retrieval."""
        prompt = (
            f"Rewrite this MLOps question to be more specific and technical for searching "
            f"a knowledge base about SageMaker, Kubernetes, CI/CD, Terraform, and model monitoring. "
            f"Keep it as one clear question.\n"
            f"Original question: {original_query}\n"
            f"Rewritten question:"
        )
        result = self._generate(prompt, max_new_tokens=MAX_NEW_TOKENS_SHORT)
        # Fall back to original if generation is empty or too short
        if not result or len(result) < 5:
            return original_query
        return result

    def generate_answer(self, query: str, context: str) -> str:
        """Generate an answer grounded in the retrieved context."""
        # Truncate context to fit within token budget
        context_tokens = self.tokenizer(context, truncation=False)["input_ids"]
        query_tokens = self.tokenizer(query, truncation=False)["input_ids"]

        # Reserve tokens for the prompt template and query
        max_context_tokens = MAX_INPUT_TOKENS - len(query_tokens) - 40
        if len(context_tokens) > max_context_tokens:
            context_tokens = context_tokens[:max_context_tokens]
            context = self.tokenizer.decode(context_tokens, skip_special_tokens=True)

        prompt = (
            f"Answer this MLOps/DevOps question using only the provided context. "
            f"Be specific and technical.\n"
            f"Context: {context}\n"
            f"Question: {query}\n"
            f"Answer:"
        )
        result = self._generate(prompt, max_new_tokens=MAX_NEW_TOKENS_ANSWER)
        if not result or len(result) < 5:
            return "I could not generate a specific answer based on the retrieved context."
        return result

    def check_relevance(self, query: str, context_snippet: str) -> bool:
        """Check if a retrieved chunk is relevant to the query."""
        snippet = context_snippet[:300]
        prompt = (
            f"Is this text relevant to the question? Answer only 'yes' or 'no'.\n"
            f"Question: {query}\n"
            f"Text: {snippet}\n"
            f"Answer:"
        )
        result = self._generate(prompt, max_new_tokens=5).lower()
        # Lenient: include the chunk unless the model explicitly says "no".
        # "yes" in result is too strict — Flan-T5 sometimes outputs synonyms or
        # fuller sentences; requiring an explicit "no" avoids false negatives.
        return "no" not in result

    def reflect_on_answer(self, query: str, answer: str) -> tuple[bool, str]:
        """Self-reflect on whether the generated answer adequately addresses the query."""
        prompt = (
            f"Does this answer fully address the question? Answer 'yes' or 'no' and give a one-sentence reason.\n"
            f"Question: {query}\n"
            f"Answer: {answer}\n"
            f"Evaluation:"
        )
        reflection = self._generate(prompt, max_new_tokens=MAX_NEW_TOKENS_SHORT)
        is_adequate = "yes" in reflection.lower()
        return is_adequate, reflection

    def generate_fallback(self, query: str) -> str:
        """Generate a general response when no relevant context is found."""
        prompt = (
            f"Provide a brief general explanation about this MLOps/DevOps topic. "
            f"Be helpful and suggest where to find more information.\n"
            f"Question: {query}\n"
            f"Answer:"
        )
        return self._generate(prompt, max_new_tokens=MAX_NEW_TOKENS_ANSWER)

"""LLM wrapper (OpenRouter, OpenAI-compatible) and the retrieve -> prompt -> generate loop."""
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

BASE_URL = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
MODEL = os.getenv("LLM_MODEL", "openai/gpt-4o-mini")

PROMPT_TEMPLATE = """You are a support assistant for Nimbus devices.
Answer the question using only the context below. Be concise (1-3 sentences).
If the context does not contain the answer, say "I don't know based on the provided documents."

Context:
{context}

Question: {question}
Answer:"""

_client = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        key = os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("Set OPENROUTER_API_KEY in .env (see .env.example)")
        _client = OpenAI(base_url=BASE_URL, api_key=key)
    return _client


def generate(prompt: str, temperature: float = 0.0, n: int = 1) -> list[str]:
    """Return n completions. Loops instead of using `n=` since not every OpenRouter provider supports it."""
    outs = []
    for _ in range(n):
        resp = get_client().chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=200,
        )
        outs.append(resp.choices[0].message.content.strip())
    return outs


def build_prompt(question: str, contexts: list[str]) -> str:
    context = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
    return PROMPT_TEMPLATE.format(context=context, question=question)


def rag_answer(question: str, retriever, n_samples: int = 0, sample_temperature: float = 1.0) -> dict:
    """Greedy answer plus optional stochastic samples (used by SelfCheckGPT)."""
    contexts = [h["text"] for h in retriever.retrieve(question)]
    prompt = build_prompt(question, contexts)
    result = {"question": question, "contexts": contexts, "answer": generate(prompt, temperature=0.0)[0]}
    if n_samples:
        result["samples"] = generate(prompt, temperature=sample_temperature, n=n_samples)
    return result


if __name__ == "__main__":
    import sys
    from src.retriever import Retriever
    q = " ".join(sys.argv[1:]) or "How long is the warranty on a Nimbus phone?"
    print(rag_answer(q, Retriever())["answer"])

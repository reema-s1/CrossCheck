"""Ragas evaluation: faithfulness, answer relevancy, context precision, context recall.

The judge LLM is the same OpenRouter model as the generator (see src/generator.py);
embeddings for answer relevancy run locally with all-MiniLM-L6-v2.
"""
import os
import warnings

os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")  # ragas sends usage analytics by default

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from ragas import EvaluationDataset, RunConfig, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

from src.generator import BASE_URL, MODEL, get_client
from src.indexing import EMBED_MODEL

warnings.filterwarnings("ignore", category=DeprecationWarning)

METRICS = [faithfulness, answer_relevancy, context_precision, context_recall]


def run_ragas(rows: list[dict]):
    """rows: dicts with question, contexts, answer, reference. Returns a per-question DataFrame."""
    get_client()  # fail fast with a clear message if the API key is missing
    judge = ChatOpenAI(model=MODEL, base_url=BASE_URL, api_key=os.environ["OPENROUTER_API_KEY"], temperature=0)
    dataset = EvaluationDataset.from_list([
        {
            "user_input": r["question"],
            "retrieved_contexts": r["contexts"],
            "response": r["answer"],
            "reference": r["reference"],
        }
        for r in rows
    ])
    result = evaluate(
        dataset,
        metrics=METRICS,
        # OpenRouter ignores n>1, so have ragas issue separate calls (answer relevancy asks for 3).
        llm=LangchainLLMWrapper(judge, bypass_n=True),
        embeddings=LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=EMBED_MODEL)),
        run_config=RunConfig(max_workers=4, timeout=120),
    )
    return result.to_pandas()

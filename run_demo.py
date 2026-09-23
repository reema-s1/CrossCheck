"""End-to-end: build index -> answer eval set (with samples) -> SelfCheckGPT -> Ragas -> comparison report."""
import argparse
import json
from pathlib import Path

import src  # noqa: F401  (sets macOS OpenMP env vars before torch/faiss load)
from src.compare import RESULTS_DIR, merge, write_report
from src.generator import MODEL, rag_answer
from src.indexing import build_index
from src.ragas_eval import run_ragas
from src.retriever import Retriever
from src.selfcheck import SelfCheckBERTScore

EVAL_FILE = Path(__file__).parent / "eval" / "eval_questions.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=5, help="stochastic samples per question for SelfCheckGPT")
    parser.add_argument("--limit", type=int, default=None, help="only run the first N questions")
    args = parser.parse_args()

    print("[1/5] Building index")
    build_index()
    retriever = Retriever(k=3)

    questions = json.loads(EVAL_FILE.read_text())[: args.limit]
    print(f"[2/5] Answering {len(questions)} questions with {MODEL} (+{args.samples} samples each)")
    rows = []
    for q in questions:
        out = rag_answer(q["question"], retriever, n_samples=args.samples)
        rows.append({**q, **out})
        print(f"  {q['id']}: {out['answer'][:100]}")

    print("[3/5] SelfCheckGPT (BERTScore) consistency scoring")
    checker = SelfCheckBERTScore()
    for r in rows:
        sc = checker.score(r["answer"], r["samples"])
        r["selfcheck_score"] = sc["selfcheck_score"]
        r["selfcheck_sentences"] = list(zip(sc["sentences"], sc["sentence_scores"]))

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "generations.json").write_text(json.dumps(rows, indent=1))

    print("[4/5] Ragas evaluation")
    ragas_df = run_ragas(rows)

    print("[5/5] Comparison report")
    report = write_report(merge(rows, ragas_df))
    print(report.read_text().split("## Per question")[0])
    print(f"Full report: {report}")


if __name__ == "__main__":
    main()

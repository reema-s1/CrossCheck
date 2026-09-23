"""Hand-rolled SelfCheckGPT, BERTScore variant.

Reproduces Manakul et al., 2023 ("SelfCheckGPT: Zero-Resource Black-Box Hallucination
Detection for Generative Large Language Models"), Section 5.1 / Eq. 1:

    S_BERT(i) = 1 - (1/N) * sum_n  max_k  B(r_i, s_k^n)

r_i is the i-th sentence of the main (greedy) answer, s_k^n the k-th sentence of the
n-th stochastic sample, and B is BERTScore F1. If a sentence is supported by the
samples, some sample sentence will match it closely and the score is near 0; a
hallucinated sentence tends to disagree with the samples and scores higher.

BERTScore (Zhang et al., 2020) is also implemented here rather than imported:
contextual token embeddings from one hidden layer, greedy cosine matching, F1, then
linear rescaling against a baseline of unrelated sentence pairs.

Reference used for validation of the formulation (not copied):
https://github.com/potsawee/selfcheckgpt/blob/main/selfcheckgpt/modeling_selfcheck.py
The official repo uses roberta-large; we default to roberta-base to keep CPU/RAM
use low on a laptop (see "Design decisions" in README.md).
"""
import random
from functools import lru_cache

import spacy
import torch
from transformers import AutoModel, AutoTokenizer

BERT_MODEL = "roberta-base"
BERT_LAYER = 10  # bert_score's recommended layer for roberta-base


class SelfCheckBERTScore:
    def __init__(self, model_name: str = BERT_MODEL, layer: int = BERT_LAYER):
        self.nlp = spacy.load("en_core_web_sm")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).eval()
        self.layer = layer
        self.baseline = 0.0
        self.baseline = self._estimate_baseline()

    def sentences(self, text: str) -> list[str]:
        return [s.text.strip() for s in self.nlp(text).sents if len(s.text.strip()) > 3]

    @lru_cache(maxsize=4096)
    def _token_embeddings(self, sentence: str) -> torch.Tensor:
        enc = self.tokenizer(sentence, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            hidden = self.model(**enc, output_hidden_states=True).hidden_states[self.layer][0]
        hidden = hidden[1:-1]  # drop <s> and </s>
        return torch.nn.functional.normalize(hidden, dim=-1)

    def bertscore_f1(self, a: str, b: str) -> float:
        ea, eb = self._token_embeddings(a), self._token_embeddings(b)
        sim = ea @ eb.T                       # cosine similarity of every token pair
        recall = sim.max(dim=1).values.mean()     # each token of a matched greedily in b
        precision = sim.max(dim=0).values.mean()  # each token of b matched greedily in a
        f1 = (2 * precision * recall / (precision + recall)).item()
        return (f1 - self.baseline) / (1 - self.baseline)

    def _estimate_baseline(self, n_pairs: int = 200) -> float:
        """Mean raw F1 between unrelated sentences, so rescaled scores spread over ~[0, 1]."""
        from src.indexing import load_chunks
        pool = [s for c in load_chunks() for s in self.sentences(c["text"])]
        rng = random.Random(0)
        pairs = [rng.sample(pool, 2) for _ in range(n_pairs)]
        return sum(self.bertscore_f1(a, b) for a, b in pairs) / n_pairs

    def score(self, answer: str, samples: list[str]) -> dict:
        """Per-sentence and passage-level hallucination scores (higher = more likely hallucinated)."""
        answer_sents = self.sentences(answer)
        sample_sents = [self.sentences(s) or [s] for s in samples]
        per_sentence = []
        for r in answer_sents:
            best_matches = [max(self.bertscore_f1(r, s) for s in sents) for sents in sample_sents]
            support = sum(best_matches) / len(best_matches)
            per_sentence.append(min(max(1 - support, 0.0), 1.0))
        passage = sum(per_sentence) / len(per_sentence) if per_sentence else 0.0
        return {"sentences": answer_sents, "sentence_scores": per_sentence, "selfcheck_score": passage}


if __name__ == "__main__":
    sc = SelfCheckBERTScore()
    print(f"baseline F1 = {sc.baseline:.3f}")
    answer = "The Nimbus Watch S is water resistant to 50 meters. It can measure blood pressure."
    consistent = [
        "The Watch S is water resistant up to 50 meters, but it does not measure blood pressure.",
        "It is water resistant to 50 meters. The watch cannot measure blood pressure.",
        "The Nimbus Watch S has 50 m water resistance. Blood pressure is not supported.",
    ]
    divergent = [
        "The watch is rated IP68 for depths of 6 meters. It tracks your sleep.",
        "The Nimbus Watch S survives splashes only. It measures ECG in the US.",
        "It is waterproof to 100 meters and supports blood glucose readings.",
    ]
    for name, samples in [("consistent samples", consistent), ("divergent samples", divergent)]:
        out = sc.score(answer, samples)
        print(name, [f"{s:.2f}" for s in out["sentence_scores"]], f"passage={out['selfcheck_score']:.2f}")

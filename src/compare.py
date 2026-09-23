"""Merge SelfCheckGPT and Ragas scores per question and highlight disagreements."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

SELFCHECK_THRESHOLD = 0.4   # above -> SelfCheckGPT flags the answer as likely hallucinated
FAITHFULNESS_THRESHOLD = 0.5  # below -> Ragas flags the answer as unfaithful to the context

VERDICTS = {
    (False, False): "agree: grounded",
    (True, True): "agree: hallucination",
    (True, False): "DISAGREE: SelfCheck only",
    (False, True): "DISAGREE: Ragas only",
}

INK, INK_MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
TYPE_STYLE = {"normal": ("#2a78d6", "o"), "trap": ("#eb6834", "^")}


def merge(rows: list[dict], ragas_df: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame([{
        "id": r["id"], "type": r["type"], "question": r["question"], "answer": r["answer"],
        "selfcheck_score": r["selfcheck_score"],
    } for r in rows])
    for col in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        df[col] = ragas_df[col].values
    df["selfcheck_flag"] = df["selfcheck_score"] > SELFCHECK_THRESHOLD
    # A refusal has no claims to verify, so Ragas may return NaN; that is not an unfaithful answer.
    df["ragas_flag"] = df["faithfulness"].fillna(1.0) < FAITHFULNESS_THRESHOLD
    df["verdict"] = [VERDICTS[(s, r)] for s, r in zip(df["selfcheck_flag"], df["ragas_flag"])]
    return df


def plot(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.axvline(FAITHFULNESS_THRESHOLD, color=INK_MUTED, lw=1, ls="--", zorder=1)
    ax.axhline(SELFCHECK_THRESHOLD, color=INK_MUTED, lw=1, ls="--", zorder=1)
    for qtype, (color, marker) in TYPE_STYLE.items():
        sub = df[df["type"] == qtype]
        ax.scatter(sub["faithfulness"].fillna(1.0), sub["selfcheck_score"], s=70, c=color, marker=marker,
                   edgecolors=SURFACE, linewidths=2, label=f"{qtype} question", zorder=3)
    for _, r in df[df["verdict"].str.startswith("DISAGREE")].iterrows():
        ax.annotate(r["id"], (r["faithfulness"] if pd.notna(r["faithfulness"]) else 1.0, r["selfcheck_score"]),
                    xytext=(6, 6), textcoords="offset points", fontsize=9, color=INK)
    quadrant = dict(fontsize=8, color=INK_MUTED, transform=ax.transAxes)
    ax.text(0.99, 0.98, "SelfCheck flags only", ha="right", va="top", **quadrant)
    ax.text(0.01, 0.02, "Ragas flags only", ha="left", va="bottom", **quadrant)
    ax.text(0.01, 0.98, "both flag", ha="left", va="top", **quadrant)
    ax.text(0.99, 0.02, "both pass", ha="right", va="bottom", **quadrant)
    ax.set(xlim=(-0.05, 1.05), ylim=(0, max(0.8, df["selfcheck_score"].max() + 0.05)))
    ax.set_xlabel("Ragas faithfulness (higher = grounded in context)", color=INK)
    ax.set_ylabel("SelfCheckGPT score (higher = likely hallucinated)", color=INK)
    ax.set_title("SelfCheckGPT vs Ragas faithfulness, per question", color=INK, loc="left")
    ax.grid(color=GRID, lw=0.5)
    ax.tick_params(colors=INK_MUTED)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(1, 1), ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_report(df: pd.DataFrame) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    df.to_csv(RESULTS_DIR / "comparison.csv", index=False)
    plot(df, RESULTS_DIR / "comparison.png")

    cols = ["id", "type", "selfcheck_score", "faithfulness", "answer_relevancy",
            "context_precision", "context_recall", "verdict"]
    lines = [
        "# SelfCheckGPT vs Ragas: comparison report", "",
        f"SelfCheck flags when score > {SELFCHECK_THRESHOLD}; Ragas flags when faithfulness < {FAITHFULNESS_THRESHOLD}.", "",
        "![comparison](comparison.png)", "",
        "## Summary", "",
        df["verdict"].value_counts().rename_axis("verdict").reset_index().to_markdown(index=False), "",
        "## Per question", "",
        df[cols].round(2).to_markdown(index=False), "",
        "## Disagreements", "",
    ]
    disagreements = df[df["verdict"].str.startswith("DISAGREE")]
    if disagreements.empty:
        lines.append("None at the current thresholds.")
    for _, r in disagreements.iterrows():
        lines += [f"### {r['id']} ({r['type']}): {r['verdict']}", "",
                  f"**Q:** {r['question']}", "", f"**A:** {r['answer']}", "",
                  f"SelfCheck = {r['selfcheck_score']:.2f}, faithfulness = {r['faithfulness']:.2f}", ""]
    out = RESULTS_DIR / "comparison.md"
    out.write_text("\n".join(lines))
    return out

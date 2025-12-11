import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

def tune_threshold(scores_path: str):
    df = pd.read_csv(scores_path)

    if "label" not in df.columns or "llm_score" not in df.columns:
        raise ValueError(f"{scores_path} must contain 'llm_score' and 'label' columns.")

    y_true = df["label"].astype(int).values
    scores = df["llm_score"].values

    taus = np.linspace(scores.min(), scores.max(), 200)
    best = None

    for tau in taus:
        preds = (scores > tau).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, preds, average="binary", zero_division=0
        )

        if best is None or f1 > best["f1"]:
            best = dict(tau=tau, precision=precision, recall=recall, f1=f1)

    print("\nBest threshold search complete:")
    print(f"  τ = {best['tau']:.4f}")
    print(f"  Precision = {best['precision']:.4f}")
    print(f"  Recall    = {best['recall']:.4f}")
    print(f"  F1        = {best['f1']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Tune decision threshold for LLM classifier.")
    parser.add_argument(
        "--scores_path",
        type=str,
        required=True,
        help="Path to CSV file containing llm_score and label columns."
    )
    args = parser.parse_args()

    tune_threshold(args.scores_path)


if __name__ == "__main__":
    main()

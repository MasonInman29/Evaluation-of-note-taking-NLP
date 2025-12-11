import argparse

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)


def evaluate(pred_path: str):
    df = pd.read_csv(pred_path)
    print(f"Loaded {len(df)} examples from {pred_path}")

    if "label" not in df.columns:
        raise ValueError(
            f"'label' column not found in {pred_path}. "
            "Make sure you ran the classifier on a split that has labels "
            "(e.g., train or a labeled test set)."
        )

    if "pred_llm" not in df.columns:
        raise ValueError(
            f"'pred_llm' column not found in {pred_path}. "
            "Make sure you ran llm_note_classifier.py to produce predictions."
        )

    # Drop rows with missing labels, if any
    mask = df["label"].notna()
    if not mask.all():
        dropped = (~mask).sum()
        print(f"Warning: dropping {dropped} rows with missing labels before evaluation.")
    df_eval = df[mask].copy()

    y_true = df_eval["label"].astype(int)
    y_pred = df_eval["pred_llm"].astype(int)

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}\n")

    print("Confusion matrix [ [TN, FP], [FN, TP] ]:")
    print(cm)
    print("\nFull classification report:")
    print(classification_report(y_true, y_pred, zero_division=0))


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM predictions.")
    parser.add_argument(
        "--pred_path",
        type=str,
        required=True,
        help="Path to CSV file with columns: pred_llm and label.",
    )
    args = parser.parse_args()
    evaluate(args.pred_path)


if __name__ == "__main__":
    main()
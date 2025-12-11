import argparse
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)


def evaluate(pred_path: str, threshold: float):
    df = pd.read_csv(pred_path)
    print(f"Loaded {len(df)} examples from {pred_path}")

    if "label" not in df.columns:
        raise ValueError("Input file must include a 'label' column for evaluation.")

    if "llm_score" not in df.columns:
        raise ValueError("Input file must contain 'llm_score' produced by the classifier.")

    # Drop missing labels if any
    df = df[df["label"].notna()].copy()

    y_true = df["label"].astype(int).values
    scores = df["llm_score"].values

    # Apply threshold
    y_pred = (scores > threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)

    print(f"\nUsing threshold τ = {threshold:.4f}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}\n")

    print("Confusion matrix [ [TN, FP], [FN, TP] ]:")
    print(cm)
    print("\nFull classification report:")
    print(classification_report(y_true, y_pred, zero_division=0))


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM classifier with thresholding.")
    parser.add_argument("--pred_path", type=str, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    args = parser.parse_args()
    evaluate(args.pred_path, args.threshold)


if __name__ == "__main__":
    main()

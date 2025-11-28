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

    if "label" not in df.columns:
        raise ValueError("No 'label' column found in the prediction file.")

    y_true = df["label"].astype(int)
    y_pred = df["pred_llm"].astype(int)

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1
    )
    cm = confusion_matrix(y_true, y_pred)

    print(f"Loaded {len(df)} examples from {pred_path}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("\nConfusion matrix [ [TN, FP], [FN, TP] ]:")
    print(cm)
    print("\nFull classification report:")
    print(classification_report(y_true, y_pred, digits=4))


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM predictions.")
    parser.add_argument(
        "--pred_path",
        type=str,
        required=True,
        help="Path to CSV with columns: label, pred_llm",
    )
    args = parser.parse_args()
    evaluate(args.pred_path)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BERT pair classifier for IdeaUnit ↔ NoteText in a 1-shot-per-(Experiment,Topic) setting.

Expected input layout:
  data/
    note_classification/
      Notes.csv        (Experiment, Topic, ID, Segment1_Notes..Segment4_Notes)
      train.csv        (Topic, ID, Segment, IdeaUnit, label)   # may lack Experiment
      test.csv         (Experiment, Topic, ID, Segment, IdeaUnit, [label])  # label may be NaN

Run (CPU example):
  python bert_note_classifier.py \
    --data_dir data/note_classification \
    --output_dir results \
    --model_name bert-base-uncased \
    --max_length 256 \
    --epochs 4 \
    --lr 2e-5 \
    --train_batch_size 8 \
    --eval_batch_size 8 \
    --seed 42

Outputs written to --output_dir:
  - bert_eval_metrics_overall.json
  - bert_eval_metrics_by_topic.csv
  - bert_eval_predictions.csv
  - bert_test_predictions.csv
  - checkpoints/  (HF Trainer artifacts, best by F1)
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Tuple

import torch
from torch.nn.functional import cross_entropy
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)

from datasets import Dataset
from transformers import (
    set_seed,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EvalPrediction,
)

# --------------------------
# Utilities
# --------------------------
SEGMENT_MAP = {
    1: "Segment1_Notes",
    2: "Segment2_Notes",
    3: "Segment3_Notes",
    4: "Segment4_Notes",
}

def read_csv_smart(path: str) -> pd.DataFrame:
    """Be tolerant of BOMs and cp1252 smart quotes."""
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, encoding="utf-8", encoding_errors="replace")

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).encode("utf-8", "ignore").decode("utf-8").strip() for c in df.columns]
    return df

def coerce_notes_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure Notes.csv has clean keys and usable note fields."""
    df = normalize_columns(df)
    # Loosely coerce types we key on
    if "Experiment" in df.columns:
        df["Experiment"] = df["Experiment"].astype(str).str.strip()
    if "Topic" in df.columns:
        df["Topic"] = df["Topic"].astype(str).str.strip()
    if "ID" in df.columns:
        df["ID"] = pd.to_numeric(df["ID"], errors="coerce")
    # Drop rows missing key
    need = [c for c in ["Topic", "ID"] if c in df.columns]
    if need:
        df = df.dropna(subset=need).copy()
    if "ID" in df.columns:
        df["ID"] = df["ID"].astype(int)
    # Leave segment note columns as-is (strings/NaN)
    return df

def coerce_train_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce train.csv to clean ints/strings and drop bad rows."""
    df = normalize_columns(df)
    # Required columns exist?
    req = {"Topic", "ID", "Segment", "IdeaUnit", "label"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"train.csv missing columns: {missing}")

    df["Topic"] = df["Topic"].astype(str).str.strip()
    df["ID"] = pd.to_numeric(df["ID"], errors="coerce")
    df["Segment"] = pd.to_numeric(df["Segment"], errors="coerce")
    df.loc[~df["Segment"].isin([1, 2, 3, 4]), "Segment"] = np.nan
    df["IdeaUnit"] = df["IdeaUnit"].astype(str)
    df["label"] = pd.to_numeric(df["label"], errors="coerce")  # accept '0','1','1.0', etc.

    n0 = len(df)
    df = df.dropna(subset=["Topic", "ID", "Segment", "IdeaUnit", "label"]).copy()
    if len(df) < n0:
        print(f"[info] Dropped {n0 - len(df)} train rows failing dtype coercion or missing required fields.")
    df["ID"] = df["ID"].astype(int)
    df["Segment"] = df["Segment"].astype(int)
    df["label"] = df["label"].astype(int)  # now safe
    return df

def coerce_test_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce test.csv; keep NaN labels if present."""
    df = normalize_columns(df)
    # Required keys for note fetch
    req = {"Topic", "ID", "Segment", "IdeaUnit"}
    miss = req - set(df.columns)
    if miss:
        raise ValueError(f"test.csv missing required columns: {miss}")

    # Optional label present
    has_label = "label" in df.columns

    # Coerce
    df["Topic"] = df["Topic"].astype(str).str.strip()
    df["ID"] = pd.to_numeric(df["ID"], errors="coerce")
    df["Segment"] = pd.to_numeric(df["Segment"], errors="coerce")
    df.loc[~df["Segment"].isin([1, 2, 3, 4]), "Segment"] = np.nan
    df["IdeaUnit"] = df["IdeaUnit"].astype(str)
    if has_label:
        df["label"] = pd.to_numeric(df["label"], errors="coerce")  # may remain NaN

    n0 = len(df)
    df = df.dropna(subset=["Topic", "ID", "Segment", "IdeaUnit"]).copy()
    if len(df) < n0:
        print(f"[info] Dropped {n0 - len(df)} test rows missing keys (Topic/ID/Segment/IdeaUnit).")
    df["ID"] = df["ID"].astype(int)
    df["Segment"] = df["Segment"].astype(int)
    return df

def attach_experiment_to_train(train_df: pd.DataFrame, notes_df: pd.DataFrame) -> pd.DataFrame:
    """Inject Experiment into train by joining Notes on (Topic, ID)."""
    if not {"Topic", "ID", "Experiment"} <= set(notes_df.columns):
        # If Notes lacks Experiment for some reason, fallback to UNK
        out = train_df.copy()
        out["Experiment"] = "UNK"
        return out
    key_cols = ["Topic", "ID", "Experiment"]
    map_df = notes_df[key_cols].drop_duplicates(subset=["Topic", "ID"])
    merged = train_df.merge(map_df, on=["Topic", "ID"], how="left", validate="many_to_one")
    merged["Experiment"] = merged["Experiment"].fillna("UNK")
    return merged

def pick_group_cols(df: pd.DataFrame):
    """Prefer 1-shot per (Experiment, Topic); fallback to per-Topic."""
    has_exp = "Experiment" in df.columns
    has_topic = "Topic" in df.columns
    if has_exp and has_topic:
        return ["Experiment", "Topic"]
    if has_topic:
        return ["Topic"]
    raise ValueError(f"No suitable grouping columns found in train: {list(df.columns)}")

def get_note_text(row, df_notes):
    """Safe lookup of Segment*_Notes by ID; return empty string if missing."""
    seg_col = SEGMENT_MAP[int(row["Segment"])]
    m = df_notes.loc[df_notes["ID"] == row["ID"], seg_col]
    if len(m) == 0:
        return ""
    val = m.values[0]
    return "" if pd.isna(val) else str(val)

def build_oneshot_split(train_df: pd.DataFrame, rng: np.random.RandomState) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Pick exactly 1 labeled example per grouping for training; rest for eval."""
    gcols = pick_group_cols(train_df)
    one_shot_idx, held_out_idx = [], []
    for _, g in train_df.groupby(gcols, dropna=False):
        # g already has clean int labels
        if len(g) == 0:
            continue
        pick = g.sample(n=1, random_state=rng)
        one_shot_idx.extend(pick.index.tolist())
        held_out_idx.extend(g.index.difference(pick.index).tolist())
    train_oneshot = train_df.loc[one_shot_idx].copy()
    eval_rest = train_df.loc[held_out_idx].copy()
    return train_oneshot, eval_rest

def preprocess_frames(df_list):
    """String-clean text; leave labels alone if NaNs exist (e.g., test)."""
    for df in df_list:
        df["IdeaUnit"] = df["IdeaUnit"].fillna("").astype(str)
        if "NoteText" in df.columns:
            df["NoteText"] = df["NoteText"].fillna("").astype(str)
        if "label" in df.columns and df["label"].notna().all():
            df["label"] = df["label"].astype(int)

def compute_metrics_binary(pred: EvalPrediction) -> Dict[str, float]:
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

def softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    ex = np.exp(x)
    return ex / ex.sum(axis=-1, keepdims=True)

class WeightedTrainer(Trainer):
    """Trainer with class-weighted CrossEntropy loss for imbalance."""
    def __init__(self, class_weights: torch.Tensor = None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.logits
        if labels is not None:
            if self.class_weights is not None:
                loss = cross_entropy(logits, labels, weight=self.class_weights.to(logits.device))
            else:
                loss = cross_entropy(logits, labels)
        else:
            loss = getattr(outputs, "loss", None)
        return (loss, outputs) if return_outputs else loss

def per_topic_eval(df_eval_with_preds: pd.DataFrame) -> pd.DataFrame:
    """Compute per-(Experiment, Topic) metrics; if Experiment missing, groups by Topic only."""
    by_cols = ["Experiment", "Topic"] if "Experiment" in df_eval_with_preds.columns else ["Topic"]
    rows = []
    for keys, g in df_eval_with_preds.groupby(by_cols):
        if isinstance(keys, tuple):
            exp, topic = keys if len(keys) == 2 else ("UNK", keys[0])
        else:
            exp, topic = ("UNK", keys)
        y_true = g["label"].to_numpy()
        y_pred = g["prediction"].to_numpy()
        acc = accuracy_score(y_true, y_pred)
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        rows.append({"Experiment": exp, "Topic": topic, "n": len(g),
                     "accuracy": acc, "precision": p, "recall": r, "f1": f1})
    return pd.DataFrame(rows).sort_values(["Experiment", "Topic", "n"], ascending=[True, True, False])

# --------------------------
# Main
# --------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../data/note_classification", help="Folder with Notes.csv/train.csv/test.csv")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    set_seed(args.seed)
    rng = np.random.RandomState(args.seed)

    # --------------------------
    # Load & normalize
    # --------------------------
    notes = coerce_notes_dtypes(read_csv_smart(os.path.join(args.data_dir, "Notes.csv")))
    train = coerce_train_dtypes(read_csv_smart(os.path.join(args.data_dir, "train.csv")))
    test  = coerce_test_dtypes(read_csv_smart(os.path.join(args.data_dir, "test.csv")))

    # Inject Experiment into train using Notes (so grouping can be by (Experiment, Topic))
    if "Experiment" not in train.columns and "Experiment" in notes.columns:
        train = attach_experiment_to_train(train, notes)

    # Attach NoteText to train/test (safe)
    train["NoteText"] = train.apply(lambda r: get_note_text(r, notes), axis=1)
    test["NoteText"]  = test.apply(lambda r: get_note_text(r, notes), axis=1)

    # Create 1-shot split (per (Experiment, Topic) if available)
    gcols = pick_group_cols(train)
    print(f"[info] One-shot grouping columns: {gcols}")
    train_oneshot, eval_rest = build_oneshot_split(train, rng)

    # Final cleanups
    preprocess_frames([train_oneshot, eval_rest, test])

    # --------------------------
    # Tokenization
    # --------------------------
    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    def encode_batch(batch):
        return tok(
            batch["IdeaUnit"],
            batch["NoteText"],
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
        )

    train_ds = Dataset.from_pandas(train_oneshot[["IdeaUnit", "NoteText", "label"]])
    eval_ds  = Dataset.from_pandas(eval_rest[["IdeaUnit", "NoteText", "label"]])
    test_ds  = Dataset.from_pandas(test[["IdeaUnit", "NoteText"]])

    train_ds = train_ds.map(encode_batch, batched=True)
    eval_ds  = eval_ds.map(encode_batch, batched=True)
    test_ds  = test_ds.map(encode_batch, batched=True)

    # set_format: be robust to models that don't use token_type_ids (e.g., RoBERTa)
    def set_torch_format(ds, with_labels: bool):
        cols = ["input_ids", "attention_mask"]
        if "token_type_ids" in ds.column_names:
            cols.append("token_type_ids")
        if with_labels:
            cols.append("label")
        ds.set_format(type="torch", columns=cols)

    set_torch_format(train_ds, with_labels=True)
    set_torch_format(eval_ds,  with_labels=True)
    set_torch_format(test_ds,  with_labels=False)

    # --------------------------
    # Model
    # --------------------------
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    # Class weights from one-shot labels (inverse-ish frequency)
    n_pos = int((train_oneshot["label"] == 1).sum())
    n_neg = int((train_oneshot["label"] == 0).sum())
    total = max(1, n_pos + n_neg)
    w0 = total / max(1, 2 * n_neg)
    w1 = total / max(1, 2 * n_pos)
    class_weights = torch.tensor([w0, w1], dtype=torch.float)

    # --------------------------
    # Training
    # --------------------------
    training_args = TrainingArguments(
        output_dir=ckpt_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        learning_rate=args.lr,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=50,
        report_to="tensorboard",  # set to "tensorboard" if you want TB logs
        seed=args.seed,
    )

    class WeightedTrainer(Trainer):
        """Trainer with class-weighted CrossEntropy loss for imbalance."""
        def __init__(self, class_weights: torch.Tensor = None, **kwargs):
            super().__init__(**kwargs)
            self.class_weights = class_weights

        # 👇 add **kwargs to swallow unknown args like num_items_in_batch
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.get("labels")
            outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
            logits = outputs.logits
            if labels is not None:
                if self.class_weights is not None:
                    loss = cross_entropy(logits, labels, weight=self.class_weights.to(logits.device))
                else:
                    loss = cross_entropy(logits, labels)
            else:
                loss = getattr(outputs, "loss", None)
            return (loss, outputs) if return_outputs else loss

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tok,
        compute_metrics=compute_metrics_binary,
    )

    trainer.train()

    # --------------------------
    # Evaluation on eval_rest
    # --------------------------
    eval_pred = trainer.predict(eval_ds)
    y_true = eval_pred.label_ids
    y_logit = eval_pred.predictions
    y_prob = softmax_np(y_logit)[:, 1]
    y_pred = y_logit.argmax(-1)

    overall = compute_metrics_binary(EvalPrediction(predictions=y_logit, label_ids=y_true))
    overall_path = os.path.join(args.output_dir, "bert_eval_metrics_overall.json")
    with open(overall_path, "w") as f:
        json.dump(overall, f, indent=2)

    # Detailed eval CSV
    eval_out = eval_rest.copy().reset_index(drop=False).rename(columns={"index": "orig_index"})
    eval_out["prediction"] = y_pred
    eval_out["prob_1"] = y_prob
    eval_out_path = os.path.join(args.output_dir, "bert_eval_predictions.csv")
    eval_out.to_csv(eval_out_path, index=False)

    # Per-topic metrics
    per_topic = per_topic_eval(eval_out)
    per_topic_path = os.path.join(args.output_dir, "bert_eval_metrics_by_topic.csv")
    per_topic.to_csv(per_topic_path, index=False)

    # Text summary for console
    cls_report = classification_report(y_true, y_pred, digits=4, zero_division=0)
    print("\n=== Overall Eval (held-out within-topic) ===")
    print(json.dumps(overall, indent=2))
    print("\n=== Classification Report (Eval) ===")
    print(cls_report)
    print(f"\nSaved overall metrics to: {overall_path}")
    print(f"Saved per-topic metrics to: {per_topic_path}")
    print(f"Saved eval predictions to: {eval_out_path}")

    # --------------------------
    # Inference on test
    # --------------------------
    test_pred = trainer.predict(test_ds)
    test_logits = test_pred.predictions
    test_prob = softmax_np(test_logits)[:, 1]
    test_labels = test_logits.argmax(-1)

    test_out = test.copy()
    test_out["prediction"] = test_labels
    test_out["prob_1"] = test_prob
    test_out_path = os.path.join(args.output_dir, "bert_test_predictions.csv")
    test_out.to_csv(test_out_path, index=False)

    print(f"Saved test predictions to: {test_out_path}")
    print("\nDone.")

if __name__ == "__main__":
    main()

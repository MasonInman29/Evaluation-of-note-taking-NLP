#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BERT pair classifier for IdeaUnit ↔ NoteText with temporal alignment.

Core idea:
  - Each Topic has a sequence of lecture IdeaUnits: u_0, u_1, ..., u_{K-1}.
  - We assume a "good" note-taker writes notes in roughly the same order
    as the lecture. So IdeaUnit u_j should appear in the corresponding
    "time slice" of the student's notes.

Mathematically:
  - For a given Topic, let there be K distinct IdeaUnits in lecture order.
  - IdeaUnit u_j is assigned the normalized interval:
        I_j = [ j/K, (j+1)/K )
  - A student's notes (for that Topic) are tokenized into L tokens:
        notes = (w_0, w_1, ..., w_{L-1})
  - We map I_j onto token indices by:
        start_idx  = floor( (j/K - margin) * L )
        end_idx    = floor( ( (j+1)/K + margin ) * L )
    clamped to [0, L].
  - We then choose a span of these note tokens (capped by BERT max_length)
    and decode back to text. This "aligned chunk" is what we feed into BERT
    together with the IdeaUnit.

Default data split (Option B):
  - We treat train.csv as a standard labeled dataset and perform a random
    80/20 split into train / validation. test.csv is held out purely for
    final testing.
  - You can still enable the original 1-shot regime via --use_oneshot.

Usage (80/20 example):
  python bert.py \
      --data_dir ../data/note_classification \
      --output_dir results_temporal_8020 \
      --model_name bert-base-uncased \
      --max_length 256 \
      --epochs 8 \
      --lr 2e-5 \
      --train_batch_size 8 \
      --eval_batch_size 8 \
      --seed 42 \
      --seeds 42,43,44 \
      --align_margin 0.1 \
      --val_frac 0.2 \
      --optimize_threshold

Outputs (per seed in output_dir/seed_<seed>/):
  - bert_eval_metrics_overall.json
  - bert_eval_metrics_by_topic.csv
  - bert_eval_predictions.csv           (held-out eval)
  - bert_eval_confusion_matrix.csv
  - bert_eval_classification_report.txt
  - bert_test_predictions.csv
  - submission_test.csv
  - checkpoints/                        (best HF checkpoint by eval F1)

If multiple seeds are provided:
  - bert_eval_seed_summary.json in output_dir/
"""

import os
import json
import argparse
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

# Quiet fork/parallelism warning from tokenizers
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
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

# ---------------------------------------------------------------------
# CSV utilities: robust reading and dtype coercion
# ---------------------------------------------------------------------

def read_csv_smart(path: str) -> pd.DataFrame:
    """
    Robust CSV reader that tolerates BOMs and cp1252 smart quotes.

    Tries a small set of common encodings and falls back to replacing
    invalid bytes if necessary.
    """
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, encoding="utf-8", encoding_errors="replace")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names by stripping whitespace and invalid bytes.

    Important because Excel / different OSes can inject weird characters
    into headers.
    """
    df = df.copy()
    df.columns = [
        str(c).encode("utf-8", "ignore").decode("utf-8").strip()
        for c in df.columns
    ]
    return df


def coerce_notes_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean Notes.csv: ensure keys (Experiment, Topic, ID) are usable.

    We do NOT touch the Segment*_Notes text content.
    """
    df = normalize_columns(df)

    if "Experiment" in df.columns:
        df["Experiment"] = df["Experiment"].astype(str).str.strip()

    if "Topic" in df.columns:
        df["Topic"] = df["Topic"].astype(str).str.strip()

    if "ID" in df.columns:
        df["ID"] = pd.to_numeric(df["ID"], errors="coerce")

    need = [c for c in ["Topic", "ID"] if c in df.columns]
    if need:
        df = df.dropna(subset=need).copy()

    if "ID" in df.columns:
        df["ID"] = df["ID"].astype(int)

    return df


def coerce_train_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean train.csv: enforce {Topic, ID, Segment, IdeaUnit, label},
    coerce numeric fields, and drop invalid rows.
    """
    df = normalize_columns(df)

    required = {"Topic", "ID", "Segment", "IdeaUnit", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"train.csv missing columns: {missing}")

    df["Topic"] = df["Topic"].astype(str).str.strip()
    df["ID"] = pd.to_numeric(df["ID"], errors="coerce")
    df["Segment"] = pd.to_numeric(df["Segment"], errors="coerce")
    df["IdeaUnit"] = df["IdeaUnit"].astype(str)
    df["label"] = pd.to_numeric(df["label"], errors="coerce")

    # Keep only segments {1,2,3,4}
    df.loc[~df["Segment"].isin([1, 2, 3, 4]), "Segment"] = np.nan

    n0 = len(df)
    df = df.dropna(subset=["Topic", "ID", "Segment", "IdeaUnit", "label"]).copy()
    if len(df) < n0:
        print(f"[info] Dropped {n0 - len(df)} train rows failing dtype coercion.")

    df["ID"] = df["ID"].astype(int)
    df["Segment"] = df["Segment"].astype(int)
    df["label"] = df["label"].astype(int)
    return df


def coerce_test_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean test.csv: label may be missing (NaN).
    Required keys: Topic, ID, Segment, IdeaUnit.
    """
    df = normalize_columns(df)

    required = {"Topic", "ID", "Segment", "IdeaUnit"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"test.csv missing required columns: {missing}")

    df["Topic"] = df["Topic"].astype(str).str.strip()
    df["ID"] = pd.to_numeric(df["ID"], errors="coerce")
    df["Segment"] = pd.to_numeric(df["Segment"], errors="coerce")
    df["IdeaUnit"] = df["IdeaUnit"].astype(str)

    if "label" in df.columns:
        df["label"] = pd.to_numeric(df["label"], errors="coerce")

    df.loc[~df["Segment"].isin([1, 2, 3, 4]), "Segment"] = np.nan

    n0 = len(df)
    df = df.dropna(subset=["Topic", "ID", "Segment", "IdeaUnit"]).copy()
    if len(df) < n0:
        print(f"[info] Dropped {n0 - len(df)} test rows missing Topic/ID/Segment/IdeaUnit.")

    df["ID"] = df["ID"].astype(int)
    df["Segment"] = df["Segment"].astype(int)
    return df


def attach_experiment_to_train(train_df: pd.DataFrame, notes_df: pd.DataFrame) -> pd.DataFrame:
    """
    If train.csv does not have Experiment, but Notes.csv does, inject it
    by joining Notes on (Topic, ID).

    If Experiment cannot be found, use "UNK".
    """
    if not {"Topic", "ID", "Experiment"} <= set(notes_df.columns):
        out = train_df.copy()
        out["Experiment"] = "UNK"
        return out

    key_cols = ["Topic", "ID", "Experiment"]
    map_df = notes_df[key_cols].drop_duplicates(subset=["Topic", "ID"])
    merged = train_df.merge(
        map_df,
        on=["Topic", "ID"],
        how="left",
        validate="many_to_one"
    )
    merged["Experiment"] = merged["Experiment"].fillna("UNK")
    return merged


# ---------------------------------------------------------------------
# Lecture ordering: IdeaUnit -> (index, count) per Topic
# ---------------------------------------------------------------------

def build_idea_order_maps(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Build two maps:

      idea_index_map[Topic][IdeaUnit] = j   (0-based lecture order)
      idea_count_map[Topic]           = K   (number of distinct IdeaUnits)

    We approximate lecture order by the first occurrence of each IdeaUnit
    per Topic in the concatenated (train + test) data.

    Mathematically:
      For each Topic t, let U_t = [u_0, ..., u_{K-1}] be the ordered list
      of distinct IdeaUnits in that Topic. Then K = |U_t| and
          idea_index_map[t][u_j] = j.
    """
    cols = ["Topic", "IdeaUnit"]
    combined = pd.concat(
        [train_df[cols], test_df[cols]],
        ignore_index=True
    ).dropna(subset=cols)

    combined["Topic"] = combined["Topic"].astype(str).str.strip()
    combined["IdeaUnit"] = combined["IdeaUnit"].astype(str)

    idea_index_map = {}
    idea_count_map = {}

    for topic, g in combined.groupby("Topic"):
        seen = set()
        ordered_ideas = []
        for idea in g["IdeaUnit"]:
            if idea not in seen:
                seen.add(idea)
                ordered_ideas.append(idea)
        idx_map = {idea: j for j, idea in enumerate(ordered_ideas)}
        idea_index_map[topic] = idx_map
        idea_count_map[topic] = len(ordered_ideas)

    print("[info] Built IdeaUnit order maps for", len(idea_index_map), "topics.")
    return idea_index_map, idea_count_map


# ---------------------------------------------------------------------
# Temporal alignment: lecture IdeaUnit ↔ note tokens
# ---------------------------------------------------------------------

SEGMENT_COLS = [f"Segment{s}_Notes" for s in (1, 2, 3, 4)]


def build_full_notes_for_row(row: pd.Series, notes_df: pd.DataFrame) -> str:
    """
    Concatenate all note segments (Segment1_Notes .. Segment4_Notes)
    for this (Topic, ID, Experiment) row into a single string.

    This models the student's notes for the entire lecture as a single
    sequence of text, which we then align against the lecture timeline.
    """
    topic = str(row["Topic"])
    sid = int(row["ID"])

    if "Experiment" in notes_df.columns and "Experiment" in row:
        exp = str(row["Experiment"])
        mask = (
            (notes_df["Topic"] == topic) &
            (notes_df["ID"] == sid) &
            (notes_df["Experiment"] == exp)
        )
    else:
        mask = (
            (notes_df["Topic"] == topic) &
            (notes_df["ID"] == sid)
        )

    m = notes_df.loc[mask]
    if len(m) == 0:
        return ""

    m = m.iloc[0]
    segments = []
    for col in SEGMENT_COLS:
        if col in m and pd.notna(m[col]):
            segments.append(str(m[col]))
    return " ".join(segments).strip()


def get_temporally_aligned_note_chunk(
    row: pd.Series,
    notes_df: pd.DataFrame,
    tokenizer,
    max_length: int,
    idea_index_map: Dict[str, Dict[str, int]],
    idea_count_map: Dict[str, int],
    margin_frac: float = 0.1,
) -> str:
    """
    Given a (train/test) row, return a note chunk that is temporally
    aligned with the IdeaUnit under the assumption that notes follow
    the lecture order.

    Steps (with math):

    1. For this Topic = t and IdeaUnit = u:
         - Let K = idea_count_map[t] be the number of distinct IdeaUnits.
         - Let j = idea_index_map[t][u] be the index of u in [0, ..., K-1].
    2. We assign u to the lecture interval:
         I_j = [ j/K, (j+1)/K ).
       Then expand by margin_frac, giving:
         I'_j = [ max(0, j/K - margin),  min(1, (j+1)/K + margin) ].
    3. The student's notes are tokenized into L tokens:
         notes = (w_0, ..., w_{L-1}).
       We map I'_j into token indices:
         start_idx = floor(alpha * L)
         end_idx   = floor(beta  * L)
       where [alpha, beta] = I'_j.
    4. We then cap the span length to max_note_tokens so that the total
       BERT input length (IdeaUnit + note chunk + specials) <= max_length.

    If the IdeaUnit was never seen in the ordering (rare), we fall back
    to approximating its position via the Segment number (1..4).
    """
    topic = str(row["Topic"])
    idea = str(row["IdeaUnit"])

    full_notes = build_full_notes_for_row(row, notes_df)
    if not full_notes:
        return ""

    # Tokenize full notes (no special tokens, we control them later).
    note_ids = tokenizer(full_notes, add_special_tokens=False)["input_ids"]
    if len(note_ids) == 0:
        return ""

    # Tokenize IdeaUnit alone (to know how many tokens it uses).
    idea_ids = tokenizer(idea, add_special_tokens=False)["input_ids"]
    # Reserve room for [CLS] idea [SEP] note_chunk [SEP] → +3 specials.
    max_note_tokens = max_length - len(idea_ids) - 3
    if max_note_tokens <= 32:
        # Safety fallback: don't let note chunk go to zero length.
        max_note_tokens = max(max_length // 2, 32)

    # --------------------------------------------------------------
    # Step 1: Retrieve j and K for this Topic and IdeaUnit.
    # --------------------------------------------------------------
    idx_map = idea_index_map.get(topic, {})
    K = idea_count_map.get(topic, 0)
    j = idx_map.get(idea, None)

    # --------------------------------------------------------------
    # Step 2: Compute normalized interval [alpha, beta] in [0,1].
    # --------------------------------------------------------------
    if j is not None and K > 0:
        base_start = j / float(K)
        base_end = (j + 1) / float(K)
        alpha = max(0.0, base_start - margin_frac)
        beta = min(1.0, base_end + margin_frac)
    else:
        # Fallback: approximate relative position using Segment (1..4).
        # For Segment s in {1,2,3,4}, we center at (s-0.5)/4 and add margin.
        seg = int(row.get("Segment", 1))
        center = (seg - 0.5) / 4.0
        alpha = max(0.0, center - margin_frac)
        beta = min(1.0, center + margin_frac)

    # --------------------------------------------------------------
    # Step 3: Map [alpha, beta] onto token indices in [0, L-1].
    # --------------------------------------------------------------
    L = len(note_ids)
    start_idx = int(np.floor(alpha * L))
    end_idx = int(np.floor(beta * L))

    if end_idx <= start_idx:
        # Ensure at least some span; extend forward if needed.
        end_idx = min(L, start_idx + max_note_tokens)

    span_ids = note_ids[start_idx:end_idx]
    if len(span_ids) > max_note_tokens:
        span_ids = span_ids[:max_note_tokens]

    chunk = tokenizer.decode(span_ids, skip_special_tokens=True)
    return chunk


# ---------------------------------------------------------------------
# Splitting strategies: 1-shot vs random 80/20
# ---------------------------------------------------------------------

def pick_group_cols(df: pd.DataFrame):
    """
    Decide grouping columns for 1-shot split.

    Preferred:
      - ["Experiment", "Topic"] if both exist.
    Fallback:
      - ["Topic"] otherwise.
    """
    has_exp = "Experiment" in df.columns
    has_topic = "Topic" in df.columns

    if has_exp and has_topic:
        return ["Experiment", "Topic"]
    if has_topic:
        return ["Topic"]
    raise ValueError(f"No suitable grouping columns found in train: {list(df.columns)}")


def build_oneshot_split(
    train_df: pd.DataFrame,
    rng: np.random.RandomState
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    1-shot per group split.

    For each group g (e.g., each (Experiment, Topic)), we:
      - randomly select exactly 1 labeled row into train_split,
      - keep all other rows in eval_rest.

    This enforces the few-shot setting:
      |train_split_g| = 1  for every group g.
    """
    gcols = pick_group_cols(train_df)
    one_shot_idx, held_out_idx = [], []

    for _, g in train_df.groupby(gcols, dropna=False):
        if len(g) == 0:
            continue
        pick = g.sample(n=1, random_state=rng)
        one_shot_idx.extend(pick.index.tolist())
        held_out_idx.extend(g.index.difference(pick.index).tolist())

    train_split = train_df.loc[one_shot_idx].copy()
    eval_rest = train_df.loc[held_out_idx].copy()
    return train_split, eval_rest


def build_random_split(
    train_df: pd.DataFrame,
    rng: np.random.RandomState,
    val_frac: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Standard random train/validation split (80/20 by default).

    Given the full training set train_df, we generate a random permutation
    of indices and split them into:

        train_split (≈ (1 - val_frac) * N rows)
        eval_rest   (≈ val_frac * N rows)

    where N = len(train_df).

    We use a NumPy RandomState rng that is seeded per run to ensure
    reproducibility across seeds.

    This ignores (Experiment, Topic) grouping and treats examples as i.i.d.
    """
    n = len(train_df)
    indices = np.arange(n)
    rng.shuffle(indices)

    n_val = int(np.floor(val_frac * n))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    train_split = train_df.iloc[train_idx].copy()
    eval_rest = train_df.iloc[val_idx].copy()

    return train_split, eval_rest


def preprocess_frames(df_list: List[pd.DataFrame]):
    """
    Light preprocessing: ensure text columns are strings, labels ints.

    This is where we do final type cleaning before wrapping data in
    HuggingFace Dataset objects.
    """
    for df in df_list:
        df["IdeaUnit"] = df["IdeaUnit"].fillna("").astype(str)
        if "NoteText" in df.columns:
            df["NoteText"] = df["NoteText"].fillna("").astype(str)
        if "label" in df.columns and df["label"].notna().all():
            df["label"] = df["label"].astype(int)


# ---------------------------------------------------------------------
# Metrics and helpers
# ---------------------------------------------------------------------

def compute_metrics_binary(pred: EvalPrediction) -> Dict[str, float]:
    """
    Basic binary classification metrics for Trainer:

      - accuracy
      - precision (positive class)
      - recall (positive class)
      - F1 (positive class)
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def softmax_np(x: np.ndarray) -> np.ndarray:
    """
    Numerically stable softmax for logits x (2D array: [n_samples, n_classes]).
    """
    x = x - x.max(axis=-1, keepdims=True)
    ex = np.exp(x)
    return ex / ex.sum(axis=-1, keepdims=True)


def get_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    """
    Choose a probability threshold t in [0,1] that maximizes F1 on
    the provided (y_true, y_prob).

    We evaluate F1 over:
      - a fine grid {0.01, 0.02, ..., 0.99}
      - all unique probabilities in y_prob

    Returns:
      (best_threshold, best_f1)
    """
    best_t = 0.5
    best_f1 = 0.0

    candidates = sorted(set(
        list(np.linspace(0.01, 0.99, 99)) + list(np.unique(y_prob))
    ))

    for t in candidates:
        y_pred = (y_prob >= t).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t, best_f1


def per_topic_eval(df_eval_with_preds: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-(Experiment, Topic) metrics (or per-Topic if Experiment
    is missing) to see performance variation across lectures.
    """
    by_cols = ["Experiment", "Topic"] if "Experiment" in df_eval_with_preds.columns else ["Topic"]

    rows = []
    for keys, g in df_eval_with_preds.groupby(by_cols):
        if isinstance(keys, tuple):
            if len(keys) == 2:
                exp, topic = keys
            else:
                exp, topic = ("UNK", keys[0])
        else:
            exp, topic = ("UNK", keys)

        y_true = g["label"].to_numpy()
        y_pred = g["prediction"].to_numpy()

        acc = accuracy_score(y_true, y_pred)
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        rows.append({
            "Experiment": exp,
            "Topic": topic,
            "n": len(g),
            "accuracy": acc,
            "precision": p,
            "recall": r,
            "f1": f1,
        })

    out = pd.DataFrame(rows)
    return out.sort_values(["Experiment", "Topic", "n"], ascending=[True, True, False])


# ---------------------------------------------------------------------
# TrainingArguments shim: keep things simple but compatible
# ---------------------------------------------------------------------

def make_training_args(args, ckpt_dir, output_dir):
    """
    Create TrainingArguments with per-epoch evaluation and checkpointing.

    If running on an older transformers version that does not support
    some of these arguments, fall back to a simpler signature.
    """
    try:
        return TrainingArguments(
            output_dir=ckpt_dir,
            evaluation_strategy="epoch",
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
            logging_dir=os.path.join(output_dir, "logs"),
            logging_steps=50,
            report_to="none",
            seed=args.seed,
        )
    except TypeError as e:
        print(f"[warn] Using legacy TrainingArguments due to: {e}")
        kwargs = dict(
            output_dir=ckpt_dir,
            learning_rate=args.lr,
            per_device_train_batch_size=args.train_batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            num_train_epochs=args.epochs,
            weight_decay=0.01,
            logging_dir=os.path.join(output_dir, "logs"),
            seed=args.seed,
        )
        return TrainingArguments(**kwargs)


# ---------------------------------------------------------------------
# One full training/eval run for a single seed
# ---------------------------------------------------------------------

def run_once(
    args,
    seed: int,
    notes: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    tokenizer,
    idea_index_map,
    idea_count_map,
):
    set_seed(seed)
    rng = np.random.RandomState(seed)

    # Inject Experiment into train if possible (for grouping / reporting).
    if "Experiment" not in train.columns and "Experiment" in notes.columns:
        train = attach_experiment_to_train(train, notes)

    # Attach temporally aligned NoteText to train and test.
    print(f"[info] (seed {seed}) Building temporally aligned note chunks...")
    train = train.copy()
    test = test.copy()

    train["NoteText"] = train.apply(
        lambda r: get_temporally_aligned_note_chunk(
            r,
            notes_df=notes,
            tokenizer=tokenizer,
            max_length=args.max_length,
            idea_index_map=idea_index_map,
            idea_count_map=idea_count_map,
            margin_frac=args.align_margin,
        ),
        axis=1,
    )
    test["NoteText"] = test.apply(
        lambda r: get_temporally_aligned_note_chunk(
            r,
            notes_df=notes,
            tokenizer=tokenizer,
            max_length=args.max_length,
            idea_index_map=idea_index_map,
            idea_count_map=idea_count_map,
            margin_frac=args.align_margin,
        ),
        axis=1,
    )

    # ----------------------------------------------------------
    # Train/eval split:
    #   Option A: 1-shot per (Experiment, Topic) (--use_oneshot)
    #   Option B: random 80/20 split on train.csv (default)
    # ----------------------------------------------------------
    if args.use_oneshot:
        gcols = pick_group_cols(train)
        print(f"[info] (seed {seed}) Using 1-shot split with grouping columns: {gcols}")
        train_split, eval_rest = build_oneshot_split(train, rng)
    else:
        print(f"[info] (seed {seed}) Using random split with val_frac = {args.val_frac:.2f}")
        train_split, eval_rest = build_random_split(train, rng, val_frac=args.val_frac)

    # Report class balance in the training split.
    if "label" in train_split.columns:
        counts = train_split["label"].value_counts().to_dict()
        n_pos = int(counts.get(1, 0))
        n_neg = int(counts.get(0, 0))
        total = max(1, n_pos + n_neg)
        pos_prev = n_pos / total
        print(f"[info] (seed {seed}) Train split class counts: pos={n_pos}, neg={n_neg}, pos_prev={pos_prev:.3f}")
    else:
        print("[warn] train_split has no label column!")

    # Final cleanup before building HF datasets.
    preprocess_frames([train_split, eval_rest, test])

    # Tokenization function for HF Dataset.map
    def encode_batch(batch):
        return tokenizer(
            batch["IdeaUnit"],
            batch["NoteText"],
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
        )

    # Wrap in HuggingFace Datasets
    train_ds = Dataset.from_pandas(train_split[["IdeaUnit", "NoteText", "label"]])
    eval_ds = Dataset.from_pandas(eval_rest[["IdeaUnit", "NoteText", "label"]])
    test_ds = Dataset.from_pandas(test[["IdeaUnit", "NoteText"]])

    train_ds = train_ds.map(encode_batch, batched=True)
    eval_ds = eval_ds.map(encode_batch, batched=True)
    test_ds = test_ds.map(encode_batch, batched=True)

    # Set tensor format for Trainer
    def set_torch_format(ds, with_labels: bool):
        cols = ["input_ids", "attention_mask"]
        if "token_type_ids" in ds.column_names:
            cols.append("token_type_ids")
        if with_labels:
            cols.append("label")
        ds.set_format(type="torch", columns=cols)

    set_torch_format(train_ds, True)
    set_torch_format(eval_ds, True)
    set_torch_format(test_ds, False)

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
    )

    # Output / checkpoint dirs per seed
    seed_dir = os.path.join(args.output_dir, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)
    ckpt_dir = os.path.join(seed_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Training arguments
    args.seed = seed
    training_args = make_training_args(args, ckpt_dir=ckpt_dir, output_dir=seed_dir)

    # Standard Trainer (no custom loss; simplicity is preferred here).
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_binary,
    )

    # Train
    trainer.train()

    # --------------------------------------------------------------
    # Evaluation on held-out validation split
    # --------------------------------------------------------------
    eval_pred = trainer.predict(eval_ds)
    y_true = eval_pred.label_ids
    logits = eval_pred.predictions
    y_prob = softmax_np(logits)[:, 1]
    y_pred = logits.argmax(-1)

    # Optional: threshold optimization on eval set.
    best_threshold = 0.5
    best_eval_f1 = None
    if args.optimize_threshold:
        best_threshold, best_eval_f1 = get_best_threshold(y_true, y_prob)
        print(f"[info] Optimized threshold on eval: t* = {best_threshold:.3f} (F1 = {best_eval_f1:.4f})")
        y_pred = (y_prob >= best_threshold).astype(int)

    # Overall metrics
    overall_basic = compute_metrics_binary(
        EvalPrediction(predictions=logits, label_ids=y_true)
    )
    balanced_acc = balanced_accuracy_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else float("nan")
    if len(np.unique(y_true)) == 2:
        auc_roc = roc_auc_score(y_true, y_prob)
        auc_pr = average_precision_score(y_true, y_prob)
    else:
        auc_roc = float("nan")
        auc_pr = float("nan")

    overall = dict(
        overall_basic,
        roc_auc=auc_roc,
        pr_auc=auc_pr,
        n_eval=int(len(y_true)),
        balanced_accuracy=balanced_acc,
        threshold=best_threshold,
    )

    overall_path = os.path.join(seed_dir, "bert_eval_metrics_overall.json")
    with open(overall_path, "w") as f:
        json.dump(overall, f, indent=2)

    # Eval predictions, per-topic metrics, confusion matrix, report
    eval_out = eval_rest.copy().reset_index(drop=False).rename(columns={"index": "orig_index"})
    eval_out["prediction"] = y_pred
    eval_out["prob_1"] = y_prob

    eval_out_path = os.path.join(seed_dir, "bert_eval_predictions.csv")
    eval_out.to_csv(eval_out_path, index=False)

    per_topic = per_topic_eval(eval_out)
    # Add per-topic positive prevalence (mean label)
    if "label" in eval_out.columns:
        group_cols = ["Experiment", "Topic"] if "Experiment" in eval_out.columns else ["Topic"]
        prevs = (
            eval_out.groupby(group_cols)["label"]
            .mean()
            .reset_index()
            .rename(columns={"label": "pos_prevalence"})
        )
        per_topic = per_topic.merge(prevs, on=group_cols, how="left")

    per_topic_path = os.path.join(seed_dir, "bert_eval_metrics_by_topic.csv")
    per_topic.to_csv(per_topic_path, index=False)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_path = os.path.join(seed_dir, "bert_eval_confusion_matrix.csv")
    pd.DataFrame(
        cm,
        index=["true_0", "true_1"],
        columns=["pred_0", "pred_1"],
    ).to_csv(cm_path, index=True)

    cls_report = classification_report(
        y_true,
        y_pred,
        digits=4,
        zero_division=0,
    )
    report_txt_path = os.path.join(seed_dir, "bert_eval_classification_report.txt")
    with open(report_txt_path, "w") as f:
        f.write(cls_report)

    print("\n=== Overall Eval (held-out validation) ===")
    print(json.dumps(overall, indent=2))
    print("\n=== Classification Report (Eval) ===")
    print(cls_report)
    print(f"\nSaved overall metrics to:      {overall_path}")
    print(f"Saved per-topic metrics to:    {per_topic_path}")
    print(f"Saved eval predictions to:     {eval_out_path}")
    print(f"Saved confusion matrix to:     {cm_path}")
    print(f"Saved classification report to:{report_txt_path}")

    # --------------------------------------------------------------
    # Inference on test set
    # --------------------------------------------------------------
    test_pred = trainer.predict(test_ds)
    test_logits = test_pred.predictions
    test_prob = softmax_np(test_logits)[:, 1]

    if args.optimize_threshold:
        test_labels = (test_prob >= best_threshold).astype(int)
    else:
        test_labels = test_logits.argmax(-1)

    test_out = test.copy()
    test_out["prediction"] = test_labels
    test_out["prob_1"] = test_prob

    test_out_path = os.path.join(seed_dir, "bert_test_predictions.csv")
    test_out.to_csv(test_out_path, index=False)

    # Submission-style file (minimal columns)
    sub_cols = ["Experiment", "Topic", "ID", "Segment", "IdeaUnit", "prediction"]
    existing = [c for c in sub_cols if c in test_out.columns]
    sub_path = os.path.join(seed_dir, "submission_test.csv")
    test_out[existing].to_csv(sub_path, index=False)

    print(f"Saved test predictions to:     {test_out_path}")
    print(f"Saved submission-style file to:{sub_path}")
    print("\nDone for seed", seed)

    return overall  # for aggregate seed statistics


# ---------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../data/note_classification",
        help="Folder with Notes.csv/train.csv/test.csv",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="bert-base-uncased",
        help="HuggingFace model name or path.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results_temporal_8020",
        help="Directory to store results and checkpoints.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
        help="Max token length for (IdeaUnit, NoteText) pair.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=8,
        help="Number of fine-tuning epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=8,
        help="Per-device train batch size.",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=8,
        help="Per-device eval batch size.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Default random seed.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="",
        help="Comma-separated list of seeds, e.g. '42,43,44'; if empty, use --seed only.",
    )
    parser.add_argument(
        "--align_margin",
        type=float,
        default=0.1,
        help=(
            "Temporal margin (fraction of lecture [0,1]) around each IdeaUnit "
            "interval. E.g. 0.1 expands [j/K, (j+1)/K] by ±0.1."
        ),
    )
    parser.add_argument(
        "--optimize_threshold",
        action="store_true",
        help="If set, tune probability threshold on eval set to maximize F1 and apply to test.",
    )
    parser.add_argument(
        "--use_oneshot",
        action="store_true",
        help="If set, use 1-shot per (Experiment,Topic). If not set, use random 80/20 split on train.csv.",
    )
    parser.add_argument(
        "--val_frac",
        type=float,
        default=0.2,
        help="Validation fraction for random split (ignored when --use_oneshot is set).",
    )

    args = parser.parse_args()

    # Prepare output_dir
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    notes = coerce_notes_dtypes(
        read_csv_smart(os.path.join(args.data_dir, "Notes.csv"))
    )
    train = coerce_train_dtypes(
        read_csv_smart(os.path.join(args.data_dir, "train.csv"))
    )
    test = coerce_test_dtypes(
        read_csv_smart(os.path.join(args.data_dir, "test.csv"))
    )

    # Shared tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    # Avoid spurious warnings when we tokenize long notes and then slice manually
    tokenizer.model_max_length = 10000

    # Build lecture IdeaUnit ordering maps
    idea_index_map, idea_count_map = build_idea_order_maps(train, test)

    # Parse seed list
    if args.seeds.strip():
        seed_list = [
            int(s.strip())
            for s in args.seeds.split(",")
            if s.strip()
        ]
    else:
        seed_list = [args.seed]

    # Run per seed and collect metrics
    all_metrics: List[Dict] = []
    for sd in seed_list:
        metrics = run_once(
            args,
            seed=sd,
            notes=notes,
            train=train,
            test=test,
            tokenizer=tokenizer,
            idea_index_map=idea_index_map,
            idea_count_map=idea_count_map,
        )
        metrics["_seed"] = sd
        all_metrics.append(metrics)

    # Aggregate over seeds if we have more than one
    if len(seed_list) > 1:
        agg_keys = ["f1", "precision", "recall", "accuracy", "roc_auc", "pr_auc"]
        summary = {"seeds": seed_list}

        for k in agg_keys:
            vals = [
                m.get(k)
                for m in all_metrics
                if isinstance(m.get(k), (int, float))
                and not (isinstance(m.get(k), float) and np.isnan(m.get(k)))
            ]
            if len(vals) > 0:
                summary[k] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals, ddof=0)),
                    "min": float(np.min(vals)),
                    "max": float(np.max(vals)),
                    "n": len(vals),
                }
            else:
                summary[k] = {"mean": float("nan"), "std": float("nan"), "n": 0}

        summary_path = os.path.join(args.output_dir, "bert_eval_seed_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        f1s = summary["f1"]
        print(f"\n=== Seed sweep summary (F1) over {f1s['n']} runs ===")
        print(
            f"F1: {f1s['mean']:.4f} ± {f1s['std']:.4f} "
            f"(min {f1s['min']:.4f}, max {f1s['max']:.4f})"
        )
        print(f"Saved seed summary to: {summary_path}")


if __name__ == "__main__":
    main()

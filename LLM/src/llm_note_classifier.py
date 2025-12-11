import argparse
import os

import pandas as pd
from tqdm import tqdm
import torch

from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM

from preprocessing import load_data, add_segment_text
from prompts import build_prompt_zero_shot, build_prompt_one_shot


# Load HF token from .env

load_dotenv() 
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise RuntimeError("HF_TOKEN is not set. Create a .env file with HF_TOKEN=...")


# Load LLaMA 3 model

MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

print(f"Loading LLaMA 3 model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)

# half-precision on GPU, full precision on CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Let HF handle device placement
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=dtype,
    device_map="auto",
    token=HF_TOKEN,
)

model.eval()
print(f"LLaMA 3 loaded. Using device: {device}")


# Helper functions 

def chunk_note_text(note: str, max_note_tokens: int = 256, stride: int = 192) -> list[str]:
    """
    Split a long note into overlapping chunks in token space.

    Args:
        note: full note text for a single segment.
        max_note_tokens: maximum number of tokens for a single note chunk.
        stride: how many tokens to move the window each step
                (stride < max_note_tokens gives overlap).

    Returns:
        List of chunk texts. For short notes, this is just [note].
    """
    # Tokenize only the note text, no special tokens
    enc = tokenizer(
        note,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"][0]

    if len(enc) <= max_note_tokens:
        return [note]

    chunks = []
    start = 0
    while start < len(enc):
        end = start + max_note_tokens
        chunk_ids = enc[start:end]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        chunks.append(chunk_text)

        if end >= len(enc):
            break
        start += stride

    return chunks

def lexical_overlap_score(idea: str, note: str) -> float:
    """
    Simple lexical overlap between IdeaUnit and note.
    Score = (# of idea tokens that appear in note) / (# of idea tokens).
    """
    idea_tokens = (
        idea.lower()
        .replace(".", " ")
        .replace(",", " ")
        .replace(";", " ")
        .split()
    )
    note_tokens = (
        note.lower()
        .replace(".", " ")
        .replace(",", " ")
        .replace(";", " ")
        .split()
    )

    idea_tokens = {t for t in idea_tokens if t}
    note_tokens = {t for t in note_tokens if t}

    if not idea_tokens:
        return 0.0

    return len(idea_tokens & note_tokens) / len(idea_tokens)


def select_one_shot_examples(train_df: pd.DataFrame) -> dict:
    """
    Select one example row per Topic from the training data.

    Improved strategy:
    - Group by Topic.
    - For each Topic:
        * Filter to rows with non-empty segment_text.
        * Prefer positive examples (label == 1).
        * Among positives, choose the one with the highest lexical overlap
          between IdeaUnit and segment_text.
        * If no positives exist for that Topic, fall back to the non-empty
          example with highest overlap.

    This automatically finds "cleanest" examples like:
        - Physics:   "the smallest quantity of energy is quantum ..."
        - Statistics:"retrospective go back in time data already collected"
        - Ecology:   "feather eyes around ..."
        - CompSci:   CU / control unit description

    Returns:
        dict: { topic: example_row }
    """
    examples = {}

    # Ensure text columns are strings
    train_df = train_df.copy()
    train_df["segment_text"] = train_df["segment_text"].fillna("").astype(str)
    train_df["IdeaUnit"] = train_df["IdeaUnit"].astype(str)

    for topic, group in train_df.groupby("Topic"):
        group = group.copy()

        # Only consider rows with actual note text
        non_empty = group[group["segment_text"].str.strip() != ""]
        if len(non_empty) == 0:
            # Fallback: no usable text, just pick the first row
            examples[topic] = group.iloc[0]
            continue

        # First try positives
        pos = non_empty[non_empty["label"] == 1].copy()
        target_df = pos if len(pos) > 0 else non_empty.copy()

        # Compute lexical overlap for each candidate
        target_df["overlap"] = target_df.apply(
            lambda r: lexical_overlap_score(str(r["IdeaUnit"]), str(r["segment_text"])),
            axis=1,
        )

        # Pick the single best example (highest overlap)
        best_row = target_df.sort_values("overlap", ascending=False).iloc[0]
        examples[topic] = best_row

    return examples


# Precompute token IDs for " YES" and " NO" once (not every call)
YES_TOKEN_ID = tokenizer(" YES", add_special_tokens=False)["input_ids"][0]
NO_TOKEN_ID = tokenizer(" NO", add_special_tokens=False)["input_ids"][0]


def call_llm(prompt: str) -> int:
    """
    Classify with LLaMA 3 by comparing the logits for ' YES' vs ' NO'
    as the next token after the prompt.

    Returns:
        1 for YES,
        0 for NO.
    """
    with torch.no_grad():
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(model.device)

        outputs = model(**inputs)
        # logits for the next token (last position)
        logits_last = outputs.logits[0, -1]  # shape: [vocab_size]

    yes_logit = logits_last[YES_TOKEN_ID].item()
    no_logit = logits_last[NO_TOKEN_ID].item()

    # Higher logit = higher probability
    return 1 if yes_logit > no_logit else 0


def run_llm_classifier(
    data_dir: str,
    split: str,
    output_path: str,
    mode: str,
):
    """
    Run the LLM classifier on either the train or test split.

    Args:
        data_dir: folder with Notes.csv, train.csv, test.csv
        split: 'train' or 'test'
        mode: 'zero_shot' or 'one_shot'
        output_path: where to save the CSV with predictions
    """
    # Load raw CSVs
    notes, train_raw, test_raw = load_data(data_dir=data_dir)

    # Add 'segment_text' column to train and test using Notes.csv
    train = add_segment_text(train_raw, notes)
    test = add_segment_text(test_raw, notes)

    # Choose which split to run on
    if split == "train":
        df = train.copy()
    elif split == "test":
        df = test.copy()
    else:
        raise ValueError("split must be 'train' or 'test'")

    # Prepare one-shot examples if needed
    if mode == "one_shot":
        examples_by_topic = select_one_shot_examples(train)
    else:
        examples_by_topic = None

    preds = []
    n = len(df)

    print(f"Running LLaMA 3 classifier on {split} split with mode={mode}, n={n} rows")

    for _, row in tqdm(df.iterrows(), total=n):
        idea = str(row["IdeaUnit"])
        full_note = str(row["segment_text"])
        topic = row["Topic"]

        # Prepare one-shot example if needed
        if mode == "one_shot":
            ex = examples_by_topic[topic]
            ex_idea = str(ex["IdeaUnit"])
            ex_note = str(ex["segment_text"])
            ex_label = int(ex["label"])
        else:
            ex_idea = ex_note = None
            ex_label = 0  # dummy

        # 1) Break the note into token-level chunks
        note_chunks = chunk_note_text(full_note, max_note_tokens=256, stride=192)

        # 2) Run the LLM on each chunk and aggregate
        window_preds = []
        for note_chunk in note_chunks:
            if mode == "one_shot":
                prompt = build_prompt_one_shot(
                    idea=idea,
                    note=note_chunk,
                    ex_idea=ex_idea,
                    ex_note=ex_note,
                    ex_label=ex_label,
                )
            else:
                prompt = build_prompt_zero_shot(idea=idea, note=note_chunk)

            window_pred = call_llm(prompt)
            window_preds.append(window_pred)

        # 3) Final prediction: YES if any window says YES
        pred_label = 1 if any(p == 1 for p in window_preds) else 0
        preds.append(pred_label)

    # Attach predictions
    df["pred_llm"] = preds

    # Keep useful columns
    cols_to_keep = ["Topic", "ID", "Segment", "IdeaUnit", "pred_llm"]
    if "label" in df.columns:
        cols_to_keep.append("label")  # available for train/test if present

    out_df = df[cols_to_keep]

    # Ensure output directory exists
    parent_dir = os.path.dirname(output_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    out_df.to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path}")

    # If labels exist (train or test split with labels), print a quick accuracy
    if "label" in out_df.columns:
        acc = (out_df["pred_llm"] == out_df["label"]).mean()
        print(f"Accuracy on {split} split: {acc:.4f}")


def main():
    parser = argparse.ArgumentParser(description="LLaMA 3-based note coverage classifier")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory containing Notes.csv, train.csv, test.csv",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "test"],
        default="test",
        help="Which split to run on (train or test).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["zero_shot", "one_shot"],
        default="one_shot",
        help="LLM prompting mode.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="outputs/llama3_predictions.csv",
        help="Path to save CSV with predictions.",
    )
    args = parser.parse_args()

    run_llm_classifier(
        data_dir=args.data_dir,
        split=args.split,
        output_path=args.output_path,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()

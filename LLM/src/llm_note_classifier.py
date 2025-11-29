import argparse
import os
import re

import pandas as pd
from tqdm import tqdm
import torch

from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM

from preprocessing import load_data, add_segment_text
from prompts import build_prompt_zero_shot, build_prompt_one_shot


# Load HF token from .env

load_dotenv()  # looks for .env in project root (LLM/.env)
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

def select_one_shot_examples(train_df: pd.DataFrame) -> dict:
    """
    Select one example row per Topic from the training data.

    Strategy:
    - Group by Topic
    - Prefer a positive (label=1) example with non-empty segment_text
    - Fallback to the first non-empty note in the group
    Returns a dict: { topic: example_row }.
    """
    examples = {}
    for topic, group in train_df.groupby("Topic"):
        # Ensure segment_text is a string and non-empty
        group = group.copy()
        group["segment_text"] = group["segment_text"].fillna("").astype(str)
        non_empty = group[group["segment_text"].str.strip() != ""]

        if len(non_empty) == 0:
            # If somehow all are empty, just use the first row
            examples[topic] = group.iloc[0]
            continue

        # Prefer positive examples
        pos = non_empty[non_empty["label"] == 1]
        if len(pos) > 0:
            examples[topic] = pos.iloc[0]
        else:
            examples[topic] = non_empty.iloc[0]

    return examples


def call_llm(prompt: str, max_new_tokens: int = 8) -> int:
    """
    Send a prompt to LLaMA 3 and map YES/NO to 1/0.

    Returns:
        1 for YES,
        0 for NO (or on failure / unexpected output).
    """
    with torch.no_grad():
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    upper = text.upper()

    # Focus on what comes after "ANSWER:"
    if "ANSWER:" in upper:
        after = upper.split("ANSWER:", 1)[1]
    else:
        after = upper

    # Look for YES or NO as standalone words in the answer section
    m = re.search(r"\b(YES|NO)\b", after)
    if m:
        ans = m.group(1)
        if ans == "YES":
            return 1
        if ans == "NO":
            return 0

    # Weak fallback: search in the whole output
    if "YES" in upper and "NO" not in upper:
        return 1
    if "NO" in upper and "YES" not in upper:
        return 0

    print(f"[WARNING!] Unexpected model output: {repr(text)} -> defaulting to 0")
    return 0


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
        note = str(row["segment_text"])
        topic = row["Topic"]

        if mode == "one_shot":
            ex = examples_by_topic[topic]
            ex_idea = str(ex["IdeaUnit"])
            ex_note = str(ex["segment_text"])
            ex_label = int(ex["label"])

            prompt = build_prompt_one_shot(
                idea=idea,
                note=note,
                ex_idea=ex_idea,
                ex_note=ex_note,
                ex_label=ex_label,
            )
        else:
            prompt = build_prompt_zero_shot(idea=idea, note=note)

        pred_label = call_llm(prompt)
        preds.append(pred_label)

    # Attach predictions
    df["pred_llm"] = preds

    # Keep useful columns
    cols_to_keep = ["Topic", "ID", "Segment", "IdeaUnit", "pred_llm"]
    if "label" in df.columns:
        cols_to_keep.append("label")  # available for train/test if present

    out_df = df[cols_to_keep]
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_df.to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path}")

    # If labels exist (train split), print a quick accuracy
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

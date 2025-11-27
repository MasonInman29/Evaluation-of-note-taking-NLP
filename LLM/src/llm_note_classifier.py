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
load_dotenv()  # looks for .env in project root (LLM/.env)
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise RuntimeError("HF_TOKEN is not set. Create a .env file with HF_TOKEN=...")


# Load LLaMA 3 model
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

print(f"Loading LLaMA 3 model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=HF_TOKEN)

# half-precision on GPU, full precision on CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# For big models, is better to let HF handle device placement so (device_map="auto")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=dtype,
    device_map="auto",          # works well on Nova GPUs
    use_auth_token=HF_TOKEN,
)

model.eval()
print(f"LLaMA 3 loaded. Using device: {device}")


# Helper functions

def select_one_shot_examples(train_df: pd.DataFrame) -> dict:
    """
    Select one example row per Topic from the training data.
    Assumes train_df already has 'segment_text'.
    Returns a dict: { topic: example_row }.
    """
    examples = {}
    for _, row in train_df.iterrows():
        topic = row["Topic"]
        if topic not in examples:
            examples[topic] = row
    return examples


def call_llm(prompt: str, max_new_tokens: int = 16) -> int:
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
        )

        text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip().upper()

    first_token = text.split()[0] if text else ""

    if "YES" in first_token:
        return 1
    if "NO" in first_token:
        return 0

    print(f"[WARNING!] Unexpected model output: '{text}' -> defaulting to 0")
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
        idea = row["IdeaUnit"]
        note = row["segment_text"]
        topic = row["Topic"]

        if mode == "one_shot":
            ex = examples_by_topic[topic]
            ex_idea = ex["IdeaUnit"]
            ex_note = ex["segment_text"]
            ex_label = ex["label"]

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

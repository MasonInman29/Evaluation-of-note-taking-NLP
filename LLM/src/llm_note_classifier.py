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


# ----------------------
# Helper functions
# ----------------------

STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for",
    "is", "are", "was", "were", "be", "been", "being", "it", "this",
    "that", "these", "those", "as", "by", "at", "from", "with"
}


def chunk_note_text(note: str, max_note_tokens: int = 256, stride: int = 192) -> list[str]:
    """Split long note text into overlapping token windows."""
    enc = tokenizer(note, add_special_tokens=False, return_tensors="pt")["input_ids"][0]

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


def truncate_text(text: str, max_tokens: int = 192) -> str:
    """Truncate overly long example notes so they don't dominate the prompt."""
    enc = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]

    if len(enc) <= max_tokens:
        return text

    trunc_ids = enc[:max_tokens]
    return tokenizer.decode(trunc_ids, skip_special_tokens=True)


def lexical_overlap_score(idea: str, note: str) -> float:
    """Compute lexical overlap, ignoring stopwords."""
    idea_tokens = {
        t for t in idea.lower().replace(".", " ").replace(",", " ").split()
        if t and t not in STOPWORDS
    }
    note_tokens = {
        t for t in note.lower().replace(".", " ").replace(",", " ").split()
        if t and t not in STOPWORDS
    }

    if not idea_tokens:
        return 0.0

    return len(idea_tokens & note_tokens) / len(idea_tokens)


def select_one_shot_examples(train_df: pd.DataFrame) -> dict:
    """Select one representative example per topic, preferring clean positive examples."""
    examples = {}
    train_df = train_df.copy()
    train_df["segment_text"] = train_df["segment_text"].fillna("").astype(str)
    train_df["IdeaUnit"] = train_df["IdeaUnit"].astype(str)

    for topic, group in train_df.groupby("Topic"):
        group = group.copy()
        non_empty = group[group["segment_text"].str.strip() != ""]

        if len(non_empty) == 0:
            examples[topic] = group.iloc[0]
            continue

        pos = non_empty[non_empty["label"] == 1]
        target_df = pos if len(pos) > 0 else non_empty

        target_df = target_df.copy()
        target_df["overlap"] = target_df.apply(
            lambda r: lexical_overlap_score(r["IdeaUnit"], r["segment_text"]),
            axis=1,
        )

        examples[topic] = target_df.sort_values("overlap", ascending=False).iloc[0]

    return examples


# ----------------------
# YES / NO sequence log-probabilities
# ----------------------

def completion_logprob(prompt: str, answer_text: str, max_length: int = 1024) -> float:
    """
    Compute the log-probability of `answer_text` as a continuation of `prompt`.

    We:
    - tokenize prompt and answer separately (no special tokens)
    - truncate from the LEFT if total length > max_length (keep the answer)
    - run the model once on [prompt_ids + answer_ids]
    - sum log-softmax probabilities for the answer tokens.
    """
    # Tokenize as plain sequences (Python lists of ids)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    answer_ids = tokenizer.encode(answer_text, add_special_tokens=False)

    # If too long, truncate the prompt from the left (keep the answer intact)
    total_len = len(prompt_ids) + len(answer_ids)
    if total_len > max_length:
        overflow = total_len - max_length
        prompt_ids = prompt_ids[overflow:]

    input_ids = torch.tensor([prompt_ids + answer_ids], device=model.device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits  # [1, seq_len, vocab_size]

    # For each answer token, use the logits from the previous position
    # Example:
    # token at index L (first answer token) uses logits at position L-1
    prompt_len = len(prompt_ids)
    logprob = 0.0

    for i, token_id in enumerate(answer_ids):
        pos = prompt_len - 1 + i
        token_logits = logits[0, pos, :]  # [vocab_size]
        log_probs = torch.log_softmax(token_logits, dim=-1)
        logprob += log_probs[token_id].item()

    return logprob


def call_llm_score(prompt: str) -> float:
    """
    Return a continuous score = log p(YES | prompt) - log p(NO | prompt),
    where YES/NO are treated as full token sequences.

    Positive score => YES is more likely than NO.
    Negative score => NO is more likely.
    """
    yes_lp = completion_logprob(prompt, " YES")
    no_lp = completion_logprob(prompt, " NO")
    return yes_lp - no_lp


# ----------------------
# Main classifier logic
# ----------------------

def run_llm_classifier(data_dir: str, split: str, output_path: str, mode: str):
    """Run the LLM classifier and produce continuous scores for threshold tuning."""
    notes, train_raw, test_raw = load_data(data_dir=data_dir)

    train = add_segment_text(train_raw, notes)
    test = add_segment_text(test_raw, notes)

    df = train.copy() if split == "train" else test.copy()

    if mode == "one_shot":
        examples_by_topic = select_one_shot_examples(train)
    else:
        examples_by_topic = None

    scores = []
    n = len(df)

    print(f"Running LLaMA classifier ({mode}) on {split} split, n={n}")

    for _, row in tqdm(df.iterrows(), total=n):
        idea = str(row["IdeaUnit"])
        full_note = str(row["segment_text"])
        topic = row["Topic"]

        if mode == "one_shot":
            ex = examples_by_topic[topic]
            ex_idea = str(ex["IdeaUnit"])
            ex_note = truncate_text(str(ex["segment_text"]), max_tokens=192)
            ex_label = int(ex["label"])
        else:
            ex_idea = ex_note = None
            ex_label = 0

        note_chunks = chunk_note_text(full_note, max_note_tokens=256, stride=192)

        window_scores = []
        for chunk in note_chunks:
            if mode == "one_shot":
                prompt = build_prompt_one_shot(
                    idea=idea,
                    note=chunk,
                    ex_idea=ex_idea,
                    ex_note=ex_note,
                    ex_label=ex_label,
                )
            else:
                prompt = build_prompt_zero_shot(idea=idea, note=chunk)

            score = call_llm_score(prompt)
            window_scores.append(score)

        # Aggregate: best (max) window score
        sample_score = max(window_scores)
        scores.append(sample_score)

    df["llm_score"] = scores

    cols = ["Topic", "ID", "Segment", "IdeaUnit", "llm_score"]
    if "label" in df.columns:
        cols.append("label")

    out_df = df[cols]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_df.to_csv(output_path, index=False)

    print(f"Saved LLM scores to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="LLaMA 3-based IdeaUnit coverage scorer")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--split", type=str, choices=["train", "test"], default="test")
    parser.add_argument("--mode", type=str, choices=["zero_shot", "one_shot"], default="one_shot")
    parser.add_argument("--output_path", type=str, default="outputs/llama3_scores.csv")
    args = parser.parse_args()

    run_llm_classifier(
        data_dir=args.data_dir,
        split=args.split,
        output_path=args.output_path,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()

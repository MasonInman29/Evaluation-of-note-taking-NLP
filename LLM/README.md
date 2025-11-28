# LLM-Based Note Classification -- COM S 5790 Final Project

This README documents the full setup, environment creation, model usage,
and evaluation process for the LLM-based model used in the
Evaluation of Note Taking and Short Answer Generation project.

------------------------------------------------------------------------

# Overview

This component of the project implements an LLM-based classifier
that determines whether a student's note segment covers a specific Idea
Unit provided by the instructor.

Using:

-   LLaMA 3 8B Instruct as thw main LLM
-   Custom 0-shot and 1-shot prompts
-   HuggingFace access via `.env` tokens
-   A reproducible Python pipeline

You can run the entire model on Nova.

------------------------------------------------------------------------

# Project Structure

    LLM/
    ├── data/
    │   ├── Notes.csv
    │   ├── train.csv
    │   └── test.csv
    │
    ├── src/
    │   ├── preprocessing.py
    │   ├── prompts.py
    │   ├── llm_note_classifier.py
    │   └── evaluate_llm.py
    │
    ├── outputs/
    │   ├── llama3_train_one_shot.csv
    │   ├── llama3_test_one_shot.csv
    │   └── ...
    │
    ├── .env                      # contains HF_TOKEN (not tracked by git)
    ├── requirements.txt
    └── README.md                 # this file

------------------------------------------------------------------------

# Requirements & Dependencies

Content of `requirements.txt`:

    pandas
    python-dotenv
    tqdm
    transformers>=4.40
    accelerate
    sentencepiece
    torch
    scikit-learn        # optional for evaluation

### What each dependency does

  Package         Purpose
  --------------- --------------------------------------------
  pandas          CSV loading & manipulation
  python-dotenv   Loads `HF_TOKEN` from `.env`
  tqdm            Progress bars
  transformers    LLaMA model loading
  accelerate      Optimized inference on Nova GPUs
  sentencepiece   Tokenizer support
  torch           Needed for all LLM inference
  scikit-learn    Evaluation metrics

Install all requirements:

``` bash
python -m pip install -r requirements.txt
```

------------------------------------------------------------------------

# Setting Up `.env`

LLaMA 3 is gated by Meta. So to use it requesting access on HuggingFace is necessary.
Please contact us if the hf token is needed.
create:

    LLM/.env

Inside:

    HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx

WARNING! Do not upload `.env` to GitHub.\
Ensure `.gitignore` contains:

    .env

------------------------------------------------------------------------

# Data Description

### `Notes.csv`

Contains full student notes segmented by lecture section:

    Experiment,Topic,ID,Segment1_Notes,Segment2_Notes,Segment3_Notes,Segment4_Notes

### `train.csv`

Supervised labels:

    Topic,ID,Segment,IdeaUnit,label

### `test.csv`

Same structure (sometimes includes labels depending on the dataset
version).

------------------------------------------------------------------------

# Setting Up the Environment

### 1. Create a Python virtual environment

``` bash
python3 -m venv LLM-venv
source LLM-venv/bin/activate
```

### 2. Install dependencies

``` bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Confirm token loading

``` bash
python - << 'EOF'
from dotenv import load_dotenv
import os
load_dotenv()
print("HF_TOKEN loaded?", os.getenv("HF_TOKEN") is not None)
EOF
```

------------------------------------------------------------------------

# Source Code Summary

## `preprocessing.py`

-   Loads Notes/train/test CSV files
-   Merges notes with IdeaUnits based on `(Topic, ID)`
-   Extracts the correct `SegmentX_Notes` into a new column:
    `segment_text`

## `prompts.py`

Defines two prompting styles:

### Zero-shot

Model sees: - Instructions
- IdeaUnit
- Student note

### One-shot

Same as above but adds a labeled example from the same topic.

## `llm_note_classifier.py`

Main pipeline:

-   Loads LLaMA 3 using HF token
-   Builds prompt per row
-   Calls LLM → YES/NO → converts to 1/0
-   Saves predictions into CSV
-   Computes accuracy when labels exist

## `evaluate_llm.py`

Computes:

-   Accuracy
-   Precision
-   Recall
-   F1-score
-   Confusion matrix

------------------------------------------------------------------------

# Running the Model (Local or Nova)

## 1. Run on train set (validation)

``` bash
python src/llm_note_classifier.py   --data_dir data   --split train   --mode one_shot   --output_path outputs/llama3_train_one_shot.csv
```

Expected output:

    LLaMA 3 loaded...
    Running LLaMA 3 classifier...
    Accuracy on train split: 0.xxx

## 2. Run on test set

``` bash
python src/llm_note_classifier.py   --data_dir data   --split test   --mode one_shot   --output_path outputs/llama3_test_one_shot.csv
```

------------------------------------------------------------------------

# Evaluation Script

Example:

``` bash
python src/evaluate_llm.py   --pred_path outputs/llama3_train_one_shot.csv
```

Outputs:

-   Accuracy
-   Precision
-   Recall
-   F1
-   Confusion matrix
-   Full classification report

------------------------------------------------------------------------

# The LLM section of the final report includes:

-   Model used: LLaMA 3 8B Instruct
-   Prompting methods: 0-shot, 1-shot
-   Accuracy on train/dev
-   Precision/Recall/F1
-   Error patterns:
    -   paraphrasing
    -   notes using synonyms
    -   overly literal false negatives
-   Effectiveness of one-shot example selection

------------------------------------------------------------------------

# Final Notes

This LLM pipeline:

-   Is fully reproducible
-   Works with LLaMA
-   Runs on CPU/GPU/Nova
-   Provides interpretable YES/NO outputs
-   Evaluation of results

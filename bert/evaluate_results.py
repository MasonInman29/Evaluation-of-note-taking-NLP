import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
)
from datetime import datetime

sns.set(style="whitegrid")

# RESULTS_DIR = "results"
RESULTS_DIR = "results_temporal_8020_4_align3/seed_42"
OUT_DIR = os.path.join(RESULTS_DIR, "eval_output")
os.makedirs(OUT_DIR, exist_ok=True)

# ------------- helper functions to find files/columns -------------
def find_prediction_csv(results_dir=RESULTS_DIR):
    # prefer explicit filename if present
    prefer = ["bert_predictions.csv", "predictions.csv", "results.csv", "bert_eval_predictions.csv"]
    for p in prefer:
        pth = os.path.join(results_dir, p)
        if os.path.exists(pth):
            return pth
    # otherwise find any csv in results dir
    csvs = glob.glob(os.path.join(results_dir, "*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSVs found in {results_dir}")
    # choose the most recently modified CSV
    csvs_sorted = sorted(csvs, key=os.path.getmtime, reverse=True)
    return csvs_sorted[0]

def infer_label_columns(df):
    # common names for ground truth and prediction columns
    label_cols = ["label", "true_label", "y_true", "ground_truth"]
    pred_cols  = ["prediction", "pred", "pred_label", "y_pred", "yhat"]
    prob_cols1 = ["prob_1", "p1", "score_1", "proba_1", "probability_1"]
    prob_cols0 = ["prob_0", "p0", "score_0", "proba_0"]
    logits_cols = ["logits", "predictions"]  # maybe stored as arrays

    found = {}
    for c in label_cols:
        if c in df.columns:
            found["label"] = c
            break
    for c in pred_cols:
        if c in df.columns:
            found["pred"] = c
            break

    # try probability of positive class
    for c in prob_cols1:
        if c in df.columns:
            found["prob_pos"] = c
            break

    # sometimes columns store a list/array as string '[-1.2, 2.3]'
    for c in df.columns:
        if c.lower().startswith("logit") or c.lower().startswith("score") or c.lower().startswith("pred"):
            if df[c].dtype == object:
                # try detect array-like strings
                sample = df[c].dropna().astype(str).iloc[0] if len(df[c].dropna())>0 else ""
                if sample.startswith("[") and "," in sample:
                    found.setdefault("logits_col", c)
                    break

    # fallback: maybe probs are in two columns like prob_0 and prob_1
    for p0 in prob_cols0:
        for p1 in prob_cols1:
            if p0 in df.columns and p1 in df.columns:
                found["prob_pos"] = p1
                found["prob_neg"] = p0
                break
        if "prob_pos" in found:
            break

    return found

def parse_logits_array(series):
    # convert strings like "[ -1.23, 2.45 ]" to numpy arrays
    parsed = []
    for v in series:
        if pd.isna(v):
            parsed.append(None)
            continue
        if isinstance(v, (list, tuple, np.ndarray)):
            parsed.append(np.array(v, dtype=float))
        else:
            s = str(v).strip()
            if s.startswith("[") and s.endswith("]"):
                inner = s[1:-1].strip()
                parts = [p.strip() for p in inner.split(",") if p.strip()!=""]
                try:
                    arr = np.array([float(x) for x in parts], dtype=float)
                except:
                    arr = None
                parsed.append(arr)
            else:
                parsed.append(None)
    return parsed

# ------------------ load predictions ------------------
csv_path = find_prediction_csv()
print("Using prediction CSV:", csv_path)
df = pd.read_csv(os.path.join(RESULTS_DIR,"bert_eval_predictions.csv"),
                 engine="python",
                 encoding="utf-8",
                 on_bad_lines="skip")

# make a copy to avoid mutating
df = df.copy()
cols_before = df.columns.tolist()

# Try to infer label/pred/prob columns
found = infer_label_columns(df)
print("Inferred columns:", found)

# If no label column found, look for numeric columns likely to be label
if "label" not in found:
    for guess in ["label", "y_true", "true", "ground_truth", "gt"]:
        if guess in df.columns:
            found["label"] = guess
            break

if "pred" not in found:
    for guess in ["prediction", "pred", "y_pred", "predicted"]:
        if guess in df.columns:
            found["pred"] = guess
            break

if "label" not in found or "pred" not in found:
    # try scanning for columns with only 0/1 values
    for c in df.columns:
        vals = pd.Series(df[c].dropna().unique())
        # if numeric and contains only 0/1
        if pd.api.types.is_numeric_dtype(df[c]) and set(vals.tolist()).issubset({0,1}):
            if "label" not in found:
                found["label"] = c
            elif "pred" not in found:
                # prefer not to overwrite label
                if c != found["label"]:
                    found["pred"] = c
    # if still missing, try object columns with 'Yes'/'No' or 'yes'/'no'
    for c in df.columns:
        if c in found.values():
            continue
        vals = set([str(x).lower() for x in df[c].dropna().unique()])
        if vals.issubset({"yes","no","true","false","1","0"}):
            if "label" not in found:
                found["label"] = c
            elif "pred" not in found:
                found["pred"] = c

if "label" not in found or "pred" not in found:
    raise ValueError(f"Could not infer label/pred columns. Columns available: {cols_before}")

# Normalize label/pred to numeric 0/1
def normalize_to_binary(series):
    s = series.copy()
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(int)
    # try mapping strings
    s = s.astype(str).str.strip().str.lower()
    s = s.replace({"yes":"1","true":"1","y":"1","no":"0","false":"0","n":"0"})
    # remove trailing .0 "1.0"
    s = s.str.replace(r"\.0$", "", regex=True)
    return s.astype(int)

# --- robust coercion and drop-missing block ---
import numpy as np

# column names inferred earlier in your script:
label_col = found.get("label")
pred_col  = found.get("pred")
prob_col  = found.get("prob_pos", None)

print(f"Found label column: {label_col}, pred column: {pred_col}, prob column: {prob_col}")

# 1) coercion to numeric with safety
def coerce_binary(series, name):
    # first try numeric coercion (handles '1.0' and '0.0')
    s_num = pd.to_numeric(series, errors="coerce")
    # if result non-null and values are 0/1 or 0.0/1.0, keep them
    # otherwise try mapping common strings
    if s_num.isin([0,1]).sum() < 1:
        s_str = series.astype(str).str.strip().str.lower().replace({
            "yes":"1", "y":"1", "true":"1", "t":"1",
            "no":"0", "n":"0", "false":"0", "f":"0",
            "nan": ""
        })
        s_num = pd.to_numeric(s_str, errors="coerce")
    # final series of ints where possible
    return s_num

# Coerce both columns
s_label = coerce_binary(df[label_col], label_col)
s_pred  = coerce_binary(df[pred_col], pred_col)

# Show counts before cleaning
print("Before cleaning: total rows =", len(df))
print("label column counts (including NaN):\n", s_label.value_counts(dropna=False).to_string())
print("pred column counts (including NaN):\n", s_pred.value_counts(dropna=False).to_string())

# Drop rows with missing ground-truth label (can't evaluate)
mask_valid_label = s_label.notna()
n_dropped = (~mask_valid_label).sum()
if n_dropped > 0:
    print(f"Dropping {n_dropped} rows with missing ground-truth '{label_col}' (cannot evaluate those).")
df = df.loc[mask_valid_label].reset_index(drop=True)
s_label = s_label.loc[mask_valid_label].reset_index(drop=True)
s_pred  = s_pred.loc[mask_valid_label].reset_index(drop=True)

# For any remaining missing predictions, you can either drop them or fill with 0/ -1.
# We'll drop rows missing predictions for clean evaluation:
mask_valid_pred = s_pred.notna()
n_pred_missing = (~mask_valid_pred).sum()
if n_pred_missing > 0:
    print(f"Dropping {n_pred_missing} rows with missing prediction '{pred_col}'.")
df = df.loc[mask_valid_pred].reset_index(drop=True)
s_label = s_label.loc[mask_valid_pred].reset_index(drop=True)
s_pred  = s_pred.loc[mask_valid_pred].reset_index(drop=True)

# Convert to int (safe now)
y_true = s_label.astype(int)
y_pred  = s_pred.astype(int)

# Attach cleaned labels/preds back to df for aligned examples and saving
df["_eval_true_label"] = y_true
df["_eval_pred_label"]  = y_pred

# Save cleaned CSV for inspection
cleaned_path = os.path.join(OUT_DIR, "bert_test_predictions.cleaned.csv")
df.to_csv(cleaned_path, index=False)
print("Saved cleaned and aligned CSV to:", cleaned_path)

# If prob column exists, coerce to numeric as well (for ROC / PR)
y_scores = None
if prob_col and prob_col in df.columns:
    y_scores = pd.to_numeric(df[prob_col], errors="coerce")
    # If probabilities in 0-100 range, scale them
    if y_scores.notna().any() and y_scores.max() > 1.5:
        y_scores = y_scores / 100.0
    # align to df index (already aligned)
    y_scores = y_scores.loc[df.index].values


# Try get positive-class probability
y_scores = None
if "prob_pos" in found:
    try:
        y_scores = pd.to_numeric(df[found["prob_pos"]], errors="coerce").values
        # if percent-like (0-100) scale, convert to 0-1
        if np.nanmax(y_scores) > 1.5:
            y_scores = y_scores / 100.0
    except Exception as e:
        print("Could not convert prob_pos to numeric:", e)
        y_scores = None

# Try to parse logits-like column to compute positive class prob via softmax
if y_scores is None and "logits_col" in found:
    parsed = parse_logits_array(df[found["logits_col"]])
    # find first non-None array length to see if binary
    if any(isinstance(x, np.ndarray) for x in parsed):
        arrs = [x for x in parsed if isinstance(x, np.ndarray)]
        arr = np.stack(arrs)  # n x C
        # compute softmax for positive class index (assume index 1)
        def softmax(z):
            e = np.exp(z - np.max(z, axis=1, keepdims=True))
            return e / e.sum(axis=1, keepdims=True)
        probs = softmax(arr)
        # map back: for rows where parsed is None set nan, else set probs[:,1]
        y_scores = []
        it = 0
        for v in parsed:
            if isinstance(v, np.ndarray):
                if v.size == probs.shape[1]:
                    y_scores.append(probs[it, 1] if probs.shape[1] > 1 else probs[it, 0])
                else:
                    y_scores.append(np.nan)
                it += 1
            else:
                y_scores.append(np.nan)
        y_scores = np.array(y_scores)

# Final fallback: if model outputs raw logits in separate .npy in results dir
if y_scores is None:
    # try to find .npy named logits or pred_probs or predictions
    for pat in ["*logits*.npy", "*predictions*.npy", "*probs*.npy", "*preds*.npy"]:
        fls = glob.glob(os.path.join(RESULTS_DIR, pat))
        if fls:
            try:
                arr = np.load(fls[0], allow_pickle=True)
                if arr.ndim == 2 and arr.shape[1] >= 2:
                    # compute softmax
                    e = np.exp(arr - np.max(arr, axis=1, keepdims=True))
                    probs = e / e.sum(axis=1, keepdims=True)
                    y_scores = probs[:, 1]
                    print("Loaded logits/probs from", fls[0])
                    break
            except Exception as e:
                pass

# ------------- compute metrics -------------
acc = accuracy_score(y_true, y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
clf_report = classification_report(y_true, y_pred, digits=4)

print("\n=== Overall metrics ===")
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1: {f1:.4f}")
print("\nClassification report:\n", clf_report)

# save metrics summary
metrics_summary = {
    "accuracy": acc,
    "precision": prec,
    "recall": rec,
    "f1": f1,
    "n_samples": int(len(y_true))
}
pd.DataFrame([metrics_summary]).to_csv(os.path.join(OUT_DIR, "metrics_summary.csv"), index=False)

# ------------- confusion matrix plot -------------
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["pred_0","pred_1"], yticklabels=["true_0","true_1"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
conf_mat_path = os.path.join(OUT_DIR, "confusion_matrix.png")
plt.tight_layout()
plt.savefig(conf_mat_path)
plt.close()
print("Saved confusion matrix to", conf_mat_path)

# ------------- confidence histogram & pr/roc if available -------------
if y_scores is not None and not np.all(np.isnan(y_scores)):
    # ensure same length
    y_scores = np.asarray(y_scores).astype(float)
    # clip nan
    valid_idx = ~np.isnan(y_scores)
    if valid_idx.sum() > 0:
        # histogram
        plt.figure(figsize=(6,3))
        plt.hist(y_scores[valid_idx], bins=20, edgecolor="k")
        plt.xlabel("Predicted probability (positive class)")
        plt.ylabel("Count")
        plt.title("Confidence distribution (positive class probability)")
        hist_path = os.path.join(OUT_DIR, "confidence_histogram.png")
        plt.tight_layout()
        plt.savefig(hist_path)
        plt.close()
        print("Saved confidence histogram to", hist_path)

        # Precision-Recall
        precs, recalls, pr_thresholds = precision_recall_curve(y_true[valid_idx], y_scores[valid_idx])
        pr_auc = auc(recalls, precs)
        plt.figure(figsize=(5,4))
        plt.plot(recalls, precs, label=f"PR AUC={pr_auc:.3f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend()
        pr_path = os.path.join(OUT_DIR, "precision_recall_curve.png")
        plt.tight_layout()
        plt.savefig(pr_path)
        plt.close()
        print("Saved PR curve to", pr_path)

        # ROC
        fpr, tpr, roc_thresholds = roc_curve(y_true[valid_idx], y_scores[valid_idx])
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(5,4))
        plt.plot(fpr, tpr, label=f"ROC AUC={roc_auc:.3f}")
        plt.plot([0,1],[0,1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        roc_path = os.path.join(OUT_DIR, "roc_curve.png")
        plt.tight_layout()
        plt.savefig(roc_path)
        plt.close()
        print("Saved ROC curve to", roc_path)
else:
    print("No probability/score information available — skipping PR/ROC/confidence plots.")

# ------------- produce examples for error analysis -------------
# create a unified examples df containing ideaunit and note text if present
# try to find columns named IdeaUnit and NoteText in the CSV if they exist
cols = df.columns.tolist()
example_cols = []
for cand in ["IdeaUnit", "ideaunit", "idea_unit", "Idea Unit"]:
    if cand in cols:
        example_cols.append(cand)
for cand in ["NoteText", "note", "Note", "Segment1_Notes", "Segment2_Notes", "Segment3_Notes", "Segment4_Notes", "note_text"]:
    if cand in cols:
        example_cols.append(cand)

# fallback: use first two text-like columns that are long strings
if not example_cols:
    # find object dtype columns and choose top2 by avg length
    text_cols = [c for c in cols if df[c].dtype == object]
    if len(text_cols) >= 2:
        avg_len = {c: df[c].astype(str).str.len().mean() for c in text_cols}
        sorted_cols = sorted(avg_len.items(), key=lambda x: x[1], reverse=True)
        example_cols = [sorted_cols[0][0], sorted_cols[1][0]]
    elif text_cols:
        example_cols = [text_cols[0]]

# build examples DataFrame
examples = pd.DataFrame({
    "true_label": y_true,
    "pred_label": y_pred
})
# add scores if available
if y_scores is not None:
    examples["pred_prob_pos"] = y_scores

# attach text columns if detected
for i, c in enumerate(example_cols[:2]):
    examples[f"text_{i+1}"] = df[c].astype(str)

# add original index/ID if present
if "ID" in df.columns:
    examples["ID"] = df["ID"]
if "Segment" in df.columns:
    examples["Segment"] = df["Segment"]

# add correctness flag
examples["correct"] = (examples["true_label"] == examples["pred_label"])

# Save full examples CSV
examples.to_csv(os.path.join(OUT_DIR, "examples_all.csv"), index=False)

# extract top false positives and false negatives with highest confidence
fp = examples[(examples["true_label"]==0) & (examples["pred_label"]==1)].copy()
fn = examples[(examples["true_label"]==1) & (examples["pred_label"]==0)].copy()

# if pred_prob_pos present, sort by descending confidence
if "pred_prob_pos" in examples.columns:
    fp = fp.sort_values("pred_prob_pos", ascending=False)
    fn = fn.sort_values("pred_prob_pos", ascending=True)  # low prob but actually pos -> show most confident negatives
else:
    # fallback: random sample
    fp = fp.sample(frac=1).reset_index(drop=True)
    fn = fn.sample(frac=1).reset_index(drop=True)

fp_head = fp.head(50)
fn_head = fn.head(50)
fp_head.to_csv(os.path.join(OUT_DIR, "false_positives_top.csv"), index=False)
fn_head.to_csv(os.path.join(OUT_DIR, "false_negatives_top.csv"), index=False)

print(f"Saved full examples to {os.path.join(OUT_DIR, 'examples_all.csv')}")
print(f"Saved top false positives to {os.path.join(OUT_DIR, 'false_positives_top.csv')}")
print(f"Saved top false negatives to {os.path.join(OUT_DIR, 'false_negatives_top.csv')}")

# ------------- generate a simple HTML report -------------
html_path = os.path.join(OUT_DIR, "evaluation_report.html")
with open(html_path, "w", encoding="utf8") as f:
    f.write("<html><head><meta charset='utf-8'><title>Evaluation Report</title></head><body>")
    f.write(f"<h1>Evaluation report ({datetime.utcnow().isoformat()} UTC)</h1>")
    f.write("<h2>Summary metrics</h2>")
    f.write(pd.DataFrame([metrics_summary]).to_html(index=False))
    f.write("<h2>Confusion matrix</h2>")
    f.write(f"<img src='confusion_matrix.png' style='max-width:700px;'>")
    if y_scores is not None and not np.all(np.isnan(y_scores)):
        f.write("<h2>Confidence & curves</h2>")
        f.write(f"<img src='confidence_histogram.png' style='max-width:700px;'><br>")
        f.write(f"<img src='precision_recall_curve.png' style='max-width:700px;'><br>")
        f.write(f"<img src='roc_curve.png' style='max-width:700px;'><br>")
    f.write("<h2>Top false positives (CSV)</h2>")
    f.write("<p>See false_positives_top.csv and false_negatives_top.csv in this folder for examples.</p>")
    f.write("<h2>Examples (first 20)</h2>")
    f.write(examples.head(20).to_html(escape=False))
    f.write("</body></html>")

print("Saved HTML report to", html_path)
print("All done. Check folder:", OUT_DIR)
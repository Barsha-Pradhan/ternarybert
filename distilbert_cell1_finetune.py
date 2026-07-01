# ============================================================
# CELL 1 — DistilBERT fine-tuning across GLUE (Stage A only)
#
# Scope: standard full fine-tuning of distilbert-base-uncased on 8 of 9
# GLUE tasks (WNLI excluded — see note below). NOT quantized. This is a
# clean baseline before we decide how to handle TernaryBERT-style
# quantization for DistilBERT (the official Huawei quant_task_glue.py /
# modeling_quant.py only supports BERT's architecture, not DistilBERT's
# 6-layer / no-token-type-id structure, so that step is deferred).
#
# Config (as specified):
#   optimizer: AdamW, weight_decay: 0.01, batch_size: 16, lr: 2e-5,
#   warmup_ratio: 10%, scheduler: linear decay, epochs: 5,
#   max_seq_length: 128, seed: 42
#
# WNLI is excluded to stay consistent with the TernaryBERT paper's own
# results table and with your existing BERT-base pipeline in this repo
# (the official codebase has no WnliProcessor). If you want WNLI numbers
# for DistilBERT specifically, that's a separate small addition — it
# doesn't depend on the official repo since this stage doesn't call it.
#
# Resumable: re-running this cell skips any task whose checkpoint
# already exists in Drive, and skips epochs already completed for a
# task via a per-task epoch marker.
# ============================================================

import os, sys, time, json, csv, gc, random
from google.colab import drive
drive.mount('/content/drive')

PROJECT_ROOT = "/content/drive/MyDrive/distilbert_glue"
for sub in ["data", "finetuned_models", "results", "logs"]:
    os.makedirs(f"{PROJECT_ROOT}/{sub}", exist_ok=True)

os.system("pip install -q datasets transformers torch scipy scikit-learn")

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    DistilBertTokenizerFast, DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, matthews_corrcoef
from scipy.stats import pearsonr, spearmanr

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type != "cuda":
    print("⚠️  No GPU detected — go to Runtime > Change runtime type > GPU before proceeding.")
else:
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    # T4 has limited fp32 throughput; use fp16 mixed precision automatically on T4/older GPUs.
    USE_FP16 = True
    print("Mixed precision (fp16) training: enabled")

# ------------------------------------------------------------
# DATA PREP — same verified TSV generation as cell1_pipeline.py,
# but we don't actually need TSVs here since we load directly from
# the HF `datasets` GLUE loader for the Stage A fine-tune (this mirrors
# what cell1_pipeline.py's finetune_bert_base() already does — the TSVs
# in that script exist for Stage B / the official script's consumption,
# not for Stage A fine-tuning itself).
# ------------------------------------------------------------

TASK_HF_MAP = {
    # task_name: (hf_subset, num_labels, is_regression, (text_key1, text_key2))
    "cola":  ("cola", 2, False, ("sentence", None)),
    "mnli":  ("mnli", 3, False, ("premise", "hypothesis")),
    "mrpc":  ("mrpc", 2, False, ("sentence1", "sentence2")),
    "sst-2": ("sst2", 2, False, ("sentence", None)),
    "sts-b": ("stsb", 1, True,  ("sentence1", "sentence2")),
    "qqp":   ("qqp", 2, False, ("question1", "question2")),
    "qnli":  ("qnli", 2, False, ("question", "sentence")),
    "rte":   ("rte", 2, False, ("sentence1", "sentence2")),
}

TASK_DISPLAY_NAMES = {
    "sst-2": "SST2", "qnli": "QNLI", "qqp": "QQP", "mnli": "MNLI",
    "mrpc": "MRPC", "sts-b": "STS-B", "rte": "RTE", "cola": "COLA",
}

MODEL_DIR_ROOT = f"{PROJECT_ROOT}/finetuned_models"
RESULTS_FILE = f"{PROJECT_ROOT}/results/finetune_results.json"
PROGRESS_FILE = f"{PROJECT_ROOT}/results/progress.json"

MAX_LEN = 128          # as specified
BATCH_SIZE = 16        # as specified
LR = 2e-5              # as specified
WEIGHT_DECAY = 0.01    # as specified
EPOCHS = 5             # as specified
WARMUP_RATIO = 0.10    # as specified

# ------------------------------------------------------------
# Eval metric computation per task, mirroring cell2_metrics.py's
# convention (STS-B reports Pearson/Spearman + their avg as the
# Accuracy/Precision/Recall/F1 proxy columns).
# ------------------------------------------------------------

def compute_metrics(task_name, preds, labels, is_regression):
    if is_regression:
        preds_flat = np.squeeze(preds)
        pearson = pearsonr(preds_flat, labels)[0]
        spearman = spearmanr(preds_flat, labels)[0]
        accuracy = (pearson + spearman) / 2
        precision, recall = pearson, spearman
        f1 = 2 * pearson * spearman / (pearson + spearman + 1e-8)
        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
    else:
        preds_cls = np.argmax(preds, axis=1)
        accuracy = accuracy_score(labels, preds_cls)
        num_labels = len(set(labels.tolist()))
        avg_method = "binary" if num_labels == 2 else "macro"
        try:
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels, preds_cls, average=avg_method, zero_division=0)
        except Exception:
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels, preds_cls, average="macro", zero_division=0)
        result = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
        if task_name == "cola":
            result["mcc"] = matthews_corrcoef(labels, preds_cls)
        return result

# ------------------------------------------------------------
# Fine-tune one task
# ------------------------------------------------------------

def finetune_distilbert(task_name):
    out_dir = f"{MODEL_DIR_ROOT}/{task_name}"
    done_marker = f"{out_dir}/DONE.marker"
    if os.path.exists(done_marker):
        print(f"  [{task_name}] already fine-tuned, skipping.")
        return json.load(open(f"{out_dir}/eval_metrics.json"))

    hf_name, num_labels, is_regression, (k1, k2) = TASK_HF_MAP[task_name]
    print(f"\n  [{task_name}] Fine-tuning DistilBERT-base ({hf_name})...")

    ds = load_dataset("glue", hf_name)
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=num_labels,
        problem_type="regression" if is_regression else "single_label_classification"
    ).to(device)

    def tokenize_batch(batch):
        if k2 is None:
            return tokenizer(batch[k1], truncation=True, max_length=MAX_LEN, padding="max_length")
        return tokenizer(batch[k1], batch[k2], truncation=True, max_length=MAX_LEN, padding="max_length")

    train_ds = ds["train"].map(tokenize_batch, batched=True)
    eval_split = "validation_matched" if task_name == "mnli" else "validation"
    eval_ds = ds[eval_split].map(tokenize_batch, batched=True)

    cols = ["input_ids", "attention_mask", "label"]  # DistilBERT has no token_type_ids
    train_ds.set_format(type="torch", columns=[c for c in cols if c in train_ds.column_names])
    eval_ds.set_format(type="torch", columns=[c for c in cols if c in eval_ds.column_names])

    g = torch.Generator()
    g.manual_seed(SEED)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=g)
    eval_loader = DataLoader(eval_ds, batch_size=BATCH_SIZE, shuffle=False)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": WEIGHT_DECAY},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=LR)

    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    log_path = f"{PROJECT_ROOT}/logs/{task_name}.log"
    epoch_marker_path = f"{out_dir}/last_epoch.json"
    start_epoch = 0
    if os.path.exists(epoch_marker_path):
        os.makedirs(out_dir, exist_ok=True)
        model.load_state_dict(torch.load(f"{out_dir}/partial_state.pt", map_location=device))
        start_epoch = json.load(open(epoch_marker_path))["epoch"] + 1
        print(f"  [{task_name}] resuming from epoch {start_epoch+1}/{EPOCHS}")

    model.train()
    for epoch in range(start_epoch, EPOCHS):
        t0 = time.time()
        total_loss = 0.0
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["label"].float().to(device) if is_regression else batch["label"].to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
                loss = out.loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.item()
            if step % 200 == 0:
                msg = f"    epoch {epoch+1}/{EPOCHS} step {step}/{len(train_loader)} loss={loss.item():.4f}"
                print(msg)

        avg_loss = total_loss / len(train_loader)
        elapsed = time.time() - t0
        print(f"  [{task_name}] epoch {epoch+1}/{EPOCHS} done in {elapsed:.0f}s, avg loss {avg_loss:.4f}")

        os.makedirs(out_dir, exist_ok=True)
        torch.save(model.state_dict(), f"{out_dir}/partial_state.pt")
        json.dump({"epoch": epoch}, open(epoch_marker_path, "w"))

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["label"]
            out = model(input_ids=input_ids, attention_mask=attn)
            all_preds.append(out.logits.detach().cpu().numpy())
            all_labels.append(labels.numpy())
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    metrics = compute_metrics(task_name, all_preds, all_labels, is_regression)
    print(f"  [{task_name}] eval metrics: { {k: round(v,4) for k,v in metrics.items()} }")

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    json.dump(metrics, open(f"{out_dir}/eval_metrics.json", "w"))
    open(done_marker, "w").write("done")
    # clean up resume artifacts now that the task is fully done
    for f in ["partial_state.pt", "last_epoch.json"]:
        p = f"{out_dir}/{f}"
        if os.path.exists(p):
            os.remove(p)

    print(f"  ✅ [{task_name}] fine-tuned checkpoint + metrics saved to {out_dir}\n")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return metrics

# ------------------------------------------------------------
# RUN — small tasks first so you get real numbers quickly even if a
# later large task (MNLI/QQP) gets cut off by a Colab disconnect.
# ------------------------------------------------------------

TASK_ORDER = ["rte", "mrpc", "cola", "sts-b", "sst-2", "qnli", "qqp", "mnli"]

progress = json.load(open(PROGRESS_FILE)) if os.path.exists(PROGRESS_FILE) else {}
all_results = json.load(open(RESULTS_FILE)) if os.path.exists(RESULTS_FILE) else {}

for task in TASK_ORDER:
    print(f"\n{'='*70}\n>>> TASK: {task}\n{'='*70}")
    try:
        metrics = finetune_distilbert(task)
        all_results[task] = metrics
        progress[task] = "done"
    except Exception as e:
        print(f"  ⚠️  [{task}] failed: {e}")
        progress[task] = "failed"
    json.dump(progress, open(PROGRESS_FILE, "w"))
    json.dump(all_results, open(RESULTS_FILE, "w"))

print("\n\n" + "#" * 70)
print("FINE-TUNING STATUS")
print("#" * 70)
print(json.dumps(progress, indent=2))
print("\nNOTE: WNLI intentionally excluded -- consistent with the TernaryBERT")
print("paper's own table and the rest of this repo's pipeline.")
print(f"\nAll checkpoints + per-task eval_metrics.json saved under: {MODEL_DIR_ROOT}")
print("Re-run this SAME cell after any disconnect — it resumes per-task")
print("(mid-task: from last completed epoch; across tasks: skips DONE ones).")

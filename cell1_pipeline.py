# ============================================================
# CELL 1 — TernaryBERT pipeline: data prep + fine-tune + quantize
# Official Huawei repo (github.com/huawei-noah/Pretrained-Language-Model).
# Verified against actual repo source (utils_glue.py processors,
# quant_task_glue.py argparse + internals, modeling_quant.py classes).
#
# Supports 8 of 9 GLUE tasks. WNLI excluded: the official script has no
# WnliProcessor (confirmed by reading utils_glue.py/quant_task_glue.py) —
# this matches the original TernaryBERT paper, which also excludes WNLI.
#
# Resumable: re-run this same cell after any disconnect. Completed work
# is skipped automatically via checkpoint-existence checks.
# ============================================================

import os, sys, subprocess, time, json, csv, gc
from google.colab import drive
drive.mount('/content/drive')

PROJECT_ROOT = "/content/drive/MyDrive/ternarybert_glue"
for sub in ["data", "finetuned_models", "ternary_models", "results", "logs"]:
    os.makedirs(f"{PROJECT_ROOT}/{sub}", exist_ok=True)

REPO_DIR = "/content/Pretrained-Language-Model"
TB_DIR = f"{REPO_DIR}/TernaryBERT"
if not os.path.exists(TB_DIR):
    subprocess.run(f"git clone https://github.com/huawei-noah/Pretrained-Language-Model.git {REPO_DIR}", shell=True)

os.system("pip install -q datasets transformers torch scipy scikit-learn")

from datasets import load_dataset
import torch
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type != "cuda":
    print("⚠️  No GPU detected — go to Runtime > Change runtime type > GPU before proceeding.")

# ------------------------------------------------------------
# DATA PREP: TSVs at {data_dir}/{lowercase_task}/, exact column layout
# verified from utils_glue.py's real Processor classes.
# ------------------------------------------------------------

DATA_ROOT = f"{PROJECT_ROOT}/data"

def w(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f, delimiter="\t").writerows(rows)

def gen_sst2():
    d = f"{DATA_ROOT}/sst-2"
    if os.path.exists(f"{d}/train.tsv"): return
    ds = load_dataset("glue", "sst2")
    for split, name in [("train","train"), ("validation","dev")]:
        rows = [["sentence", "label"]] + [[ex["sentence"], ex["label"]] for ex in ds[split]]
        w(f"{d}/{name}.tsv", rows)

def gen_mrpc():
    d = f"{DATA_ROOT}/mrpc"
    if os.path.exists(f"{d}/train.tsv"): return
    ds = load_dataset("glue", "mrpc")
    for split, name in [("train","train"), ("validation","dev")]:
        rows = [["Quality", "#1 ID", "#2 ID", "#1 String", "#2 String"]]
        for ex in ds[split]:
            rows.append([ex["label"], 0, 0, ex["sentence1"], ex["sentence2"]])
        w(f"{d}/{name}.tsv", rows)

def gen_cola():
    d = f"{DATA_ROOT}/cola"
    if os.path.exists(f"{d}/train.tsv"): return
    ds = load_dataset("glue", "cola")
    for split, name in [("train","train"), ("validation","dev")]:
        rows = [["dummy_src", ex["label"], "dummy_note", ex["sentence"]] for ex in ds[split]]
        w(f"{d}/{name}.tsv", rows)  # NO header row -- matches real CoLA processor

def gen_stsb():
    d = f"{DATA_ROOT}/sts-b"
    if os.path.exists(f"{d}/train.tsv"): return
    ds = load_dataset("glue", "stsb")
    for split, name in [("train","train"), ("validation","dev")]:
        rows = [["c0","c1","c2","c3","c4","c5","c6","sentence1","sentence2","label"]]
        for ex in ds[split]:
            rows.append([0,0,0,0,0,0,0, ex["sentence1"], ex["sentence2"], ex["label"]])
        w(f"{d}/{name}.tsv", rows)

def gen_qqp():
    d = f"{DATA_ROOT}/qqp"
    if os.path.exists(f"{d}/train.tsv"): return
    ds = load_dataset("glue", "qqp")
    for split, name in [("train","train"), ("validation","dev")]:
        rows = [["id","qid1","qid2","question1","question2","is_duplicate"]]
        for ex in ds[split]:
            rows.append([0, 0, 0, ex["question1"], ex["question2"], ex["label"]])
        w(f"{d}/{name}.tsv", rows)

def gen_qnli():
    d = f"{DATA_ROOT}/qnli"
    if os.path.exists(f"{d}/train.tsv"): return
    ds = load_dataset("glue", "qnli")
    label_map = {0: "entailment", 1: "not_entailment"}
    for split, name in [("train","train"), ("validation","dev")]:
        rows = [["index","question","sentence","label"]]
        for ex in ds[split]:
            rows.append([0, ex["question"], ex["sentence"], label_map[ex["label"]]])
        w(f"{d}/{name}.tsv", rows)

def gen_rte():
    d = f"{DATA_ROOT}/rte"
    if os.path.exists(f"{d}/train.tsv"): return
    ds = load_dataset("glue", "rte")
    label_map = {0: "entailment", 1: "not_entailment"}
    for split, name in [("train","train"), ("validation","dev")]:
        rows = [["index","sentence1","sentence2","label"]]
        for ex in ds[split]:
            rows.append([0, ex["sentence1"], ex["sentence2"], label_map[ex["label"]]])
        w(f"{d}/{name}.tsv", rows)

def gen_mnli():
    d = f"{DATA_ROOT}/mnli"
    if os.path.exists(f"{d}/train.tsv"): return
    ds = load_dataset("glue", "mnli")
    label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}
    for split, name, fname in [("train","train","train.tsv"), ("validation_matched","dev_matched","dev_matched.tsv")]:
        rows = [["index","c1","c2","c3","c4","c5","c6","c7","sentence1","sentence2","label"]]
        for ex in ds[split]:
            rows.append([0,0,0,0,0,0,0,0, ex["premise"], ex["hypothesis"], label_map[ex["label"]]])
        w(f"{d}/{fname}", rows)

print("Generating TSVs (skips tasks already prepared in Drive)...")
for fn in [gen_sst2, gen_mrpc, gen_cola, gen_stsb, gen_qqp, gen_qnli, gen_rte, gen_mnli]:
    fn()
print("✅ Data prep done.\n")

# ------------------------------------------------------------
# STAGE A: fine-tune standard BERT-base per task with HuggingFace
# `transformers`, saved in the exact format quant_task_glue.py expects
# to load as teacher+student init: {model_dir}/{task_name}/.
# ------------------------------------------------------------

MODEL_DIR_ROOT = f"{PROJECT_ROOT}/finetuned_models"

TASK_HF_MAP = {
    "cola":  ("cola", 2, False, ("sentence", None)),
    "mnli":  ("mnli", 3, False, ("premise", "hypothesis")),
    "mrpc":  ("mrpc", 2, False, ("sentence1", "sentence2")),
    "sst-2": ("sst2", 2, False, ("sentence", None)),
    "sts-b": ("stsb", 1, True,  ("sentence1", "sentence2")),
    "qqp":   ("qqp", 2, False, ("question1", "question2")),
    "qnli":  ("qnli", 2, False, ("question", "sentence")),
    "rte":   ("rte", 2, False, ("sentence1", "sentence2")),
}

def finetune_bert_base(task_name):
    out_dir = f"{MODEL_DIR_ROOT}/{task_name}"
    if os.path.exists(f"{out_dir}/pytorch_model.bin"):
        print(f"  [{task_name}] fine-tuned checkpoint already exists, skipping.")
        return
    hf_name, num_labels, is_regression, (k1, k2) = TASK_HF_MAP[task_name]
    print(f"  [{task_name}] Fine-tuning BERT-base ({hf_name})...")

    ds = load_dataset("glue", hf_name)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=num_labels,
        problem_type="regression" if is_regression else "single_label_classification"
    ).to(device)

    max_len = 64 if task_name in ("cola", "sst-2") else 128
    bsz = 16 if task_name == "cola" else 32

    def tokenize_batch(batch):
        if k2 is None:
            return tokenizer(batch[k1], truncation=True, max_length=max_len, padding="max_length")
        return tokenizer(batch[k1], batch[k2], truncation=True, max_length=max_len, padding="max_length")

    train_ds = ds["train"].map(tokenize_batch, batched=True)
    cols = ["input_ids", "attention_mask", "token_type_ids", "label"]
    train_ds.set_format(type="torch", columns=[c for c in cols if c in train_ds.column_names])

    loader = DataLoader(train_ds, batch_size=bsz, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    total_steps = len(loader) * 3
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    model.train()
    for epoch in range(3):
        t0 = time.time()
        total_loss = 0
        for step, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            tok_type = batch.get("token_type_ids")
            tok_type = tok_type.to(device) if tok_type is not None else None
            labels = batch["label"].float().to(device) if is_regression else batch["label"].to(device)

            optimizer.zero_grad()
            out = model(input_ids=input_ids, attention_mask=attn, token_type_ids=tok_type, labels=labels)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += out.loss.item()
            if step % 200 == 0:
                print(f"    epoch {epoch+1}/3 step {step}/{len(loader)} loss={out.loss.item():.4f}")
        print(f"  [{task_name}] epoch {epoch+1} done in {time.time()-t0:.0f}s, avg loss {total_loss/len(loader):.4f}")

    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"  ✅ [{task_name}] fine-tuned checkpoint saved to {out_dir}\n")
    del model
    gc.collect()
    torch.cuda.empty_cache()

# ------------------------------------------------------------
# STAGE B: official quant_task_glue.py per task.
# --data_dir is the PARENT dir (script appends lowercase task_name itself).
# --model_dir is the PARENT dir containing {task_name}/ fine-tuned checkpoint.
# ------------------------------------------------------------

def run_ternarybert_quant(task_name):
    output_dir = f"{PROJECT_ROOT}/ternary_models/{task_name}"
    marker = f"{output_dir}/DONE.marker"
    if os.path.exists(marker):
        print(f"  [{task_name}] TernaryBERT quantization already completed, skipping.")
        return True
    os.makedirs(output_dir, exist_ok=True)

    cmd = (
        f"cd {TB_DIR} && python quant_task_glue.py "
        f"--data_dir {DATA_ROOT} "
        f"--model_dir {MODEL_DIR_ROOT} "
        f"--task_name {task_name} "
        f"--output_dir {output_dir} "
        f"--learning_rate 2e-5 "
        f"--num_train_epochs 3 "
        f"--weight_bits 2 "
        f"--input_bits 8 "
        f"--pred_distill "
        f"--intermediate_distill "
        f"--save_quantized_model"
    )
    print(f"  [{task_name}] Running TernaryBERT QAT...")
    ret = subprocess.run(cmd, shell=True)
    if ret.returncode == 0:
        open(marker, "w").write("done")
        print(f"  ✅ [{task_name}] TernaryBERT quantization complete.\n")
        return True
    else:
        print(f"  ⚠️  [{task_name}] quant_task_glue.py failed (exit {ret.returncode}).\n")
        return False

# ------------------------------------------------------------
# RUN: small tasks first so you have real results quickly even if
# later (larger) tasks get interrupted by a Colab disconnect.
# ------------------------------------------------------------

OFFICIAL_TASKS = ["rte", "mrpc", "cola", "sts-b", "sst-2", "qnli", "mnli", "qqp"]

PROGRESS_FILE = f"{PROJECT_ROOT}/results/progress.json"
progress = json.load(open(PROGRESS_FILE)) if os.path.exists(PROGRESS_FILE) else {}

for task in OFFICIAL_TASKS:
    print(f"\n{'='*70}\n>>> TASK: {task}\n{'='*70}")
    if progress.get(task) == "done":
        print(f"⏭️  Already marked done, skipping.")
        continue
    finetune_bert_base(task)
    success = run_ternarybert_quant(task)
    progress[task] = "done" if success else "failed"
    json.dump(progress, open(PROGRESS_FILE, "w"))

print("\n\n" + "#"*70)
print("PIPELINE STATUS")
print("#"*70)
print(json.dumps(progress, indent=2))
print("\nNOTE: WNLI is intentionally excluded -- the official TernaryBERT")
print("script has no processor for it (verified in utils_glue.py), matching")
print("the original paper's own results table, which also omits WNLI.")
print(f"\nAll checkpoints saved under: {PROJECT_ROOT}")
print("Re-run this SAME cell after any disconnect to resume automatically.")

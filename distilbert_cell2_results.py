# ============================================================
# CELL 2 — Build final results table for DistilBERT (fp32 baseline,
# pre-quantization) from the checkpoints/metrics saved by Cell 1.
#
# Adds Memory Size, Latency, Throughput by loading each saved checkpoint
# and timing a real eval pass (same approach as cell2_metrics.py in the
# BERT-base pipeline). Energy Consumption uses codecarbon if installed;
# otherwise reports "N/A" honestly rather than a placeholder number.
#
# Run this AFTER Cell 1 has completed at least one task.
# ============================================================

import os, json, time
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

PROJECT_ROOT = "/content/drive/MyDrive/distilbert_glue"
MODEL_DIR_ROOT = f"{PROJECT_ROOT}/finetuned_models"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
TASK_DISPLAY_NAMES = {
    "sst-2": "SST2", "qnli": "QNLI", "qqp": "QQP", "mnli": "MNLI",
    "mrpc": "MRPC", "sts-b": "STS-B", "rte": "RTE", "cola": "COLA",
}
MAX_LEN = 128

def measure_latency_throughput(task_name, out_dir):
    hf_name, num_labels, is_regression, (k1, k2) = TASK_HF_MAP[task_name]
    tokenizer = DistilBertTokenizerFast.from_pretrained(out_dir)
    model = DistilBertForSequenceClassification.from_pretrained(out_dir).to(device)
    model.eval()

    ds = load_dataset("glue", hf_name)
    eval_split = "validation_matched" if task_name == "mnli" else "validation"

    def tokenize_batch(batch):
        if k2 is None:
            return tokenizer(batch[k1], truncation=True, max_length=MAX_LEN, padding="max_length")
        return tokenizer(batch[k1], batch[k2], truncation=True, max_length=MAX_LEN, padding="max_length")

    eval_ds = ds[eval_split].map(tokenize_batch, batched=True)
    eval_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    eval_loader = DataLoader(eval_ds, batch_size=32, shuffle=False)

    batch_times, n_samples = [], 0
    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            model(input_ids=input_ids, attention_mask=attn)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.time()
            batch_times.append(t1 - t0)
            n_samples += input_ids.shape[0]

    avg_latency_ms = (sum(batch_times) / len(batch_times)) * 1000
    throughput = n_samples / sum(batch_times)

    mem_mb = os.path.getsize(f"{out_dir}/pytorch_model.bin") / (1024 ** 2)

    energy_kwh = None
    try:
        from codecarbon import EmissionsTracker
        tracker = EmissionsTracker(log_level="error", save_to_file=False)
        tracker.start()
        with torch.no_grad():
            for batch in list(eval_loader)[:20]:
                input_ids = batch["input_ids"].to(device)
                attn = batch["attention_mask"].to(device)
                model(input_ids=input_ids, attention_mask=attn)
        emissions_kg = tracker.stop()
        energy_kwh = (emissions_kg * 1000 / 0.475) if emissions_kg else None
    except Exception:
        pass

    del model
    torch.cuda.empty_cache()
    return mem_mb, avg_latency_ms, throughput, energy_kwh

final_rows = {}
for task in ["sst-2", "qnli", "qqp", "mnli", "mrpc", "sts-b", "rte", "cola"]:
    out_dir = f"{MODEL_DIR_ROOT}/{task}"
    metrics_path = f"{out_dir}/eval_metrics.json"
    if not os.path.exists(metrics_path):
        final_rows[task] = {"error": "not yet fine-tuned (run Cell 1 first)"}
        continue
    metrics = json.load(open(metrics_path))
    print(f"Measuring latency/throughput/memory for {task}...")
    mem_mb, lat_ms, thr, energy_kwh = measure_latency_throughput(task, out_dir)
    final_rows[task] = {
        "Model name": "DistilBERT (fp32, pre-quantization)",
        "Memory Size (MB)": round(mem_mb, 2),
        "Latency (ms/batch)": round(lat_ms, 2),
        "Accuracy": round(metrics["accuracy"], 4),
        "Bits": 32,
        "Precision": round(metrics["precision"], 4),
        "Recall": round(metrics["recall"], 4),
        "F1-score": round(metrics["f1"], 4),
        "Throughput (samples/sec)": round(thr, 2),
        "Energy Consumption (kWh, approx)": round(energy_kwh, 6) if energy_kwh else "N/A",
    }
    if "mcc" in metrics:
        final_rows[task]["MCC (CoLA)"] = round(metrics["mcc"], 4)

print("\n\n" + "#" * 70)
print("FINAL RESULTS TABLE — DistilBERT, fp32 baseline (Stage A only)")
print("#" * 70)
for task in ["sst-2", "qnli", "qqp", "mnli", "mrpc", "sts-b", "rte", "cola"]:
    print(f"\nDataset name: {TASK_DISPLAY_NAMES[task]}")
    r = final_rows[task]
    if "error" in r:
        print(f"  {r['error']}")
        continue
    headers = list(r.keys())
    print("  " + "\t".join(headers))
    print("  " + "\t".join(str(r[h]) for h in headers))

with open(f"{PROJECT_ROOT}/results/final_results_table.json", "w") as f:
    json.dump(final_rows, f, indent=2)
print(f"\nSaved to {PROJECT_ROOT}/results/final_results_table.json")
print("\nNOTE: this is the fp32 DistilBERT baseline, not yet quantized with")
print("TernaryBERT. We'll add the quantization stage once this baseline is")
print("validated.")

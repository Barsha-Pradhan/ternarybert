# TernaryBERT GLUE Benchmarking Pipeline

Benchmarks **TernaryBERT** (2-bit weight, 8-bit activation quantization of BERT-base) across the GLUE benchmark, using the official [Huawei Noah's Ark Lab TernaryBERT repository](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/TernaryBERT).

Built as part of a research internship benchmarking compressed/quantized NLP models across GLUE tasks.

## What this does

1. **Data preparation** — generates legacy-format GLUE TSV files from the [Hugging Face `datasets`](https://huggingface.co/datasets/glue) library, matching the exact column layout expected by the official TernaryBERT codebase's `utils_glue.py` processors (verified by direct inspection of the source, since the original NYU GLUE download script is no longer reliably available).
2. **Stage A — Fine-tuning** — fine-tunes standard BERT-base per task using Hugging Face `transformers`, saved in the checkpoint format (`config.json`, `pytorch_model.bin`, vocab files) that the official TernaryBERT script expects as its teacher/student initialization.
3. **Stage B — Quantization-aware training** — runs the official `quant_task_glue.py` script (unmodified) to perform distillation-aware ternary quantization (2-bit weights, 8-bit activations) on top of the fine-tuned checkpoint.
4. **Evaluation** — since the official script does not compute Precision/Recall, model size, latency, throughput, or energy consumption, and writes no results file to disk, a separate evaluation pass loads each saved quantized checkpoint and computes the full metric set.



- Google Colab with GPU runtime (T4 or better recommended), or any environment with a CUDA GPU
- Google Drive (for checkpoint persistence across sessions — training is long-running)
- Python packages: `datasets`, `transformers`, `torch`, `scipy`, `scikit-learn` (installed automatically by the script)

## Usage

Run as two Colab cells in order, or as two sequential scripts:

```bash
python cell1_pipeline.py   # data prep + fine-tuning + TernaryBERT quantization (long-running)
python cell2_metrics.py    # evaluates saved checkpoints, produces the final results table
```

`cell1_pipeline.py` is **resumable**: if interrupted (Colab disconnect, timeout), re-running it skips any task whose checkpoint already exists, and resumes from the next incomplete task. Progress is tracked in `results/progress.json`.

### Runtime expectations

Training time depends heavily on task size. Approximate full-data, 3-epoch fine-tune + 3-epoch quantization-aware training times on a single T4 GPU:

| Task | Approx. time |
|---|---|
| RTE, MRPC, WNLI* | 20–35 min |
| CoLA, STS-B | 30–40 min |
| SST-2 | 1–1.5 hr |
| QNLI | 1.5–2 hr |
| MNLI, QQP | 3–5 hr |



## Output structure

All artifacts are written under `{Drive}/ternarybert_glue/`:

```
ternarybert_glue/
├── data/                  # generated GLUE TSVs, per task (lowercase folder names)
├── finetuned_models/      # Stage A: fine-tuned full-precision BERT-base checkpoints, per task
├── ternary_models/        # Stage B: TernaryBERT quantized checkpoints, per task (.../quant/)
├── results/
│   ├── progress.json      # resumability tracking
│   └── final_results.json # final metrics table, all tasks
└── logs/
```

## Metrics reported

| Metric | Source |
|---|---|
| Accuracy | Computed (sklearn) |
| Precision / Recall / F1 | Computed (sklearn); **not** provided by the official script, which only reports accuracy/F1 (for MRPC/QQP)/MCC (CoLA)/Pearson-Spearman (STS-B) |
| Memory Size | Saved checkpoint (`pytorch_model.bin`) file size |
| Latency / Throughput | Measured via `torch.cuda.synchronize()`-wrapped timing during evaluation |
| Bits | Fixed at 2 (weight quantization bit-width used) |
| Energy Consumption | Approximate, via [`codecarbon`](https://github.com/mlco2/codecarbon) where available; falls back to `N/A` |

For STS-B (a regression task), Accuracy/Precision/Recall/F1 columns report Pearson/Spearman correlation and their average as proxies, consistent with how the task is conventionally evaluated in the GLUE benchmark.

## Verification notes

The data schema and script invocation in this pipeline were verified by directly reading the official repository's source files (`utils_glue.py`, `quant_task_glue.py`, `transformer/modeling_quant.py`) rather than inferred from documentation alone, since the README-level instructions in the upstream repo do not fully specify TSV column conventions or script internals.

## Known limitations

- This pipeline has been verified against the official source line-by-line but full end-to-end execution across all 8 tasks may surface environment-specific issues (e.g., out-of-memory on large batch sizes for MNLI/QQP) not yet encountered.
- Energy consumption figures are approximate estimates, not calibrated hardware measurements.
- Fine-tuning (Stage A) uses standard Hugging Face `transformers`, not the original repository's bundled legacy `transformer` package — this is consistent with how the upstream TernaryBERT authors describe preparing their own fine-tuned checkpoints (per repository maintainer correspondence in [issue #146](https://github.com/huawei-noah/Pretrained-Language-Model/issues/146)).

## Citation

```bibtex
@inproceedings{zhang-etal-2020-ternarybert,
  title     = {TernaryBERT: Distillation-aware Ultra-low Bit BERT},
  author    = {Zhang, Wei and Hou, Lu and Yin, Yichun and Shang, Lifeng and Chen, Xiao and Jiang, Xin and Liu, Qun},
  booktitle = {Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year      = {2020}
}
```

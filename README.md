## 1. Environment

Recommended system requirements:

- Linux with an NVIDIA GPU and a CUDA-compatible PyTorch installation.
- Python 3.10 or newer.
- Enough GPU memory for the selected 7B/13B checkpoint. Use
  `--device-map auto` (the default), a smaller debug subset, or quantization
  where the runner supports it if one GPU is insufficient.

Create an isolated environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Experiment 1: localization and relation evaluation

```bash
python exp1/run_exp1_qwen.py \
  --model-path "$QWEN_MODEL" \
  --data-root "$DATA_ROOT" \
  --output-dir exp1/results_qwen

python exp1/run_exp1_llava.py \
  --model-path "$LLAVA_7B_MODEL" \
  --data-root "$DATA_ROOT" \
  --output-dir exp1/results_llava
```

Each run generates a result JSONL plus JSON/CSV analysis files. To rebuild the
analysis from an existing result file:

```bash
python exp1/analyze_exp1.py \
  --results-jsonl exp1/results_qwen/qwen_exp1_results.jsonl \
  --model-name qwen
```

Optional figure scripts:

```bash
python exp1/visualize_llava_1d_abs_pe_tsne.py \
  --model-dir "$LLAVA_7B_MODEL" \
  --output-dir exp1/outputs

python exp1/qwen_pca.py \
  --model-dir "$QWEN_MODEL" \
  --data-root "$DATA_ROOT" \
  --object-token-source mask \
  --pca-normalization zscore \
  --output-dir exp1/outputs

python exp1/tsne.py \
  --llava-model-dir "$LLAVA_7B_MODEL" \
  --qwen-model-dir "$QWEN_MODEL" \
  --data-root "$DATA_ROOT" \
  --output-dir exp1/outputs
```

### Experiment 2: object-token ablation

```bash
python exp2/run_exp2_qwen.py \
  --model-path "$QWEN_MODEL" \
  --data-root "$DATA_ROOT" \
  --output-dir exp2/results_qwen

python exp2/run_exp2_llava.py \
  --model-path "$LLAVA_7B_MODEL" \
  --data-root "$DATA_ROOT" \
  --output-dir exp2/results_llava
```

The 13B runner uses the official LLaVA code/model format:

```bash
python exp2/run_exp2_llava13b.py \
  --llava-repo "$LLAVA_REPO" \
  --model-path "$LLAVA_13B_OFFICIAL_MODEL" \
  --data-root "$DATA_ROOT" \
  --output-dir exp2/results_llava13b
```

Summarize token-removal ratios either from an existing run or directly from
masks:

```bash
python exp2/summarize_removed_token_ratios.py \
  --results-jsonl exp2/results_qwen/qwen_exp2_results.jsonl \
  --output-csv exp2/results_qwen/removed_token_ratios.csv

python exp2/summarize_removed_token_ratios.py \
  --data-root "$DATA_ROOT" \
  --model-grid qwen \
  --output-csv exp2/results_qwen/removed_token_ratios_from_masks.csv
```

### Experiment 3: position probe and relation logit lens

```bash
python exp3/run_exp3_qwen.py \
  --model-path "$QWEN_MODEL" \
  --data-root "$DATA_ROOT" \
  --imagenet-root "$IMAGENET_ROOT" \
  --save-probes \
  --output-dir exp3/results_qwen

python exp3/run_exp3_llava.py \
  --model-path "$LLAVA_7B_MODEL" \
  --data-root "$DATA_ROOT" \
  --imagenet-root "$IMAGENET_ROOT" \
  --save-probes \
  --output-dir exp3/results_llava
```

Use `--skip-position-probe` to run only the relation logit lens, or
`--skip-relation-probe` to run only the ImageNet position probe. For a bounded
test, pass `--max-position-images 20 --max-relation-samples 20`.

### Experiment 5: decoder-attention knockout

```bash
python exp5/run_exp5_qwen.py \
  --model-path "$QWEN_MODEL" \
  --data-root "$DATA_ROOT" \
  --output-dir exp5/results_qwen

python exp5/run_exp5_llava.py \
  --model-path "$LLAVA_7B_MODEL" \
  --data-root "$DATA_ROOT" \
  --output-dir exp5/results_llava
```

The default layer granularities are `group4,single,group6,all`. Use
`--conditions`, `--layer-granularities`, `--max-samples`, and `--random-seeds`
to create smaller or targeted runs. `filter_qwen_padding_results.py` is a
recovery utility for legacy Qwen JSONL files affected by right padding; new
runs already use the corrected padding path.

### Experiment 6: causal mediation analysis

Experiment 6 has four stages: `prepare`, `filter`, `cma`, and `all`. The source
image contains the target object; the base image is its target-removed version.
The CMA stage patches one source attention-head activation into the base run.

```bash
python exp6/run_exp6_qwen.py \
  --stage all \
  --model-path "$QWEN_MODEL" \
  --qa-path "$DATA_ROOT/qa.json" \
  --bbox-mask-path "$DATA_ROOT/bbox_mask_xyxy.json" \
  --source-image-root "$DATA_ROOT/images" \
  --removed-image-root "$REMOVED_IMAGE_ROOT" \
  --output-dir exp6/results_qwen

python exp6/run_exp6_llava.py \
  --stage all \
  --model-path "$LLAVA_7B_MODEL" \
  --qa-path "$DATA_ROOT/qa.json" \
  --bbox-mask-path "$DATA_ROOT/bbox_mask_xyxy.json" \
  --source-image-root "$DATA_ROOT/images" \
  --removed-image-root "$REMOVED_IMAGE_ROOT" \
  --output-dir exp6/results_llava

python exp6/run_exp6_llava13b.py \
  --stage all \
  --model-path "$LLAVA_13B_HF_MODEL" \
  --qa-path "$DATA_ROOT/qa.json" \
  --bbox-mask-path "$DATA_ROOT/bbox_mask_xyxy.json" \
  --source-image-root "$DATA_ROOT/images" \
  --removed-image-root "$REMOVED_IMAGE_ROOT" \
  --output-dir exp6/results_llava13b
```

The default score is teacher-forced mean NLL, and a valid mediation gap
requires `base_score > source_score`. Relation and localization use separate
curated subsets. A useful smoke test is:

```bash
python exp6/run_exp6_qwen.py \
  --stage all \
  --model-path "$QWEN_MODEL" \
  --qa-path "$DATA_ROOT/qa.json" \
  --bbox-mask-path "$DATA_ROOT/bbox_mask_xyxy.json" \
  --source-image-root "$DATA_ROOT/images" \
  --removed-image-root "$REMOVED_IMAGE_ROOT" \
  --output-dir exp6/debug_qwen \
  --max-candidates 8 --max-curated 2 --layers 0,1 --heads 0,1
```

### Experiment 7: progressive ablation of CMA-ranked heads

Experiment 7 consumes Experiment 6's CMA summary and task-specific curated
subsets. Run the matching Experiment 6 configuration first.

```bash
python exp7/run_exp7_qwen.py \
  --model-path "$QWEN_MODEL" \
  --exp6-results-dir exp6/results_qwen \
  --output-dir exp7/results_qwen

python exp7/run_exp7_llava.py \
  --model-path "$LLAVA_7B_MODEL" \
  --exp6-results-dir exp6/results_llava \
  --output-dir exp7/results_llava

python exp7/run_exp7_llava13b.py \
  --model-path "$LLAVA_13B_HF_MODEL" \
  --exp6-results-dir exp6/results_llava13b \
  --output-dir exp7/results_llava13b
```

The default ranking statistic is mean mediation fraction, and the default
ablation scope is prompt prefill. A bounded smoke test can use
`--fractions 0,0.01 --max-samples-per-task 2`.

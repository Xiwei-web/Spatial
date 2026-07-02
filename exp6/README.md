# Experiment 6: Causal Mediation Analysis

This folder contains code for Experiment 6, modeled after Sec. 3.3.2 of
`Mechanisms of Object Localization in Vision-Language Models`.

## What The Scripts Do

1. Convert the original multi-choice `qa.json` into binary yes/no relation QA:

   ```text
   Is the {target} {true_relation} the {reference}? Answer yes or no.
   ```

   The converted file is written to:

   ```text
   exp6/qa.json
   ```

2. Build source/base counterfactual pairs:

   - source image: original image, target object present
   - base image: target-removed image from LaMa inpainting

   The candidate pair file is written to:

   ```text
   exp6/candidate_pairs.json
   ```

3. Run the model on all candidate source/base pairs with the binary prompt.

4. Select up to 50 curated samples with fixed seed where:

   - source prediction is correct (`yes`)
   - base prediction is wrong (`no`/unparsed), or base yes-perplexity is worse than source yes-perplexity

5. Run head-level CMA on the curated subset by patching source-run attention-head activations into the base run at prompt-token positions only.

## Scripts

LLaVA:

```bash
python exp6/run_exp6_llava.py --stage all
```

Qwen:

```bash
python exp6/run_exp6_qwen.py --stage all
```

The LLaVA and Qwen implementations are intentionally separate.

## Default HPC Paths

The scripts assume these default data paths, which can be overridden:

```text
--source-image-root autodl-tmp/whatsUP/images
--removed-image-root autodl-tmp/whatsUP_lama_504/output_dilate20
```

Model defaults:

```text
--model-path models/llava-1.5-7b-hf
--model-path models/Qwen2.5-VL-7B-Instruct
```

If your HPC layout differs, pass explicit paths.

## Useful Stage Commands

Only prepare binary QA and candidate pairs:

```bash
python exp6/run_exp6_llava.py --stage prepare
```

Only run source/base filtering:

```bash
python exp6/run_exp6_llava.py --stage filter
```

Only run CMA from an existing curated subset:

```bash
python exp6/run_exp6_llava.py --stage cma
```

Debug with a tiny subset:

```bash
python exp6/run_exp6_llava.py --stage all --max-candidates 8 --max-curated 2 --layers 0,1 --heads 0,1
```

## Main Outputs

LLaVA outputs:

```text
exp6/results_llava/llava-1.5-7b-hf_source_base_behavior.jsonl
exp6/results_llava/llava-1.5-7b-hf_curated_subset.json
exp6/results_llava/llava-1.5-7b-hf_cma_prompt_tokens.jsonl
exp6/results_llava/llava-1.5-7b-hf_cma_summary.json
exp6/results_llava/llava-1.5-7b-hf_cma_summary.csv
```

Qwen outputs:

```text
exp6/results_qwen/Qwen2.5-VL-7B-Instruct_source_base_behavior.jsonl
exp6/results_qwen/Qwen2.5-VL-7B-Instruct_curated_subset.json
exp6/results_qwen/Qwen2.5-VL-7B-Instruct_cma_prompt_tokens.jsonl
exp6/results_qwen/Qwen2.5-VL-7B-Instruct_cma_summary.json
exp6/results_qwen/Qwen2.5-VL-7B-Instruct_cma_summary.csv
```

## Notes

- CMA patches only prompt-token positions, as requested.
- Relation CMA uses the converted binary yes/no prompt.
- Localization CMA uses the target object's ground-truth bbox as the teacher-forcing answer.
- The scripts do not require the curated subset to contain exactly 50 samples; if fewer are eligible, all eligible samples are used.

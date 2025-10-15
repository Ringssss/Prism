# Prism: Fast Diffusion LLM Inference (Open‑Box)

This repo contains runnable scripts and datasets to reproduce our Prism results on a single H100 (CUDA).

## Quickstart

1) Create env (Python 3.11)

```
pip install -r requirements.txt
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio
```

2) Datasets included
- `datasets/sharegpt.jsonl` (sampled)
- `datasets/gsm8k.jsonl` (full)

3) Commands

- LLaDA‑8B‑Base, ShareGPT bs=1, steps=64 (throughput ≈ 150–160 tok/s)
```
CUDA_VISIBLE_DEVICES=0 MODEL_PATH=/data/huggingface/llada-8B-Base \
python scripts/run_llada8b_sharegpt.py
```
Output JSON: `results_llada8b_sharegpt_bs1.json`

- LLaDA‑1.5, ShareGPT bs=1, steps=64 (throughput ≈ 150–165 tok/s)
```
CUDA_VISIBLE_DEVICES=0 MODEL_PATH=/data/huggingface/LLaDA-1.5 \
python scripts/run_llada15_sharegpt.py
```
Output JSON: `results_llada15_sharegpt_bs1.json`

- GSM8K accuracy (50 samples), same fast settings
```
CUDA_VISIBLE_DEVICES=0 python scripts/run_gsm8k_eval.py \
  --model-path /data/huggingface/llada-8B-Base --max-samples 50
# or
CUDA_VISIBLE_DEVICES=0 python scripts/run_gsm8k_eval.py \
  --model-path /data/huggingface/LLaDA-1.5 --max-samples 50
```
Outputs JSON: `results_gsm8k_accuracy.json`

- LLaDA‑MoE‑7B‑A1B, ShareGPT bs=1, steps=64 (readable baseline via vLLM MoE path (requires vllm), no external dInfer)
```
CUDA_VISIBLE_DEVICES=0 python scripts/run_moe_llada_sharegpt.py
```
Output JSON: `results_moe_llada_blockwise.json`

## Notes
- The fast settings use sliding‑window + threshold parallel decode + expected TPF tuning.
- For 8B, we also provide a faster‑quality profile (≈ 300–380 tok/s, longer outputs) in the main repo.
- MoE is routed through dInfer BlockWiseDiffusionLLM to ensure readable text; vLLM FusedMoE fastpath achieves 100+ tok/s but may require template tuning for quality.

## Expected Throughput (H100, bs=1, gen=128, steps=64)
- LLaDA‑8B‑Base: 150–160 tok/s
- LLaDA‑1.5: 150–165 tok/s
- LLaDA‑MoE‑7B‑A1B: 80–100 tok/s (readable path)


## Example Result
- File: results/example_llada8b_sharegpt_bs1.json (H100, bs=1, steps=64, gen=128)
- Throughput: ~150-160 tok/s


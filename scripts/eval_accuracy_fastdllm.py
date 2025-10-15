#!/usr/bin/env python3
"""
Evaluate Fast-dLLM LLaDA diffusion generation accuracy on GSM8K/MGSM JSONL.

Usage:
  python scripts/eval_accuracy_fastdllm.py --dataset gsm8k --dataset-path datasets/gsm8k.jsonl \
    --model-path /data/huggingface/llada-8B-Base --max-samples 32 --steps 64 --gen-length 128
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional

import torch

# Fast-dLLM model path
sys.path.insert(0, '/home/zhujianian/Fast-dLLM-main/llada')
from transformers import AutoTokenizer
from model.modeling_llada import LLaDAModelLM  # type: ignore
from generate import generate  # type: ignore

_NUM_RE = re.compile(r"[-+]?\d+(?:[\,\d]*\d)?(?:\.\d+)?")


def normalize_number_str(s: str) -> str:
    s = s.replace(',', '').strip()
    s = re.sub(r"[\s\)\]\.:，。]+$", "", s)
    return s


def extract_final_number(text: str) -> Optional[str]:
    t = text
    if 'Answer:' in t:
        t = t.split('Answer:')[-1]
    matches = list(_NUM_RE.finditer(t))
    if not matches:
        return None
    return normalize_number_str(matches[-1].group(0))


def load_jsonl(path: str, limit: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if len(rows) >= limit:
                break
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return rows


def build_items(dataset: str, rows: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    for r in rows:
        if dataset in ('gsm8k', 'mgsm'):
            q = r.get('question') or r.get('input') or r.get('prompt') or ''
            a = r.get('answer') or r.get('label') or ''
            if not q:
                continue
            prompt = f"Question: {q}\nAnswer:"
            items.append({'prompt': prompt, 'gold': str(a)})
        else:
            raise ValueError(f'Unsupported dataset: {dataset}')
    return items


@torch.no_grad()
def fastdllm_generate_texts(model_path: str, items: List[Dict[str, str]], steps: int, gen_length: int) -> List[str]:
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = LLaDAModelLM.from_pretrained(model_path, torch_dtype=torch.float16).cuda().eval()
    texts: List[str] = []
    block_length = gen_length
    for it in items:
        ids = tok.encode(it['prompt'], return_tensors='pt').cuda()
        out, _ = generate(model, ids, steps=steps, gen_length=gen_length, block_length=block_length,
                          temperature=0.0, remasking='low_confidence', mask_id=int(getattr(tok, 'mask_token_id', 126336) or 126336))
        seq = out[0]
        text = tok.decode(seq, skip_special_tokens=True)
        # Extract only generated segment
        plen = ids.shape[1]
        gen_text = tok.decode(seq[plen:], skip_special_tokens=True)
        texts.append(gen_text)
    return texts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['gsm8k', 'mgsm'], required=True)
    ap.add_argument('--dataset-path', required=True)
    ap.add_argument('--model-path', required=True)
    ap.add_argument('--max-samples', type=int, default=32)
    ap.add_argument('--steps', type=int, default=64)
    ap.add_argument('--gen-length', type=int, default=128)
    ap.add_argument('--out', default='results/fastdllm_accuracy.json')
    args = ap.parse_args()

    rows = load_jsonl(args.dataset_path, args.max_samples)
    items = build_items(args.dataset, rows)
    print(f"Loaded {len(items)} items from {args.dataset_path}")
    t0 = time.time()
    pred_texts = fastdllm_generate_texts(args.model_path, items, args.steps, args.gen_length)
    elapsed = time.time() - t0
    correct = 0
    detail = []
    for txt, it in zip(pred_texts, items):
        g = extract_final_number(it['gold'])
        p = extract_final_number(txt)
        ok = (g is not None and p is not None and g == p)
        correct += int(ok)
        detail.append({'gold': it['gold'], 'gold_num': g, 'pred_num': p, 'pred_text': txt, 'ok': ok})
    acc = correct / max(1, len(items))
    out = {
        'dataset': args.dataset,
        'num_samples': len(items),
        'model_path': args.model_path,
        'accuracy': acc,
        'correct': correct,
        'elapsed_s': elapsed,
    }
    os.makedirs('results', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump({'summary': out, 'details': detail}, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()


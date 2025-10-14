#!/usr/bin/env python3
import os, sys, subprocess
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
os.environ.setdefault('DISCARD_FIRST','1')
os.environ.setdefault('USE_CUDAGRAPH','0')
os.environ.setdefault('USE_KV_CACHE','0')
os.environ.setdefault('USE_SLIDING_WINDOW','1')
os.environ.setdefault('USE_THRESHOLD_DECODING','1')
os.environ.setdefault('CONF_THRESHOLD','0.95')
os.environ.setdefault('SLIDING_PREFIX','64')
os.environ.setdefault('SLIDING_AFTER','32')
os.environ.setdefault('SLIDING_WARMUP','8')
os.environ.setdefault('EXPECTED_TPF','2')
os.environ.setdefault('USE_EOS_GATING','1')
os.environ.setdefault('EOS_MIN_STEP','16')
os.environ.setdefault('TORCH_COMPILE_MODE','reduce-overhead')
model = os.environ.get('MODEL_PATH','/data/huggingface/llada-8B-Base')
cmd = [sys.executable,'-u','../../test_full_optimization.py',
       '--model-path', model,
       '--dataset','sharegpt','--dataset-path','../datasets/sharegpt.jsonl',
       '--batch-size','1','--steps','64','--gen-length','128',
       '--out','../results_llada8b_sharegpt_bs1.json']
os.makedirs('../results',exist_ok=True)
print('Running:', ' '.join(cmd))
subprocess.run(cmd, check=True)

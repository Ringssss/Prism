#!/usr/bin/env python3
import os, sys, subprocess
# Prefer vLLM MoE path without external dInfer; requires vllm installed.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
os.environ.setdefault('DISCARD_FIRST','1')
os.environ.setdefault('USE_CUDAGRAPH','0')
os.environ.setdefault('USE_VLLM_MOE','1')
# MoE + compile often conflicts; disable compile by default
model = os.environ.get('MODEL_PATH','/data/huggingface/LLaDA-MoE-7B-A1B-Instruct')
cmd = [sys.executable,'-u','test_full_optimization.py',
       '--model-path', model,
       '--dataset','sharegpt','--dataset-path','datasets/sharegpt.jsonl',
       '--batch-size','1','--steps','64','--gen-length','128','--no-compile',
       '--out','results/results_moe_sharegpt_bs1.json']
os.makedirs('results',exist_ok=True)
print('Running:', ' '.join(cmd))
try:
    subprocess.run(cmd, check=True)
except subprocess.CalledProcessError as e:
    print('[WARN] vLLM MoE path failed. Ensure vllm is installed. Error:', e)
    sys.exit(1)

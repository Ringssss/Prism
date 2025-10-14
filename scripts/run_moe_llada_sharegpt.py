#!/usr/bin/env python3
import os, sys, subprocess
# Use dInfer BlockWiseDiffusionLLM for readable outputs
os.environ.setdefault('PYTHONPATH', os.getcwd()+':'+os.path.abspath('..')+':'+os.path.abspath('../../')+':/home/zhujianian/dInfer/python')
cmd=[sys.executable,'-u','../../scripts/bench_moe_llada_blockwise.py',
     '--model-path', '/data/huggingface/LLaDA-MoE-7B-A1B-Instruct',
     '--dataset-path','../datasets/sharegpt.jsonl','--count','8',
     '--gen-length','128','--steps','64','--block-length','32','--threshold','0.95',
     '--out','../results_moe_llada_blockwise.json']
print('Running:', ' '.join(cmd))
subprocess.run(cmd, check=True)

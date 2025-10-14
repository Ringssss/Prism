#!/usr/bin/env python3
import os, sys, subprocess, argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model-path', required=True)
parser.add_argument('--max-samples', type=int, default=50)
args = parser.parse_args()
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
cmd=[sys.executable,'-u','../../scripts/eval_accuracy_fastdllm.py',
     '--dataset','gsm8k','--dataset-path','../datasets/gsm8k.jsonl',
     '--model-path', args.model_path,
     '--max-samples', str(args.max_samples), '--steps','64','--gen-length','128',
     '--out','../results_gsm8k_accuracy.json']
print('Running:', ' '.join(cmd))
subprocess.run(cmd, check=True)

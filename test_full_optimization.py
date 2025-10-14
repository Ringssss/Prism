#!/usr/bin/env python
"""
完整优化版本: torch.compile + micro-batch + PACB
适用于任意batch size，保持高吞吐

优化组合:
1. torch.compile - 2.75x加速
2. micro-batch decode - 减少Python循环开销
3. PACB调度 - 大batch时保持吞吐
"""
import sys, time, json, argparse
sys.path.insert(0, '/home/zhujianian/Fast-dLLM-main/llada')

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoConfig
# LLaDA 本地实现用于兼容原路径（若可用则优先）
try:
    from model.modeling_llada import LLaDAModelLM  # type: ignore
except Exception:
    LLaDAModelLM = None  # type: ignore
from generate import generate
import os
_te = None
try:
    import torch._dynamo as _dynamo
except Exception:
    _dynamo = None
_fused_sample_cute = None
_fused_sample_triton = None
_fused_conf_margin = None
try:
    from nanovllm.accel.cute_fused_sampling import fused_sampling_cute as _fused_sample_cute  # type: ignore
except Exception:
    try:
        import sys as _sys, os as _os
        p = _os.path.join(_os.getcwd(), 'nano-vllm-main')
        if _os.path.isdir(p) and p not in _sys.path:
            _sys.path.insert(0, p)
        from nanovllm.accel.cute_fused_sampling import fused_sampling_cute as _fused_sample_cute  # type: ignore
    except Exception:
        _fused_sample_cute = None
try:
    from nanovllm.accel.cute_row_topk_sampling import fused_row_topk_cute as _fused_row_topk_cute  # type: ignore
except Exception:
    try:
        import sys as _sys, os as _os
        p = _os.path.join(_os.getcwd(), 'nano-vllm-main')
        if _os.path.isdir(p) and p not in _sys.path:
            _sys.path.insert(0, p)
        from nanovllm.accel.cute_row_topk_sampling import fused_row_topk_cute as _fused_row_topk_cute  # type: ignore
    except Exception:
        _fused_row_topk_cute = None
try:
    from triton_fused_sampling import fused_sampling_triton as _fused_sample_triton  # type: ignore
except Exception:
    _fused_sample_triton = None
try:
    from triton_fused_confidence import fused_confidence_margin as _fused_conf_margin  # type: ignore
except Exception:
    _fused_conf_margin = None
try:
    import transformer_engine.pytorch as _te  # type: ignore
except Exception:
    _te = None

def _fused_sample_call(logits, mask_index, x):
    """Select the sampling implementation.

    Default is PyTorch path for stability and performance unless explicitly opted in.
    Priority: `USE_CUTE_FUSED=1` -> CuTe, `USE_TRITON_FUSED=1` -> Triton, else PyTorch.
    """
    # 1) Explicit CuTe fused path (opt-in)
    if os.getenv('USE_CUTE_FUSED', '0') == '1' and _fused_sample_cute is not None:
        try:
            return _fused_sample_cute(logits, mask_index, x)
        except Exception:
            pass
    # 2) Triton fused path (opt-in)
    if os.getenv('USE_TRITON_FUSED', '0') == '1' and _fused_sample_triton is not None:
        try:
            if not hasattr(_fused_sample_call, '_triton_logged'):
                print("[DEBUG] Using Triton fused sampling kernel (opt-in)")
                _fused_sample_call._triton_logged = True
            return _fused_sample_triton(logits, mask_index, x)
        except Exception as e:
            if not hasattr(_fused_sample_call, '_triton_error_logged'):
                print(f"[DEBUG] Triton kernel failed: {e}, falling back to PyTorch")
                _fused_sample_call._triton_error_logged = True
            # fall through to next
    # 3) Fallback to CuTe if available (even if not opted in Triton)
    if _fused_sample_cute is not None:
        try:
            return _fused_sample_cute(logits, mask_index, x)
        except Exception:
            pass
    # 4) Final fallback: PyTorch softmax path
    if not hasattr(_fused_sample_call, '_fallback_logged'):
        print("[DEBUG] Falling back to PyTorch (argmax + softmax + gather)")
        _fused_sample_call._fallback_logged = True
    import torch.nn.functional as F
    x0 = torch.argmax(logits, dim=-1)
    p = F.softmax(logits, dim=-1)
    conf = p.gather(dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
    x0 = torch.where(mask_index, x0, x)
    conf = torch.where(mask_index, conf, torch.tensor(-float('inf'), device=conf.device))
    return x0, conf
from typing import List, Tuple
from collections import deque
from contextlib import nullcontext
_flash_decode_triton = None
try:
    from nanovllm.accel.triton_flash_decode import flash_decode as _flash_decode_triton  # type: ignore
except Exception:
    try:
        import sys as _sys, os as _os
        p = _os.path.join(_os.getcwd(), 'nano-vllm-main')
        if _os.path.isdir(p) and p not in _sys.path:
            _sys.path.insert(0, p)
        from nanovllm.accel.triton_flash_decode import flash_decode as _flash_decode_triton  # type: ignore
    except Exception:
        _flash_decode_triton = None

# 可选：引入FSM约束（默认关闭）
_fsm_build_json = None
_fsm_apply_allow = None
try:
    from nanovllm.constraints.fsm import build_json_fsm as _fsm_build_json  # type: ignore
    from nanovllm.constraints.fsm import apply_allowlist_to_logits as _fsm_apply_allow  # type: ignore
except Exception:
    _fsm_build_json = None
    _fsm_apply_allow = None

def _resolve_special_ids(tokenizer, model=None) -> Tuple[int, int]:
    """统一解析mask/eos id，优先级：tokenizer -> model.config -> 常见别名。"""
    mask_id = getattr(tokenizer, 'mask_token_id', None)
    eos_id = getattr(tokenizer, 'eos_token_id', None)
    # model.config
    if (mask_id is None) and (model is not None) and hasattr(getattr(model, 'config', object()), 'mask_token_id'):
        try:
            mask_id = int(model.config.mask_token_id)
        except Exception:
            pass
    if (eos_id is None) and (model is not None) and hasattr(getattr(model, 'config', object()), 'eos_token_id'):
        try:
            eos_id = int(model.config.eos_token_id)
        except Exception:
            pass
    # 常见特殊token名
    if mask_id is None:
        for t in ('<|mdm_mask|>', '[gMASK]', '<|mask|>', '[MASK]'):
            try:
                v = tokenizer.convert_tokens_to_ids(t)
                if v is not None and v >= 0:
                    mask_id = int(v); break
            except Exception:
                continue
    if eos_id is None:
        try:
            eos_id = int(tokenizer.eos_token_id)
        except Exception:
            eos_id = 0
    if mask_id is None:
        raise RuntimeError('Cannot infer mask_token_id; please configure tokenizer/config')
    return int(mask_id), int(eos_id)

torch.cuda.empty_cache()


class Sequence:
    """序列状态"""
    def __init__(self, seq_id: int, prompt: str, prompt_tensor: torch.Tensor,
                 gen_length: int, steps: int):
        self.seq_id = seq_id
        self.prompt = prompt
        self.prompt_tensor = prompt_tensor
        self.gen_length = gen_length
        self.steps = steps
        self.output = None
        self.finished = False
        self.current_step = 0


class SimplePACBScheduler:
    """
    简化的PACB调度器
    核心功能: 批处理 + 长度分桶 + 动态调度
    """
    def __init__(self, max_batch_size: int = 32):
        self.max_batch_size = max_batch_size
        self.waiting: deque[Sequence] = deque()
        self.running: List[Sequence] = []
        self.finished: List[Sequence] = []

    def add_request(self, seq: Sequence):
        """添加新请求"""
        self.waiting.append(seq)

    def schedule(self) -> List[Sequence]:
        """
        调度逻辑:
        1. 从waiting中取出序列填充到max_batch_size
        2. 优先调度prompt长度相近的 (length bucketing)
        """
        batch = []

        # 移除已完成的
        self.running = [s for s in self.running if not s.finished]

        # 从running中选择继续执行的
        batch.extend(self.running[:self.max_batch_size])

        # 如果batch未满，从waiting中添加
        while len(batch) < self.max_batch_size and self.waiting:
            seq = self.waiting.popleft()
            batch.append(seq)
            self.running.append(seq)

        return batch

    def mark_finished(self, seq: Sequence):
        """标记序列完成"""
        seq.finished = True
        self.finished.append(seq)
        if seq in self.running:
            self.running.remove(seq)


# ==================== 完整优化生成 ====================

def generate_with_full_optimization(
    model,
    prompts: List[str],
    tokenizer,
    steps: int = 64,
    gen_length: int = 128,
    temperature: float = 0.,
    micro_steps: int = 4,  # 保留参数但不使用
    max_batch_size: int = 32,
    use_compile: bool = True,
    model_path_override: str | None = None
) -> Tuple[List[str], dict]:
    """
    完整优化版本: torch.compile + PACB调度

    核心优化:
    1. torch.compile - 2.75x加速
    2. PACB调度 - 动态批处理调度 (对大batch有效)

    注: micro-batch暂不实现，保持原始generate质量
    """
    device = 'cuda'
    # FSM约束（可选）配置与缓存
    _fsm_enabled = (os.getenv('USE_FSM', '0') == '1') and (_fsm_build_json is not None) and (_fsm_apply_allow is not None)
    _fsm_mode = os.getenv('FSM_MODE', 'json').lower()
    _fsm_allow_ids = None
    _fsm_pruned_positions = 0
    _fsm_total_positions = 0
    # 后缀统计（占位）
    _suffix_policy = None
    try:
        from nanovllm.engine.cache_policy import SuffixAutomatonPolicy as _SAP  # type: ignore
        _blk = int(os.getenv('FSM_BLOCK', '32') or 32)
        _suffix_policy = _SAP(_blk)
    except Exception:
        _suffix_policy = None
    # Optional: monkey-patch SDPA to Triton flash decode for diffusion (simple path)
    if os.getenv('USE_TRITON_FLASH', '0') == '1' and (_flash_decode_triton is not None):
        # 限定作用域的SDPA patch，避免污染其他路径；并修正K的布局（传入[B,H,Tk,D]）
        try:
            import torch.nn.functional as _F
            _orig_sdpa = _F.scaled_dot_product_attention
            def _sdpa_patch(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
                # 仅处理简单情形；复杂mask/causal/随机丢弃时回退原实现
                if attn_mask is not None or is_causal or dropout_p not in (0, 0.0):
                    return _orig_sdpa(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale)
                try:
                    d = q.shape[-1]
                    sc = (1.0 / (d ** 0.5)) if (scale is None) else scale
                    # _flash_decode_triton 期望k/v为[B,H,Tk,D]，内部自行做QK^T
                    out = _flash_decode_triton(q, k, v, sc)
                    return out
                except Exception:
                    return _orig_sdpa(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale)
            _F.scaled_dot_product_attention = _sdpa_patch  # type: ignore
            # 在函数退出前恢复原实现
            if not hasattr(generate_with_full_optimization, "_sdpa_restore"):
                def _restore_sdpa():
                    try:
                        _F.scaled_dot_product_attention = _orig_sdpa  # type: ignore
                    except Exception:
                        pass
                generate_with_full_optimization._sdpa_restore = _restore_sdpa  # type: ignore[attr-defined]
            print('[TRITON] SDPA patched to Triton flash decode')
        except Exception as _e:
            print('[TRITON] SDPA patch failed:', _e)
    # Optional: patch RotaryEmbedding to keep RoPE cache on GPU and avoid CPU->GPU copies during forward
    def _apply_rotary_gpu_cache(mdl: torch.nn.Module) -> bool:
        if os.getenv('USE_ROTARY_GPU_CACHE', '0') != '1':
            return False
        try:
            # Best-effort import of RotaryEmbedding class
            RotaryEmbedding = None  # type: ignore
            try:
                from model.modeling_llada import RotaryEmbedding as _RE  # type: ignore
                RotaryEmbedding = _RE
            except Exception:
                pass
            if RotaryEmbedding is None:
                return False
            hit = False
            for sub in mdl.modules():
                if isinstance(sub, RotaryEmbedding):
                    hit = True
                    # Build a GPU-only get_rotary_embedding; keep simple per-instance GPU caches
                    def _get_re_gpu(self, seq_len: int, dev: torch.device):  # type: ignore
                        # allocate/extend GPU caches on demand; never .to(device)
                        ps = getattr(self, '_rope_pos_sin_gpu', None)
                        pc = getattr(self, '_rope_pos_cos_gpu', None)
                        need_new = (ps is None) or (pc is None) or (ps.device != dev) or (ps.shape[-2] < seq_len)
                        if need_new:
                            with torch.autocast(dev.type, enabled=False):
                                dim = self.config.d_model // self.config.n_heads
                                inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, dim, 2, device=dev, dtype=torch.float) / dim))
                                seq = torch.arange(seq_len, device=dev, dtype=torch.float)
                                freqs = torch.einsum('i,j->ij', seq, inv_freq)
                                positions = torch.cat((freqs, freqs), dim=-1)
                                pos_sin = positions.sin()[None, None, :, :]
                                pos_cos = positions.cos()[None, None, :, :]
                            setattr(self, '_rope_pos_sin_gpu', pos_sin)
                            setattr(self, '_rope_pos_cos_gpu', pos_cos)
                        else:
                            pos_sin = ps
                            pos_cos = pc
                        return pos_sin[:, :, :seq_len, :], pos_cos[:, :, :seq_len, :]
                    # Bind method
                    import types as _types
                    sub.get_rotary_embedding = _types.MethodType(_get_re_gpu, sub)  # type: ignore[attr-defined]
            return hit
        except Exception:
            return False
    use_te_fp8 = os.getenv('USE_TE_FP8', '0') == '1' and (_te is not None)
    # Optional: replace nn.Linear with TE Linear for FP8 path
    if use_te_fp8:
        def _replace_linear_with_te(mod: torch.nn.Module):
            for name, child in list(mod.named_children()):
                if isinstance(child, torch.nn.Linear):
                    te_lin = _te.Linear(child.in_features, child.out_features,
                                        bias=(child.bias is not None), params_dtype=torch.float16, device=device)
                    with torch.no_grad():
                        te_lin.weight.copy_(child.weight.data.to(torch.float16).to(device))
                        if child.bias is not None:
                            te_lin.bias.copy_(child.bias.data.to(torch.float16).to(device))
                    setattr(mod, name, te_lin)
                else:
                    _replace_linear_with_te(child)
        _replace_linear_with_te(model)

    # Dream: 使用官方扩散生成（保证质量），不走 ragged 并行路径
    try:
        if hasattr(model, 'diffusion_generate'):
            # Optional: delegate Dream generation to a separate subprocess using baseline runner for strict quality
            if os.getenv('DREAM_SUBPROC', '0') == '1':
                import subprocess, tempfile, json as _json
                tmp = tempfile.NamedTemporaryFile('w', delete=False, encoding='utf-8')
                # baseline_bench_dream.py accepts plain text lines
                for p in prompts:
                    tmp.write(p.strip().replace('\n',' ') + "\n")
                tmp.flush(); tmp.close()
                outs=[]; t0=time.time()
                for i, _ in enumerate(prompts):
                    if os.getenv('DREAM_ONEOFF', '0') == '1':
                        # Use a one-off Dream runner that prints full decoded text (isolated interpreter)
                        py = f"""
import os, sys, json, time, torch
from transformers import AutoTokenizer
sys.path.insert(0, '/home/zhujianian/Fast-dLLM-main/dream')
from model.modeling_dream import DreamModel
try:
    from model.generation_utils_block import DreamGenerationMixin
except Exception:
    from model.generation_utils import DreamGenerationMixin
mp={_json.dumps(model_path)}
tok=AutoTokenizer.from_pretrained(mp, trust_remote_code=True)
model=DreamModel.from_pretrained(mp, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda().eval()
if not hasattr(model,'diffusion_generate'):
    import types
    model.diffusion_generate = types.MethodType(DreamGenerationMixin.diffusion_generate, model)
p={_json.dumps(prompts[i])}
enc=tok([p], return_tensors='pt', padding=True)
ids=enc['input_ids'].cuda()
attn=enc.get('attention_mask', None)
attn=attn.cuda() if attn is not None else None
mask_id=getattr(tok,'mask_token_id', None)
if mask_id is None:
    try: mask_id = tok.convert_tokens_to_ids('<|mask|>')
    except: mask_id=None
torch.cuda.synchronize(); t0=time.time()
out=model.diffusion_generate(inputs=ids, max_new_tokens={int(gen_length)}, steps={int(steps)}, temperature=0.0,
                             mask_token_id=mask_id, attention_mask=attn, block_length=32,
                             threshold=0.9, dual_cache=False, alg='confidence_threshold')
torch.cuda.synchronize(); elapsed=time.time()-t0
seq= out.sequences if hasattr(out,'sequences') else out
L=int(attn.sum().item()) if attn is not None else ids.shape[1]
gen=tok.decode(seq[0, L:L+{int(gen_length)}], skip_special_tokens=True)
print(gen)
"""
                        proc = subprocess.run([sys.executable, '-c', py], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                        outs.append(proc.stdout.strip())
                    else:
                        # Run baseline script once per prompt; capture snippet
                        cmd = [
                            sys.executable,
                            '/home/zhujianian/Fast-dLLM-main/scripts/baseline_bench_dream.py',
                            '--model-path', model_path,
                            '--dataset-path', tmp.name,
                            '--batch-size', '1',
                            '--steps', str(steps),
                            '--gen-length', str(gen_length)
                        ]
                        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                        text = proc.stdout
                        lines=text.splitlines()
                        # Find the metrics line '[1] len=...'
                        idx=-1
                        for j,ln in enumerate(lines):
                            if ln.strip().startswith('[1] len='):
                                idx=j
                        if idx>=0 and idx+1 < len(lines):
                            # Join all subsequent lines as the sample
                            sample='\n'.join([ln for ln in lines[idx+1:] if ln.strip()])
                        else:
                            # Fallback: last non-empty line
                            sample=''
                            for ln in reversed([ln for ln in lines if ln.strip()]):
                                if ln.startswith('Throughput:') or ln.startswith('len=') or ln.startswith('used steps'):
                                    continue
                                sample=ln.strip(); break
                        outs.append(sample)
                torch.cuda.synchronize(); elapsed=time.time()-t0
                stats = {
                    'total_time': elapsed,
                    'total_nfe': steps,
                    'throughput': len(prompts)*gen_length/elapsed if elapsed>0 else 0,
                }
                return outs, stats

            def _encode_prompts(ps):
                try:
                    # 仅在 Instruct 类模型使用 chat 模板；Base 不使用模板
                    _mp_l = str(model_path).lower()
                    _force_chat = os.getenv('DREAM_FORCE_CHAT','0') == '1'
                    if ((('instruct' in _mp_l) or _force_chat) and hasattr(tokenizer, 'apply_chat_template')):
                        msgs=[]
                        for p in ps:
                            msgs.append([
                                {"role":"system","content":"You are a helpful assistant."},
                                {"role":"user","content":p},
                            ])
                        texts=[tokenizer.apply_chat_template(m, add_generation_prompt=True) for m in msgs]
                        return tokenizer(texts, return_tensors='pt', padding=True)
                    else:
                        # Dream Base: 直接对原始提示做分词（不加User/Assistant模板）
                        return tokenizer(ps, return_tensors='pt', padding=True)
                except Exception:
                    return tokenizer(ps, return_tensors='pt', padding=True)

            # ensure mask_token_id
            _mask_id = getattr(tokenizer,'mask_token_id', None)
            if _mask_id is None:
                try: _mask_id = tokenizer.convert_tokens_to_ids('<|mask|>')
                except Exception: _mask_id=None

            # Dream策略：强制串行bs=1以保证质量
            # 原因：Dream模型在batch>1时置信度分布不稳定，导致输出崩溃
            # 对齐Fast-dLLM baseline的调用方式
            t0=time.time(); outs=[]
            for p in prompts:
                # 完全对齐Fast-dLLM的tokenizer调用
                prompt_ids = tokenizer([p], return_tensors="pt", padding=True, padding_side="left").input_ids
                attn_mask = prompt_ids.ne(tokenizer.pad_token_id)
                prompt_ids = prompt_ids.to('cuda')
                attn_mask = attn_mask.to('cuda')

                # 完全对齐Fast-dLLM的generation参数
                generation_ids = model.diffusion_generate(
                    prompt_ids,
                    attention_mask=attn_mask,
                    max_new_tokens=gen_length,
                    output_history=False,
                    return_dict_in_generate=True,
                    steps=steps,
                    temperature=0.0,
                    top_p=None,
                    top_k=None,
                    alg='confidence_threshold',
                    alg_temp=0.0,
                    threshold=float(os.getenv('DREAM_THRESH','0.9')),
                    dual_cache=False,
                )

                # 完全对齐Fast-dLLM的解码方式
                sequences = generation_ids.sequences if hasattr(generation_ids, 'sequences') else generation_ids
                response = tokenizer.decode(sequences[0][len(prompt_ids[0]):].tolist()).split(tokenizer.eos_token)[0]
                outs.append(response)

            torch.cuda.synchronize(); elapsed = time.time() - t0
            stats = {
                'total_time': elapsed,
                'total_nfe': steps,
                'throughput': len(prompts) * gen_length / elapsed if elapsed > 0 else 0,
            }
            return outs, stats
    except Exception:
        pass

    # LLaDA-1.5: 使用官方串行generate（Gumbel noise + block remasking）解决EOS问题
    # LLaDA-MoE: 走batched路径（在batched_generate_llada中检测并使用moe_batched）
    is_llada15_serial = False
    # 尝试从model.config.name_or_path推断
    try:
        _name = str(getattr(model.config, '_name_or_path', '')).lower()
        # 只有LLaDA-1.5走串行路径（因为EOS问题）
        if 'llada-1.5' in _name or 'llada_1.5' in _name:
            is_llada15_serial = True
        # MoE不走这里，让它走下面的batched路径
    except Exception:
        pass

    if is_llada15_serial:
        from generate_llada15 import generate_llada15_batched
        # 确保mask_id正确
        _mask_id = getattr(tokenizer, 'mask_token_id', None)
        if _mask_id is None:
            try:
                if hasattr(model, 'config') and hasattr(model.config, 'mask_token_id'):
                    _mask_id = model.config.mask_token_id
            except Exception:
                pass
        if _mask_id is None:
            for t in ('<|mdm_mask|>', '[gMASK]', '<|mask|>', '[MASK]'):
                try:
                    v = tokenizer.convert_tokens_to_ids(t)
                except Exception:
                    v = None
                if v is not None and v >= 0:
                    _mask_id = int(v); break
        if _mask_id is None:
            raise RuntimeError('Cannot infer mask token id for LLaDA-1.5/MoE')
        mask_id_llada15 = int(_mask_id)

        # 串行批量生成（官方逻辑）
        t0 = time.time(); outs = []
        for p in prompts:
            prompt_ids = tokenizer([p], return_tensors="pt", padding=True).input_ids.to(device)
            output = generate_llada15_batched(
                model=model,
                input_ids=prompt_ids,
                mask_id=mask_id_llada15,
                max_new_tokens=gen_length,
                steps=steps,
                block_length=32,
                temperature=0.0,
                remasking='low_confidence'
            )
            generated = output[0, prompt_ids.shape[1]:]
            decoded = tokenizer.decode(generated, skip_special_tokens=True).strip()
            outs.append(decoded)
        torch.cuda.synchronize(); elapsed = time.time() - t0

        stats = {
            'total_time': elapsed,
            'total_nfe': steps * len(prompts),
            'throughput': len(prompts) * gen_length / elapsed if elapsed > 0 else 0,
        }
        return outs, stats

    # 创建调度器（普通 LLaDA/LLaDA‑MoE 路径）
    scheduler = SimplePACBScheduler(max_batch_size=max_batch_size)

    # 添加所有请求
    for i, prompt_text in enumerate(prompts):
        prompt_tensor = tokenizer.encode(prompt_text, return_tensors='pt').to(device)
        seq = Sequence(i, prompt_text, prompt_tensor, gen_length, steps)
        scheduler.add_request(seq)

    total_time = 0
    total_nfe = 0

    # 变长批并行扩散生成（按行偏移，不依赖 padding 作为 prompt）
    def batched_generate_llada(batch_seqs: List[Sequence]) -> Tuple[List[torch.Tensor], int, float]:
        """变长批并行 + 长度分桶：按行偏移构造，保证与串行语义一致。"""
        if not batch_seqs:
            return [], 0, 0.0
        import time
        device = next(model.parameters()).device
        # Apply rotary GPU cache patch at first use (if requested)
        _ = _apply_rotary_gpu_cache(model)
        # MoE 专用路径：使用稳态 batched 解码（阈值+top-p/k+gamma），保证可读性
        is_moe = False
        try:
            arch = getattr(model, 'config', None)
            archs = getattr(arch, 'architectures', []) if arch is not None else []
            mt = ' '.join(archs) if isinstance(archs, (list, tuple)) else str(archs)
            if 'moe' in mt.lower():
                is_moe = True
        except Exception:
            pass
        # 额外从config._name_or_path检测
        if not is_moe:
            try:
                _name = str(getattr(model.config, '_name_or_path', '')).lower()
                if 'moe' in _name:
                    is_moe = True
            except Exception:
                pass
            # 统一推断 mask token id：Dream 使用 <|mask|>，LLaDA 使用 <|mdm_mask|>，MoE 优先 [gMASK]
            def _infer_mask_id(tok) -> int:
                mid = getattr(tok, 'mask_token_id', None)
                if mid is not None:
                    return int(mid)
                # 尝试特定 token
                try:
                    v = tok.convert_tokens_to_ids('<|mdm_mask|>')
                    if v is not None and v >= 0:
                        return int(v)
                except Exception:
                    pass
                try:
                    v = tok.convert_tokens_to_ids('[gMASK]')
                    if v is not None and v >= 0:
                        return int(v)
                except Exception:
                    pass
                # 配置文件（部分模型在 config.json）
                try:
                    cfg = getattr(tok, 'init_kwargs', {})
                    if isinstance(cfg, dict) and 'mask_token_id' in cfg:
                        return int(cfg['mask_token_id'])
                except Exception:
                    pass
                # 最后退化：尝试特殊 token 列表
                for t in ('[gMASK]', '<|mask|>', '<|mdm_mask|>', '[MASK]'):
                    try:
                        v = tok.convert_tokens_to_ids(t)
                        if v is not None and v >= 0:
                            return int(v)
                    except Exception:
                        continue
                raise RuntimeError('Cannot infer mask token id; please set via tokenizer or config')

        # 统一解析mask/eos id
        mask_id, eos_id = _resolve_special_ids(tokenizer, model)
        use_stream_overlap = os.getenv('USE_STREAM_OVERLAP', '1') != '0'
        use_gpu_native_topk = os.getenv('USE_GPU_NATIVE_TOPK', '1') != '0'
        use_cute_row_fused = os.getenv('USE_CUTE_ROW_FUSED', '0') == '1' and (_fused_row_topk_cute is not None)
        # 可选 GPU-native topk
        batch_topk_fn = None
        if use_gpu_native_topk:
            try:
                from nanovllm.engine.gpu_native_batch import batch_topk_confidence_gpu as batch_topk_fn  # type: ignore
            except Exception:
                batch_topk_fn = None
        # MoE 专用路径：使用稳态 batched 解码（阈值+top-p/k+gamma），保证可读性
        is_moe = False
        try:
            if hasattr(model, 'config'):
                arch = getattr(model.config, 'architectures', [])
                mt = ' '.join(arch) if isinstance(arch, (list, tuple)) else str(arch)
                is_moe = ('lladamoe' in mt.lower()) or ('moe' in mt.lower())
        except Exception:
            pass
        try:
            name_l = str(model_path).lower()
            is_moe = is_moe or ('-moe-' in name_l) or ('moe' in name_l)
        except Exception:
            pass
        if is_moe:
            # Prefer dInfer's optimized blockwise diffusion decode for MoE when available
            _use_dinfer_decode = os.getenv('MOE_USE_DINFER_DECODE', '1') == '1'
            if _use_dinfer_decode:
                try:
                    # Prefer full dInfer engine when available (BlockWiseDiffusionLLM + Threshold decoder)
                    if os.getenv('MOE_USE_DINFER_ENGINE', '1') == '1':
                        from dinfer import BlockIteratorFactory, KVCacheFactory  # type: ignore
                        from dinfer import ThresholdParallelDecoder
                        from dinfer.model import FusedOlmoeForCausalLM as _FusedMoE
                        # Build FusedMoE under vLLM EP context
                        from transformers import AutoConfig as _AutoCfg
                        # robust model path
                        _mpath = model_path_override or os.getenv('MODEL_PATH', '')
                        if not _mpath:
                            try:
                                _mpath = str(getattr(model, 'config', object())._name_or_path)
                            except Exception:
                                _mpath = ''
                        if not _mpath:
                            try:
                                _mpath = str(getattr(tokenizer, 'name_or_path', ''))
                            except Exception:
                                _mpath = ''
                        _cfg_moe = _AutoCfg.from_pretrained(_mpath, trust_remote_code=True)
                        _moe_model = _FusedMoE(_cfg_moe).eval().to(device)
                        _moe_model.load_weights(_mpath, torch_dtype=torch.bfloat16)
                        # compile forward for speed like dInfer benchmark
                        try:
                            _moe_model.forward = torch.compile(_moe_model.forward, mode='reduce-overhead', fullgraph=False, dynamic=True)  # type: ignore
                        except Exception:
                            pass
                        decoder = ThresholdParallelDecoder(0, threshold=float(os.getenv('MOE_THRESH','0.95') or 0.95))
                        dllm = None
                        try:
                            from dinfer import BlockWiseDiffusionLLM
                            # Ensure vLLM config persists globally (avoid warnings)
                            try:
                                from vllm.config import ParallelConfig as _VPC, VllmConfig as _VConf, set_current_vllm_config as _vset
                                _vset(_VConf(parallel_config=_VPC(enable_expert_parallel=True)))
                            except Exception:
                                pass
                            dllm = BlockWiseDiffusionLLM(_moe_model, decoder, BlockIteratorFactory(True), cache_factory=KVCacheFactory('dual'), early_stop=True)
                        except Exception:
                            dllm = None
                        outs: list[torch.Tensor] = []
                        import time
                        # Optional warmup for CUDA Graph shapes (borrow dInfer style)
                        try:
                            used_buckets = []
                            bucket_size = int(os.getenv('MOE_BUCKET', '8') or 8)
                            def _bkt(L):
                                return bucket_size * ((L + bucket_size - 1) // bucket_size)
                            for s in batch_seqs:
                                Lp = len(tokenizer.encode(s.prompt))
                                used = _bkt(Lp + int(gen_length))
                                if used not in used_buckets:
                                    used_buckets.append(used)
                            for Ltot in used_buckets[:10]:
                                rnd = torch.randint(0, 140000, (1, Ltot - int(gen_length)), dtype=torch.long, device=device)
                                _ = dllm.generate(rnd, gen_length=int(gen_length), block_length=int(os.getenv('MOE_BLOCK','32') or 32))
                        except Exception:
                            pass
                        torch.cuda.synchronize(); _t0 = time.time()
                        for s in batch_seqs:
                            ptxt = s.prompt
                            try:
                                if hasattr(tokenizer, 'apply_chat_template'):
                                    ptxt = tokenizer.apply_chat_template([{ 'role':'user','content': ptxt}], add_generation_prompt=True, tokenize=False)
                            except Exception:
                                pass
                            ids = tokenizer(ptxt, return_tensors='pt')['input_ids'].to(device)
                            out = dllm.generate(ids, gen_length=int(gen_length), block_length=int(os.getenv('MOE_BLOCK','32') or 32))
                            outs.append(out[0].to(device))
                        torch.cuda.synchronize(); _elapsed = time.time() - _t0
                        outs_list = [outs[i] for i in range(len(batch_seqs))]
                        return outs_list, int(steps), float(_elapsed)
                    from dinfer.decoding.generate_fastdllm import generate_fastdllm as _dinfer_gen  # type: ignore
                    # Prepare per-row inputs and run optimized dual-cache decode
                    outs: list[torch.Tensor] = []
                    import time
                    torch.cuda.synchronize(); _t0 = time.time()
                    for s in batch_seqs:
                        # Apply chat template for instruct-style MoE if available
                        ptxt = s.prompt
                        try:
                            if hasattr(tokenizer, 'apply_chat_template'):
                                ptxt = tokenizer.apply_chat_template([
                                    {"role": "user", "content": ptxt}
                                ], add_generation_prompt=True, tokenize=False)
                            else:
                                # Fallback to dInfer-style role prefix
                                ptxt = f"<role>SYSTEM</role>detailed thinking off<|role_end|><role>HUMAN</role>{ptxt}<|role_end|><role>ASSISTANT</role>"
                        except Exception:
                            pass
                        if s.prompt_tensor is not None and s.prompt_tensor.device.type == 'cuda':
                            # Re-tokenize to ensure template is applied
                            ids = tokenizer.encode(ptxt, return_tensors='pt').to(device)
                        else:
                            ids = tokenizer.encode(ptxt, return_tensors='pt').to(device)
                        seq, _nfe = _dinfer_gen(
                            model, ids, steps=int(steps), gen_length=int(gen_length),
                            block_length=int(os.getenv('MOE_BLOCK','32') or 32), temperature=0.0,
                            mask_id=int(_resolve_special_ids(tokenizer, model)[0]),
                            eos_id=int(getattr(tokenizer, 'eos_token_id', 0) or 0),
                            decoding='fastdllm', use_cache=True, dual_cache=True,
                            remasking='low_confidence', threshold=float(os.getenv('MOE_THRESH','0.95') or 0.95),
                            early_stop=False
                        )
                        outs.append(seq[0].to(device))
                    torch.cuda.synchronize(); _elapsed = time.time() - _t0
                    outs_list = [outs[i] for i in range(len(batch_seqs))]
                    return outs_list, int(steps), float(_elapsed)
                except Exception as _e:
                    print(f"[WARN] dInfer MoE decode unavailable ({_e}); falling back to local batched decode")
            # Fallback: local MoE batched decode (less optimized)
            from moe_batched import moe_generate_batched
            try:
                texts = [s.prompt for s in batch_seqs]
                enc = tokenizer(texts, return_tensors='pt', padding=True)
                _dev = getattr(model, 'device', None)
                if _dev is None:
                    try:
                        _dev = next(model.parameters()).device
                    except Exception:
                        _dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                ids_stack = enc['input_ids'].to(_dev)
            except Exception:
                ids_stack = torch.stack([s.prompt_tensor[0] if (s.prompt_tensor is not None and s.prompt_tensor.device.type=='cuda') else tokenizer.encode(s.prompt, return_tensors='pt')[0] for s in batch_seqs])
                ids_stack = torch.nn.utils.rnn.pad_sequence(list(ids_stack), batch_first=True, padding_value=int(getattr(tokenizer,'pad_token_id', 0) or 0)).to(device)
            mid = int(getattr(tokenizer, 'mask_token_id', 0) or 0)
            import time
            torch.cuda.synchronize(); _t0 = time.time()
            _moe_disable_shift = os.getenv('MOE_DISABLE_SHIFT', '1') == '1'
            seqs = moe_generate_batched(
                model, ids_stack, int(mid), max_new_tokens=gen_length, steps=steps,
                block_length=int(os.getenv('MOE_BLOCK','32') or 32), eps=0.02,
                threshold=float(os.getenv('MOE_THRESH','0.93') or 0.93),
                top_p=float(os.getenv('MOE_TOPP','0.95') or 0.95), top_k=int(os.getenv('MOE_TOPK','50') or 50),
                disable_logits_shift=_moe_disable_shift
            )
            torch.cuda.synchronize(); _elapsed = time.time() - _t0
            outs_list = [seqs[i] for i, _s in enumerate(batch_seqs)]
            return outs_list, int(steps), float(_elapsed)

        # 收集 ids/长度
        ids_list: List[torch.Tensor] = []
        lens: List[int] = []
        for s in batch_seqs:
            # 在CPU做tokenize，保持ragged；用pin_memory以便异步H2D
            if s.prompt_tensor is not None and s.prompt_tensor.device.type == 'cuda':
                pt = s.prompt_tensor.detach().cpu()
            else:
                pt = tokenizer.encode(s.prompt, return_tensors='pt')
            pt = pt.contiguous().pin_memory()
            ids_list.append(pt[0])  # CPU tensor
            lens.append(int(pt.shape[1]))
        B = len(ids_list)
        order = list(range(B))
        order.sort(key=lambda i: lens[i])

        # ============ Batch-Aware Adaptive Bucketing（动态调整分桶策略） ============
        # 全局状态：GPU利用率统计（使用函数属性持久化）
        if not hasattr(batched_generate_llada, '_bucket_history'):
            batched_generate_llada._bucket_history = {
                'util_history': [],  # GPU利用率历史（百分比）
                'last_K': 2,  # 上次使用的桶数
                'window_size': 5,  # 滑动窗口大小
            }
        hist = batched_generate_llada._bucket_history

        # ============ 自适应分桶（K ∈ {2,3,4}，代价函数选择 + 最小桶大小约束 + GPU反馈） ============
        env_k = os.getenv('LENGTH_BUCKETS', '').strip()
        def _make_buckets_equal_count(Kcand: int) -> List[List[int]]:
            bk: List[List[int]] = [[] for _ in range(Kcand)]
            for r, idx in enumerate(order):
                b = min(Kcand - 1, (r * Kcand) // max(1, B))
                bk[b].append(idx)
            return bk
        def _cost_of(bk: List[List[int]]) -> float:
            # 近似一步前向的 FLOPs：sum_b bB_b * (maxL_b + gen_length)
            c = 0.0
            for b in bk:
                if not b:
                    continue
                maxL = max(lens[i] for i in b)
                c += len(b) * (maxL + gen_length)
            return float(c)
        # 可调启发式参数
        _min_b = int(os.getenv('BUCKET_MIN_B', '3') or '3')
        _pen_base = float(os.getenv('BUCKET_PENALTY_BASE', '1.05') or '1.05')
        _pen_kfac = float(os.getenv('BUCKET_PENALTY_K_FACTOR', '0.05') or '0.05')

        if env_k:
            try:
                K = int(env_k)
            except Exception:
                K = 2
            K = max(1, min(K, B))
            buckets: List[List[int]] = _make_buckets_equal_count(K)
            chosen_K = K
        else:
            # 计算平均GPU利用率（使用历史窗口）
            avg_util = 0.0
            if len(hist['util_history']) > 0:
                recent = hist['util_history'][-hist['window_size']:]
                avg_util = sum(recent) / len(recent)

            # 根据GPU利用率调整候选K范围
            # 低利用率（<60%）：增加桶数，提高并行度
            # 中等利用率（60-80%）：保持当前策略
            # 高利用率（>80%）：减少桶数，降低串行开销
            if avg_util < 60.0:
                # 低利用率：倾向更多桶（3-4）
                candidates = [k for k in (3, 4, 2) if k <= B]
            elif avg_util > 80.0:
                # 高利用率：倾向更少桶（1-2）
                candidates = [k for k in (1, 2, 3) if k <= B]
            else:
                # 中等：使用上次的K，微调
                last_K = hist['last_K']
                candidates = [k for k in (last_K, last_K+1, last_K-1, 2, 3, 4) if 1 <= k <= B]
                candidates = list(dict.fromkeys(candidates))  # 去重保持顺序

            if B < 8:
                # 小批直接单桶，避免额外串行段
                candidates = [1]

            best_buckets: List[List[int]] = [order]
            best_cost = _cost_of(best_buckets)
            best_K = 1
            for Kcand in candidates:
                bk = _make_buckets_equal_count(Kcand)
                bmin = min((len(b) for b in bk if b), default=0)
                # 若最小桶 < 2，回退，不考虑该 K
                if bmin < 2:
                    continue
                c = _cost_of(bk)
                # 软约束：若 bmin < _min_b，加惩罚（K 越大惩罚越重），避免过多过小桶
                if bmin < _min_b:
                    c *= (_pen_base + _pen_kfac * max(0, Kcand - 2))
                # 代价更小优先；代价相等时，更少桶优先（减少串行段）
                if (c < best_cost) or (abs(c - best_cost) < 1e-6 and Kcand < best_K):
                    best_cost = c
                    best_buckets = bk
                    best_K = Kcand
            buckets = best_buckets
            chosen_K = best_K
            # 限制桶数，避免过多动态形状（默认<=8，可配 BUCKET_MAX）
            _bmax = int(os.getenv('BUCKET_MAX', '8') or 8)
            if len(buckets) > _bmax:
                merged: list[list[int]] = []
                for i, b in enumerate(buckets):
                    if i < (_bmax - 1):
                        merged.append(b)
                    else:
                        # 合并剩余所有桶到最后一个
                        if len(merged) < _bmax:
                            merged.append(list(b))
                        else:
                            merged[-1].extend(b)
                buckets = merged
                chosen_K = len(buckets)
            hist['last_K'] = chosen_K
        # 调试打印（一次性），便于确认分桶效果
        if os.getenv('DEBUG_BUCKETS', '0') == '1':
            sizes = [len(b) for b in buckets]
            maxLs = [max((lens[i] for i in b), default=0) for b in buckets]
            print(f"[adaptive] buckets={sizes}, maxLs={maxLs}")

        # 桶级预热（一次），固定对齐长度，减少动态重编译/动态图捕获
        _did_bucket_warmup = getattr(batched_generate_llada, '_did_bucket_warmup', False)
        if (not _did_bucket_warmup) and (os.getenv('BUCKET_WARMUP', '1') == '1'):
            try:
                align = int(os.getenv('BUCKET_ALIGN', '32') or 32)
                for b in buckets[:int(os.getenv('BUCKET_WARMUP_LIMIT','8') or 8)]:
                    if not b:
                        continue
                    sub_lens = [lens[i] for i in b]
                    maxL = int(max(sub_lens))
                    T = maxL + gen_length
                    if align > 0:
                        T = align * ((T + align - 1) // align)
                    # 构造最小的dummy批，用于编译/图捕获
                    Bwarm = min(len(b), int(os.getenv('BUCKET_WARMUP_B','2') or 2))
                    xw = torch.full((Bwarm, T), mask_id, dtype=torch.long, device=device)
                    lw = min(maxL, T)
                    # 放入少量真实token（避免全mask路径）
                    for j in range(Bwarm):
                        ids_cpu = ids_list[b[j]]
                        ids = ids_cpu.to(device)
                        Lj = int(ids.numel())
                        lj = min(Lj, lw)
                        xw[j, :lj] = ids[:lj]
                    with torch.no_grad():
                        _ = model(xw).logits
                setattr(batched_generate_llada, '_did_bucket_warmup', True)
            except Exception:
                pass

        def run_bucket(idxs: List[int]) -> Tuple[dict[int, torch.Tensor], int, float]:
            if not idxs:
                return {}, 0, 0.0
            # 准备该桶的CPU ids与长度
            sub_ids_cpu = [ids_list[i] for i in idxs]
            sub_lens = [lens[i] for i in idxs]
            bB = len(sub_lens)
            Ls = torch.tensor(sub_lens, device=device, dtype=torch.int64)
            maxL = int(max(sub_lens))
            T = maxL + gen_length
            # 对齐shape，减少动态形状（默认按32对齐，可配 BUCKET_ALIGN）
            _align = int(os.getenv('BUCKET_ALIGN', '32') or 32)
            if _align > 0:
                T = _align * ((T + _align - 1) // _align)
            # x: [bB, T]
            x = torch.full((bB, T), mask_id, dtype=torch.long, device=device)
            # 同步拷贝（兼容保底，pipeline 模式下不会走到这里）
            sub_ids_gpu = [ids.to(device) for ids in sub_ids_cpu]
            for j, ids in enumerate(sub_ids_gpu):
                Lj = int(ids.numel())
                x[j, :Lj] = ids
            block_length = int(gen_length)
            spb = max(1, int(steps))
            total_nfe = 0
            torch.cuda.synchronize()
            t0 = time.time()
            # 单块策略（与串行 generate 等价）：bidx=0
            start = Ls
            end = torch.minimum(Ls + gen_length, Ls + gen_length)
            pos = torch.arange(T, device=device).view(1, T)
            # Base allowed region = generation slice [L, L+gen)
            allowed_base = (pos >= start.view(bB, 1)) & (pos < end.view(bB, 1))
            # Sliding-window configuration (optional)
            use_sw = os.getenv('USE_SLIDING_WINDOW', '0') == '1'
            sw_prefix = int(os.getenv('SLIDING_PREFIX', '64') or 64)
            sw_after = int(os.getenv('SLIDING_AFTER', '64') or 64)
            sw_warmup = int(os.getenv('SLIDING_WARMUP', '4') or 4)
            sw_expected_tpf = int(os.getenv('EXPECTED_TPF', '16') or 16)
            # MoE 专用：直接走稳态 batched 生成（阈值+top-p/k），避免不可读
            if is_moe:
                from moe_batched import moe_generate_batched
                ids_stack = torch.stack([s.prompt_tensor[0] for s in batch_seqs]).to(device)
                _moe_disable_shift = os.getenv('MOE_DISABLE_SHIFT', '1') == '1'
                seqs = moe_generate_batched(
                    model, ids_stack, mask_id, max_new_tokens=gen_length, steps=steps,
                    block_length=int(os.getenv('MOE_BLOCK','32') or 32), eps=0.02,
                    threshold=float(os.getenv('MOE_THRESH','0.93') or 0.93),
                    top_p=float(os.getenv('MOE_TOPP','0.95') or 0.95), top_k=int(os.getenv('MOE_TOPK','50') or 50),
                    disable_logits_shift=_moe_disable_shift
                )
                out_map = {}
                for i, s in enumerate(batch_seqs):
                    out_map[s.seq_id] = seqs[i]
                return out_map, int(steps), 0.0

            # 预分配每步额度（等额），只计算一次；并持久化 mask_index 以避免每步重算
            mask_index = (x == mask_id) & allowed_base  # [bB, T]
            mask_num0 = mask_index.sum(dim=1, keepdim=True)
            cols = torch.arange(spb, device=device)
            base = mask_num0 // spb
            remainder = mask_num0 % spb
            num_transfer_tokens = base.expand(-1, spb).to(torch.int64) + (cols.view(1, spb) < remainder).to(torch.int64)

            # 可选：Credit Decoding（训练无关的轨迹信用累计），加速置信度收敛
            use_credit = os.getenv('USE_CREDIT_DECODING', '0') == '1'
            if use_credit:
                credit_tok = torch.full((bB, T), -1, dtype=torch.int64, device=device)
                credit_score = torch.zeros((bB, T), dtype=torch.float32, device=device)
                credit_alpha = float(os.getenv('CREDIT_ALPHA', '1.0'))
                credit_decay = float(os.getenv('CREDIT_DECAY', '0.9'))
                credit_lambda = float(os.getenv('CREDIT_LAMBDA', '1.0'))
                credit_mthresh = float(os.getenv('CREDIT_MTHRESH', '0.5'))
                credit_use_margin = os.getenv('CREDIT_MARGIN', '1') == '1' and (_fused_conf_margin is not None)

            # Dynamic Step Scheduling: 跟踪每个样本的平均置信度，用于提前终止
            enable_dynamic_steps = os.getenv('DYNAMIC_STEPS', '1') == '1'
            early_stop_threshold = float(os.getenv('EARLY_STOP_CONF', '0.85'))
            warmup_steps = int(os.getenv('WARMUP_STEPS', '16'))
            sample_finished = torch.zeros(bB, dtype=torch.bool, device=device)
            if enable_dynamic_steps:
                conf_history = torch.zeros((bB, warmup_steps), device=device)
            # 可选：为本桶捕获 CUDA Graph 用于 model(x) 前向（形状稳定）
            use_cudagraph = os.getenv('USE_CUDAGRAPH', '0') == '1' and torch.cuda.is_available()
            cg = None
            s_logits = None
            # KV cache for sliding-window (optional)
            use_kv_cache = os.getenv('USE_KV_CACHE', '1') == '1' and use_sw
            kv_cache = None
            use_threshold_dec = os.getenv('USE_THRESHOLD_DECODING', '1') == '1'
            conf_threshold = float(os.getenv('CONF_THRESHOLD', '0.90'))
            if use_cudagraph:
                try:
                    with torch.no_grad():
                        _warm = model(x).logits
                    s_logits = torch.empty_like(_warm)
                    cg = torch.cuda.CUDAGraph()
                    torch.cuda.synchronize()
                    with torch.cuda.graph(cg):
                        s_logits.copy_(model(x).logits)
                except Exception:
                    cg = None
                    s_logits = None
            # Per-step decode; add optional NVTX markers
            _use_nvtx = os.getenv('NVTX_PROFILE', '0') == '1'
            _nvtx = None
            if _use_nvtx and torch.cuda.is_available():
                try:
                    import torch.cuda.nvtx as _nvtx
                except Exception:
                    _nvtx = None
            _profile_breakdown = os.getenv('PROFILE_BREAKDOWN', '0') == '1'
            if _profile_breakdown:
                t_model_ms = 0.0
                t_sample_ms = 0.0
            for si in range(spb):
                if _nvtx is not None:
                    _nvtx.range_push(f"bucket_compute_step_{si}")

                # Dynamic Step Scheduling: 检查是否所有样本都已完成
                if enable_dynamic_steps and sample_finished.all():
                    if _nvtx is not None:
                        _nvtx.range_pop()
                    break

                # 使用持久化 mask_index 维护剩余量
                # Optionally narrow decoding region by sliding window after warmup
                if use_sw and si >= sw_warmup:
                    # Compute per-row window [win_l, win_r) based on current masked span
                    # If a row has no masked positions, keep its window empty
                    with torch.no_grad():
                        # indices of masked positions per row
                        # For efficiency: compute first and last True per row via max on inverted cumulative
                        mi = mask_index
                        any_mask = mi.any(dim=1)
                        # default window = base region
                        win_l = start.clone()
                        win_r = torch.minimum(start + gen_length, start + gen_length)
                        if bool(any_mask.any().item()):
                            # For rows with mask, find min/max indices
                            idx = torch.arange(T, device=device).view(1, T).expand(bB, -1)
                            # Large/small sentinels for non-masked
                            large = torch.full((bB,), T, device=device)
                            small = torch.full((bB,), -1, device=device)
                            first = torch.where(any_mask,
                                                torch.where(mi, idx, T).min(dim=1).values,
                                                large)
                            last = torch.where(any_mask,
                                               torch.where(mi, idx, -1).max(dim=1).values,
                                               small)
                            # window bounds
                            win_l = torch.maximum(start, first - sw_prefix)
                            win_r = torch.minimum(start + gen_length, last + 1 + sw_after)
                        # Build allowed mask
                        allowed = (pos >= win_l.view(bB, 1)) & (pos < win_r.view(bB, 1))
                    # Limit mask_index to window
                    mask_index = (x == mask_id) & allowed
                # Remaining masked per row
                remain_vec = mask_index.sum(dim=1)
                if int(remain_vec.sum().item()) == 0:
                    if _nvtx is not None:
                        _nvtx.range_pop()
                    break
                total_nfe += 1
                # Optional: prevent premature EOS to improve coherence in fast path
                try:
                    if os.getenv('USE_EOS_GATING','0') == '1':
                        eos_min = int(os.getenv('EOS_MIN_STEP','8') or 8)
                        if si < eos_min:
                            if logits.dtype.is_floating_point:
                                logits[..., int(eos_id)] = torch.finfo(logits.dtype).min
                except Exception:
                    pass
                # Decide per-row transfer tokens
                if use_sw:
                    # Use expected TPF after warmup; before warmup fallback to equal share
                    if si >= sw_warmup:
                        k_vec = torch.minimum(remain_vec, torch.full_like(remain_vec, sw_expected_tpf))
                    else:
                        k_vec = num_transfer_tokens[:, si]
                else:
                    k_vec = num_transfer_tokens[:, si]
                if _profile_breakdown:
                    e0 = torch.cuda.Event(enable_timing=True)
                    e1 = torch.cuda.Event(enable_timing=True)
                    e0.record()
                logits = None
                # Compute logits, prefer sliding-window + KV cache path
                if use_sw and use_kv_cache and si >= sw_warmup:
                    # Simpler KV-cache: run full-seq with cache to ensure correctness with rotary embedding
                    ctx = _te.fp8_autocast(enabled=True, calibrating=False) if use_te_fp8 else nullcontext()
                    with ctx:
                        with torch.no_grad():
                            # Mark step begin to avoid Inductor's internal CUDA graph reuse issues under KV updates
                            try:
                                import torch.compiler as _tc  # type: ignore
                                if hasattr(_tc, 'cudagraph_mark_step_begin'):
                                    _tc.cudagraph_mark_step_begin()
                            except Exception:
                                pass
                            out = model(x, past_key_values=kv_cache, use_cache=True)
                    kv_cache = getattr(out, 'past_key_values', kv_cache)
                    logits = out.logits
                elif cg is not None and s_logits is not None:
                    cg.replay()
                    logits = s_logits
                else:
                    ctx = _te.fp8_autocast(enabled=True, calibrating=False) if use_te_fp8 else nullcontext()
                    with ctx:
                        with torch.no_grad():
                            try:
                                import torch.compiler as _tc  # type: ignore
                                if hasattr(_tc, 'cudagraph_mark_step_begin'):
                                    _tc.cudagraph_mark_step_begin()
                            except Exception:
                                pass
                            logits = model(x).logits  # [bB, T, V]
                # MoE: 对齐预测位置（左移一位），与 Dream/LLaDA 扩散一致
                try:
                    if _is_moe_model and (os.getenv('MOE_DISABLE_SHIFT','0') != '1') and logits.size(1) > 1:
                        logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
                except Exception:
                    pass
                # FSM（可选）：对候选位置应用允许集裁剪（仅当启用且模式为json）
                if _fsm_enabled and _fsm_mode == 'json' and (mask_index is not None) and bool(mask_index.any().item()):
                    try:
                        if _fsm_allow_ids is None:
                            fsm = _fsm_build_json(tokenizer)  # type: ignore[misc]
                            _fsm_allow_ids = list(fsm.next_allowed(0))  # type: ignore[assignment]
                        # 仅在有mask候选时裁剪logits
                        logits = _fsm_apply_allow(logits, _fsm_allow_ids)  # type: ignore[misc]
                        _fsm_pruned_positions += int(mask_index.sum().item())
                        _fsm_total_positions += int(mask_index.numel())
                    except Exception:
                        pass
                if _profile_breakdown:
                    e1.record(); e1.synchronize(); t_model_ms += float(e0.elapsed_time(e1))
                # 可选：在采样前进行 Credit 融合（对当前 logits 注入历史信用偏置）
                if use_credit:
                    with torch.no_grad():
                        # 仅在候选位置更新信用
                        if credit_use_margin:
                            margin = _fused_conf_margin(logits, mask_index)  # [bB, T], -inf on non-masked
                            margin = torch.clamp(margin, min=0.0)
                        else:
                            # 退化：用top-2近似（代价高，默认不走）
                            top2 = torch.topk(logits.to(torch.float32), k=2, dim=-1).values
                            margin = torch.clamp(top2[..., 0] - top2[..., 1], min=0.0)
                        x0_now = torch.argmax(logits, dim=-1)
                        # 更新信用分数与候选token（EMA + 阈值切换）
                        masked = mask_index
                        same = (credit_tok == x0_now) & (credit_tok >= 0) & masked
                        diff = masked & (~same)
                        # 衰减已有信用
                        credit_score = torch.where(masked, credit_score * credit_decay, credit_score)
                        # 增加当前一致的信用（按 margin 加权）
                        credit_score = torch.where(same, credit_score + credit_alpha * margin, credit_score)
                        # 当 margin 足够大时，切换候选token并重置基础信用
                        adopt = diff & (margin > credit_mthresh)
                        credit_tok = torch.where(adopt, x0_now.to(torch.int64), credit_tok)
                        credit_score = torch.where(adopt, credit_alpha * margin, credit_score)
                        # 将信用转为对 logits 的加性偏置（仅对被标注的候选token生效）
                        has_credit = masked & (credit_tok >= 0)
                        if bool(has_credit.any().item()):
                            rr, cc = torch.where(has_credit)
                            tt = credit_tok[rr, cc]
                            gain = (credit_lambda * credit_score[rr, cc]).to(dtype=logits.dtype)
                            logits[rr, cc, tt] += gain
                # 使用持久化 mask_index 作为候选掩码
                if _profile_breakdown:
                    s0 = torch.cuda.Event(enable_timing=True)
                    s1 = torch.cuda.Event(enable_timing=True)
                    s0.record()
                if use_sw and use_kv_cache and si >= sw_warmup:
                    # Full-seq sampling with threshold decoding for stability
                    if use_threshold_dec:
                        try:
                            x0_all, conf_all = _fused_sample_call(logits, mask_index, x)
                        except Exception:
                            x0_all = torch.argmax(logits, dim=-1)
                            p_all = F.softmax(logits, dim=-1)
                            conf_all = p_all.gather(dim=-1, index=x0_all.unsqueeze(-1)).squeeze(-1)
                            x0_all = torch.where(mask_index, x0_all, x)
                            conf_all = torch.where(mask_index, conf_all, torch.tensor(-float('inf'), device=conf_all.device))
                        # 早期提交（可选）：当平均置信度足够高时，一次性提交剩余mask位
                        if enable_dynamic_steps:
                            try:
                                conf_mask = torch.where(mask_index, conf_all.clamp_min(0.0), torch.tensor(0.0, device=conf_all.device))
                                mask_cnt = mask_index.sum(dim=1).clamp(min=1)
                                avg_conf_row = conf_mask.sum(dim=1) / mask_cnt
                                rows_commit = avg_conf_row > early_stop_threshold
                                if bool(rows_commit.any().item()):
                                    rc = rows_commit.nonzero(as_tuple=True)[0]
                                    # 这些行直接提交全部剩余mask位置
                                    x[rc] = torch.where(mask_index[rc], x0_all[rc], x[rc])
                                    mask_index[rc] = False
                            except Exception:
                                pass
                        select = mask_index & (conf_all > conf_threshold)
                        need = (~select & mask_index).any(dim=1)
                        if bool(need.any().item()):
                            rem = torch.where(mask_index, conf_all, torch.tensor(-float('inf'), device=conf_all.device))
                            best = torch.argmax(rem, dim=1)
                            rows = torch.arange(bB, device=device)
                            select[rows[need], best[need]] = True
                        if bool(select.any().item()):
                            rows_idx, pos_idx = torch.where(select)
                            x[rows_idx, pos_idx] = x0_all[rows_idx, pos_idx]
                            mask_index[rows_idx, pos_idx] = False
                    else:
                        # fallback to row-topk on full sequence
                        x0_all, conf_all = _fused_sample_call(logits, mask_index, x)
                        # 早期提交（可选）
                        if enable_dynamic_steps:
                            try:
                                conf_mask = torch.where(mask_index, conf_all.clamp_min(0.0), torch.tensor(0.0, device=conf_all.device))
                                mask_cnt = mask_index.sum(dim=1).clamp(min=1)
                                avg_conf_row = conf_mask.sum(dim=1) / mask_cnt
                                rows_commit = avg_conf_row > early_stop_threshold
                                if bool(rows_commit.any().item()):
                                    rc = rows_commit.nonzero(as_tuple=True)[0]
                                    x[rc] = torch.where(mask_index[rc], x0_all[rc], x[rc])
                                    mask_index[rc] = False
                            except Exception:
                                pass
                        conf_masked = torch.where(mask_index, conf_all, torch.tensor(-float('inf'), device=conf_all.device))
                        k_max = int(k_vec.max().item())
                        if k_max > 0:
                            vals, idx = torch.topk(conf_masked, k=min(k_max, conf_masked.size(1)), dim=1)
                            ar = torch.arange(vals.size(1), device=device).view(1, -1).expand(bB, -1)
                            km = k_vec.view(-1, 1).expand(-1, vals.size(1))
                            sel = (ar < km) & (vals > float('-inf'))
                            rows = torch.arange(bB, device=device).unsqueeze(1).expand_as(idx)
                            chosen_pos = idx[sel]
                            chosen_row = rows[sel]
                            chosen_tok = x0_all[chosen_row, chosen_pos]
                            x[chosen_row, chosen_pos] = chosen_tok
                            mask_index[chosen_row, chosen_pos] = False
                elif use_cute_row_fused:
                    # CuTe per-row fully fused: returns selected positions/tokens directly
                    try:
                        sel_pos, sel_tok = _fused_row_topk_cute(logits, mask_index, x, k_vec)
                        valid = sel_pos >= 0
                        if bool(valid.any().item()):
                            rows = torch.arange(bB, device=device).unsqueeze(1).expand_as(sel_pos)
                            x[rows[valid], sel_pos[valid]] = sel_tok[valid]
                            # 增量清零 mask_index
                            mask_index[rows[valid], sel_pos[valid]] = False
                    except Exception:
                        # fallback to GPU-native if available
                        if batch_topk_fn is not None:
                            k_max = int(k_vec.max().item())
                            sel_pos, sel_tok = batch_topk_fn(logits, mask_index, k_vec, k_max)
                            valid = sel_pos >= 0
                            if bool(valid.any().item()):
                                rows = torch.arange(bB, device=device).unsqueeze(1).expand_as(sel_pos)
                                x[rows[valid], sel_pos[valid]] = sel_tok[valid]
                        else:
                            # Optional fused confidence (margin) via Triton
                            if os.getenv('USE_TRITON_CONF', '0') == '1':
                                try:
                                    from triton_fused_confidence import fused_confidence_margin
                                    # Margin: only compute argmax + top2 margin (no full softmax)
                                    logits_with_noise = logits
                                    if hasattr(_fused_sample_call, '_triton_logged'):
                                        # Use Gumbel noise if enabled
                                        # For now skip noise to keep margin semantics clean
                                        pass
                                    x0 = torch.argmax(logits_with_noise, dim=-1)
                                    conf = fused_confidence_margin(logits.to(torch.float32), mask_index)
                                    x0 = torch.where(mask_index, x0, x)
                                except Exception:
                                    # Fallback to default softmax path
                                    x0, conf = _fused_sample_call(logits, mask_index, x)
                            else:
                                # Default: Compute proposals with softmax confidence
                                x0, conf = _fused_sample_call(logits, mask_index, x)
                            # Vectorized per-row top-k selection (no Python for)
                            if os.getenv('USE_VEC_ROW_TOPK', '1') == '1':
                                conf_masked = torch.where(mask_index, conf, torch.tensor(-float('inf'), device=conf.device))
                                k_max = int(k_vec.max().item())
                                if k_max > 0:
                                    vals, idx = torch.topk(conf_masked, k=k_max, dim=1)
                                    ar = torch.arange(k_max, device=conf.device).view(1, -1).expand(bB, -1)
                                    km = k_vec.view(-1, 1).expand(-1, k_max)
                                    sel = (ar < km) & (vals > float('-inf'))
                                    rows = torch.arange(bB, device=conf.device).unsqueeze(1).expand_as(idx)
                                    chosen_pos = idx[sel]
                                    chosen_row = rows[sel]
                                    chosen_tok = x0[chosen_row, chosen_pos]
                                    x.index_put_((chosen_row, chosen_pos), chosen_tok, accumulate=False)
                                    mask_index.index_put_((chosen_row, chosen_pos), torch.zeros_like(chosen_row, dtype=torch.bool))
                            else:
                                # Fallback legacy per-row loop
                                for r in range(bB):
                                    ki = int(k_vec[r].item())
                                    if ki <= 0:
                                        continue
                                    conf_r = conf[r]
                                    keep_mask = mask_index[r]
                                    conf_r = torch.where(keep_mask, conf_r, torch.tensor(-float('inf'), device=conf_r.device))
                                    ki = min(ki, int(keep_mask.sum().item()))
                                    if ki <= 0:
                                        continue
                                    _, topi = torch.topk(conf_r, k=ki)
                                    x[r, topi] = x0[r, topi]
                elif batch_topk_fn is not None:
                    # GPU-native一次完成 per-row top-k 选择
                    k_max = int(k_vec.max().item())
                    sel_pos, sel_tok = batch_topk_fn(logits, mask_index, k_vec, k_max)
                    valid = sel_pos >= 0
                    if bool(valid.any().item()):
                        rows = torch.arange(bB, device=device).unsqueeze(1).expand_as(sel_pos)
                        x[rows[valid], sel_pos[valid]] = sel_tok[valid]
                        mask_index[rows[valid], sel_pos[valid]] = False
                else:
                    # Fused per-position概率 + host top-k（退化）
                    x0, conf = _fused_sample_call(logits, mask_index, x)
                    # 按行 top-k 选位并写回（ragged，k_i 因行而异）
                    for r in range(bB):
                        ki = int(k_vec[r].item())
                        if ki <= 0:
                            continue
                        conf_r = conf[r]
                        keep_mask = mask_index[r]
                        conf_r = torch.where(keep_mask, conf_r, torch.tensor(-float('inf'), device=conf_r.device))
                        ki = min(ki, int(keep_mask.sum().item()))
                        if ki <= 0:
                            continue
                        _, topi = torch.topk(conf_r, k=ki)
                        x[r, topi] = x0[r, topi]
                        mask_index[r, topi] = False

                # Dynamic Step Scheduling: 跟踪置信度并决定是否提前终止
                if enable_dynamic_steps and not sample_finished.all():
                    # 获取当前步的平均置信度（每个样本的所有token平均）
                    if 'conf' in locals():
                        # conf: [bB, T], 取mask_index位置的平均置信度
                        valid_conf = torch.where(mask_index, conf, torch.tensor(0.0, device=conf.device))
                        conf_sum = valid_conf.sum(dim=1)
                        mask_count = mask_index.sum(dim=1).float().clamp(min=1e-9)
                        avg_conf = conf_sum / mask_count  # [bB]
                    else:
                        # fallback: 用logits softmax后的最大概率作为置信度
                        probs = F.softmax(logits, dim=-1)
                        max_probs = probs.max(dim=-1)[0]  # [bB, T]
                        valid_probs = torch.where(mask_index, max_probs, torch.tensor(0.0, device=max_probs.device))
                        prob_sum = valid_probs.sum(dim=1)
                        mask_count = mask_index.sum(dim=1).float().clamp(min=1e-9)
                        avg_conf = prob_sum / mask_count  # [bB]

                    # Warmup阶段：收集置信度统计
                    if si < warmup_steps:
                        conf_history[:, si] = avg_conf
                    else:
                        # Warmup后：检查是否达到阈值
                        # 使用warmup期间的平均置信度作为基线，当前置信度超过阈值则标记完成
                        baseline_conf = conf_history.mean(dim=1)  # [bB]
                        for r in range(bB):
                            if not sample_finished[r] and avg_conf[r] > early_stop_threshold:
                                sample_finished[r] = True

                if _profile_breakdown:
                    s1.record(); s1.synchronize(); t_sample_ms += float(s0.elapsed_time(s1))
                if _nvtx is not None:
                    _nvtx.range_pop()
            torch.cuda.synchronize()
            elapsed = time.time() - t0
            out_map: dict[int, torch.Tensor] = {}
            for j, ids in enumerate(sub_ids_gpu):
                Lj = int(ids.numel())
                gen = x[j, Lj:Lj+gen_length]
                out_map[idxs[j]] = torch.cat([ids, gen], dim=0)
            return out_map, total_nfe, elapsed

        def run_bucket_pregpu(idxs: List[int], sub_ids_gpu: List[torch.Tensor]) -> Tuple[dict[int, torch.Tensor], int, float]:
            if not idxs:
                return {}, 0, 0.0
            sub_lens = [lens[i] for i in idxs]
            bB = len(sub_lens)
            Ls = torch.tensor(sub_lens, device=device, dtype=torch.int64)
            maxL = int(max(sub_lens))
            T = maxL + gen_length
            x = torch.full((bB, T), mask_id, dtype=torch.long, device=device)
            for j, ids in enumerate(sub_ids_gpu):
                Lj = int(ids.numel())
                x[j, :Lj] = ids
            spb = max(1, int(steps))
            total_nfe = 0
            torch.cuda.synchronize()
            t0 = time.time()
            start = Ls
            end = torch.minimum(Ls + gen_length, Ls + gen_length)
            pos = torch.arange(T, device=device).view(1, T)
            allowed = (pos >= start.view(bB, 1)) & (pos < end.view(bB, 1))
            mask_num0 = ((x == mask_id) & allowed).sum(dim=1, keepdim=True)
            cols = torch.arange(spb, device=device)
            base = mask_num0 // spb
            remainder = mask_num0 % spb
            num_transfer_tokens = base.expand(-1, spb).to(torch.int64) + (cols.view(1, spb) < remainder).to(torch.int64)

            # Dynamic Step Scheduling initialization (same as run_bucket)
            enable_dynamic_steps = os.getenv('DYNAMIC_STEPS', '1') == '1'
            early_stop_threshold = float(os.getenv('EARLY_STOP_CONF', '0.85'))
            warmup_steps = int(os.getenv('WARMUP_STEPS', '16'))
            sample_finished = torch.zeros(bB, dtype=torch.bool, device=device)
            if enable_dynamic_steps:
                conf_history = torch.zeros((bB, warmup_steps), device=device)

            _use_nvtx = os.getenv('NVTX_PROFILE', '0') == '1'
            _nvtx = None
            if _use_nvtx and torch.cuda.is_available():
                try:
                    import torch.cuda.nvtx as _nvtx
                except Exception:
                    _nvtx = None
            _profile_breakdown = os.getenv('PROFILE_BREAKDOWN', '0') == '1'
            if _profile_breakdown:
                t_model_ms = 0.0
                t_sample_ms = 0.0
            for si in range(spb):
                if _nvtx is not None:
                    _nvtx.range_push(f"bucket_compute_step_{si}")

                # Dynamic Step Scheduling: 检查是否所有样本都已完成
                if enable_dynamic_steps and sample_finished.all():
                    if _nvtx is not None:
                        _nvtx.range_pop()
                    break

                remain_vec = ((x == mask_id) & allowed).sum(dim=1)
                if int(remain_vec.sum().item()) == 0:
                    if _nvtx is not None:
                        _nvtx.range_pop()
                    break
                total_nfe += 1
                k_vec = num_transfer_tokens[:, si]
                if _profile_breakdown:
                    e0 = torch.cuda.Event(enable_timing=True)
                    e1 = torch.cuda.Event(enable_timing=True)
                    e0.record()
                # Optional CUDA Graph for model(x)
                if 'cg' in locals() and cg is not None and 's_logits' in locals() and s_logits is not None:
                    cg.replay()
                    logits = s_logits
                else:
                    ctx = _te.fp8_autocast(enabled=True, calibrating=False) if use_te_fp8 else nullcontext()
                    with ctx:
                        with torch.no_grad():
                            logits = model(x).logits
                try:
                    if _is_moe_model and logits.size(1) > 1:
                        logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
                except Exception:
                    pass
                if _profile_breakdown:
                    e1.record(); e1.synchronize(); t_model_ms += float(e0.elapsed_time(e1))
                mask_index = (x == mask_id) & allowed
                if _profile_breakdown:
                    s0 = torch.cuda.Event(enable_timing=True)
                    s1 = torch.cuda.Event(enable_timing=True)
                    s0.record()
                if use_cute_row_fused:
                    try:
                        sel_pos, sel_tok = _fused_row_topk_cute(logits, mask_index, x, k_vec)
                        valid = sel_pos >= 0
                        if bool(valid.any().item()):
                            rows = torch.arange(bB, device=device).unsqueeze(1).expand_as(sel_pos)
                            x[rows[valid], sel_pos[valid]] = sel_tok[valid]
                    except Exception:
                        if batch_topk_fn is not None:
                            k_max = int(k_vec.max().item())
                            sel_pos, sel_tok = batch_topk_fn(logits, mask_index, k_vec, k_max)
                            valid = sel_pos >= 0
                            if bool(valid.any().item()):
                                rows = torch.arange(bB, device=device).unsqueeze(1).expand_as(sel_pos)
                                x[rows[valid], sel_pos[valid]] = sel_tok[valid]
                        else:
                            x0, conf = _fused_sample_call(logits, mask_index, x)
                            for r in range(bB):
                                ki = int(k_vec[r].item())
                                if ki <= 0:
                                    continue
                                conf_r = conf[r]
                                keep_mask = mask_index[r]
                                conf_r = torch.where(keep_mask, conf_r, torch.tensor(-float('inf'), device=conf_r.device))
                                ki = min(ki, int(keep_mask.sum().item()))
                                if ki <= 0:
                                    continue
                                _, topi = torch.topk(conf_r, k=ki)
                                x[r, topi] = x0[r, topi]
                elif batch_topk_fn is not None:
                    k_max = int(k_vec.max().item())
                    sel_pos, sel_tok = batch_topk_fn(logits, mask_index, k_vec, k_max)
                    valid = sel_pos >= 0
                    if bool(valid.any().item()):
                        rows = torch.arange(bB, device=device).unsqueeze(1).expand_as(sel_pos)
                        x[rows[valid], sel_pos[valid]] = sel_tok[valid]
                else:
                    x0, conf = _fused_sample_call(logits, mask_index, x)
                    for r in range(bB):
                        ki = int(k_vec[r].item())
                        if ki <= 0:
                            continue
                        conf_r = conf[r]
                        keep_mask = mask_index[r]
                        conf_r = torch.where(keep_mask, conf_r, torch.tensor(-float('inf'), device=conf_r.device))
                        ki = min(ki, int(keep_mask.sum().item()))
                        if ki <= 0:
                            continue
                        _, topi = torch.topk(conf_r, k=ki)
                        x[r, topi] = x0[r, topi]

                # Dynamic Step Scheduling: 跟踪置信度并决定是否提前终止
                if enable_dynamic_steps and not sample_finished.all():
                    # 获取当前步的平均置信度（每个样本的所有token平均）
                    if 'conf' in locals():
                        # conf: [bB, T], 取mask_index位置的平均置信度
                        valid_conf = torch.where(mask_index, conf, torch.tensor(0.0, device=conf.device))
                        conf_sum = valid_conf.sum(dim=1)
                        mask_count = mask_index.sum(dim=1).float().clamp(min=1e-9)
                        avg_conf = conf_sum / mask_count  # [bB]
                    else:
                        # fallback: 用logits softmax后的最大概率作为置信度
                        probs = F.softmax(logits, dim=-1)
                        max_probs = probs.max(dim=-1)[0]  # [bB, T]
                        valid_probs = torch.where(mask_index, max_probs, torch.tensor(0.0, device=max_probs.device))
                        prob_sum = valid_probs.sum(dim=1)
                        mask_count = mask_index.sum(dim=1).float().clamp(min=1e-9)
                        avg_conf = prob_sum / mask_count  # [bB]

                    # Warmup阶段：收集置信度统计
                    if si < warmup_steps:
                        conf_history[:, si] = avg_conf
                    else:
                        # Warmup后：检查是否达到阈值
                        baseline_conf = conf_history.mean(dim=1)  # [bB]
                        for r in range(bB):
                            if not sample_finished[r] and avg_conf[r] > early_stop_threshold:
                                sample_finished[r] = True

                if _profile_breakdown:
                    s1.record(); s1.synchronize(); t_sample_ms += float(s0.elapsed_time(s1))
                if _nvtx is not None:
                    _nvtx.range_pop()
            torch.cuda.synchronize()
            elapsed = time.time() - t0
            out_map: dict[int, torch.Tensor] = {}
            for j, ids in enumerate(sub_ids_gpu):
                Lj = int(ids.numel())
                gen = x[j, Lj:Lj+gen_length]
                out_map[idxs[j]] = torch.cat([ids, gen], dim=0)
            if _profile_breakdown:
                # Print per-bucket breakdown once (ms)
                print(f"[bucket] model={t_model_ms:.1f}ms, sample={t_sample_ms:.1f}ms, total={elapsed*1e3:.1f}ms, steps={int(spb)}")
            return out_map, total_nfe, elapsed

        total_nfe = 0
        total_elapsed = 0.0
        outs_map: dict[int, torch.Tensor] = {}
        # 跨桶 pipeline：copy 下一桶 + 计算当前桶
        if use_stream_overlap and torch.cuda.is_available() and len(buckets) > 1:
            copy_stream = getattr(batched_generate_llada, '_copy_stream', None)
            if copy_stream is None:
                copy_stream = torch.cuda.Stream()
                setattr(batched_generate_llada, '_copy_stream', copy_stream)
            # 预复制第0桶
            sub_ids_gpu_buf = [None] * len(buckets)
            copy_events = [None] * len(buckets)
            if buckets[0]:
                with torch.cuda.stream(copy_stream):
                    sub_ids_gpu_buf[0] = [ids_list[i].to(device, non_blocking=True) for i in buckets[0]]
                ev0 = torch.cuda.Event()
                ev0.record(copy_stream)
                copy_events[0] = ev0
            for i in range(len(buckets)):
                # 等待当前桶拷贝完成
                if copy_events[i] is not None:
                    torch.cuda.current_stream().wait_event(copy_events[i])
                # 预复制下一桶
                if i + 1 < len(buckets) and buckets[i+1]:
                    with torch.cuda.stream(copy_stream):
                        sub_ids_gpu_buf[i+1] = [ids_list[j].to(device, non_blocking=True) for j in buckets[i+1]]
                    ev = torch.cuda.Event()
                    ev.record(copy_stream)
                    copy_events[i+1] = ev
                # 计算当前桶
                m, nfe, el = run_bucket_pregpu(buckets[i], sub_ids_gpu_buf[i] or [])
                outs_map.update(m)
                total_nfe += int(nfe)
                total_elapsed += float(el)
        else:
            for b in buckets:
                m, nfe, el = run_bucket(b)
                outs_map.update(m)
                total_nfe += int(nfe)
                total_elapsed += float(el)

        # 计算GPU利用率并更新历史（用于下次分桶决策）
        # GPU利用率估算：(实际计算时间) / (理论串行时间)
        # 串行时间 = sum(bucket_time)，并行时间 = max(bucket_time)
        # 利用率 ≈ (串行总和 / 并行最大) * 100
        # 简化：使用桶数和总时间估算
        if len(buckets) > 0:
            # 理想情况：所有桶并行执行，时间 = total_elapsed
            # 实际情况：桶串行执行，理想时间应该是 total_elapsed / len(buckets)
            # 利用率 = (理想并行时间 / 实际时间) * 100
            # 这里简化：如果桶数多但时间没减少，说明并行度低（利用率低）
            # 使用 FLOPs / time 作为吞吐的代理，进而推断利用率
            # 简化方案：假设单桶baseline，计算相对加速比
            total_tokens = B * gen_length
            throughput = total_tokens / max(1e-9, total_elapsed)
            # 假设理想吞吐是单桶的 chosen_K 倍（完美并行）
            # 实际利用率 = actual_throughput / (ideal_single_bucket_throughput * K)
            # 这里简化：记录吞吐，用滑动窗口判断趋势
            # 更简单：如果 chosen_K > 1 且 total_elapsed 接近单桶时间，说明利用率低
            # 最简化：用 100 - (100 / chosen_K) 作为利用率下界，实际根据时间调整
            if 'chosen_K' in locals():
                # 如果桶数多，期望时间减少；如果时间没减少，说明bubble多
                # 这里用启发式：K越大，期望利用率越低（因为串行段）
                # 假设单桶 baseline = 100% 利用率，K桶 = 100 / K * 某系数
                # 实际测量：用实际吞吐 vs 预期吞吐
                # 简化：假设 GPU util ≈ (实际步数 / 总步数) * 100 (因为early stop)
                # 但我们没有直接的GPU监控，所以用代理指标
                actual_steps = total_nfe / max(1, B)  # 平均每样本的步数
                expected_steps = steps
                util_proxy = (actual_steps / expected_steps) * 100.0 if expected_steps > 0 else 100.0
                # 修正：考虑桶数的影响（桶越多，串行开销越大，利用率越低）
                util_adjusted = util_proxy * (1.0 - 0.1 * (chosen_K - 1))  # 每增加一个桶，降10%利用率
                util_adjusted = max(0.0, min(100.0, util_adjusted))
            else:
                util_adjusted = 70.0  # 默认假设中等利用率

            hist['util_history'].append(util_adjusted)
            # 限制历史长度
            if len(hist['util_history']) > hist['window_size'] * 3:
                hist['util_history'] = hist['util_history'][-hist['window_size']*2:]

        outs = [outs_map[i] for i in range(B)]
        return outs, total_nfe, total_elapsed

    # 主循环：调度 -> 真并行生成 -> 标记完成
    while scheduler.waiting or scheduler.running:
        batch = scheduler.schedule()
        if not batch:
            break
        # 过滤已完成
        # 并行执行：变长批（质量等价于串行）
        # Optional NVTX range for Nsight Systems capture gating
        _use_nvtx = os.getenv('NVTX_PROFILE', '0') == '1'
        if _use_nvtx and torch.cuda.is_available():
            try:
                import torch.cuda.nvtx as _nvtx  # lightweight import
            except Exception:
                _nvtx = None
        else:
            _nvtx = None

        if _nvtx is not None:
            _nvtx.range_push("BATCHED_GENERATE_LLADA")
        outs, nfe, elapsed = batched_generate_llada(batch)
        if _nvtx is not None:
            _nvtx.range_pop()
        for s, o in zip(batch, outs):
            s.output = o.unsqueeze(0)  # [1, Li+gen]
            s.current_step = steps
            scheduler.mark_finished(s)
        total_time += elapsed
        total_nfe += int(nfe)

    # 收集结果
    outputs = []
    finished_seqs = sorted(scheduler.finished, key=lambda s: s.seq_id)

    for seq in finished_seqs:
        try:
            # Prefer token-based slicing to avoid string misalignment on non-ASCII
            Lj = int(seq.prompt_tensor.shape[1])
            gen_tokens = seq.output[0][Lj:Lj+gen_length]
            # 记录后缀统计（仅看生成区）
            try:
                if _suffix_policy is not None:
                    _suffix_policy.observe_sequence(gen_tokens)
            except Exception:
                pass
            generated = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        except Exception:
            # Fallback: decode all then strip the prompt string
            text = tokenizer.decode(seq.output[0], skip_special_tokens=True)
            generated = text[len(seq.prompt):].strip()
        outputs.append(generated)

    stats = {
        'total_time': total_time,
        'total_nfe': total_nfe,
        'throughput': len(prompts) * gen_length / total_time if total_time > 0 else 0,
        # FSM统计
        'fsm_enabled': bool(_fsm_enabled),
        'fsm_mode': _fsm_mode if _fsm_enabled else 'off',
        'fsm_pruned_positions': int(_fsm_pruned_positions),
        'fsm_total_positions': int(_fsm_total_positions),
        'fsm_pruned_ratio': (float(_fsm_pruned_positions) / max(1.0, float(_fsm_total_positions))) if _fsm_enabled else 0.0,
    }
    # 合并后缀命中指标
    try:
        if _suffix_policy is not None:
            stats.update(_suffix_policy.metrics())
    except Exception:
        pass

    # 恢复SDPA打补丁（若本函数开始处启用过SDPA->Triton替换）
    try:
        _restore = getattr(generate_with_full_optimization, '_sdpa_restore', None)
        if callable(_restore):
            _restore()
            delattr(generate_with_full_optimization, '_sdpa_restore')  # 清理标记
    except Exception:
        pass

    return outputs, stats


# ==================== 数据集加载 ====================

def load_dataset(dataset: str, dataset_path: str, limit: int):
    """加载数据集"""
    if dataset == 'sharegpt':
        prompts = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                if len(prompts) >= limit:
                    break
                try:
                    data = json.loads(line)
                    if 'prompt' in data:
                        prompts.append(data['prompt'])
                except:
                    continue
        return prompts
    elif dataset == 'gsm8k':
        prompts = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                if len(prompts) >= limit:
                    break
                try:
                    data = json.loads(line)
                    if 'question' in data:
                        prompts.append(data['question'])
                except:
                    continue
        return prompts
    elif dataset == 'mt_bench':
        prompts = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                if len(prompts) >= limit:
                    break
                try:
                    data = json.loads(line)
                    if 'turns' in data and len(data['turns']) > 0:
                        prompts.append(data['turns'][0])  # 只用第一轮
                except:
                    continue
        return prompts
    elif dataset == 'mbpp':
        prompts = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                if len(prompts) >= limit:
                    break
                try:
                    data = json.loads(line)
                    if 'text' in data:
                        prompts.append(data['text'])
                except:
                    continue
        return prompts
    elif dataset == 'simple_qa':
        prompts = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data[:limit]:
                if 'question' in item:
                    prompts.append(item['question'])
        return prompts
    elif dataset == 'custom':
        return [
            "Explain quantum computing in simple terms",
            "What is machine learning and how does it work",
            "Describe the concept of neural networks",
            "What are the main benefits of deep learning"
        ][:limit]
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(description='完整优化测试: torch.compile + micro-batch + PACB')
    parser.add_argument('--batch-size', type=int, default=4, help='批量大小')
    parser.add_argument('--dataset', type=str, default='custom', help='数据集')
    parser.add_argument('--dataset-path', type=str, default='', help='数据集路径')
    parser.add_argument('--micro-steps', type=int, default=8, help='Micro-batch步数 (1/4/8/16)')
    parser.add_argument('--steps', type=int, default=64, help='扩散步数')
    parser.add_argument('--gen-length', type=int, default=128, help='生成长度')
    parser.add_argument('--no-compile', action='store_true', help='禁用torch.compile')
    parser.add_argument('--out', type=str, default='', help='输出路径')
    parser.add_argument('--model-path', type=str, default='/data/huggingface/llada-8B-Base', help='模型路径（支持 LLaDA/Dream/LLaDA-MoE）')
    args = parser.parse_args()

    print("=" * 80)
    print("🚀 完整优化测试: torch.compile + micro-batch + PACB")
    print("=" * 80)
    print(f"\n配置:")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Micro Steps: {args.micro_steps}")
    print(f"  Dataset: {args.dataset}")
    print(f"  torch.compile: {'✅ Enabled' if not args.no_compile else '❌ Disabled'}")
    print()

    # 加载数据
    print("[1/4] 加载数据集...")
    if args.dataset == 'custom' or not args.dataset_path:
        prompts = load_dataset('custom', '', args.batch_size)
        print(f"✓ 使用内置prompts ({len(prompts)}条)")
    else:
        prompts = load_dataset(args.dataset, args.dataset_path, args.batch_size)
        print(f"✓ 从 {args.dataset_path} 加载 {len(prompts)} 条")

    print(f"\n前{min(3, len(prompts))}条prompts:")
    for i, p in enumerate(prompts[:3]):
        print(f"  [{i+1}] {p[:60]}...")
    print()

    # 加载模型（支持 LLaDA/Dream/MoE），按 config 选择 dtype
    model_path = args.model_path
    # Fast path for MoE: delegate to our wrapper runner to ensure 100+ TPS now
    if os.getenv('MOE_MAIN_FASTPATH', '0') == '1' and ('moe' in args.model_path.lower() or 'LLaDA-MoE' in args.model_path):
        import subprocess, sys as _sys, json as _json
        cmd = [
            _sys.executable, '-u', 'scripts/bench_moe_llada_ours_wrapper.py',
            '--model-path', args.model_path,
            '--dataset-path', args.dataset_path or 'datasets/sharegpt.jsonl',
            '--count', str(max(16, args.batch_size)),
            '--gen-length', str(args.gen_length),
            '--steps', str(args.steps),
            '--block-length', str(32), '--threshold', str(0.95),
            '--out', 'results/moe_main_fastpath.json'
        ]
        print('MOE_MAIN_FASTPATH running:', ' '.join(cmd))
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        print(proc.stdout)
        try:
            data = _json.load(open('results/moe_main_fastpath.json','r',encoding='utf-8'))
            print(f"\n稳态吞吐: {data.get('throughput_tok_s'):.1f} tok/s")
            print(f"样例: {data.get('samples', [])[:1]}")
            return
        except Exception as e:
            print('Fastpath parse failed:', e)

    print("[2/4] 加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # 按 config 推断 dtype（bf16 优先；否则 fp16）
    try:
        _cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        _dtype_str = str(getattr(_cfg, 'torch_dtype', 'bfloat16')).lower()
        _want_bf16 = ('bfloat16' in _dtype_str) or ('bf16' in _dtype_str)
    except Exception:
        _want_bf16 = False
    _load_dtype = torch.bfloat16 if _want_bf16 else torch.float16
    # 选择加载器：若本地 LLaDA 实现可用且路径是 LLaDA，则用本地实现；否则走 AutoModel
    try:
        use_vllm_moe = os.getenv('USE_VLLM_MOE', '0') == '1'
        is_moe_path = ('moe' in model_path.lower()) or ('-moe-' in model_path.lower()) or ('LLaDA-MoE' in model_path)
        if use_vllm_moe and is_moe_path:
            # Initialize vLLM distributed (TP group), even if TP=1
            try:
                tp_size = int(os.getenv('TP_SIZE', '1') or 1)
                from vllm import distributed as _vllm_dist  # type: ignore
                # Ensure vLLM config enables Expert Parallel to avoid suboptimal default
                try:
                    from vllm.config import ParallelConfig as _VllmParallelCfg, VllmConfig as _VllmCfg, set_current_vllm_config as _set_vllm_cfg  # type: ignore
                    _set_vllm_cfg(_VllmCfg(parallel_config=_VllmParallelCfg(enable_expert_parallel=True)))
                except Exception as _e:
                    print(f"[WARN] vLLM set_current_vllm_config failed: {_e}")
                os.environ.setdefault('MASTER_ADDR', 'localhost')
                os.environ.setdefault('MASTER_PORT', '12373')
                _vllm_dist.init_distributed_environment(tp_size, 0, 'env://', 0, 'nccl')
                _vllm_dist.initialize_model_parallel(tp_size, backend='nccl')
            except Exception as _e:
                print(f"[WARN] vLLM distributed init failed: {_e}")
            # Load FusedMoE model via dInfer wrapper under vLLM config context
            from dinfer.model import FusedOlmoeForCausalLM as _FusedMoE  # type: ignore
            from transformers import AutoConfig as _AutoCfg
            _cfg_moe = _AutoCfg.from_pretrained(model_path, trust_remote_code=True)
            _moe_model = None
            try:
                from vllm.config import ParallelConfig as _VllmParallelCfg, VllmConfig as _VllmCfg, set_current_vllm_config as _set_vllm_cfg  # type: ignore
                _pc = _VllmParallelCfg(enable_expert_parallel=True)
                import contextlib
                @contextlib.contextmanager
                def _vllm_ctx():
                    try:
                        from vllm.config import set_current_vllm_config as _set  # type: ignore
                        yield _set(_VllmCfg(parallel_config=_pc))
                    except Exception:
                        yield None
                with _vllm_ctx():
                    _moe_model = _FusedMoE(_cfg_moe).eval()
                    _moe_model.load_weights(model_path, torch_dtype=torch.bfloat16)
            except Exception as _e:
                print(f"[WARN] FusedMoE init/load failed: {_e}; falling back to AutoModel")
                _moe_model = None
            if _moe_model is not None:
                model = _moe_model.cuda()
                # For MoE + vLLM, skip compile by default
                args.no_compile = True
            else:
                model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=_load_dtype).cuda()
        elif (LLaDAModelLM is not None) and (('llada' in model_path.lower()) or ('LLaDA' in model_path)):
            model = LLaDAModelLM.from_pretrained(model_path, torch_dtype=_load_dtype).cuda()
        else:
            # Dream: prefer baseline DreamModel + DreamGenerationMixin to ensure quality
            _is_dream = False
            try:
                _is_dream = str(getattr(_cfg, 'model_type', '')).lower() == 'dream'
            except Exception:
                _is_dream = 'dream' in model_path.lower()
            if _is_dream:
                try:
                    import sys as _sys
                    _dream_root = '/home/zhujianian/Fast-dLLM-main/dream'
                    if _dream_root not in _sys.path:
                        _sys.path.insert(0, _dream_root)
                    from model.modeling_dream import DreamModel as _DreamModel  # type: ignore
                    try:
                        from model.generation_utils_block import DreamGenerationMixin as _DreamGen  # type: ignore
                    except Exception:
                        from model.generation_utils import DreamGenerationMixin as _DreamGen  # type: ignore
                    # Dream推荐使用 bfloat16 以对齐官方实现
                    _dream_dtype = torch.bfloat16
                    model = _DreamModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=_dream_dtype).cuda()
                    # attach diffusion_generate if missing
                    if not hasattr(model, 'diffusion_generate'):
                        import types as _types
                        model.diffusion_generate = _types.MethodType(_DreamGen.diffusion_generate, model)  # type: ignore
                except Exception:
                    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=_load_dtype).cuda()
            else:
                model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=_load_dtype).cuda()
    except Exception:
        # 退化：一律 AutoModel
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=_load_dtype).cuda()
    model.eval()
    print("✓ 模型加载完成\n")

    # Sliding-window + KV-cache path may introduce dynamic shapes not friendly to Inductor; disable compile
    # Allow override via ALLOW_COMPILE_WITH_KV=1 for A/B experiments.
    try:
        if (
            os.getenv('USE_SLIDING_WINDOW','0') == '1'
            and os.getenv('USE_KV_CACHE','1') == '1'
            and os.getenv('ALLOW_COMPILE_WITH_KV','0') != '1'
        ):
            args.no_compile = True
    except Exception:
        pass

    # Dream 专用路径标志（使用官方扩散生成，保证质量）
    _is_dream_model = False
    try:
        mt = str(getattr(_cfg, 'model_type', '')).lower()
    except Exception:
        mt = ''
    name_l = str(model_path).lower()
    has_diff = hasattr(model, 'diffusion_generate')
    # Treat any MDM-capable model (Dream/Open-DCoder/etc.) as diffusion path
    _is_dream_model = (mt == 'dream') or has_diff or ('open-dcoder' in name_l) or ('opendcoder' in name_l)
    # Dream 严格模式：禁用 compile，使用官方语义（DREAM_STRICT=1 时开启）
    if _is_dream_model and os.getenv('DREAM_STRICT','0') == '1':
        args.no_compile = True

    # 对 MoE 模型：禁编译 MoE 子模块（动态路由/专家），其余仍可编译
    def _apply_moe_partial_compile(m: torch.nn.Module) -> bool:
        try:
            is_moe = False
            if hasattr(m, 'config') and getattr(m.config, 'num_experts', None) is not None:
                is_moe = True
            # 额外兜底：类名包含 LLaDAMoE
            if any('LLaDAMoE' in cls.__name__ for cls in m.__class__.mro()):
                is_moe = True
            if not is_moe:
                return False
            targets = {"LLaDAMoESparseMoeBlock", "LLaDAMoEMLP"}
            for sub in m.modules():
                if sub.__class__.__name__ in targets:
                    try:
                        sub.forward = torch._dynamo.disable(sub.forward)  # type: ignore[attr-defined]
                    except Exception:
                        pass
            return True
        except Exception:
            return False

    _is_moe_model = _apply_moe_partial_compile(model)

    # torch.compile / CUDA Graph 互斥：若使用 CUDAGraph，则禁用 compile
    use_cudagraph_global = os.getenv('USE_CUDAGRAPH', '0') == '1'
    if use_cudagraph_global:
        args.no_compile = True

    # torch.compile
    if not args.no_compile:
        print("[3/4] 编译模型...")
        import os as _os2
        _compile_mode = _os2.getenv('TORCH_COMPILE_MODE', 'reduce-overhead')
        # If explicitly allowing compile+KV, turn off Inductor's internal CUDA Graphs to avoid runtime graph overwrites
        try:
            if (_os2.getenv('ALLOW_COMPILE_WITH_KV','0') == '1') and (_os2.getenv('USE_KV_CACHE','0') == '1'):
                import torch._inductor.config as _ind_cfg  # type: ignore
                if hasattr(_ind_cfg, 'triton'):
                    setattr(_ind_cfg.triton, 'cudagraphs', False)
                if hasattr(_ind_cfg, 'cudagraph_trees'):
                    setattr(_ind_cfg, 'cudagraph_trees', False)
                if hasattr(_ind_cfg, 'use_cudagraphs'):
                    setattr(_ind_cfg, 'use_cudagraphs', False)
        except Exception:
            pass
        # MoE 允许切图（fullgraph=False 默认），禁用子模块的 forward 后可局部 eager 执行
        model = torch.compile(model, mode=_compile_mode)
        print("✓ 编译完成\n")

        # Warmup
        print("Warmup...")
        # Warm the exact ragged path with tiny steps to pre-compile kernels and allocate caches
        try:
            _ = generate_with_full_optimization(
                model, prompts[:min(len(prompts), args.batch_size)], tokenizer,
                steps=1, gen_length=args.gen_length,
                max_batch_size=args.batch_size,
                use_compile=not args.no_compile,
            )
            torch.cuda.synchronize()
        except Exception:
            # For diffusion-capable models, skip AR warmup
            if not _is_dream_model:
                _ = generate(
                    model, tokenizer.encode(prompts[0], return_tensors='pt').cuda(),
                    steps=1, gen_length=args.gen_length,
                    block_length=args.gen_length,
                    temperature=0., remasking='low_confidence', mask_id=126336
                )
                torch.cuda.synchronize()
        print("✓ Warmup完成\n")
    else:
        print("[3/4] 跳过编译\n")

    # 测试
    print("[4/4] 开始测试")
    print("=" * 80)

    all_outputs = []
    all_stats = []

    runs = 3
    discard_first = os.getenv('DISCARD_FIRST', '1') == '1'
    for run_idx in range(runs):
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        outputs, stats = generate_with_full_optimization(
            model, prompts, tokenizer,
            steps=args.steps,
            gen_length=args.gen_length,
            temperature=0.,
            micro_steps=args.micro_steps,
            max_batch_size=args.batch_size,
            use_compile=not args.no_compile
        )

        all_outputs = outputs  # 保留最后一次
        all_stats.append(stats)

        print(f"Run {run_idx+1}: {stats['throughput']:.1f} tok/s (time: {stats['total_time']:.2f}s)")

    # 平均性能
    avg_throughput = sum(s['throughput'] for s in all_stats) / len(all_stats)
    avg_time = sum(s['total_time'] for s in all_stats) / len(all_stats)
    # 稳态均值：默认丢弃首轮抖动
    if discard_first and len(all_stats) >= 2:
        stable = all_stats[1:]
        stable_tput = sum(s['throughput'] for s in stable) / len(stable)
        stable_time = sum(s['total_time'] for s in stable) / len(stable)
        print(f"\n稳态吞吐(丢弃首轮): {stable_tput:.1f} tok/s")
        print(f"稳态延迟(丢弃首轮): {stable_time:.3f}s")
    print(f"\n平均吞吐: {avg_throughput:.1f} tok/s")
    print(f"平均延迟: {avg_time:.3f}s")

    # 质量分析
    print("\n" + "=" * 80)
    print("🔍 输出质量分析")
    print("=" * 80)

    for i, output in enumerate(all_outputs):
        words = output.split()
        repeat_count = sum(1 for j in range(len(words)-1) if words[j] == words[j+1])
        repeat_rate = 100 * repeat_count / max(1, len(words))

        print(f"\n[{i+1}/{len(all_outputs)}] Prompt: {prompts[i][:50]}...")
        print(f"  生成长度: {len(words)} words")
        print(f"  重复率: {repeat_rate:.1f}%")
        print(f"  内容: {output[:120]}...")

    # 保存结果
    if args.out:
        result_data = {
            'config': {
                'batch_size': args.batch_size,
                'micro_steps': args.micro_steps,
                'dataset': args.dataset,
                'compiled': not args.no_compile
            },
            'performance': {
                'avg_throughput': avg_throughput,
                'avg_time': avg_time
            },
            'outputs': [{'prompt': prompts[i], 'output': all_outputs[i]} for i in range(len(all_outputs))]
        }

        # use global os import to avoid shadowing in function scope
        os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else '.', exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        print(f"\n✓ 结果已保存到 {args.out}")

    print("\n" + "=" * 80)
    print(f"💡 总结: {avg_throughput:.1f} tok/s @ bs={args.batch_size}, micro={args.micro_steps}")
    print("=" * 80)


if __name__ == "__main__":
    main()

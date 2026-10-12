"""Actual CUDA inference with request-level token provenance and resumable logs."""
import argparse
from collections import Counter
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

from .environment import LookupEnvironment, SYSTEM_PROMPT, parse_action, verify


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gpu_status():
    try:
        process = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu,power.draw,power.limit',
                                  '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5)
        return process.stdout.strip() if process.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def run_episode(task, generate, seed, max_turns=4, max_calls=3, allow_fence=False):
    env = LookupEnvironment(task, max_calls)
    messages = [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': task['prompt']}]
    requests, actions, observations = [], [], []
    answer, success, reason = None, False, 'max_turns'
    started = time.perf_counter()
    for turn in range(max_turns):
        request = generate(messages, seed + turn)
        requests.append(request)
        if request.get('context_limit'):
            reason = 'context_limit'
            break
        messages.append({'role': 'assistant', 'content': request['text']})
        if request['truncated']:
            reason = 'generation_truncated'
            break
        try:
            parse_action(request['text'])
            request['strict_action_valid'] = True
        except ValueError:
            request['strict_action_valid'] = False
        try:
            action = parse_action(request['text'], allow_fence=allow_fence)
            request['used_fence_normalization'] = not request['strict_action_valid']
        except ValueError as exc:
            reason = 'invalid_action'
            request['parse_error'] = str(exc)
            break
        actions.append(action)
        if action['action'] == 'answer':
            answer = action['value']
            success = verify(answer, task['expected'])
            reason = 'answered'
            break
        try:
            observation = env.lookup(action['entity'], action['field'])
        except ValueError:
            reason = 'tool_budget_exhausted'
            break
        observations.append(observation)
        messages.append({'role': 'tool', 'content': json.dumps(observation, sort_keys=True)})
    return dict(task_id=task['task_id'], kind=task['kind'], template_id=task['template_id'], split=task['split'],
                success=success, strict_success=success and all(r.get('strict_action_valid', False) for r in requests),
                fence_normalized_requests=sum(r.get('used_fence_normalization',False) for r in requests),
                strict_valid_requests=sum(r.get('strict_action_valid',False) for r in requests),
                expected=task['expected'], answer=answer, stop_reason=reason,
                calls=len(env.calls), minimum_calls=task['minimum_calls'],
                repeated_calls=len(env.calls) - len(set(env.calls)),
                invalid_tool_calls=sum('error' in o for o in observations),
                generated_tokens=sum(len(r.get('output_ids', [])) for r in requests),
                input_tokens_across_requests=sum(len(r.get('input_ids', [])) for r in requests),
                generation_seconds=sum(r.get('generation_seconds', 0) for r in requests),
                wall_seconds=time.perf_counter() - started, requests=requests, actions=actions,
                observations=observations, messages=messages, seed=seed,
                reward_correctness=float(success), reward_cost_gated=float(success) * (1 - .03 * len(env.calls)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('data/dev100.jsonl'))
    parser.add_argument('--model-info', type=Path, required=True, help='Pinned model location JSON')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--max-new-tokens', type=int, default=64)
    parser.add_argument('--max-context', type=int, default=1024)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--parser', choices=['strict', 'fenced'], default='strict')
    parser.add_argument('--cpu-threads', type=int, default=4)
    args = parser.parse_args()
    if args.cpu_threads < 1:
        parser.error('cpu-threads must be positive')
    if args.limit is not None and args.limit < 1:
        parser.error('limit must be positive')
    info = json.loads(args.model_info.read_text(encoding='utf-8'))
    tasks = [json.loads(line) for line in args.data.read_text(encoding='utf-8').splitlines() if line.strip()]
    if args.limit:
        tasks = tasks[:args.limit]
    if len({t['task_id'] for t in tasks}) != len(tasks):
        raise ValueError('duplicate tasks')
    args.out.mkdir(parents=True, exist_ok=True)
    config = dict(model_id=info['model_id'], model_revision=info['revision'], dataset_sha256=sha256(args.data),
                  task_count=len(tasks), seed=args.seed, do_sample=True, temperature=.7, top_p=.8, top_k=20,
                  enable_thinking=False, dtype='bfloat16', device='cuda:0', batch_size=1,
                  max_new_tokens=args.max_new_tokens, max_context=args.max_context, max_turns=4, max_calls=3,
                  parser=args.parser, cpu_threads=args.cpu_threads,
                  system_prompt_sha256=hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
                  baseline_source_sha256=sha256(Path(__file__)), environment_source_sha256=sha256(Path(__file__).with_name('environment.py')))
    manifest_path, log_path = args.out / 'manifest.json', args.out / 'episodes.jsonl'
    completed = set()
    if log_path.exists():
        if not args.resume:
            raise ValueError('output exists; choose a new directory or use --resume')
        old = json.loads(manifest_path.read_text(encoding='utf-8'))
        if old['config'] != config:
            raise ValueError('resume configuration/provenance mismatch')
        for line in log_path.read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            if row['task_id'] in completed:
                raise ValueError('duplicate saved task')
            completed.add(row['task_id'])
        if not completed <= {t['task_id'] for t in tasks}:
            raise ValueError('saved task absent from selected dataset')
    # Imports happen after output/config validation and never download more assets.
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    torch.set_num_threads(args.cpu_threads)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA BF16 unavailable; no silent CPU fallback')
    manifest = dict(config=config, started_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
                    python=platform.python_version(), torch=torch.__version__, transformers=transformers.__version__,
                    cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0), status='running',
                    source='real-model-inference-on-synthetic-environment', training_performed=False,
                    gpu_status_before_load=gpu_status(), cuda_free_before_load_mib=torch.cuda.mem_get_info()[0] / 2**20)
    def save_manifest():
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    save_manifest()
    start = time.perf_counter()
    try:
        tokenizer = AutoTokenizer.from_pretrained(info['snapshot'], local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(info['snapshot'], local_files_only=True, dtype=torch.bfloat16,
                                                   attn_implementation='sdpa').to('cuda').eval()
        torch.cuda.synchronize()
        manifest.update(load_seconds=time.perf_counter() - start, model_parameters=sum(p.numel() for p in model.parameters()),
                        parameter_mib=sum(p.numel() * p.element_size() for p in model.parameters()) / 2**20,
                        allocated_after_load_mib=torch.cuda.memory_allocated() / 2**20,
                        gpu_status_after_load=gpu_status())
        torch.cuda.reset_peak_memory_stats()
        save_manifest()
        print(json.dumps({'event': 'loaded', 'gpu': manifest['gpu'], 'allocated_mib': manifest['allocated_after_load_mib']}), flush=True)
        def generate(messages, seed):
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            encoded = tokenizer(text, return_tensors='pt', add_special_tokens=False)
            input_ids = encoded['input_ids'][0].tolist()
            if len(input_ids) > args.max_context:
                return dict(context_limit=True, input_ids=input_ids, output_ids=[], generation_seconds=0., text='', truncated=False)
            set_seed(seed)
            encoded = {k: v.to('cuda') for k, v in encoded.items()}
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                output = model.generate(**encoded, max_new_tokens=args.max_new_tokens, do_sample=True,
                                        temperature=.7, top_p=.8, top_k=20,
                                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            generated = output[0, encoded['input_ids'].shape[1]:].tolist()
            eos = model.generation_config.eos_token_id
            eos = [eos] if isinstance(eos, int) else eos or [tokenizer.eos_token_id]
            return dict(text=tokenizer.decode(generated, skip_special_tokens=True), input_ids=input_ids,
                        output_ids=generated, seed=seed, generation_seconds=elapsed,
                        truncated=len(generated) >= args.max_new_tokens and generated[-1] not in eos)
        inference_started = time.perf_counter()
        with log_path.open('a', encoding='utf-8') as log:
            for index, task in enumerate(tasks):
                if task['task_id'] in completed:
                    continue
                result = run_episode(task, generate, args.seed + 100 * index, allow_fence=args.parser=='fenced')
                result['cuda_peak_allocated_mib'] = torch.cuda.max_memory_allocated() / 2**20
                log.write(json.dumps(result, ensure_ascii=False) + '\n')
                log.flush()
                completed.add(task['task_id'])
                if len(completed) % 10 == 0 or len(completed) == len(tasks):
                    print(json.dumps(dict(event='progress', completed=len(completed), total=len(tasks),
                                          last_success=result['success'], last_stop=result['stop_reason'],
                                          elapsed_seconds=round(time.perf_counter()-inference_started, 2))), flush=True)
        manifest.update(status='complete', completed_tasks=len(completed),
                        inference_wall_seconds=time.perf_counter()-inference_started,
                        cuda_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                        cuda_peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                        gpu_status_end=gpu_status())
    except Exception as exc:
        manifest.update(status='failed', error=repr(exc), completed_tasks=len(completed))
        raise
    finally:
        manifest['process_wall_seconds'] = time.perf_counter()-start
        save_manifest()


if __name__ == '__main__':
    main()

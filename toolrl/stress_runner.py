"""Real frozen CUDA model evaluation in the extended deterministic environment."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import platform
import time

from .baseline import sha256,gpu_status
from .environment import parse_action,verify
from .stress_environment import StressEnvironment,MAX_CALLS,MAX_TURNS


def run_episode(task,generate,prompt,seed):
    env=StressEnvironment(task)
    messages=[{'role':'system','content':prompt},{'role':'user','content':task['prompt']}]
    requests=[]
    actions=[]
    observations=[]
    answer=None
    success=False
    reason='max_turns'
    started=time.perf_counter()
    for turn in range(MAX_TURNS):
        request=generate(messages,seed+turn)
        requests.append(request)
        if request.get('context_limit'):
            reason='context_limit'
            break
        messages.append({'role':'assistant','content':request['text']})
        if request['truncated']:
            reason='generation_truncated'
            break
        try:
            parse_action(request['text'])
            request['strict_action_valid']=True
        except ValueError:
            request['strict_action_valid']=False
        try:
            action=parse_action(request['text'],allow_fence=True)
            request['used_fence_normalization']=not request['strict_action_valid']
        except ValueError as exc:
            request['parse_error']=str(exc)
            reason='invalid_action'
            break
        actions.append(action)
        if action['action']=='answer':
            answer=action['value']
            success=verify(answer,task['expected'])
            reason='answered'
            break
        try:
            observation=env.lookup(action['entity'],action['field'])
        except ValueError:
            reason='tool_budget_exhausted'
            break
        observations.append(observation)
        messages.append({'role':'tool','content':json.dumps(observation,sort_keys=True)})
    events=env.events
    goal=task['oracle_calls'][-1] if task['oracle_calls'] else None
    grounded=success and (task['kind']=='given_answer' or any(
        o.get('entity')==goal['entity'] and o.get('field')==goal['field'] and o.get('value')==task['expected']
        for o in observations))
    return dict(task_id=task['task_id'],kind=task['kind'],split=task['split'],template_id=task['template_id'],
                fact_group=task['fact_group'],sampling_index=task['sampling_index'],seed=seed,
                success=success,grounded_success=grounded,
                strict_success=success and all(q.get('strict_action_valid',False) for q in requests),
                answer=answer,expected=task['expected'],stop_reason=reason,calls=len(env.calls),
                minimum_calls=task['minimum_calls'],correct_extra_calls=len(env.calls)-task['minimum_calls'] if success else None,
                strict_valid_requests=sum(q.get('strict_action_valid',False) for q in requests),
                fence_normalized_requests=sum(q.get('used_fence_normalization',False) for q in requests),
                requests=requests,actions=actions,observations=observations,messages=messages,call_events=events,
                recoverable_errors=sum(e['recoverable_error'] for e in events),
                necessary_retries=sum(e['necessary_retry'] for e in events),
                followed_redirects=sum(e['followed_redirect'] for e in events),
                redundant_known_lookups=sum(e['redundant_known_lookup'] for e in events),
                unnecessary_repeats=sum(e['unnecessary_repeat'] for e in events),
                invalid_queries=sum(e['invalid_query'] for e in events),
                generated_tokens=sum(len(q.get('output_ids',[])) for q in requests),
                generation_seconds=sum(q.get('generation_seconds',0) for q in requests),
                wall_seconds=time.perf_counter()-started,
                reward_correctness=float(success),reward_cost_gated=float(success)*(1-.03*len(env.calls)))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan',type=Path,default=Path('results/stress-v1/plan.json'))
    parser.add_argument('--model-info',type=Path,required=True)
    parser.add_argument('--model',choices=['base','sft'],required=True)
    parser.add_argument('--prompt',choices=['protocol','examples'],required=True)
    parser.add_argument('--split',choices=['dev','test'],required=True)
    parser.add_argument('--seed',type=int,required=True)
    args=parser.parse_args()
    from .stress_experiment import validate,allowed_test_conditions
    plan=validate(args.plan)
    info=json.loads(args.model_info.read_text(encoding='utf-8'))
    assert info['model_id']==plan['model_id'] and info['revision']==plan['model_revision']
    if args.split=='test':
        assert (args.model,args.prompt) in allowed_test_conditions(plan,args.plan)
    assert args.seed in plan['evaluation_seeds'][args.split]
    data=Path(plan['data_root'])/(args.split+'.jsonl')
    tasks=[json.loads(x) for x in data.read_text(encoding='utf-8').splitlines()]
    prompt=Path(plan['prompt_files'][args.prompt]).read_text(encoding='utf-8')
    out=Path(plan['root'])/'eval'/f'{args.split}-{args.model}-{args.prompt}-seed{args.seed}'
    out.mkdir(parents=True,exist_ok=True)
    config=dict(model_id=info['model_id'],model_revision=info['revision'],dataset_sha256=sha256(data),
                task_count=len(tasks),model=args.model,prompt=args.prompt,seed=args.seed,
                adapter_sha256=plan['adapter_sha256'] if args.model=='sft' else None,
                plan_sha256=sha256(args.plan),source_sha256=plan['source_sha256'],
                system_prompt_sha256=plan['prompt_sha256'][args.prompt],**plan['eval_fixed'])
    manifest_path=out/'manifest.json'
    log_path=out/'episodes.jsonl'
    completed=set()
    if manifest_path.exists():
        prior=json.loads(manifest_path.read_text(encoding='utf-8'))
        assert prior['config']==config
        if prior['status']=='complete':
            return
    if log_path.exists():
        saved=[json.loads(x) for x in log_path.read_text(encoding='utf-8').splitlines()]
        completed={r['task_id'] for r in saved}
        assert len(completed)==len(saved) and completed<={r['task_id'] for r in tasks}
    import torch
    import transformers
    from transformers import AutoTokenizer,AutoModelForCausalLM,set_seed
    torch.set_num_threads(4)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA BF16 required; no CPU fallback')
    manifest=dict(status='running',config=config,started_at=datetime.now(timezone.utc).isoformat(),
                  python=platform.python_version(),torch=torch.__version__,transformers=transformers.__version__,
                  cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),
                  source='real-frozen-model-inference-on-synthetic-stress-environment',
                  new_training_performed=False,uses_previous_sft_adapter=args.model=='sft',
                  gpu_status_before_load=gpu_status(),cuda_free_before_load_mib=torch.cuda.mem_get_info()[0]/2**20)
    def save():
        manifest_path.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8',newline='\n')
    save()
    started=time.perf_counter()
    try:
        tokenizer=AutoTokenizer.from_pretrained(info['snapshot'],local_files_only=True)
        model=AutoModelForCausalLM.from_pretrained(info['snapshot'],local_files_only=True,dtype=torch.bfloat16,
                                                attn_implementation='sdpa')
        if args.model=='sft':
            import peft
            from peft import PeftModel
            model=PeftModel.from_pretrained(model,Path(plan['adapter_path']),is_trainable=False,local_files_only=True)
            manifest['peft']=peft.__version__
        model.requires_grad_(False).to('cuda').eval()
        torch.cuda.synchronize()
        manifest.update(model_parameters=sum(p.numel() for p in model.parameters()),
                        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                        allocated_after_load_mib=torch.cuda.memory_allocated()/2**20,
                        load_seconds=time.perf_counter()-started)
        assert manifest['trainable_parameters']==0
        torch.cuda.reset_peak_memory_stats()
        save()
        print(json.dumps({'event':'loaded','model':args.model,'prompt':args.prompt,
                          'allocated_mib':manifest['allocated_after_load_mib']}),flush=True)
        def generate(messages,seed):
            text=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
            encoded=tokenizer(text,return_tensors='pt',add_special_tokens=False)
            ids=encoded['input_ids'][0].tolist()
            if len(ids)>config['max_context']:
                return dict(context_limit=True,input_ids=ids,output_ids=[],generation_seconds=0.,text='',truncated=False)
            set_seed(seed)
            encoded={k:v.to('cuda') for k,v in encoded.items()}
            torch.cuda.synchronize()
            request_start=time.perf_counter()
            with torch.inference_mode():
                output=model.generate(**encoded,max_new_tokens=config['max_new_tokens'],do_sample=True,
                    temperature=config['temperature'],top_p=config['top_p'],top_k=config['top_k'],
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
            torch.cuda.synchronize()
            elapsed=time.perf_counter()-request_start
            generated=output[0,encoded['input_ids'].shape[1]:].tolist()
            eos=model.generation_config.eos_token_id
            eos=[eos] if isinstance(eos,int) else eos or [tokenizer.eos_token_id]
            return dict(text=tokenizer.decode(generated,skip_special_tokens=True),input_ids=ids,output_ids=generated,
                        seed=seed,generation_seconds=elapsed,
                        truncated=len(generated)>=config['max_new_tokens'] and generated[-1] not in eos)
        inference_start=time.perf_counter()
        with log_path.open('a',encoding='utf-8',newline='\n') as log:
            for task in tasks:
                if task['task_id'] in completed:
                    continue
                row=run_episode(task,generate,prompt,args.seed+100*task['sampling_index'])
                row['cuda_peak_allocated_mib']=torch.cuda.max_memory_allocated()/2**20
                log.write(json.dumps(row)+'\n')
                log.flush()
                completed.add(task['task_id'])
                if len(completed)%8==0 or len(completed)==len(tasks):
                    print(json.dumps(dict(event='progress',completed=len(completed),total=len(tasks),
                        last_kind=task['kind'],last_success=row['success'],last_stop=row['stop_reason'],
                        elapsed_seconds=round(time.perf_counter()-inference_start,2))),flush=True)
        manifest.update(status='complete',completed_tasks=len(completed),
                        inference_wall_seconds=time.perf_counter()-inference_start,
                        cuda_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                        cuda_peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                        gpu_status_end=gpu_status())
    except Exception as exc:
        manifest.update(status='failed',error=repr(exc),completed_tasks=len(completed))
        raise
    finally:
        manifest['process_wall_seconds']=time.perf_counter()-started
        save()


if __name__=='__main__':
    main()

"""Measured CUDA evaluation with replay verification and fresh rollout seeds."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import platform
import time

from .baseline import gpu_status
from .sft4_episode import run_episode
from .sft4_verify import verify_rows
from .sft4_experiment import read,rows,write,validate,checkpoint,sha256,CONDITIONS


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan',type=Path,default=Path('results/sft-v4/plan.json'))
    parser.add_argument('--model-info',type=Path,required=True)
    parser.add_argument('--model',choices=['base','old','new'],required=True)
    parser.add_argument('--prompt',choices=['protocol','examples'],required=True)
    parser.add_argument('--split',choices=['collect','reserve','dev','test','recovery','signal'],required=True)
    parser.add_argument('--seed',type=int,required=True)
    args=parser.parse_args();plan=validate(args.plan);training=checkpoint(plan,args.plan) if args.model=='new' else None
    assert args.seed in plan['evaluation_seeds'][args.split]
    assert (args.model,args.prompt) in ([("old","protocol")] if args.split in ('collect','reserve') else [("new","protocol")] if args.split=='signal' else CONDITIONS)
    if args.split=='signal':
        from .sft4_experiment import signal_gate
        signal_gate(plan)
    info=read(args.model_info)
    assert info['model_id']==plan['model_id'] and info['revision']==plan['model_revision']
    out=Path(plan['root'])/'eval'/f'{args.split}-{args.model}-{args.prompt}-seed{args.seed}'
    adapter=None
    hashes=None
    if args.model=='old':
        adapter=Path(plan['old_adapter_path']);hashes=plan['old_adapter_files_sha256']
    if args.model=='new':
        adapter=Path(plan['root'])/'training'/'adapter';hashes=training['adapter_files_sha256']
    tasks=rows(Path(plan['data_root'])/(args.split+'.jsonl'))
    prompt=Path(plan['prompt_files'][args.prompt]).read_text(encoding='utf-8')
    config=dict(model=args.model,prompt=args.prompt,split=args.split,seed=args.seed,
                model_id=info['model_id'],model_revision=info['revision'],task_count=len(tasks),
                plan_sha256=sha256(args.plan),dataset_sha256=plan['dataset_sha256'][args.split],
                system_prompt_sha256=plan['prompt_sha256'][args.prompt],adapter_files_sha256=hashes,
                source_sha256=plan['source_sha256'],**plan['eval_fixed'])
    completed=set()
    if (out/'manifest.json').exists():
        prior=read(out/'manifest.json');assert prior['config']==config
        if prior['status']=='complete' and (out/'verification.json').exists():return
    if (out/'episodes.jsonl').exists():
        saved=rows(out/'episodes.jsonl');completed={r['task_id'] for r in saved}
        assert len(saved)==len(completed) and completed<={t['task_id'] for t in tasks}
    out.mkdir(parents=True,exist_ok=True)
    import torch
    import transformers
    from transformers import AutoTokenizer,AutoModelForCausalLM,set_seed
    torch.set_num_threads(4)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError('CUDA BF16 required')
    manifest=dict(status='running',config=config,started_at=datetime.now(timezone.utc).isoformat(),
                  python=platform.python_version(),torch=torch.__version__,transformers=transformers.__version__,
                  gpu=torch.cuda.get_device_name(0),cuda=torch.version.cuda,
                  gpu_status_before=gpu_status(),new_training_performed=False,
                  source='real CUDA inference, deterministic synthetic environment',
                  checkpoint_manifest_sha256=sha256(Path(plan['root'])/'training'/'manifest.json') if training else plan['old_training_manifest_sha256'])
    start=time.perf_counter()
    write(out/'manifest.json',manifest)
    try:
        tok=AutoTokenizer.from_pretrained(info['snapshot'],local_files_only=True)
        model=AutoModelForCausalLM.from_pretrained(info['snapshot'],local_files_only=True,
                    dtype=torch.bfloat16,attn_implementation='sdpa')
        if adapter:
            import peft
            from peft import PeftModel
            model=PeftModel.from_pretrained(model,adapter,is_trainable=False,local_files_only=True)
            manifest['peft']=peft.__version__
        model.requires_grad_(False).to('cuda').eval()
        assert not any(p.requires_grad for p in model.parameters())
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        manifest.update(trainable_parameters=0,load_seconds=time.perf_counter()-start)
        print(json.dumps(dict(event='loaded',model=args.model,prompt=args.prompt,split=args.split)),flush=True)
        eos=model.generation_config.eos_token_id
        eos=[eos] if isinstance(eos,int) else eos or [tok.eos_token_id]
        def generate(messages,seed):
            text=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
            encoded=tok(text,return_tensors='pt',add_special_tokens=False)
            ids=encoded['input_ids'][0].tolist()
            if len(ids)>config['max_context']:
                return dict(context_limit=True,input_ids=ids,output_ids=[],text='',truncated=False,generation_seconds=0.)
            set_seed(seed);encoded={k:v.to('cuda') for k,v in encoded.items()}
            torch.cuda.synchronize();started=time.perf_counter()
            with torch.inference_mode():
                output=model.generate(**encoded,max_new_tokens=config['max_new_tokens'],do_sample=True,
                    temperature=config['temperature'],top_p=config['top_p'],top_k=config['top_k'],
                    pad_token_id=tok.pad_token_id or tok.eos_token_id)
            torch.cuda.synchronize()
            generated=output[0,len(ids):].tolist()
            return dict(seed=seed,text=tok.decode(generated,skip_special_tokens=True),input_ids=ids,output_ids=generated,
                        generation_seconds=time.perf_counter()-started,
                        truncated=len(generated)>=config['max_new_tokens'] and generated[-1] not in eos)
        loop=time.perf_counter()
        with (out/'episodes.jsonl').open('a',encoding='utf-8',newline='\n') as log:
            for task in tasks:
                if task['task_id'] in completed:continue
                row=run_episode(task,generate,prompt,args.seed+100*task['sampling_index'])
                log.write(json.dumps(row)+'\n');log.flush();completed.add(task['task_id'])
                if len(completed)%8==0 or len(completed)==len(tasks):
                    print(json.dumps(dict(event='progress',split=args.split,model=args.model,prompt=args.prompt,
                        seed=args.seed,completed=len(completed),total=len(tasks),last_success=row['success'],
                        seconds=round(time.perf_counter()-loop,2))),flush=True)
        rs=rows(out/'episodes.jsonl')
        count=verify_rows(rs,{t['task_id']:t for t in tasks},config,tok,prompt)
        # Independently check flags against the model's actual EOS configuration.
        for row in rs:
            for q in row['requests']:
                if not q.get('context_limit'):
                    assert q['truncated']==(len(q['output_ids'])>=config['max_new_tokens'] and q['output_ids'][-1] not in eos)
        write(out/'verification.json',dict(tasks_verified=len(rs),requests_verified=count,
            episodes_sha256=sha256(out/'episodes.jsonl'),tool_replay_matches=True,
            answer_and_cost_regrading_matches=True,input_token_provenance_matches=True,
            output_token_decode_matches=True,recovery_accounting_matches=True,eos_truncation_flags_match=True))
        manifest.update(status='complete',completed_tasks=len(rs),inference_wall_seconds=time.perf_counter()-loop,
            cuda_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
            cuda_peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,gpu_status_after=gpu_status())
        print(json.dumps(dict(event='condition_complete',split=args.split,model=args.model,prompt=args.prompt,
            seed=args.seed,tasks=len(rs),successes=sum(r['grounded_success'] for r in rs),
            correct_extra_cost=sum(r['correct_extra_cost'] or 0 for r in rs))),flush=True)
    except Exception as exc:
        manifest.update(status='failed',error=repr(exc));raise
    finally:
        manifest['process_wall_seconds']=time.perf_counter()-start
        write(out/'manifest.json',manifest)


if __name__=='__main__':main()

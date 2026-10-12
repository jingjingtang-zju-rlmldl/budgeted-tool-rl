"""Continuation of the fixed existing adapter on fresh oracle decisions."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import random
import time

from .sft3_data import request_examples, encode_example


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_hash(state):
    digest=hashlib.sha256()
    for name,value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(value.detach().contiguous().view(__import__('torch').uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def export_requests(tasks,tokenizer,prompt,max_context):
    rows=[encode_example(row,tokenizer,max_context) for task in tasks for row in request_examples(task,prompt)]
    for row in rows:
        assert row['labels'][:row['prefix_tokens']]==[-100]*row['prefix_tokens']
        assert row['labels'][row['prefix_tokens']:]==row['input_ids'][row['prefix_tokens']:]
        assert tokenizer.decode(row['input_ids'][row['prefix_tokens']:-1],skip_special_tokens=True)==row['target']
        assert row['input_ids'][-1]==tokenizer.eos_token_id
    return rows


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['smoke','train'])
    parser.add_argument('--plan',type=Path,default=Path('results/sft-v3/plan.json'))
    parser.add_argument('--model-info',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--checkpointing',action='store_true')
    args=parser.parse_args()
    if args.out.exists():
        raise ValueError('output exists; choose a new directory')
    from .sft3_experiment import validate
    plan=validate(args.plan)
    assert args.checkpointing,'Frozen memory configuration requires checkpointing'
    from .sft3_experiment import paired_gate
    paired_gate(plan)
    config=plan['training']
    info=json.loads(args.model_info.read_text(encoding='utf-8'))
    assert info['model_id']==plan['model_id'] and info['revision']==plan['model_revision']
    train_path=Path(plan['data_root'])/'train.jsonl'
    dev_path=Path(plan['data_root'])/'dev.jsonl'
    assert sha(train_path)==plan['dataset_sha256']['train'] and sha(dev_path)==plan['dataset_sha256']['dev']
    assert sha(Path(__file__))==plan['source_sha256']['sft3_train.py']
    prompt=Path(plan['prompt_files']['protocol']).read_text(encoding='utf-8')
    assert hashlib.sha256(prompt.encode()).hexdigest()==plan['prompt_sha256']['protocol']
    train=[json.loads(x) for x in train_path.read_text(encoding='utf-8').splitlines()]
    dev=[json.loads(x) for x in dev_path.read_text(encoding='utf-8').splitlines()]
    import torch
    import transformers
    import peft
    from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
    from peft import PeftModel, get_peft_model_state_dict
    torch.set_num_threads(4)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA BF16 required; no CPU fallback')
    set_seed(config['seed'])
    args.out.mkdir(parents=True)
    manifest=dict(status='running',mode=args.mode,started_at=datetime.now(timezone.utc).isoformat(),
                  plan_sha256=sha(args.plan),config=config,gradient_checkpointing=args.checkpointing,
                  model_id=info['model_id'],model_revision=info['revision'],train_sha256=sha(train_path),dev_sha256=sha(dev_path),
                  python=platform.python_version(),torch=torch.__version__,transformers=transformers.__version__,peft=peft.__version__,
                  accelerate=importlib.metadata.version('accelerate'),gpu=torch.cuda.get_device_name(0),cuda=torch.version.cuda,
                  source_sha256=sha(Path(__file__)),cuda_allocator_environment=__import__('os').environ.get('PYTORCH_CUDA_ALLOC_CONF'),training_performed=False,continuation=True,old_adapter_files_sha256=plan['old_adapter_files_sha256'])
    def save():
        (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8',newline='\n')
    save()
    started=time.perf_counter()
    try:
        tokenizer=AutoTokenizer.from_pretrained(info['snapshot'],local_files_only=True)
        rows=export_requests(train,tokenizer,prompt,config['max_context'])
        dev_rows=export_requests(dev,tokenizer,prompt,config['max_context'])
        assert len(rows)==plan['decision_targets'] and len({r['task_id'] for r in rows})==300
        (args.out/'supervision.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows),encoding='utf-8',newline='\n')
        manifest.update(supervision_sha256=sha(args.out/'supervision.jsonl'),training_tasks=300,decision_targets=len(rows),
                        max_sequence_tokens=max(len(r['input_ids']) for r in rows),
                        supervised_tokens=sum(r['target_tokens'] for r in rows),
                        recovery_next_states=sum(bool(r['messages'][-1]['role']=='tool' and 'error' in json.loads(r['messages'][-1]['content'])) for r in rows))
        base=AutoModelForCausalLM.from_pretrained(info['snapshot'],local_files_only=True,dtype=torch.bfloat16,
                                               attn_implementation='sdpa').to('cuda')
        base.config.use_cache=False
        model=PeftModel.from_pretrained(base,Path(plan['old_adapter_path']),is_trainable=True,local_files_only=True)
        if args.checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        parameters=[p for p in model.parameters() if p.requires_grad]
        assert parameters and all('lora_' in name for name,p in model.named_parameters() if p.requires_grad)
        initial=tensor_hash(get_peft_model_state_dict(model))
        assert initial==plan['old_adapter_state_sha256'],'Training did not start from the existing adapter'
        manifest.update(trainable_parameters=sum(p.numel() for p in parameters),
                        total_parameters=sum(p.numel() for p in model.parameters()),
                        initial_adapter_sha256=initial,
                        parameter_dtypes=dict((str(dtype),sum(p.numel() for p in model.parameters() if p.dtype==dtype))
                                              for dtype in {p.dtype for p in model.parameters()}))
        optimizer=torch.optim.AdamW(parameters,lr=config['learning_rate'],weight_decay=0.,betas=(.9,.999))
        print(json.dumps({'event':'training_model_loaded','mode':args.mode,
                          'trainable_parameters':manifest['trainable_parameters'],
                          'max_sequence_tokens':manifest['max_sequence_tokens']}),flush=True)
        def forward(row):
            ids=torch.tensor([row['input_ids']],device='cuda',dtype=torch.long)
            labels=torch.tensor([row['labels']],device='cuda',dtype=torch.long)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                return model(input_ids=ids,attention_mask=torch.ones_like(ids),labels=labels,use_cache=False).loss
        def dev_loss():
            model.eval()
            losses=[]
            with torch.no_grad():
                for row in dev_rows:
                    losses.append(float(forward(row)))
            model.train()
            return sum(losses)/len(losses)
        if args.mode=='train':
            manifest['dev_loss_before']=dev_loss()
        accumulation=config['gradient_accumulation']
        if args.mode=='smoke':
            longest=sorted(rows,key=lambda r:len(r['input_ids']),reverse=True)[:accumulation]
            selected=longest*8
        else:
            selected=list(rows)
            random.Random(config['seed']).shuffle(selected)
        manifest.update(distinct_tasks_used=len({r['task_id'] for r in selected}),
                        distinct_decision_targets_used=len({(r['task_id'],r['turn']) for r in selected}))
        model.train()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        loop_started=time.perf_counter()
        records=[]
        with (args.out/'steps.jsonl').open('w',encoding='utf-8',newline='\n') as log:
            for step,offset in enumerate(range(0,len(selected),accumulation),1):
                group=selected[offset:offset+accumulation]
                losses=[]
                for row in group:
                    loss=forward(row)
                    if not torch.isfinite(loss):
                        raise RuntimeError('nonfinite loss')
                    losses.append(float(loss.detach()))
                    (loss/len(group)).backward()
                    del loss
                norm=torch.nn.utils.clip_grad_norm_(parameters,1.)
                if not torch.isfinite(norm) or float(norm)==0:
                    raise RuntimeError('nonfinite or zero adapter gradient')
                if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
                    raise RuntimeError('base model unexpectedly has gradients')
                optimizer.step()
                manifest.update(training_performed=True,optimizer_steps=step,microbatches_seen=offset+len(group))
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()  # Frozen per-step allocator maintenance on Windows.
                torch.cuda.synchronize()
                record=dict(step=step,loss=sum(losses)/len(losses),gradient_norm_before_clip=float(norm),
                            request_ids=[r['task_id']+':'+str(r['turn']) for r in group],
                            elapsed_seconds=time.perf_counter()-loop_started,
                            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                            peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
                records.append(record)
                log.write(json.dumps(record)+'\n')
                log.flush()
                if args.mode=='smoke' or step%5==0:
                    print(json.dumps({'event':'optimizer_step','mode':args.mode,**{k:record[k] for k in
                                     ['step','loss','gradient_norm_before_clip','elapsed_seconds','peak_allocated_mib']}}),flush=True)
        final=tensor_hash(get_peft_model_state_dict(model))
        if final==initial:
            raise RuntimeError('adapter weights did not change')
        manifest.update(training_performed=True,optimizer_steps=len(records),microbatches=len(selected),
                        adapter_changed=True,final_adapter_state_sha256=final,
                        optimization_wall_seconds=time.perf_counter()-loop_started,
                        cuda_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                        cuda_peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                        frozen_base_gradients_absent=True)
        if args.mode=='train':
            assert len(records)==math.ceil(len(rows)/accumulation) and len({i for r in records for i in r['request_ids']})==len(rows)
            manifest['dev_loss_after']=dev_loss()
            adapter=args.out/'adapter'
            model.save_pretrained(adapter,safe_serialization=True)
            manifest['adapter_files_sha256']={p.name:sha(p) for p in adapter.iterdir() if p.is_file()}
            manifest['checkpoint_selection']='fixed final checkpoint after one epoch; no test-based selection'
        manifest['status']='complete'
    except Exception as exc:
        manifest.update(status='failed',error=repr(exc))
        raise
    finally:
        manifest['process_wall_seconds']=time.perf_counter()-started
        save()
    print(json.dumps({'event':'finished','mode':args.mode,'steps':manifest['optimizer_steps'],
                      'peak_allocated_mib':manifest['cuda_peak_allocated_mib']}),flush=True)


if __name__=='__main__':
    main()

"""Same baseline engine, with a pinned prompt and optional frozen LoRA adapter."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from . import baseline


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan',type=Path,default=Path('results/sft-v1/plan.json'))
    parser.add_argument('--model-info',type=Path,required=True)
    parser.add_argument('--model',choices=['base','sft'],required=True)
    parser.add_argument('--prompt',choices=['neutral','balanced_examples'],required=True)
    parser.add_argument('--split',choices=['dev','test'],required=True)
    parser.add_argument('--seed',type=int,required=True)
    args=parser.parse_args()
    plan=json.loads(args.plan.read_text(encoding='utf-8'))
    info=json.loads(args.model_info.read_text(encoding='utf-8'))
    assert info['model_id']==plan['model_id'] and info['revision']==plan['model_revision']
    assert baseline.sha256(Path(__file__))==plan['source_sha256']['sft_eval.py']
    data=Path(plan['data_root'])/(args.split+'.jsonl')
    assert baseline.sha256(data)==plan['dataset_sha256'][args.split]
    assert args.seed in plan['evaluation_seeds'][args.split]
    root=Path(plan['root'])
    adapter=root/'training'/'adapter'
    training_manifest=root/'training'/'manifest.json'
    # Test outputs are allowed only after the final adapter is frozen, even for base.
    if args.model=='sft' or args.split=='test':
        training=json.loads(training_manifest.read_text(encoding='utf-8'))
        assert training['status']=='complete' and training['optimizer_steps']==75
        for name,sha in training['adapter_files_sha256'].items():
            assert baseline.sha256(adapter/name)==sha
    adapter_sha=baseline.sha256(adapter/'adapter_model.safetensors') if args.model=='sft' else None
    out=root/'eval'/f'{args.split}-{args.model}-{args.prompt}-seed{args.seed}'
    prompt=Path(plan['prompt_files'][args.prompt]).read_text(encoding='utf-8')
    assert hashlib.sha256(prompt.encode()).hexdigest()==plan['prompt_sha256'][args.prompt]
    condition=dict(plan_sha256=baseline.sha256(args.plan),split=args.split,model=args.model,prompt=args.prompt,
                   seed=args.seed,adapter_sha256=adapter_sha,source_sha256=baseline.sha256(Path(__file__)))
    if out.exists() and (out/'condition.json').exists():
        assert json.loads((out/'condition.json').read_text(encoding='utf-8'))==condition
        if (out/'manifest.json').exists() and json.loads((out/'manifest.json').read_text())['status']=='complete':
            return
    else:
        out.mkdir(parents=True,exist_ok=True)
        (out/'condition.json').write_text(json.dumps(condition,indent=2)+'\n',encoding='utf-8',newline='\n')
    baseline.SYSTEM_PROMPT=prompt
    if args.model=='sft':
        from transformers import AutoModelForCausalLM
        from peft import PeftModel
        original_loader=AutoModelForCausalLM.from_pretrained
        def load_with_adapter(*values,**kwargs):
            base=original_loader(*values,**kwargs)
            return PeftModel.from_pretrained(base,adapter,is_trainable=False,local_files_only=True)
        AutoModelForCausalLM.from_pretrained=load_with_adapter
    sys.argv=['toolrl.baseline','--data',str(data),'--model-info',str(args.model_info),
              '--out',str(out),'--parser','fenced','--seed',str(args.seed)]
    if (out/'episodes.jsonl').exists():
        sys.argv.append('--resume')
    baseline.main()
    manifest=json.loads((out/'manifest.json').read_text(encoding='utf-8'))
    manifest['sft_experiment']=condition|dict(adapter_storage='FP32' if args.model=='sft' else None,
                                             inference_weights_updated=args.model=='sft')
    # The engine's training_performed field refers to this inference process.
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8',newline='\n')


if __name__=='__main__':
    main()

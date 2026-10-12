"""Frozen data/configuration and staged local SFT experiment orchestration."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys

from .baseline import sha256
from .prompt_study import PROMPTS
from .sft_data import write_splits

SOURCE_FILES=['baseline.py','environment.py','sft_data.py','sft_train.py','sft_eval.py','sft_experiment.py','verify_run.py']


def prepare(root,data_root,info):
    if root.exists():
        raise ValueError('experiment exists; choose a new directory')
    audit=write_splits(data_root)
    model_info=json.loads(info.read_text())
    root.mkdir(parents=True)
    (root/'prompts').mkdir()
    prompts={}
    for name in ['neutral','balanced_examples']:
        path=root/'prompts'/(name+'.txt')
        path.write_text(PROMPTS[name],encoding='utf-8',newline='\n')
        prompts[name]=str(path)
    plan=dict(prepared_at=datetime.now(timezone.utc).isoformat(),root=str(root),data_root=str(data_root),
              model_id=model_info['model_id'],model_revision=model_info['revision'],
              dataset_sha256=audit['sha256'],data_counts={'train':300,'dev':30,'test':60},
              prompt_files=prompts,prompt_sha256={k:sha256(Path(p)) for k,p in prompts.items()},
              source_sha256={name:sha256(Path(__file__).with_name(name)) for name in SOURCE_FILES},
              training=dict(seed=42,rank=8,alpha=16,dropout=.05,target_modules=['q_proj','v_proj'],
                            learning_rate=1e-4,micro_batch=1,gradient_accumulation=8,epochs=1,
                            max_context=768,optimizer='AdamW',weight_decay=0.,gradient_clip=1.,
                            dtype='BF16 backbone, FP32 LoRA parameters',loss='mean assistant-token loss per request, averaged over accumulation group'),
              memory_test=dict(optimizer_steps=8,microbatches=64,selection='eight longest train decision contexts, repeated eight times',
                               fresh_model_for_full_training=True,fallback='retry with non-reentrant gradient checkpointing only on CUDA OOM'),
              evaluation_seeds={'dev':[7],'test':[7,10007,20007]},
              eval_conditions=['base/neutral','base/balanced_examples','sft/neutral','sft/balanced_examples'],
              eval_fixed=dict(parser='fenced',temperature=.7,top_p=.8,top_k=20,max_context=1024,max_new_tokens=64,
                              max_turns=4,max_calls=3,thinking=False,batch_size=1),
              checkpoint_selection='fixed final one-epoch checkpoint; no hyperparameter or checkpoint selection using test',
              test_gate='all base and SFT test generation happens only after the final adapter has been saved and hashed',
              limitation='One training seed; three evaluation seeds share 60 held-out-template tasks. Controlled synthetic English schema only.')
    (root/'plan.json').write_text(json.dumps(plan,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({'event':'sft_prepared','data_counts':plan['data_counts'],'test_seeds':plan['evaluation_seeds']['test']}),flush=True)


def validate(plan_path):
    plan=json.loads(plan_path.read_text())
    for name,sha in plan['source_sha256'].items():
        if sha256(Path(__file__).with_name(name))!=sha:
            raise ValueError('frozen source changed: '+name)
    for split,sha in plan['dataset_sha256'].items():
        assert sha256(Path(plan['data_root'])/(split+'.jsonl'))==sha
    for name,sha in plan['prompt_sha256'].items():
        assert sha256(Path(plan['prompt_files'][name]))==sha
    return plan


def evaluate(plan_path,info,models,splits):
    plan=validate(plan_path)
    for split in splits:
        for seed in plan['evaluation_seeds'][split]:
            for model in models:
                # Rotate prompts on alternate seeds; same task/turn seeds across conditions.
                prompts=['neutral','balanced_examples'] if seed!=10007 else ['balanced_examples','neutral']
                for prompt in prompts:
                    print(json.dumps({'event':'eval_started','split':split,'model':model,'prompt':prompt,'seed':seed}),flush=True)
                    subprocess.run([sys.executable,'-m','toolrl.sft_eval','--plan',str(plan_path),'--model-info',str(info),
                                    '--model',model,'--prompt',prompt,'--split',split,'--seed',str(seed)],check=True)
                    out=Path(plan['root'])/'eval'/f'{split}-{model}-{prompt}-seed{seed}'
                    subprocess.run([sys.executable,'-m','toolrl.analyze',str(out)],check=True,capture_output=True,text=True)
                    subprocess.run([sys.executable,'-m','toolrl.verify_run',str(out),'--data',str(Path(plan['data_root'])/(split+'.jsonl')),
                                    '--model-info',str(info)],check=True,capture_output=True,text=True)
                    m=json.loads((out/'summary.json').read_text())
                    print(json.dumps({'event':'eval_complete','split':split,'model':model,'prompt':prompt,'seed':seed,
                                      'correct':m['overall']['successes'],'strict_correct':m['overall']['strict_successes'],
                                      'by_kind':{k:v['successes'] for k,v in m['by_kind'].items()}}),flush=True)


def train(plan_path,info):
    plan=validate(plan_path)
    root=Path(plan['root'])
    smoke=root/'smoke'
    flag=[]
    if not (smoke/'manifest.json').exists():
        result=subprocess.run([sys.executable,'-m','toolrl.sft_train','smoke','--plan',str(plan_path),
                               '--model-info',str(info),'--out',str(smoke)])
        if result.returncode:
            failure=json.loads((smoke/'manifest.json').read_text())
            if 'out of memory' not in failure.get('error','').lower():
                raise RuntimeError('memory probe failed for a reason other than OOM: '+failure.get('error','unknown'))
    status=json.loads((smoke/'manifest.json').read_text())
    if status['status']!='complete':
        smoke=root/'smoke-checkpointed'
        flag=['--checkpointing']
        if not (smoke/'manifest.json').exists():
            subprocess.run([sys.executable,'-m','toolrl.sft_train','smoke','--plan',str(plan_path),
                            '--model-info',str(info),'--out',str(smoke),*flag],check=True)
        assert json.loads((smoke/'manifest.json').read_text())['status']=='complete'
    if not (root/'training'/'manifest.json').exists():
        subprocess.run([sys.executable,'-m','toolrl.sft_train','train','--plan',str(plan_path),
                        '--model-info',str(info),'--out',str(root/'training'),*flag],check=True)
    assert json.loads((root/'training'/'manifest.json').read_text())['status']=='complete'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['prepare','baseline-dev','train','evaluate-final'])
    parser.add_argument('--root',type=Path,default=Path('results/sft-v1'))
    parser.add_argument('--data-root',type=Path,default=Path('data/sft-v1'))
    parser.add_argument('--model-info',type=Path,required=True)
    args=parser.parse_args()
    plan_path=args.root/'plan.json'
    if args.stage=='prepare':
        prepare(args.root,args.data_root,args.model_info)
    elif args.stage=='baseline-dev':
        evaluate(plan_path,args.model_info,['base'],['dev'])
    elif args.stage=='train':
        train(plan_path,args.model_info)
    else:
        evaluate(plan_path,args.model_info,['sft'],['dev'])
        evaluate(plan_path,args.model_info,['base','sft'],['test'])


if __name__=='__main__':
    main()

"""Record a discarded operational attempt, then launch the unchanged trainer."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone

project=Path(__file__).resolve().parents[1]/'outputs/budgeted-tool-rl'
root=project/'results/sft-v2'
old=root/'training-attempt-gc-cache'
manifest=json.loads((old/'manifest.json').read_text(encoding='utf-8'))
steps=[json.loads(line) for line in (old/'steps.jsonl').read_text(encoding='utf-8').splitlines()]
manifest.update(status='aborted_resource_contention',
    aborted_at=datetime.now(timezone.utc).isoformat(),
    reason='Discarded partial run to enable native allocator unused-cache garbage collection under concurrent GPU load; not OOM or checkpoint selection.',
    completed_logged_optimizer_steps=len(steps),adapter_saved=False)
(old/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
config='backend:native,max_split_size_mb:64,garbage_collection_threshold:0.6'
prior_record=json.loads((root/'resource-retry.json').read_text(encoding='utf-8'))
(root/'resource-retry-first.json').write_text(json.dumps(prior_record,indent=2)+'\n',encoding='utf-8')
record=dict(started_at=datetime.now(timezone.utc).isoformat(),
    partial_attempt='training-attempt-gc-cache',discarded_optimizer_steps=len(steps),
    previous_retry_record='resource-retry-first.json',previous_discarded_steps=prior_record['discarded_optimizer_steps'],
    partial_steps_sha256=hashlib.sha256((old/'steps.jsonl').read_bytes()).hexdigest(),
    reason=manifest['reason'],environment={'PYTORCH_CUDA_ALLOC_CONF':config},
    training_source_unchanged=True,all_hyperparameters_unchanged=True,
    restarts_from='results/sft-v1/training/adapter',
    checkpoint_selection='Fixed final one epoch after fresh restart; no partial weights reused.',
    documentation='https://docs.pytorch.org/docs/2.9/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf')
(root/'resource-retry.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
env=os.environ.copy();env['PYTORCH_CUDA_ALLOC_CONF']=config
command=[sys.executable,'-m','toolrl.sft2_train','train','--model-info','../../work/model-location.json','--out','results/sft-v2/training']
print(json.dumps(dict(event='resource_restart',allocator=config,discarded_steps=len(steps))),flush=True)
raise SystemExit(subprocess.call(command,cwd=project,env=env))

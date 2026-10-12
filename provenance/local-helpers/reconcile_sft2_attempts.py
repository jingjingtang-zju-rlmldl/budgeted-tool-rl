"""Repair stale initial manifests using flushed partial-progress evidence."""
from pathlib import Path
import json,shutil
root=Path(__file__).resolve().parents[1]/'outputs/budgeted-tool-rl/results/sft-v2'
for name in ('training-attempt-default-cache','training-attempt-gc-cache','training-attempt-split-cache'):
    path=root/name/'manifest.json'
    before=path.with_name('manifest-before-progress-reconciliation.json')
    assert not before.exists();shutil.copyfile(path,before)
    m=json.loads(path.read_text(encoding='utf-8'))
    steps=[json.loads(line) for line in path.with_name('steps.jsonl').read_text(encoding='utf-8').splitlines()]
    assert m['status']=='aborted_resource_contention' and steps
    m.update(training_performed=True,progress_reconciled_from_flushed_steps=True,
        minimum_completed_optimizer_steps=len(steps),
        minimum_microbatches_seen=sum(len(s['request_ids']) for s in steps),
        aborted_before_final_manifest_save=True,
        progress_scope='Lower bounds from flushed completed-step records; in-flight work is uncounted. No adapter saved or reused.')
    path.write_text(json.dumps(m,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'attempt':name,'logged_steps':len(steps),'partial_weights_reused':False}))

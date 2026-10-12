"""Preserve provenance for a resource-only inference restart on native Windows."""
from pathlib import Path
from datetime import datetime,timezone
import json
import hashlib
import shutil
import sys

project=Path(__file__).resolve().parents[1]/'outputs/budgeted-tool-rl'
root=project/'results/sft-v4';out=root/'resource-maintenance';out.mkdir(exist_ok=True)
cohort=root/'eval/test-old-examples-seed10017'
m=json.loads((cohort/'manifest.json').read_text(encoding='utf-8'))
assert m['status']=='running' and m['config']['seed']==10017 and m['config']['model']=='old' and m['config']['prompt']=='examples'
shutil.copy2(cohort/'manifest.json',out/'test-old-examples-seed10017.manifest-before-restart.json')
data=(cohort/'episodes.jsonl').read_bytes()
assert data.endswith(b'\n')
rs=[json.loads(x) for x in data.splitlines()];assert len({r['task_id'] for r in rs})==len(rs)
(out/'test-old-examples-seed10017.prefix-before-restart.jsonl').write_bytes(data)
record=dict(registered_at=datetime.now(timezone.utc).isoformat(),
    reason='Completed first strong-prompt cohort peaked at allocated2355.25MiB/reserved7496MiB on an8151MiB GPU; subsequent inference uses allocator garbage collection to reduce unused-cache pressure.',
    allocator='backend:native,garbage_collection_threshold:0.5',model_sampling_data_prompts_rewards_unchanged=True,
    original_completed_prefix_episodes=len(rs),prefix_sha256=hashlib.sha256(data).hexdigest(),
    resumed_cohort=cohort.name,interrupted_inflight_episode_restarted=True,
    counts='Only complete recorded episodes and their recorded requests enter replay totals; any unlogged in-flight request is not claimed as an audited episode.',
    plan_sha256=hashlib.sha256((root/'plan.json').read_bytes()).hexdigest())
(out/'amendment.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps(record))

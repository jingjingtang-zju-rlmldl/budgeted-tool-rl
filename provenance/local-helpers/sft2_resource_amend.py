"""Preserve original plan and partial evidence before a memory-only amendment."""
from pathlib import Path
import json,hashlib,shutil
from datetime import datetime,timezone
root=Path(__file__).resolve().parents[1]/'outputs/budgeted-tool-rl/results/sft-v2'
training=root/'training';target=root/'training-attempt-split-cache'
assert root.resolve() in training.resolve().parents and not target.exists()
training.rename(target)
manifest=json.loads((target/'manifest.json').read_text(encoding='utf-8'))
steps=[json.loads(line) for line in (target/'steps.jsonl').read_text(encoding='utf-8').splitlines()]
manifest.update(status='aborted_resource_contention',aborted_at=datetime.now(timezone.utc).isoformat(),
    completed_logged_optimizer_steps=len(steps),adapter_saved=False,
    reason='Discarded native split-cache attempt after severe shared-memory paging; no test observed.')
(target/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
original=root/'plan-original.json';assert not original.exists()
shutil.copyfile(root/'plan.json',original)
p=json.loads(original.read_text(encoding='utf-8'))
p['memory_test']['fallback']='non-reentrant gradient checkpointing on CUDA OOM OR severe memory pressure / Windows shared-memory paging'
p['memory_test']['final_gradient_checkpointing']=True
p['memory_test']['resource_amendment_path']='results/sft-v2/resource-amendment.json'
record=dict(amended_at=datetime.now(timezone.utc).isoformat(),
    original_plan_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
    reason='Memory-only trigger expanded before any evaluation: WDDM paging can severely slow training without CUDA OOM.',
    changed_fields=['memory_test.fallback','memory_test.final_gradient_checkpointing','memory_test.resource_amendment_path'],
    frozen_sources_unchanged=True,data_prompts_seeds_hyperparameters_unchanged=True,
    checkpoint_selection_unchanged=True,no_test_or_signal_output_exists=True,
    final_environment={'PYTORCH_CUDA_ALLOC_CONF':'backend:native,garbage_collection_threshold:0.5'},
    final_training_flag='--checkpointing',
    additional_discarded_attempt='training-attempt-split-cache',
    additional_discarded_optimizer_steps=len(steps),
    additional_steps_sha256=hashlib.sha256((target/'steps.jsonl').read_bytes()).hexdigest(),
    peak_reserved_mib=max(s['peak_reserved_mib'] for s in steps),
    retries_before_this=['resource-retry-first.json','resource-retry.json'],
    all_partial_weights_discarded=True)
assert not (root/'eval').exists()
(root/'resource-amendment.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
(root/'plan.json').write_text(json.dumps(p,indent=2)+'\n',encoding='utf-8')
print(json.dumps(record,indent=2))

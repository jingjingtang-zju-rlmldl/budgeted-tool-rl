from pathlib import Path

root=Path(__file__).resolve().parents[1]/'outputs/budgeted-tool-rl/toolrl'
for name in ['episode','verify','train','run']:
    text=(root/f'sft3_{name}.py').read_text(encoding='utf-8').replace('sft3_', 'sft4_').replace('sft-v3','sft-v4')
    text=text.replace('from .stress_environment import StressEnvironment','from .sft4_environment import BranchEnvironment as StressEnvironment')
    text=text.replace('MAX_CALLS,MAX_TURNS','MAX_CALLS,MAX_TURNS,grounded as is_grounded')
    if name=='episode':
        text=text.replace("text=json.dumps(dict(action='lookup',**a),separators=(',',':'))", "text=task['bootstrap_texts'][len(bootstrap)] if task.get('bootstrap_texts') else json.dumps(dict(action='lookup',**a),separators=(',',':'))")
        start=text.index("    goal=task['oracle_calls'][-1]")
        end=text.index('    return dict(',start)
        text=text[:start]+"    grounded=is_grounded(task,env,answer)\n"+text[end:]
        text=text.replace("minimum_calls=task['minimum_calls'],", "minimum_calls=task['minimum_calls'],minimum_cost=task['minimum_cost'],total_cost=env.total_cost,\n                correct_extra_cost=env.total_cost-task['minimum_cost'] if grounded else None,")
        text=text.replace("reward_cost_gated=float(success)*(1-.03*len(env.calls))", "reward_call_cost_gated=float(success)*(1-.03*len(env.calls)),\n                reward_cost_gated=float(grounded)*(1-.02*env.total_cost)")
    elif name=='verify':
        text=text.replace('import BranchEnvironment as StressEnvironment','import BranchEnvironment as StressEnvironment,grounded as is_grounded')
        text=text.replace("'expected','minimum_calls'", "'expected','minimum_calls','minimum_cost'")
        text=text.replace("content=json.dumps(dict(action='lookup',**a),separators=(',',':'))", "content=task['bootstrap_texts'][len(bootstrap)-1] if task.get('bootstrap_texts') else json.dumps(dict(action='lookup',**a),separators=(',',':'))")
        start=text.index("        goal=task['oracle_calls'][-1]"); end=text.index("        assert row['grounded_success']",start)
        text=text[:start]+"        grounded=is_grounded(task,env,answer)\n"+text[end:]
        text=text.replace("assert row['reward_cost_gated']==float(success)*(1-.03*len(env.calls))", "assert row['total_cost']==env.total_cost\n        assert row['correct_extra_cost']==(env.total_cost-task['minimum_cost'] if grounded else None)\n        assert row['reward_call_cost_gated']==float(success)*(1-.03*len(env.calls))\n        assert row['reward_cost_gated']==float(grounded)*(1-.02*env.total_cost)")
    elif name=='train':
        text=text.replace('from .sft4_experiment import paired_gate\n    paired_gate(plan)', 'from .sft4_experiment import training_gate\n    commit=training_gate(plan)\n    plan=plan|dict(decision_targets=commit["decision_targets"])')
        text=text.replace("/'train.jsonl'", "/'derived_train.jsonl'")
        text=text.replace("plan['dataset_sha256']['train']", "commit['derived_train_sha256']")
        text=text.replace('==300', "==commit['training_tasks']").replace('training_tasks=300', "training_tasks=commit['training_tasks']")
        text=text.replace("old_adapter_files_sha256=plan['old_adapter_files_sha256']", "old_adapter_files_sha256=plan['old_adapter_files_sha256'],training_set_manifest_sha256=sha(Path(plan['root'])/'training-set-manifest.json')")
    elif name=='run':
        text=text.replace("choices=['dev','test','recovery','pair','signal']", "choices=['collect','reserve','dev','test','recovery','signal']")
        start=text.index("    assert (args.model,args.prompt)"); end=text.index("    if args.split=='signal':",start)
        text=text[:start]+"    assert (args.model,args.prompt) in ([(\"old\",\"protocol\")] if args.split in ('collect','reserve') else [(\"new\",\"protocol\")] if args.split=='signal' else CONDITIONS)\n"+text[end:]
        start=text.index("    if args.split=='pair'"); end=text.index("    prompt=",start);text=text[:start]+text[end:]
    (root/f'sft4_{name}.py').write_text(text,encoding='utf-8',newline='\n')
print('Four new modules created; historical files untouched.')

from pathlib import Path
root=Path(__file__).resolve().parents[1]/'outputs/budgeted-tool-rl/toolrl'
s=(root/'stress_runner.py').read_text(encoding='utf-8').split('\ndef main():')[0]
s=s.replace("    requests=[]", "    bootstrap=[]\n    for a in task.get('bootstrap_calls',[]):\n        text=json.dumps(dict(action='lookup',**a),separators=(',',':'))\n        o=env.lookup(**a);bootstrap.append(o)\n        messages += [dict(role='assistant',content=text),dict(role='tool',content=json.dumps(o,sort_keys=True))]\n    requests=[]",1)
s=s.replace('range(MAX_TURNS):',"range(MAX_TURNS-len(task.get('bootstrap_calls',[]))):")
s=s.replace("return dict(task_id=", "return dict(bootstrap_observations=bootstrap,bootstrap_calls=task.get('bootstrap_calls',[]),task_id=",1)
(root/'sft3_episode.py').write_text(s,encoding='utf-8',newline='\n')
s=(root/'stress_verify.py').read_text(encoding='utf-8').split('\ndef main():')[0]
s=s.replace('        actions=[]', "        bootstrap=[]\n        for a in task.get('bootstrap_calls',[]):\n            o=env.lookup(**a);bootstrap.append(o)\n            prefix += [dict(role='assistant',content=json.dumps(dict(action='lookup',**a),separators=(',',':'))),\n                       dict(role='tool',content=json.dumps(o,sort_keys=True))]\n        assert row['bootstrap_observations']==bootstrap and row['bootstrap_calls']==task.get('bootstrap_calls',[])\n        actions=[]",1)
s=s.replace("len(row['requests'])<=config['max_turns']", "len(row['requests'])<=config['max_turns']-len(task.get('bootstrap_calls',[]))")
(root/'sft3_verify.py').write_text(s,encoding='utf-8',newline='\n')
s=(root/'sft2_train.py').read_text(encoding='utf-8')
s=s.replace('.sft2_data','.sft3_data').replace('.sft2_experiment','.sft3_experiment').replace('results/sft-v2/plan.json','results/sft-v3/plan.json').replace("'sft2_train.py'","'sft3_train.py'")
s=s.replace("    config=plan['training']", "    assert args.checkpointing,'Frozen memory configuration requires checkpointing'\n    from .sft3_experiment import paired_gate\n    paired_gate(plan)\n    config=plan['training']")
s=s.replace('                optimizer.zero_grad(set_to_none=True)\n                torch.cuda.synchronize()', '                optimizer.zero_grad(set_to_none=True)\n                torch.cuda.empty_cache()  # Frozen per-step allocator maintenance on Windows.\n                torch.cuda.synchronize()')
s=s.replace("source_sha256=sha(Path(__file__)),training_performed", "source_sha256=sha(Path(__file__)),cuda_allocator_environment=__import__('os').environ.get('PYTORCH_CUDA_ALLOC_CONF'),training_performed")
(root/'sft3_train.py').write_text(s,encoding='utf-8',newline='\n')
s=(root/'sft2_run.py').read_text(encoding='utf-8')
s=s.replace('.stress_runner','.sft3_episode').replace('.stress_verify','.sft3_verify').replace('.sft2_experiment','.sft3_experiment').replace('results/sft-v2/plan.json','results/sft-v3/plan.json')
s=s.replace("choices=['dev','test','signal']", "choices=['dev','test','recovery','pair','signal']")
s=s.replace("args=parser.parse_args();plan=validate(args.plan);training=checkpoint(plan,args.plan)", "args=parser.parse_args();plan=validate(args.plan);training=checkpoint(plan,args.plan) if args.model=='new' else None")
start=s.index("    assert (args.model,args.prompt)")
end=s.index("    info=read(args.model_info)",start)
s=s[:start]+'''    assert (args.model,args.prompt) in (CONDITIONS if args.split not in ('pair','signal') else [('old','protocol'),('new','protocol')] if args.split=='pair' else [('new','protocol')])
    if args.split=='signal':
        from .sft3_experiment import signal_gate
        signal_gate(plan)
'''+s[end:]
s=s.replace("checkpoint_manifest_sha256=sha256(Path(plan['root'])/'training'/'manifest.json')", "checkpoint_manifest_sha256=sha256(Path(plan['root'])/'training'/'manifest.json') if training else plan['old_training_manifest_sha256']")
s=s.replace("row=run_episode(task,generate,prompt,args.seed+100*task['sampling_index'])", "row=run_episode(task,generate,prompt,args.seed+100*task['sampling_index'])")
(root/'sft3_run.py').write_text(s,encoding='utf-8',newline='\n')

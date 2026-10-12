from pathlib import Path
root=Path(__file__).resolve().parent.parent/'outputs'/'budgeted-tool-rl'/'toolrl'
text=(root/'sft_train.py').read_text(encoding='utf-8')
changes={
 '"""Measured CUDA LoRA smoke test and one-epoch, assistant-only SFT."""':'"""Continuation of the fixed existing adapter on fresh oracle decisions."""',
 'from .sft_data import request_examples, encode_example':'from .sft2_data import request_examples, encode_example',
 "default=Path('results/sft-v1/plan.json')":"default=Path('results/sft-v2/plan.json')",
 "plan['source_sha256']['sft_train.py']":"plan['source_sha256']['sft2_train.py']",
 "plan['prompt_files']['neutral']":"plan['prompt_files']['protocol']",
 "plan['prompt_sha256']['neutral']":"plan['prompt_sha256']['protocol']",
 'from peft import LoraConfig, get_peft_model, get_peft_model_state_dict':'from peft import PeftModel, get_peft_model_state_dict',
 "assert len(rows)==600 and len({r['task_id'] for r in rows})==300":"assert len(rows)==plan['decision_targets'] and len({r['task_id'] for r in rows})==300",
 "second_lookup_states=sum(r['kind']=='two_lookup' and r['turn']==1 for r in rows)":"recovery_next_states=sum(bool(r['messages'][-1]['role']=='tool' and 'error' in json.loads(r['messages'][-1]['content'])) for r in rows)",
 "model=get_peft_model(base,LoraConfig(r=config['rank'],lora_alpha=config['alpha'],lora_dropout=config['dropout'],\n                                            target_modules=config['target_modules'],bias='none',task_type='CAUSAL_LM'))":"model=PeftModel.from_pretrained(base,Path(plan['old_adapter_path']),is_trainable=True,local_files_only=True)",
 "assert len(records)==75 and len({i for r in records for i in r['request_ids']})==600":"assert len(records)==math.ceil(len(rows)/accumulation) and len({i for r in records for i in r['request_ids']})==len(rows)",
 "training_performed=False)":"training_performed=False,continuation=True,old_adapter_files_sha256=plan['old_adapter_files_sha256'])",
 "manifest.update(trainable_parameters=sum":"manifest.update(trainable_parameters=sum"
}
for old,new in changes.items():
    assert old in text,old
    text=text.replace(old,new)
path=root/'sft2_train.py'
assert not path.exists()
path.write_text(text,encoding='utf-8',newline='\n')
print(path)

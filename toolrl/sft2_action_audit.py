"""Post-hoc size audit of compact oracle actions, without model generation."""
import argparse,json
from collections import Counter
from pathlib import Path
from .sft2_experiment import read,rows,write,validate,sha256
from .sft2_data import request_examples

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('results/sft-v2'))
    p.add_argument('--model-info',type=Path,required=True);a=p.parse_args()
    plan=validate(a.root/'plan.json')
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(read(a.model_info)['snapshot'],local_files_only=True)
    prompt=Path(plan['prompt_files']['protocol']).read_text(encoding='utf-8')
    sizes=[];by_split={};longest=None
    for split in plan['dataset_sha256']:
        local=[]
        for task in rows(Path(plan['data_root'])/(split+'.jsonl')):
            for item in request_examples(task,prompt):
                ids=tokenizer(item['target'],add_special_tokens=False)['input_ids']+[tokenizer.eos_token_id]
                size=len(ids);sizes.append(size);local.append(size)
                if longest is None or size>longest['tokens_including_eos']:
                    longest=dict(task_id=task['task_id'],split=split,target=item['target'],tokens_including_eos=size)
        by_split[split]=dict(target_strings=len(local),max_tokens=max(local))
    assert max(sizes)<=plan['eval_fixed']['max_new_tokens']
    result=dict(scope='Compact executed oracle next-action JSON spellings plus EOS only; not every possible legal whitespace/key spelling.',
        post_hoc=True,model_generation_performed=False,training_or_checkpoint_selection_performed=False,
        plan_sha256=sha256(a.root/'plan.json'),source_sha256=sha256(Path(__file__)),
        dataset_sha256=plan['dataset_sha256'],model_revision=plan['model_revision'],
        target_strings=len(sizes),max_tokens_including_eos=max(sizes),
        configured_max_new_tokens=plan['eval_fixed']['max_new_tokens'],
        all_checked_spellings_fit=True,by_split=by_split,
        length_histogram=dict(sorted(Counter(sizes).items())),longest=longest)
    write(a.root/'action-budget-audit.json',result)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()

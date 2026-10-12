"""Post-hoc check that the recorded token budget fits each oracle action."""
import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from .baseline import sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-info', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=Path('results/stress-v1'))
    args = parser.parse_args()
    info = json.loads(args.model_info.read_text(encoding='utf-8'))
    snapshot = info.get('snapshot_path', info.get('snapshot'))
    if snapshot is None:
        raise ValueError('Missing snapshot path in model-location metadata')
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    plan = json.loads((args.root / 'plan.json').read_text(encoding='utf-8'))
    records = []
    for split in ('dev', 'test'):
        path = Path(plan['data_root']) / (split + '.jsonl')
        assert sha256(path) == plan['dataset_sha256'][split]
        for line in path.read_text(encoding='utf-8').splitlines():
            task = json.loads(line)
            actions = [dict(action='lookup', **call) for call in task['oracle_calls']]
            actions.append(dict(action='answer', value=task['expected']))
            for action in actions:
                # Both representations are illustrative, not a universal bound
                # over every legal spelling or tool argument.
                for compact in (False, True):
                    text = json.dumps(action, separators=(',', ':') if compact else None)
                    count = len(tokenizer.encode(text, add_special_tokens=False)) + 1
                    records.append(dict(split=split, task_id=task['task_id'],
                                        action=text, tokens_with_one_eos=count))
    longest = max(records, key=lambda r: r['tokens_with_one_eos'])
    result = dict(post_hoc=True, weights_updated=False,
                  purpose='Diagnostic only; evaluation output limit was not changed',
                  plan_sha256=sha256(args.root / 'plan.json'),
                  audit_source_sha256=sha256(Path(__file__)),
                  model_revision=plan['model_revision'],
                  action_spellings_checked=len(records),
                  max_oracle_action_tokens_with_one_eos=longest['tokens_with_one_eos'],
                  max_new_tokens=plan['eval_fixed']['max_new_tokens'],
                  all_checked_actions_fit=all(r['tokens_with_one_eos'] <= plan['eval_fixed']['max_new_tokens'] for r in records),
                  longest_checked_action=longest,
                  limitation='Oracle action spellings only; not a claim that all legal outputs fit')
    (args.root / 'action-budget-audit.json').write_text(
        json.dumps(result, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

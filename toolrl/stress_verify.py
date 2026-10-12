"""Replay actual tools and rebuild exact token/message provenance for stress runs."""
import argparse
import json
from pathlib import Path

from .baseline import sha256
from .environment import parse_action,verify
from .stress_environment import StressEnvironment


def verify_rows(rows,tasks,config,tokenizer,prompt):
    count=0
    assert len(rows)==len(tasks) and {r['task_id'] for r in rows}==tasks.keys()
    for row in rows:
        task=tasks[row['task_id']]
        assert all(row[k]==task[k] for k in ('kind','split','template_id','fact_group','sampling_index','expected','minimum_calls'))
        assert row['seed']==config['seed']+100*task['sampling_index']
        prefix=[{'role':'system','content':prompt},{'role':'user','content':task['prompt']}]
        env=StressEnvironment(task,config['max_calls'])
        actions=[]
        observations=[]
        answer=None
        reason='max_turns'
        strict=[]
        assert len(row['requests'])<=config['max_turns']
        for turn,request in enumerate(row['requests']):
            text=tokenizer.apply_chat_template(prefix,tokenize=False,add_generation_prompt=True,enable_thinking=False)
            ids=tokenizer(text,add_special_tokens=False)['input_ids']
            assert ids==request['input_ids']
            if request.get('context_limit'):
                assert len(ids)>config['max_context'] and not request['output_ids']
                reason='context_limit'
                break
            assert request['seed']==row['seed']+turn and len(ids)<=config['max_context']
            assert tokenizer.decode(request['output_ids'],skip_special_tokens=True)==request['text']
            assert len(request['output_ids'])<=config['max_new_tokens']
            count+=1
            prefix.append({'role':'assistant','content':request['text']})
            if request['truncated']:
                reason='generation_truncated'
                break
            try:
                parse_action(request['text'])
                valid=True
            except ValueError:
                valid=False
            assert request['strict_action_valid']==valid
            strict.append(valid)
            try:
                action=parse_action(request['text'],allow_fence=True)
            except ValueError:
                reason='invalid_action'
                break
            assert request['used_fence_normalization']==(not valid)
            actions.append(action)
            if action['action']=='answer':
                answer=action['value']
                reason='answered'
                break
            try:
                observation=env.lookup(action['entity'],action['field'])
            except ValueError:
                reason='tool_budget_exhausted'
                break
            observations.append(observation)
            prefix.append({'role':'tool','content':json.dumps(observation,sort_keys=True)})
        success=answer is not None and verify(answer,task['expected'])
        goal=task['oracle_calls'][-1] if task['oracle_calls'] else None
        grounded=success and (task['kind']=='given_answer' or any(
            o.get('entity')==goal['entity'] and o.get('field')==goal['field'] and o.get('value')==task['expected']
            for o in observations))
        assert row['grounded_success']==grounded
        assert row['answer']==answer and row['success']==success and row['stop_reason']==reason
        assert row['actions']==actions and row['observations']==observations and row['messages']==prefix
        assert row['calls']==len(env.calls) and row['call_events']==env.events
        assert row['strict_valid_requests']==sum(strict)
        assert row['strict_success']==(success and all(strict) and len(strict)==len(row['requests']))
        assert row['correct_extra_calls']==(len(env.calls)-task['minimum_calls'] if success else None)
        assert row['reward_correctness']==float(success)
        assert row['reward_cost_gated']==float(success)*(1-.03*len(env.calls))
        for key,event_key in [('recoverable_errors','recoverable_error'),('necessary_retries','necessary_retry'),
                             ('followed_redirects','followed_redirect'),('redundant_known_lookups','redundant_known_lookup'),
                             ('unnecessary_repeats','unnecessary_repeat'),('invalid_queries','invalid_query')]:
            assert row[key]==sum(e[event_key] for e in env.events)
    return count


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('run',type=Path)
    parser.add_argument('--plan',type=Path,default=Path('results/stress-v1/plan.json'))
    parser.add_argument('--model-info',type=Path,required=True)
    args=parser.parse_args()
    from .stress_experiment import validate
    plan=validate(args.plan)
    manifest=json.loads((args.run/'manifest.json').read_text(encoding='utf-8'))
    assert manifest['status']=='complete' and manifest['trainable_parameters']==0 and not manifest['new_training_performed']
    config=manifest['config']
    data=Path(plan['data_root'])/(args.run.name.split('-')[0]+'.jsonl')
    assert config['dataset_sha256']==sha256(data)
    assert config['plan_sha256']==sha256(args.plan) and config['source_sha256']==plan['source_sha256']
    tasks={r['task_id']:r for r in map(json.loads,data.read_text(encoding='utf-8').splitlines())}
    rows=[json.loads(x) for x in (args.run/'episodes.jsonl').read_text(encoding='utf-8').splitlines()]
    info=json.loads(args.model_info.read_text(encoding='utf-8'))
    assert info['revision']==config['model_revision'] and info['model_id']==config['model_id']
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(info['snapshot'],local_files_only=True)
    prompt=Path(plan['prompt_files'][config['prompt']]).read_text(encoding='utf-8')
    count=verify_rows(rows,tasks,config,tokenizer,prompt)
    result=dict(tasks_verified=len(rows),requests_verified=count,tool_replay_matches=True,
                answer_and_cost_regrading_matches=True,input_token_provenance_matches=True,output_token_decode_matches=True,
                recovery_and_redundancy_accounting_matches=True,episodes_sha256=sha256(args.run/'episodes.jsonl'))
    (args.run/'verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()

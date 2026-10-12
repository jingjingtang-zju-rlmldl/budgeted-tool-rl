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
        bootstrap=[]
        for a in task.get('bootstrap_calls',[]):
            o=env.lookup(**a);bootstrap.append(o)
            prefix += [dict(role='assistant',content=json.dumps(dict(action='lookup',**a),separators=(',',':'))),
                       dict(role='tool',content=json.dumps(o,sort_keys=True))]
        assert row['bootstrap_observations']==bootstrap and row['bootstrap_calls']==task.get('bootstrap_calls',[])
        actions=[]
        observations=[]
        answer=None
        reason='max_turns'
        strict=[]
        assert len(row['requests'])<=config['max_turns']-len(task.get('bootstrap_calls',[]))
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


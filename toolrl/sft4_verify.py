"""Replay actual tools and rebuild exact token/message provenance for branch/cost runs."""
import argparse
import json
from pathlib import Path

from .baseline import sha256
from .environment import parse_action,verify
from .sft4_environment import BranchEnvironment,grounded as is_grounded


def verify_rows(rows,tasks,config,tokenizer,prompt):
    count=0
    assert len(rows)==len(tasks) and {r['task_id'] for r in rows}==tasks.keys()
    for row in rows:
        task=tasks[row['task_id']]
        assert all(row[k]==task[k] for k in ('kind','split','template_id','fact_group','sampling_index','expected','minimum_calls','minimum_cost'))
        assert row['seed']==config['seed']+100*task['sampling_index']
        prefix=[{'role':'system','content':prompt},{'role':'user','content':task['prompt']}]
        env=BranchEnvironment(task,config['max_calls'])
        bootstrap=[]
        for a in task.get('bootstrap_calls',[]):
            o=env.lookup(**a);bootstrap.append(o)
            prefix += [dict(role='assistant',content=task['bootstrap_texts'][len(bootstrap)-1] if task.get('bootstrap_texts') else json.dumps(dict(action='lookup',**a),separators=(',',':'))),
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
        grounded=is_grounded(task,env,answer)
        assert row['grounded_success']==grounded
        assert row['answer']==answer and row['success']==success and row['stop_reason']==reason
        assert row['actions']==actions and row['observations']==observations and row['messages']==prefix
        assert row['calls']==len(env.calls) and row['call_events']==env.events
        assert row['strict_valid_requests']==sum(strict)
        assert row['strict_success']==(success and all(strict) and len(strict)==len(row['requests']))
        assert row['call_delta_from_cost_oracle']==(len(env.calls)-task['minimum_calls'] if success else None)
        assert row['reward_correctness']==float(success)
        assert row['total_cost']==env.total_cost
        assert row['correct_extra_cost']==(env.total_cost-task['minimum_cost'] if grounded else None)
        assert row['reward_call_cost_gated']==float(success)*(1-.03*len(env.calls))
        assert row['reward_cost_gated']==float(grounded)*(1-.02*env.total_cost)
        for key,event_key in [('recoverable_errors','recoverable_error'),('necessary_retries','necessary_retry'),
                             ('followed_redirects','followed_redirect'),('redundant_known_lookups','redundant_known_lookup'),
                             ('unnecessary_repeats','unnecessary_repeat'),('invalid_queries','invalid_query')]:
            assert row[key]==sum(e[event_key] for e in env.events)
    return count


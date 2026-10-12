"""Real frozen CUDA model evaluation in the extended deterministic environment."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import platform
import time

from .baseline import sha256,gpu_status
from .environment import parse_action,verify
from .sft4_environment import BranchEnvironment,MAX_CALLS,MAX_TURNS,grounded as is_grounded


def run_episode(task,generate,prompt,seed):
    env=BranchEnvironment(task)
    messages=[{'role':'system','content':prompt},{'role':'user','content':task['prompt']}]
    bootstrap=[]
    for a in task.get('bootstrap_calls',[]):
        text=task['bootstrap_texts'][len(bootstrap)] if task.get('bootstrap_texts') else json.dumps(dict(action='lookup',**a),separators=(',',':'))
        o=env.lookup(**a);bootstrap.append(o)
        messages += [dict(role='assistant',content=text),dict(role='tool',content=json.dumps(o,sort_keys=True))]
    requests=[]
    actions=[]
    observations=[]
    answer=None
    success=False
    reason='max_turns'
    started=time.perf_counter()
    for turn in range(MAX_TURNS-len(task.get('bootstrap_calls',[]))):
        request=generate(messages,seed+turn)
        requests.append(request)
        if request.get('context_limit'):
            reason='context_limit'
            break
        messages.append({'role':'assistant','content':request['text']})
        if request['truncated']:
            reason='generation_truncated'
            break
        try:
            parse_action(request['text'])
            request['strict_action_valid']=True
        except ValueError:
            request['strict_action_valid']=False
        try:
            action=parse_action(request['text'],allow_fence=True)
            request['used_fence_normalization']=not request['strict_action_valid']
        except ValueError as exc:
            request['parse_error']=str(exc)
            reason='invalid_action'
            break
        actions.append(action)
        if action['action']=='answer':
            answer=action['value']
            success=verify(answer,task['expected'])
            reason='answered'
            break
        try:
            observation=env.lookup(action['entity'],action['field'])
        except ValueError:
            reason='tool_budget_exhausted'
            break
        observations.append(observation)
        messages.append({'role':'tool','content':json.dumps(observation,sort_keys=True)})
    events=env.events
    grounded=is_grounded(task,env,answer)
    return dict(bootstrap_observations=bootstrap,bootstrap_calls=task.get('bootstrap_calls',[]),task_id=task['task_id'],kind=task['kind'],split=task['split'],template_id=task['template_id'],
                fact_group=task['fact_group'],sampling_index=task['sampling_index'],seed=seed,
                success=success,grounded_success=grounded,
                strict_success=success and all(q.get('strict_action_valid',False) for q in requests),
                answer=answer,expected=task['expected'],stop_reason=reason,calls=len(env.calls),
                minimum_calls=task['minimum_calls'],minimum_cost=task['minimum_cost'],total_cost=env.total_cost,
                correct_extra_cost=env.total_cost-task['minimum_cost'] if grounded else None,call_delta_from_cost_oracle=len(env.calls)-task['minimum_calls'] if success else None,
                strict_valid_requests=sum(q.get('strict_action_valid',False) for q in requests),
                fence_normalized_requests=sum(q.get('used_fence_normalization',False) for q in requests),
                requests=requests,actions=actions,observations=observations,messages=messages,call_events=events,
                recoverable_errors=sum(e['recoverable_error'] for e in events),
                necessary_retries=sum(e['necessary_retry'] for e in events),
                followed_redirects=sum(e['followed_redirect'] for e in events),
                redundant_known_lookups=sum(e['redundant_known_lookup'] for e in events),
                unnecessary_repeats=sum(e['unnecessary_repeat'] for e in events),
                invalid_queries=sum(e['invalid_query'] for e in events),
                generated_tokens=sum(len(q.get('output_ids',[])) for q in requests),
                generation_seconds=sum(q.get('generation_seconds',0) for q in requests),
                wall_seconds=time.perf_counter()-started,
                reward_correctness=float(success),reward_call_cost_gated=float(success)*(1-.03*len(env.calls)),
                reward_cost_gated=float(grounded)*(1-.02*env.total_cost))


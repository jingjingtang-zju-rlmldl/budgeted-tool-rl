"""Fresh split-safe supervision, evaluation and sealed RL signal tasks."""
import copy
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from .stress_environment import StressEnvironment, oracle
from .stress_data import DEV_TEMPLATES, TEST_TEMPLATES, sets, prior_sets
from .sft_data import encode_example

KINDS=['anchor_two','chain_three','chain_four','retry_two','redirect_two',
       'skip_owner','skip_prefix','given_answer','one_lookup']
TRAIN_COUNTS=dict(anchor_two=20,chain_three=50,chain_four=50,retry_two=40,
                  redirect_two=40,skip_owner=20,skip_prefix=20,given_answer=40,one_lookup=20)
DEV={
 'anchor_two':['Determine which department employs the registered owner of {p}.'],
 'chain_three':['Determine the cost center used by the registered owner of {p}, through their department.'],
 'chain_four':['Determine the spending limit attached to the department cost center of the registered owner of {p}.'],
 'skip_owner':['Ownership has been established: {e} owns {p}. Which department employs {e}?'],
 'skip_prefix':['The department for the owner of {p} is {d}, with cost center {c}. Give the limit assigned to {c}.'],
 'given_answer':['Budget entry for {p}: {b}; ownership entry: {e}. Give the budget amount.'],
 'one_lookup':['Determine the budget entry stored against {p}.']}
TEST={
 'anchor_two':['In which department is the registered owner of {p} employed?',
               'Resolve the ownership record for {p}, then give the department of that employee.'],
 'chain_three':['Name the cost center used by the department employing the registered owner of {p}.',
                'For {p}, retrieve its owner, their department, and then that department\'s cost_center value.'],
 'chain_four':['Give the monetary spending limit for the cost center used by the department employing the registered owner of {p}.',
               'Starting with {p}, follow owner, department, and cost_center to retrieve spending_limit.'],
 'skip_owner':['Ownership record already supplied: {e} is responsible for {p}. In which department is this employee employed?',
               'Take {e} as the known owner of {p}; find their department without resolving ownership again.'],
 'skip_prefix':['Already resolved for {p}: owner department = {d}, department cost center = {c}. What limit applies to {c}?',
                'No ownership resolution is needed: {p}\'s owner works in {d}, which uses {c}. Obtain the spending limit for {c}.'],
 'given_answer':['Budget amount already available for {p}: {b}. The owner is {e}. Give only that budget amount.',
                 'Stored information is supplied here: project {p}, owner {e}, budget {b}. What is the budget value?'],
 'one_lookup':['Give the monetary budget entry held in the database for {p}.',
               'Read the private budget field belonging to {p} and return its value.']}
SIGNAL={
 'chain_three':['Trace the ownership chain of {p} to its departmental cost center and identify that center.',
                'Identify the cost center that services the department of the employee responsible for {p}.'],
 'chain_four':['Trace the ownership chain of {p} to the department\'s cost center and obtain its spending limit.',
               'Identify the spending limit of the cost center serving the department of the employee responsible for {p}.'],
 'skip_prefix':['Context for {p}: owner department {d}, cost center {c}. Obtain the monetary spending limit at {c}.',
                'Use the resolved chain {p} -> owner -> {d} -> {c}. What spending limit belongs to {c}?']}


def templates(split,kind):
    key='anchor_two' if kind in ('retry_two','redirect_two') else kind
    if split=='train':
        if key=='one_lookup':
            return ['What budget is saved for {p}?','Retrieve the project budget for {p}.',
                    'Look up the budget amount stored under {p}.']
        return [DEV_TEMPLATES[key]]+TEST_TEMPLATES[key]
    return (DEV if split=='dev' else TEST)[key]


def build(prior_entities=(),prior_values=(),seed=2026101103):
    rng=random.Random(seed)
    used_e=set(prior_entities)
    used_v=set(prior_values)|{'321','875','24761','department_74123','department_48216','9101'}
    suffixes={e.rsplit('_',1)[-1] for e in used_e}
    def ident(role):
        while True:
            tail=f'{rng.getrandbits(48):012x}'
            e=role+'_'+tail
            if tail not in suffixes and e not in used_v:
                suffixes.add(tail);used_e.add(e)
                return e
    def number(lo,hi):
        while True:
            v=rng.randint(lo,hi)
            if str(v) not in used_v:
                used_v.add(str(v));return v
    def department():
        while True:
            e='department_'+str(rng.randint(10000,99999))
            if e not in used_e and e not in used_v:
                used_e.add(e);used_v.add(e);return e
    def graph():
        db={}
        main=None
        for _ in range(3):
            p,e,d,c=ident('project'),ident('employee'),department(),ident('cost_center')
            b,limit=number(101,9999),number(10001,99999)
            db.update({p:dict(owner=e,budget=b),e:dict(department=d),d:dict(cost_center=c),c:dict(spending_limit=limit)})
            if main is None:main=dict(p=p,e=e,d=d,c=c,b=b,limit=limit)
        return main|dict(database=db)
    def task(f,split,kind,index,signal=False):
        f=copy.deepcopy(f)
        p,e,d,c=f['p'],f['e'],f['d'],f['c']
        route=[dict(entity=p,field='owner'),dict(entity=e,field='department'),
               dict(entity=d,field='cost_center'),dict(entity=c,field='spending_limit')]
        aliases={};faults=[];given=[];provided=None
        if kind in ('anchor_two','retry_two','redirect_two'):
            calls=route[:2];expected=d
        elif kind in ('chain_three','retry_three','redirect_three'):
            calls=route[:3];expected=c
        elif kind in ('chain_four','retry_four','redirect_four'):
            calls=route;expected=f['limit']
        elif kind=='skip_owner':
            calls=[route[1]];expected=d;given=[dict(entity=p,field='owner',value=e)]
        elif kind in ('skip_prefix','prefix_retry','prefix_redirect'):
            calls=[route[3]];expected=f['limit']
            given=[dict(entity=e,field='department',value=d),dict(entity=d,field='cost_center',value=c)]
        elif kind=='one_lookup':
            calls=[dict(entity=p,field='budget')];expected=f['b']
        else:
            calls=[];expected=f['b'];provided=expected
            given=[dict(entity=p,field='budget',value=expected),dict(entity=p,field='owner',value=e)]
        if kind.startswith('retry_') or kind=='prefix_retry':
            fault_index=0 if kind=='prefix_retry' else 1
            call=calls[fault_index]
            faults=[[call['entity'],call['field']]]
            calls=calls[:fault_index+1]+[call]+calls[fault_index+1:]
        if kind.startswith('redirect_') or kind=='prefix_redirect':
            at=0 if kind=='prefix_redirect' else 1
            call=calls[at]
            role='cost_center' if kind=='prefix_redirect' else 'employee'
            alias=ident(role);aliases[alias]=call['entity']
            if kind=='prefix_redirect':
                given[-1]['value']=alias
                f['c']=alias
            else:
                f['database'][p]['owner']=alias
            calls=calls[:at]+[dict(entity=alias,field=call['field']),call]+calls[at+1:]
        if signal:
            key='skip_prefix' if kind.startswith('prefix_') else 'chain_three' if kind.endswith('three') else 'chain_four'
            wording=SIGNAL[key]
        else:
            wording=templates(split,kind)
        tid=index%len(wording)
        question=wording[tid].format(**f)
        r=dict(task_id=f'sft2_{split}_{kind}_{index:04d}',split=split,kind=kind,
               template_id=tid,sampling_index=index,fact_group=p,prompt=question,
               database=f['database'],aliases=aliases,transient_once=faults,
               provided_facts=given,provided_value=provided,expected=expected,
               oracle_calls=calls,minimum_calls=len(calls))
        oracle(r)
        return r
    result={}
    for split,counts in [('train',TRAIN_COUNTS),('dev',{k:3 for k in KINDS}),('test',{k:8 for k in KINDS})]:
        rows=[];paired={}
        for kind,n in counts.items():
            for i in range(n):
                # Forty within-train goal contrasts; never share between splits.
                pairkey=('old',i) if kind in ('anchor_two','given_answer') and i<20 else ('long',i) if kind in ('chain_three','chain_four') and i<20 else None
                if split=='train' and pairkey in paired:
                    f=paired[pairkey]
                else:
                    f=graph()
                    if split=='train' and pairkey:paired[pairkey]=f
                r=task(f,split,kind,i)
                r['sampling_index']=len(rows)
                rows.append(r)
        rng.shuffle(rows);result[split]=rows
    # Eight families, four independent tasks each. Composition is fixed
    # before training, not adapted after observing test scores.
    kinds=['chain_three','chain_four','retry_three','retry_four',
           'redirect_three','redirect_four','prefix_retry','prefix_redirect']
    result['signal']=[]
    for kind in kinds:
        for i in range(4):
            r=task(graph(),'signal',kind,i,True)
            r['sampling_index']=len(result['signal'])
            result['signal'].append(r)
    rng.shuffle(result['signal'])
    audit(result,set(prior_entities),set(prior_values))
    return result


def audit(splits,old_e,old_v):
    for a,rs in splits.items():
        e,v=sets(rs)
        assert not e&old_e and not v&old_v
        assert len({r['task_id'] for r in rs})==len(rs)
        for b,others in splits.items():
            if a!=b:
                oe,ov=sets(others)
                assert not e&oe and not v&ov
        for r in rs:
            oracle(r)
            for entity,fields in r['database'].items():
                for key in ('owner','cost_center'):
                    if key in fields:assert entity.rsplit('_',1)[-1]!=fields[key].rsplit('_',1)[-1]
    return dict(split_entities_disjoint=True,split_private_values_disjoint=True,
                historical_disjoint=True,counts={s:dict(Counter(r['kind'] for r in rs)) for s,rs in splits.items()},
                fact_groups={s:len({r['fact_group'] for r in rs}) for s,rs in splits.items()},
                train_goal_contrast_pairs=40,signal_used_for_training=False,
                template_sets={'train':'previous dev/test phrasings on new facts','dev':DEV,'test':TEST,'signal':SIGNAL})


def request_examples(task,prompt):
    env=StressEnvironment(task)
    messages=[dict(role='system',content=prompt),dict(role='user',content=task['prompt'])]
    examples=[]
    for turn,call in enumerate(task['oracle_calls']+[None]):
        action=dict(action='lookup',**call) if call else dict(action='answer',value=task['expected'])
        target=json.dumps(action,separators=(',',':'))
        examples.append(dict(task_id=task['task_id'],kind=task['kind'],turn=turn,
                             messages=copy.deepcopy(messages),target=target))
        if call:
            messages+=[dict(role='assistant',content=target),
                       dict(role='tool',content=json.dumps(env.lookup(**call),sort_keys=True))]
    return examples


def write_splits(root):
    if root.exists():raise ValueError('New data directory required')
    historical=sorted(Path('data').rglob('*.jsonl'))
    old_e,old_v=prior_sets(historical)
    for path in historical:
        for r in map(json.loads,path.read_text(encoding='utf-8').splitlines()):
            old_e.update(r.get('aliases',{}))
    splits=build(old_e,old_v)
    root.mkdir(parents=True)
    hashes={}
    for split,rs in splits.items():
        path=root/(split+'.jsonl')
        path.write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rs),encoding='utf-8',newline='\n')
        hashes[split]=hashlib.sha256(path.read_bytes()).hexdigest()
    result=audit(splits,old_e,old_v)|dict(seed=2026101103,sha256=hashes,
        historical_files_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in historical})
    (root/'audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    return result

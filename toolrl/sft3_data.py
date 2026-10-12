"""Same-fact wording controls and executed invalid-query correction states."""
import copy
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from .sft2_data import SIGNAL, TEST as V2_TEST
from .stress_data import prior_sets, sets
from .stress_environment import StressEnvironment, oracle
from .sft_data import encode_example

REGULAR_COUNTS=dict(anchor_two=20,chain_three=40,chain_four=40,retry_two=10,
                    redirect_two=10,skip_owner=5,skip_prefix=5,given_answer=5,one_lookup=5)
TRAIN_WORDING={
 'anchor_two':['Which department employs the employee responsible for {p}?','Find the department of the person managing {p}.'],
 'chain_three':SIGNAL['chain_three']+['Which departmental cost center serves the person managing {p}?'],
 'chain_four':SIGNAL['chain_four']+['What is the cost-center spending limit for the team of the person managing {p}?'],
 'skip_owner':['The responsible employee for {p} is {e}. Find their department.'],
 'skip_prefix':['For {p}, the responsible employee works in {d}, which uses {c}. Retrieve the limit for {c}.'],
 'given_answer':['For {p}, the responsible employee is {e} and the budget is {b}. Return the budget.'],
 'one_lookup':['Retrieve the budget of {p}.']}
TEST_WORDING={
 'anchor_two':['Report the organizational department of the staff member accountable for {p}.','Whoever is in charge of {p} works in which department?'],
 'chain_three':['Report the cost center of the organizational department of the staff member accountable for {p}.','Which cost center belongs to the department of whoever is in charge of {p}?'],
 'chain_four':['Report the spending limit of the cost center of the organizational department of the staff member accountable for {p}.','What spending ceiling applies to the cost center of the department of whoever is in charge of {p}?'],
 'skip_owner':['The staff member accountable for {p} is {e}. Report the organizational department of this staff member.','For {p}, the staff member in charge is already identified as {e}. Retrieve the department of {e}.'],
 'skip_prefix':['Resolved records for {p}: accountable staff member department {d}; its cost center {c}. Report the spending limit for {c}.','The department of whoever is in charge of {p} is {d}, assigned to {c}. Read the spending ceiling for {c}.'],
 'given_answer':['Recorded budget for {p} is {b}; staff member accountable is {e}. Report the recorded budget.','The budget is supplied: {p} has {b}. The staff member in charge is {e}. Return the budget.'],
 'one_lookup':['Report the financial budget recorded for {p}.','Read the budget allocation assigned to project {p}.']}
FRESH_SIGNAL={
 'chain_three':['Which cost center supports the department of the staff member overseeing {p}?','Find the departmental cost-center record for the individual entrusted with {p}.'],
 'chain_four':['What spending limit is associated with the cost center supporting the department of the staff member overseeing {p}?','Find the spending limit attached to the departmental cost center of the individual entrusted with {p}.'],
 'skip_prefix':['For {p}, the overseeing staff member is in {d}, linked to {c}. Read the limit of {c}.','Resolved for {p}: entrusted individual department {d}; department cost center {c}. Report its spending limit.']}
SIGNAL_KINDS=['chain_three','chain_four','retry_three','retry_four','redirect_three','redirect_four','prefix_retry','prefix_redirect']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def create(prior_e=(),prior_v=(),seed=2026101104):
    rng=random.Random(seed);used_e=set(prior_e);used_v=set(prior_v)|{'321','875','24761','9101'}
    suffix={e.rsplit('_',1)[-1] for e in used_e}
    def ident(role):
        while True:
            tail=f'{rng.getrandbits(48):012x}';e=role+'_'+tail
            if tail not in suffix and e not in used_e and e not in used_v:
                suffix.add(tail);used_e.add(e);return e
    def num(lo,hi):
        while True:
            v=rng.randint(lo,hi)
            if str(v) not in used_v:used_v.add(str(v));return v
    def dept():
        while True:
            d='department_'+str(rng.randint(10000,99999))
            if d not in used_e and d not in used_v:used_e.add(d);used_v.add(d);return d
    def graph():
        db={};f=None
        for _ in range(3):
            p,e,d,c=ident('project'),ident('employee'),dept(),ident('cost_center');b,l=num(101,9999),num(10001,99999)
            db.update({p:dict(owner=e,budget=b),e:dict(department=d),d:dict(cost_center=c),c:dict(spending_limit=l)})
            if f is None:f=dict(p=p,e=e,d=d,c=c,b=b,limit=l)
        return f|dict(database=db)
    def task(f,split,kind,i,wording):
        f=copy.deepcopy(f);p,e,d,c=f['p'],f['e'],f['d'],f['c']
        route=[dict(entity=p,field='owner'),dict(entity=e,field='department'),dict(entity=d,field='cost_center'),dict(entity=c,field='spending_limit')]
        aliases={};faults=[];given=[];provided=None
        if kind in ('anchor_two','retry_two','redirect_two'):calls=route[:2];expected=d;key='anchor_two'
        elif kind.endswith('three'):calls=route[:3];expected=c;key='chain_three'
        elif kind.endswith('four'):calls=route;expected=f['limit'];key='chain_four'
        elif kind=='skip_owner':calls=route[1:2];expected=d;key=kind;given=[dict(entity=p,field='owner',value=e)]
        elif kind in ('skip_prefix','prefix_retry','prefix_redirect'):
            calls=route[3:];expected=f['limit'];key='skip_prefix';given=[dict(entity=e,field='department',value=d),dict(entity=d,field='cost_center',value=c)]
        elif kind=='one_lookup':calls=[dict(entity=p,field='budget')];expected=f['b'];key=kind
        elif kind=='given_answer':calls=[];expected=f['b'];provided=expected;key=kind;given=[dict(entity=p,field='budget',value=expected),dict(entity=p,field='owner',value=e)]
        else:raise ValueError(kind)
        if 'retry' in kind:
            at=0 if kind.startswith('prefix') else 1;call=calls[at];faults=[[call['entity'],call['field']]]
            calls=calls[:at+1]+[call]+calls[at+1:]
        if 'redirect' in kind:
            at=0 if kind.startswith('prefix') else 1;call=calls[at];alias=ident('cost_center' if at==0 else 'employee');aliases[alias]=call['entity']
            if at==0:given[-1]['value']=alias;f['c']=alias
            else:f['database'][p]['owner']=alias
            calls=calls[:at]+[dict(entity=alias,field=call['field']),call]+calls[at+1:]
        tid=i%len(wording[key]);q=wording[key][tid].format(**f)
        r=dict(task_id=f'sft3_{split}_{kind}_{i:04d}',split=split,kind=kind,template_id=tid,sampling_index=i,fact_group=p,
               prompt=q,database=f['database'],aliases=aliases,transient_once=faults,provided_facts=given,provided_value=provided,
               expected=expected,oracle_calls=calls,minimum_calls=len(calls),bootstrap_calls=[])
        oracle(r);return r
    def correction(split,i,wording):
        t=task(graph(),split,'chain_four',i,wording)
        typ='unknown_field' if i%2==0 else 'unknown_entity';stage=(i//2)%4;mode=(i//8)%2
        good=t['oracle_calls'][stage];bad=copy.deepcopy(good)
        if typ=='unknown_field':bad['field']=['employee','owner','department','cost_center'][stage] if mode==0 else ['department','cost_center','owner','budget'][stage]
        else:bad['entity']=['project','employee','department','cost_center'][stage] if mode==0 else good['entity']+'_typo'
        t.update(task_id=f'sft3_{split}_{typ}_{i:04d}',correction_type=typ,correction_stage=stage,
                 correction_mode='role_literal' if typ=='unknown_entity' and mode==0 else 'typo' if typ=='unknown_entity' else 'wrong_field',
                 bootstrap_calls=t['oracle_calls'][:stage]+[bad],remaining_oracle_calls=t['oracle_calls'][stage:])
        env=StressEnvironment(t)
        obs=[env.lookup(**a) for a in t['bootstrap_calls']]
        assert all('error' not in o for o in obs[:-1]) and obs[-1]['error']==typ
        for a in t['remaining_oracle_calls']:env.lookup(**a)
        assert len(env.calls)==5 and env.events[-1]['observation']['value']==t['expected']
        return t
    result={}
    result['train']=[correction('train',i,TRAIN_WORDING) for i in range(160)]
    for k,n in REGULAR_COUNTS.items():
        for i in range(n):result['train'].append(task(graph(),'train',k,i,TRAIN_WORDING))
    result['dev']=[task(graph(),'dev',k,i,V2_TEST) for k in REGULAR_COUNTS for i in range(2)]
    result['dev'] += [correction('dev',i,V2_TEST) for i in range(8)]
    result['test']=[task(graph(),'test',k,i,TEST_WORDING) for k in REGULAR_COUNTS for i in range(8)]
    result['recovery']=[correction('recovery',i,TEST_WORDING) for i in range(32)]
    result['pair']=[]
    for kind in SIGNAL_KINDS:
        for i in range(4):
            t=task(graph(),'pair',kind,i,SIGNAL)
            p=t['fact_group'];e=t['database'][p]['owner'];e=t['aliases'].get(e,e);d=t['database'][e]['department'];c=t['database'][d]['cost_center']
            if kind=='prefix_redirect':c=next(iter(t['aliases']))
            key='skip_prefix' if kind.startswith('prefix') else 'chain_three' if kind.endswith('three') else 'chain_four'
            for tid in (0,1):
                v=copy.deepcopy(t);v.update(task_id=t['task_id']+f'_word{tid}',pair_id=t['task_id'],template_id=tid,prompt=SIGNAL[key][tid].format(p=p,e=e,d=d,c=c))
                result['pair'].append(v)
    result['signal']=[task(graph(),'signal',k,i,FRESH_SIGNAL) for k in SIGNAL_KINDS for i in range(4)]
    for s,rs in result.items():
        # A pair shares its rollout root and every non-wording experimental input.
        for j,t in enumerate(rs):t['sampling_index']=j//2 if s=='pair' else j
        if s!='pair':rng.shuffle(rs)
    audit(result,set(prior_e),set(prior_v));return result

def audit(splits,old_e,old_v):
    for a,rs in splits.items():
        e,v=sets(rs);e.update(t['entity'] for r in rs for t in r['bootstrap_calls'])
        assert not e&old_e and not v&old_v
        assert len({r['task_id'] for r in rs})==len(rs)
        for b,os in splits.items():
            if a!=b:
                oe,ov=sets(os);oe.update(t['entity'] for r in os for t in r['bootstrap_calls'] if '_' in t['entity'])
                # Literal entity-type words are errors, not private identifiers.
                assert not {x for x in e&oe if x not in ('project','employee','department','cost_center')} and not v&ov
        for r in rs:oracle(r)
    for j in range(0,len(splits['pair']),2):
        a,b=splits['pair'][j:j+2]
        excluded={'task_id','prompt','template_id'}
        assert {k:v for k,v in a.items() if k not in excluded}=={k:v for k,v in b.items() if k not in excluded}
        assert a['prompt']!=b['prompt']
    errors=Counter(r.get('correction_type','none') for r in splits['train'])
    assert errors==dict(unknown_field=80,unknown_entity=80,none=140)
    return dict(counts={s:len(rs) for s,rs in splits.items()},train_error_prefix_counts=dict(errors),
                historical_and_cross_split_private_facts_disjoint=True,pair_only_changes_wording=True,
                training_error_prefixes='scripted counterfactual assistant queries, actually executed; not student-generated rollouts',
                seed=2026101104,signal_used_for_training=False)

def request_examples(task,prompt):
    env=StressEnvironment(task);messages=[dict(role='system',content=prompt),dict(role='user',content=task['prompt'])]
    for a in task['bootstrap_calls']:
        messages += [dict(role='assistant',content=json.dumps(dict(action='lookup',**a),separators=(',',':'))),
                     dict(role='tool',content=json.dumps(env.lookup(**a),sort_keys=True))]
    examples=[]
    for turn,a in enumerate(task.get('remaining_oracle_calls',task['oracle_calls'])+[None]):
        target=json.dumps(dict(action='lookup',**a) if a else dict(action='answer',value=task['expected']),separators=(',',':'))
        examples.append(dict(task_id=task['task_id'],kind=task['kind'],turn=turn,messages=copy.deepcopy(messages),target=target))
        if a:messages += [dict(role='assistant',content=target),dict(role='tool',content=json.dumps(env.lookup(**a),sort_keys=True))]
    return examples

def write_splits(root):
    assert not root.exists();paths=sorted(Path('data').rglob('*.jsonl'));old_e,old_v=prior_sets(paths)
    for p in paths:
        for t in map(json.loads,p.read_text(encoding='utf-8').splitlines()):old_e.update(t.get('aliases',{}))
    splits=create(old_e,old_v);root.mkdir(parents=True)
    for s,rs in splits.items():(root/(s+'.jsonl')).write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rs),encoding='utf-8',newline='\n')
    a=audit(splits,old_e,old_v)|dict(sha256={s:sha(root/(s+'.jsonl')) for s in splits},historical_files_sha256={str(p):sha(p) for p in paths},
                                   templates=dict(train=TRAIN_WORDING,test=TEST_WORDING,pair=SIGNAL,signal=FRESH_SIGNAL))
    (root/'audit.json').write_text(json.dumps(a,indent=2)+'\n',encoding='utf-8',newline='\n');return a

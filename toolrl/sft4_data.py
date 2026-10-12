"""Disjoint facts, observable oracle decisions, and student-copy error extraction."""
import copy
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from .sft_data import encode_example
from .sft4_environment import BranchEnvironment, completion, oracle, MAX_CALLS, next_action

KINDS=['policy_team','policy_partner','policy_local','service_team','service_partner','known_policy','known_prefix','given_result']
WORDING = {
    'train':['Find the effective spending limit for {p}.','What effective limit applies to {p}?'],
    'collect':['Retrieve the effective spending limit of {p}.','Determine the applicable effective limit for {p}.'],
    'reserve':['Return the effective limit attached to {p}.','Identify the effective spending ceiling for {p}.'],
    'dev':['Report the effective limit for {p}.','Resolve the applicable limit of {p}.'],
    'test':['Which numeric ceiling governs spending for {p}?','Establish the effective expenditure cap for {p}.'],
    'recovery':['Identify the operative spending cap for {p}.','Report the governing spending ceiling of {p}.'],
    'signal':['What monetary ceiling is ultimately in force for {p}?','Determine the operative expenditure allowance applicable to {p}.']}


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p): return [json.loads(x) for x in Path(p).read_text(encoding='utf-8').splitlines()]
def save_rows(p, rs): Path(p).write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rs),encoding='utf-8',newline='\n')


def private_sets(tasks):
    entities, values = set(), set()
    for t in tasks:
        db = t.get('database',{})
        entities.update(db)
        entities.update(t.get('aliases',{}))
        for fields in db.values():
            for v in fields.values():
                if isinstance(v,(int,float)): values.add(str(v))
                elif isinstance(v,str) and '_' in v: entities.add(v)
        if isinstance(t.get('expected'),(int,float)): values.add(str(t['expected']))
    return entities, values


def historical():
    paths=sorted(Path('data').rglob('*.jsonl'))
    es,vs=set(),set()
    for path in paths:
        e,v=private_sets(rows(path)); es.update(e); vs.update(v)
    return paths,es,vs


def create(old_e=(), old_v=(), seed=2026101205):
    rng=random.Random(seed); used_e=set(old_e); used_v=set(old_v)|{'47261','68312','29418','83725'}
    def ident(role,n):
        while True:
            tail=f'{rng.getrandbits(4*n):0{n}x}'; s=role+'_'+tail
            if s not in used_e: used_e.add(s); return s
    def number():
        while True:
            v=rng.randint(100001,9999999)
            if str(v) not in used_v: used_v.add(str(v)); return v
    def graph(mode,n):
        db={}; root=None
        for j in range(3):
            p,e,d,c,v=[ident(role,n) for role in ('project','employee','department','cost_center','vendor')]
            local,team,partner,budget=[number() for _ in range(4)]
            actual=mode if j==0 else rng.choice(['local','team','partner'])
            limit={'local':local,'team':team,'partner':partner}[actual]
            db.update({p:dict(policy=actual,owner=e,partner=v,local_limit=local,resolved_limit=limit,budget=budget),
                       e:dict(department=d),d:dict(cost_center=c),c:dict(spending_limit=team),v:dict(spending_limit=partner)})
            if j==0: root=dict(p=p,e=e,d=d,c=c,v=v,limit=limit)
        return root|dict(database=db)
    def task(split,kind,i):
        mode={'policy_team':'team','policy_partner':'partner','policy_local':'local','service_team':'team','service_partner':'partner',
              'known_prefix':'team'}.get(kind,['local','team','partner'][i%3])
        price=2 if kind.startswith('service_') else 6
        if kind in ('known_policy','given_result'): price=[2,6][(i//3)%2]
        n=[16,24,32][i%3]; f=graph(mode,n); p=f['p']; given=[]; supplied=[]
        if kind in ('known_policy','known_prefix'):
            given.append(dict(entity=p,field='policy',value=mode)); supplied.append('Policy '+mode+'.')
        if kind=='known_prefix':
            given.append(dict(entity=p,field='owner',value=f['e'])); supplied.append('Owner '+f['e']+'.')
            if i%3>=1:
                given.append(dict(entity=f['e'],field='department',value=f['d'])); supplied.append('Owner department '+f['d']+'.')
            if i%3==2:
                given.append(dict(entity=f['d'],field='cost_center',value=f['c'])); supplied.append('That department uses '+f['c']+'.')
        if kind=='given_result':
            given.append(dict(entity=p,field='resolved_limit',value=f['limit'])); supplied.append('The verified effective limit is '+str(f['limit'])+'.')
        template=i%2
        q=WORDING[split][template].format(p=p)+' '+' '.join(supplied)+f' The resolved_limit service price is {price}; every other query costs 1.'
        t=dict(task_id=f'sft4_{split}_{kind}_{i:04d}',split=split,kind=kind,template_id=template,
               prompt=q,database=f['database'],root_entity=p,fact_group=p,provided_facts=given,
               service_price=price,expected=f['limit'],bootstrap_calls=[],id_tail_length=n)
        calls,env,value=completion(t)
        t.update(oracle_calls=calls,minimum_calls=len(calls),minimum_cost=env.total_cost)
        oracle(t); return t
    splits={s:[task(s,k,i) for k in KINDS for i in range(n)] for s,n in
            [('train',30),('collect',16),('reserve',16),('dev',2),('test',8),('signal',4)]}
    recovery=[]
    for edit_i,edit in enumerate(['deletion','substitution','insertion','transposition']):
        for stage in range(4):
            for template in range(2):
                i=edit_i*8+stage*2+template
                t=task('recovery','policy_team',i)
                good=t['oracle_calls'][stage]; bad=dict(good)
                s=good['entity']; start=s.rfind('_')+1; at=start+(len(s)-start)//2
                if edit=='deletion': s=s[:at]+s[at+1:]
                elif edit=='substitution': s=s[:at]+('0' if s[at]!='0' else '1')+s[at+1:]
                elif edit=='insertion': s=s[:at]+'a'+s[at:]
                else:
                    at=next(j for j in range(start,len(s)-1) if s[j]!=s[j+1])
                    s=s[:at]+s[at+1]+s[at]+s[at+2:]
                bad['entity']=s
                t.update(task_id=f'sft4_recovery_{edit}_stage{stage}_word{template}',correction_type='unknown_entity',
                         correction_mode=edit,correction_stage=stage,bootstrap_calls=t['oracle_calls'][:stage]+[bad])
                calls,env,_=completion(t); t['remaining_oracle_calls']=calls
                t['minimum_remaining_cost']=env.total_cost-(stage+1)
                assert env.events[stage]['observation']['error']=='unknown_entity'
                oracle(t); recovery.append(t)
    splits['recovery']=recovery # 4 edits x 4 positions x 2 phrasings = 32, fully crossed.
    for s,rs in splits.items():
        for index,t in enumerate(rs): t['sampling_index']=index
        rng.shuffle(rs)
    audit(splits,set(old_e),set(old_v)); return splits


def audit(splits,old_e,old_v):
    private={s:private_sets(rs) for s,rs in splits.items()}
    for s,rs in splits.items():
        assert len({r['task_id'] for r in rs})==len(rs)
        es,vs=private[s]; assert not es&old_e and not vs&old_v
        for other,(oe,ov) in private.items():
            if other!=s: assert not es&oe and not vs&ov
        for t in rs: oracle(t)
    return dict(counts={s:len(rs) for s,rs in splits.items()},historical_and_cross_split_facts_disjoint=True,
                seed=2026101205,oracle_uses_public_state_only=True,
                recovery='scripted single-character edit states, not natural model failures; edit/position/wording crossed',
                id_lengths=[16,24,32],cost='unit 1 for ordinary/error queries, quoted 2 or 6 for resolved_limit including errors',
                signal_used_for_training=False)


def request_examples(task,prompt):
    env=BranchEnvironment(task); messages=[dict(role='system',content=prompt),dict(role='user',content=task['prompt'])]
    for j,a in enumerate(task['bootstrap_calls']):
        text=task.get('bootstrap_texts',[])[j] if task.get('bootstrap_texts') else json.dumps(dict(action='lookup',**a),separators=(',',':'))
        messages += [dict(role='assistant',content=text),dict(role='tool',content=json.dumps(env.lookup(**a),sort_keys=True))]
    examples=[]
    for turn in range(7-len(task['bootstrap_calls'])):
        a=next_action(env)
        target=json.dumps(a,separators=(',',':'))
        examples.append(dict(task_id=task['task_id'],kind=task['kind'],turn=turn,messages=copy.deepcopy(messages),target=target))
        if a['action']=='answer':
            assert a['value']==task['expected']; return examples
        messages += [dict(role='assistant',content=target),dict(role='tool',content=json.dumps(env.lookup(a['entity'],a['field']),sort_keys=True))]
    raise AssertionError('oracle correction exceeds real budget')


def edit_type(a,b):
    if a==b: return None
    if len(a)==len(b)-1 and any(b[:i]+b[i+1:]==a for i in range(len(b))): return 'deletion'
    if len(a)==len(b)+1 and any(a[:i]+a[i+1:]==b for i in range(len(a))): return 'insertion'
    if len(a)==len(b):
        diffs=[i for i in range(len(a)) if a[i]!=b[i]]
        if len(diffs)==1:return 'substitution'
        if len(diffs)==2 and diffs[1]==diffs[0]+1 and a[diffs[0]]==b[diffs[1]] and a[diffs[1]]==b[diffs[0]]:return 'transposition'
    return None


def harvest(task,row):
    """First eligible actual copy mistake, unique repair among previously visible IDs."""
    known={task['root_entity']}
    for r in task['provided_facts']:
        known.add(r['entity'])
        if isinstance(r['value'],str) and r['value'] in task['database']: known.add(r['value'])
    calls,texts=[],[]
    for j,event in enumerate(row['call_events']):
        a=dict(entity=event['entity'],field=event['field']); calls.append(a); texts.append(row['requests'][j]['text'])
        obs=event['observation']
        if obs.get('error')=='unknown_entity':
            candidates=[(e,edit_type(a['entity'],e)) for e in known if a['field'] in task['database'][e] and edit_type(a['entity'],e)]
            if len(candidates)==1:
                t=copy.deepcopy(task); t.update(task_id=task['task_id']+'_natural_copy',bootstrap_calls=copy.deepcopy(calls),
                    bootstrap_texts=list(texts),natural_copy_type=candidates[0][1],correct_copied_entity=candidates[0][0],
                    source_task_id=task['task_id'],source_request_index=j,source_seed=row['seed'],
                    prefix_origin='actual frozen v3 student CUDA rollout')
                try:
                    remaining,env,_=completion(t)
                    if len(env.calls)<=MAX_CALLS and remaining:
                        t['remaining_oracle_calls']=remaining; oracle(t); return t
                except ValueError: pass
        if 'error' not in obs and isinstance(obs.get('value'),str) and obs['value'] in task['database']: known.add(obs['value'])
    return None


def write_splits(root):
    assert not root.exists(); paths,es,vs=historical(); splits=create(es,vs); root.mkdir(parents=True)
    for s,rs in splits.items():save_rows(root/(s+'.jsonl'),rs)
    out=audit(splits,es,vs)|dict(sha256={s:sha(root/(s+'.jsonl')) for s in splits},
         historical_files_sha256={str(p):sha(p) for p in paths},templates=WORDING)
    (root/'audit.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8',newline='\n'); return out

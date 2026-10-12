"""Split-safe, paired difficulty probes; none are used to update weights."""
import copy
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from .stress_environment import oracle

KINDS=['anchor_two','chain_three','chain_four','retry_two','redirect_two',
       'skip_owner','skip_prefix','given_answer']
DEV_TEMPLATES={
 'anchor_two':"What department does the owner of {p} belong to?",
 'chain_three':"Find the cost center of the department that the owner of {p} belongs to.",
 'chain_four':"Find the spending limit of the cost center of the department of the owner of {p}.",
 'skip_owner':"{p} is owned by {e}. What department does this owner belong to?",
 'skip_prefix':"The owner of {p} belongs to {d}, whose cost center is {c}. What is that cost center's spending limit?",
 'given_answer':"The budget of {p} is {b}. Its owner is {e}. What is this project's budget?"}
TEST_TEMPLATES={
 'anchor_two':[
  "Using company records, identify the department of the employee who owns {p}.",
  "Report the department associated with the person listed as the owner of {p}."],
 'chain_three':[
  "Which cost center is linked to the department of the employee listed as the owner of {p}?",
  "From the records for {p}, follow its owner to that person's department and report the department's cost center."],
 'chain_four':[
  "What spending limit is recorded for the cost center linked to the department of the owner of {p}?",
  "For {p}, report the spending limit of its owner's department's cost center."],
 'skip_owner':[
  "The supplied ownership entry is {p} -> {e}. Report the department of this employee.",
  "You already know that {e} owns {p}. Retrieve that employee's department."],
 'skip_prefix':[
  "Known entries: the owner of {p} belongs to {d}; {d} uses {c}. Retrieve the spending limit stored for {c}.",
  "The project is {p}. Its owner's department is {d} and that department's cost center is {c}. Report the spending limit for this cost center."],
 'given_answer':[
  "Provided entries: {p} has budget {b} and owner {e}. Return the budget.",
  "For {p}, the budget is already stated as {b}, while the owner is {e}. Answer with this project's budget amount."]}


def historical_paths():
    return sorted(p for p in Path('data').rglob('*.jsonl') if 'stress-v1' not in p.parts)


def prior_sets(paths):
    entities=set()
    values=set()
    for path in paths:
        for row in map(json.loads,path.read_text(encoding='utf-8').splitlines()):
            entities.update(row.get('database',{}))
            for record in row.get('database',{}).values():
                for value in record.values():
                    values.add(str(value))
            values.add(str(row.get('expected')))
    return entities,values


def make_splits(prior_entities=(),prior_values=(),seed=2026101102):
    rng=random.Random(seed)
    used_entities=set(prior_entities)
    used_values=set(prior_values)|{'321','875','department_48216','department_74123','9101','24761'}
    suffixes={e.rsplit('_',1)[-1] for e in used_entities}
    def ident(role):
        while True:
            suffix=f'{rng.getrandbits(48):012x}'
            entity=role+'_'+suffix
            if suffix not in suffixes and entity not in used_values:
                suffixes.add(suffix)
                used_entities.add(entity)
                return entity
    def value(lo,hi):
        while True:
            x=rng.randint(lo,hi)
            if str(x) not in used_values:
                used_values.add(str(x))
                return x
    def department():
        while True:
            entity=f'department_{rng.randint(10000,99999)}'
            if entity not in used_entities and entity not in used_values:
                used_entities.add(entity)
                used_values.add(entity)
                return entity
    def graph():
        p,e,d,c=ident('project'),ident('employee'),department(),ident('cost_center')
        b,limit=value(101,9999),value(10001,99999)
        database={p:{'budget':b,'owner':e},e:{'department':d},d:{'cost_center':c},c:{'spending_limit':limit}}
        for _ in range(2):
            dp,de,dd,dc=ident('project'),ident('employee'),department(),ident('cost_center')
            database.update({dp:{'budget':value(101,9999),'owner':de},de:{'department':dd},
                             dd:{'cost_center':dc},dc:{'spending_limit':value(10001,99999)}})
        return dict(p=p,e=e,d=d,c=c,b=b,limit=limit,database=database)
    splits={}
    for split,n in [('dev',3),('test',8)]:
        rows=[]
        for index in range(n):
            facts=[graph() for _ in range(3)]
            for kind in KINDS:
                gidx=1 if kind in ('chain_four','skip_prefix') else 2 if kind=='chain_three' else 0
                f=copy.deepcopy(facts[gidx])
                p,e,d,c,b=f['p'],f['e'],f['d'],f['c'],f['b']
                route=[{'entity':p,'field':'owner'},{'entity':e,'field':'department'},
                       {'entity':d,'field':'cost_center'},{'entity':c,'field':'spending_limit'}]
                aliases={}
                transient=[]
                supplied=[]
                supplied_value=None
                expected=d
                if kind=='anchor_two':
                    calls=route[:2]
                elif kind=='chain_three':
                    calls=route[:3]
                    expected=c
                elif kind=='chain_four':
                    calls=route
                    expected=f['limit']
                elif kind=='retry_two':
                    calls=[route[0],route[1],route[1]]
                    transient=[[e,'department']]
                elif kind=='redirect_two':
                    alias=ident('employee')
                    aliases={alias:e}
                    f['database'][p]['owner']=alias
                    calls=[route[0],{'entity':alias,'field':'department'},route[1]]
                elif kind=='skip_owner':
                    calls=[route[1]]
                    supplied=[{'entity':p,'field':'owner','value':e}]
                elif kind=='skip_prefix':
                    calls=[route[3]]
                    supplied=[{'entity':e,'field':'department','value':d},
                              {'entity':d,'field':'cost_center','value':c}]
                    expected=f['limit']
                else:
                    calls=[]
                    supplied_value=b
                    expected=b
                    supplied=[{'entity':p,'field':'budget','value':b},{'entity':p,'field':'owner','value':e}]
                template_kind='anchor_two' if kind in ('retry_two','redirect_two') else kind
                templates=[DEV_TEMPLATES[template_kind]] if split=='dev' else TEST_TEMPLATES[template_kind]
                tid=index%len(templates)
                prompt=templates[tid].format(p=p,e=e,d=d,c=c,b=b)
                row=dict(task_id=f'stress_{split}_{len(rows):04d}',split=split,kind=kind,template_id=tid,
                         sampling_index=3*index+gidx,fact_group=p,prompt=prompt,database=f['database'],
                         aliases=aliases,transient_once=transient,provided_facts=supplied,provided_value=supplied_value,
                         expected=expected,oracle_calls=calls,minimum_calls=len(calls))
                oracle(row)
                rows.append(row)
        rng.shuffle(rows)
        splits[split]=rows
    audit(splits,set(prior_entities),set(prior_values))
    return splits


def sets(rows):
    entities={e for r in rows for e in r['database']}|{e for r in rows for e in r['aliases']}
    values={str(v) for r in rows for record in r['database'].values() for v in record.values()}
    return entities,values


def audit(splits,prior_entities,prior_values):
    dev_e,dev_v=sets(splits['dev'])
    test_e,test_v=sets(splits['test'])
    assert not dev_e&test_e and not dev_v&test_v
    assert not (dev_e|test_e)&prior_entities and not (dev_v|test_v)&prior_values
    for split,rows in splits.items():
        assert len({r['task_id'] for r in rows})==len(rows)
        for r in rows:
            assert r['split']==split and r['minimum_calls']<=6
            assert all('demo' not in e and 'distractor' not in e for e in r['database'])
            for e,record in r['database'].items():
                for field in ('owner','cost_center'):
                    if field in record:
                        assert e.rsplit('_',1)[-1]!=record[field].rsplit('_',1)[-1]
            oracle(r)
    return dict(dev_test_entity_overlap=False,dev_test_value_overlap=False,historical_entity_overlap=False,
                historical_value_overlap=False,
                counts={s:dict(Counter(r['kind'] for r in rs)) for s,rs in splits.items()},
                fact_groups={s:len({r['fact_group'] for r in rs}) for s,rs in splits.items()},
                templates={'dev':DEV_TEMPLATES,'test':TEST_TEMPLATES},
                sampling='Matched fact-group scenarios share task/turn seeds; repeated seed trials and paired scenarios are correlated.')


def write_splits(root):
    if root.exists():
        raise ValueError('new dataset directory required')
    paths=historical_paths()
    old_e,old_v=prior_sets(paths)
    splits=make_splits(old_e,old_v)
    root.mkdir(parents=True)
    hashes={}
    for split,rows in splits.items():
        path=root/(split+'.jsonl')
        path.write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows),encoding='utf-8',newline='\n')
        hashes[split]=hashlib.sha256(path.read_bytes()).hexdigest()
    result=audit(splits,old_e,old_v)|dict(seed=2026101102,sha256=hashes,
        historical_files_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    (root/'audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    return result

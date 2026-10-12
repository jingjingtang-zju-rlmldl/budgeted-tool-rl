"""Independent opaque entities, split-safe facts, and oracle decision supervision."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from .environment import LookupEnvironment, oracle, verify

TRAIN_TEMPLATES = {
    'direct': [
        'The budget of {project} is {budget}. What is its budget?',
        'Given: {project} has a budget of {budget}. Return the budget of {project}.',
        'For {project}, the budget is already provided: {budget}. What is that value?',
        'The company record shown here gives {project} a budget of {budget}. Return that budget.'],
    'one_lookup': [
        'What is the budget of {project}?',
        'Find the budget recorded for {project}.',
        'Return the budget of {project} from the private database.',
        'Look up the budget amount assigned to {project}.'],
    'two_lookup': [
        'What department does the owner of {project} belong to?',
        'Find the department of the employee who owns {project}.',
        'Return the department of the owner recorded for {project}.',
        'Which department does the employee listed as the owner of {project} work in?'],
}
TEST_TEMPLATES = {
    'direct': [
        'For reference, the allocation for {project} is {budget}. Give its budget as a number.',
        'The supplied budget entry for {project} reads {budget}. Report this value.',
        '{project} has {budget} as its stated budget. Answer with that amount.'],
    'one_lookup': [
        'Please retrieve the stored budget amount for {project}.',
        'Using the company records, report the budget for {project}.',
        'What number is saved in the budget field of {project}?'],
    'two_lookup': [
        'Which department is associated with the employee listed as the owner of {project}?',
        'Identify the owner of {project} and report that person\'s department.',
        'From company records, give the department of the person marked as the owner of {project}.'],
}


def make_splits(seed=2026101201):
    rng=random.Random(seed)
    # Exclude fixed prompt-demo answers so copying them cannot score by chance.
    used_ids,used_budgets,used_departments=set(),{321,875},{'department_48216'}
    def identifier(prefix):
        while True:
            suffix=f'{rng.getrandbits(48):012x}'
            if suffix not in used_ids:
                used_ids.add(suffix)
                return prefix+'_'+suffix
    def budget():
        while True:
            value=rng.randint(101,9999)
            if value not in used_budgets:
                used_budgets.add(value)
                return value
    def facts():
        project,employee=identifier('project'),identifier('employee')
        def department():
            while True:
                value=f'department_{rng.randrange(10000,99999)}'
                if value not in used_departments:
                    used_departments.add(value)
                    return value
        value,dept=budget(),department()
        db={project:{'budget':value,'owner':employee},employee:{'department':dept}}
        for _ in range(2):
            decoy_project,decoy_employee=identifier('project'),identifier('employee')
            db[decoy_project]={'budget':budget(),'owner':decoy_employee}
            db[decoy_employee]={'department':department()}
        return project,employee,value,dept,db
    splits={}
    for split,n in [('train',100),('dev',10),('test',20)]:
        templates=TEST_TEMPLATES if split=='test' else TRAIN_TEMPLATES
        rows=[]
        paired_facts=[]
        for kind in templates:
            for index in range(n):
                if split=='train' and kind=='two_lookup' and index<40:
                    project,employee,value,department,db=paired_facts[index]
                else:
                    project,employee,value,department,db=facts()
                if split=='train' and kind=='one_lookup' and index<40:
                    paired_facts.append((project,employee,value,department,db))
                calls=[] if kind=='direct' else [{'entity':project,'field':'budget'}] if kind=='one_lookup' else [
                    {'entity':project,'field':'owner'},{'entity':employee,'field':'department'}]
                template_id=index%len(templates[kind])
                row=dict(task_id=f'{split}_{len(rows):04d}',split=split,kind=kind,template_id=template_id,
                         template_family='held_out' if split=='test' else 'train_family',
                         fact_group=project,prompt=templates[kind][template_id].format(project=project,budget=value),
                         database=db,expected=department if kind=='two_lookup' else value,
                         provided_value=value if kind=='direct' else None,oracle_calls=calls,minimum_calls=len(calls))
                answer,observations=oracle(row)
                assert verify(answer,row['expected']) and all('error' not in o for o in observations)
                rows.append(row)
        rng.shuffle(rows)
        splits[split]=rows
    audit_splits(splits)
    return splits


def audit_splits(splits):
    entities={s:{e for t in rows for e in t['database']} for s,rows in splits.items()}
    values={s:{str(v) for t in rows for record in t['database'].values() for k,v in record.items() if k!='owner'}
            for s,rows in splits.items()}
    for a,b in [('train','dev'),('train','test'),('dev','test')]:
        assert not entities[a]&entities[b]
        assert not values[a]&values[b]
    for rows in splits.values():
        for row in rows:
            for entity,record in row['database'].items():
                if 'owner' in record:
                    assert entity.split('_')[-1]!=record['owner'].split('_')[-1]
            assert all('distractor' not in e and 'train' not in e and 'test' not in e and 'dev' not in e for e in row['database'])
    for kind in TRAIN_TEMPLATES:
        assert not set(TRAIN_TEMPLATES[kind])&set(TEST_TEMPLATES[kind])
    return dict(entity_overlap=False,private_value_overlap=False,owner_suffix_relation=False,
                held_out_test_templates=True,
                counts={s:dict(Counter(t['kind'] for t in rows)) for s,rows in splits.items()})


def request_examples(task,system_prompt):
    """Only preceding oracle actions/observations enter a decision's input."""
    env=LookupEnvironment(task)
    messages=[{'role':'system','content':system_prompt},{'role':'user','content':task['prompt']}]
    rows=[]
    observations=[]
    for turn,call in enumerate(task['oracle_calls']):
        action={'action':'lookup',**call}
        text=json.dumps(action,separators=(',',':'))
        rows.append(dict(task_id=task['task_id'],kind=task['kind'],turn=turn,
                         messages=list(messages),target=text))
        messages.append({'role':'assistant','content':text})
        observation=env.lookup(call['entity'],call['field'])
        assert 'error' not in observation
        observations.append(observation)
        messages.append({'role':'tool','content':json.dumps(observation,sort_keys=True)})
    answer=task['provided_value'] if task['kind']=='direct' else observations[-1]['value']
    assert verify(answer,task['expected'])
    rows.append(dict(task_id=task['task_id'],kind=task['kind'],turn=len(task['oracle_calls']),
                     messages=list(messages),target=json.dumps({'action':'answer','value':answer},separators=(',',':'))))
    return rows


def encode_example(row,tokenizer,max_context=768):
    text=tokenizer.apply_chat_template(row['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    prefix=tokenizer(text,add_special_tokens=False)['input_ids']
    target=tokenizer(row['target'],add_special_tokens=False)['input_ids']+[tokenizer.eos_token_id]
    if len(prefix)+len(target)>max_context:
        raise ValueError('training example exceeds context; never silently truncate')
    labels=[-100]*len(prefix)+target
    assert sum(x!=-100 for x in labels)==len(target)
    return dict(task_id=row['task_id'],kind=row['kind'],turn=row['turn'],messages=row['messages'],target=row['target'],
                input_ids=prefix+target,labels=labels,prefix_tokens=len(prefix),target_tokens=len(target))


def write_splits(root):
    if root.exists():
        raise ValueError('choose new split directory; no overwrite')
    splits=make_splits()
    root.mkdir(parents=True)
    hashes={}
    for split,rows in splits.items():
        path=root/(split+'.jsonl')
        path.write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows),encoding='utf-8',newline='\n')
        hashes[split]=hashlib.sha256(path.read_bytes()).hexdigest()
    audit=audit_splits(splits)|dict(seed=2026101201,sha256=hashes,
                                 training_goal_contrast_pairs=40,train_templates=TRAIN_TEMPLATES,test_templates=TEST_TEMPLATES,
                                 limitation='Synthetic English schema and two-hop compositions only; test holds out paraphrases, not new tools.')
    (root/'audit.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8',newline='\n')
    return audit

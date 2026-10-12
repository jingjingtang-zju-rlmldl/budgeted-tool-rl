"""Generated tasks are development probes, not a held-out final benchmark."""
import argparse
import hashlib
import json
from pathlib import Path
import random

from .environment import oracle, verify

TEMPLATES = {
    'direct': [
        'The budget of {project} is {budget}. What is its budget?',
        'Given: {project} has a budget of {budget}. Return the budget of {project}.',
        'For {project}, the budget is already provided: {budget}. What is that value?'],
    'one_lookup': [
        'What is the budget of {project}?',
        'Find the budget recorded for {project}.',
        'Return the budget of {project} from the private database.'],
    'two_lookup': [
        'What department does the owner of {project} belong to?',
        'Find the department of the employee who owns {project}.',
        'Return the department of the owner recorded for {project}.'],
}


def make_tasks(seed=20261011, counts=(34, 33, 33), namespace='dev'):
    rng = random.Random(seed)
    tasks = []
    # Each episode has independent private facts and a distinct entity namespace.
    for kind, count in zip(TEMPLATES, counts):
        for index in range(count):
            suffix = f'{namespace}_{rng.getrandbits(48):012x}'
            project = f'project_{suffix}'
            employee = f'employee_{suffix}'
            budget = rng.randint(101, 9999)
            department = f'department_{rng.randrange(10000,99999)}'
            database = {project: {'budget': budget, 'owner': employee}, employee: {'department': department}}
            for distractor in range(2):
                database[f'project_distractor_{suffix}_{distractor}'] = {
                    'budget': rng.randint(101, 9999), 'owner': f'employee_distractor_{suffix}_{distractor}'}
                database[f'employee_distractor_{suffix}_{distractor}'] = {'department': f'department_{rng.randrange(10000,99999)}'}
            if kind == 'direct':
                calls, expected = [], budget
            elif kind == 'one_lookup':
                calls, expected = [{'entity': project, 'field': 'budget'}], budget
            else:
                calls = [{'entity': project, 'field': 'owner'}, {'entity': employee, 'field': 'department'}]
                expected = department
            template_id = index % len(TEMPLATES[kind])
            task = dict(task_id=suffix, split='dev', kind=kind, template_id=template_id,
                        prompt=TEMPLATES[kind][template_id].format(project=project, budget=budget),
                        database=database, expected=expected, provided_value=budget if kind == 'direct' else None,
                        oracle_calls=calls, minimum_calls=len(calls))
            answer, observations = oracle(task)
            assert verify(answer, expected) and all('error' not in x for x in observations)
            tasks.append(task)
    rng.shuffle(tasks)
    return tasks


def write_tasks(path, tasks, seed):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = ''.join(json.dumps(t, sort_keys=True) + '\n' for t in tasks)
    path.write_text(raw, encoding='utf-8', newline='\n')
    manifest = dict(seed=seed, task_count=len(tasks), sha256=hashlib.sha256(raw.encode()).hexdigest(),
                    purpose='development baseline; used to decide subsequent training', language='English',
                    model_visible_fields=['prompt'], hidden_fields=['database', 'expected', 'oracle_calls', 'kind'],
                    limitation='Controlled synthetic English tasks; no evidence about broad real-world agent ability.',
                    oracle_success=sum(verify(oracle(t)[0], t['expected']) for t in tasks))
    path.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=Path('data/dev100.jsonl'))
    args = parser.parse_args()
    write_tasks(args.out, make_tasks(), 20261011)
    write_tasks(args.out.parent / 'calibration9.jsonl', make_tasks(707, (3, 3, 3), 'calibration'), 707)
    print('Generated 100 development tasks and 9 separate calibration tasks; oracle solves every task.')


if __name__ == '__main__':
    main()

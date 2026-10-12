import json
import re
from decimal import Decimal, InvalidOperation

SYSTEM_PROMPT = '''Answer the question using a private company database when necessary.
Output exactly one JSON object per turn. No explanation and no Markdown code fences.
There are two possible actions:
1. lookup: the object has exactly three keys: "action" (value "lookup"), "entity" (an entity ID copied exactly from the current question or a tool result), and "field" (the field to query).
2. answer: the object has exactly two keys: "action" (value "answer") and "value" (the exact requested answer).
Project fields: "budget" and "owner". Employee field: "department".
An owner is an employee ID. To find the owner's department, first look up the project's owner, then look up that employee's department.
Tool messages contain the result of your lookup. If the requested answer is already in the question, answer directly.
Do not invent private facts or entity IDs. You may make at most 3 lookup calls.
Return only the JSON object, without enclosing it in triple backticks.'''


def parse_action(text, allow_fence=False):
    candidate = text.strip()
    if allow_fence:
        block = re.fullmatch(r'```(?:json)?\s*\n([\s\S]*?)\n```', candidate)
        if block:
            candidate = block.group(1).strip()
    try:
        action = json.loads(candidate)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError('output is not one JSON object') from exc
    if not isinstance(action, dict):
        raise ValueError('action must be a JSON object')
    if action.get('action') == 'answer':
        if set(action) != {'action', 'value'} or type(action['value']) not in (str, int, float):
            raise ValueError('answer requires only action and scalar value')
    elif action.get('action') == 'lookup':
        if set(action) != {'action', 'entity', 'field'} or not all(isinstance(action[k], str) for k in ('entity', 'field')):
            raise ValueError('lookup requires only action, string entity and string field')
    else:
        raise ValueError('unknown action')
    return action


def verify(value, expected):
    if type(expected) is int:
        try:
            number = Decimal(str(value).strip())
            return number.is_finite() and number == Decimal(expected)
        except (InvalidOperation, ValueError):
            return False
    return isinstance(value, str) and value.strip() == expected


class LookupEnvironment:
    def __init__(self, task, max_calls=3):
        self.task = task
        self.max_calls = max_calls
        self.calls = []

    def lookup(self, entity, field):
        if len(self.calls) >= self.max_calls:
            raise ValueError('tool budget exhausted')
        self.calls.append((entity, field))
        if entity not in self.task['database']:
            return {'error': 'unknown entity', 'entity': entity}
        record = self.task['database'][entity]
        if field not in record:
            return {'error': 'unknown field', 'entity': entity, 'field': field}
        return {'entity': entity, 'field': field, 'value': record[field]}


def oracle(task):
    env = LookupEnvironment(task)
    observations = [env.lookup(x['entity'], x['field']) for x in task['oracle_calls']]
    answer = task['provided_value'] if task['kind'] == 'direct' else observations[-1]['value']
    return answer, observations

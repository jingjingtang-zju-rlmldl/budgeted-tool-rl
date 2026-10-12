"""Observable branches and disclosed weighted query costs; no external services."""
from .environment import verify

MAX_CALLS = 6
MAX_TURNS = 7
FIELDS = {'project': ['policy', 'owner', 'partner', 'local_limit', 'resolved_limit', 'budget'],
          'employee': ['department'], 'department': ['cost_center'],
          'cost_center': ['spending_limit'], 'vendor': ['spending_limit']}


class BranchEnvironment:
    def __init__(self, task, max_calls=MAX_CALLS):
        self.task = task
        self.max_calls = max_calls
        self.calls, self.events = [], []
        self.total_cost = 0
        self.known = {(r['entity'], r['field']): r['value'] for r in task['provided_facts']}

    def lookup(self, entity, field):
        if len(self.calls) >= self.max_calls:
            raise ValueError('tool budget exhausted')
        key = (entity, field)
        seen = key in self.calls
        redundant = key in self.known
        cost = self.task['service_price'] if field == 'resolved_limit' else 1
        self.calls.append(key)
        self.total_cost += cost
        if entity not in self.task['database']:
            result = dict(error='unknown_entity', entity=entity, field=field)
        elif field not in self.task['database'][entity]:
            result = dict(error='unknown_field', entity=entity, field=field,
                          allowed_fields=sorted(self.task['database'][entity]))
        else:
            result = dict(entity=entity, field=field, value=self.task['database'][entity][field])
            self.known[key] = result['value']
        result.update(query_cost=cost, total_cost=self.total_cost)
        self.events.append(dict(entity=entity, field=field, observation=result, query_cost=cost,
                                necessary_retry=False, followed_redirect=False,
                                redundant_known_lookup=redundant, unnecessary_repeat=seen,
                                recoverable_error=False, invalid_query='error' in result))
        return result


def next_action(env):
    """Planner reads ONLY public state, never private DB or hidden expected value."""
    p = env.task['root_entity']
    known = env.known
    def action(e, f):
        return dict(action='lookup', entity=e, field=f)
    def answer(v):
        return dict(action='answer', value=v)
    if (p, 'resolved_limit') in known:
        return answer(known[p, 'resolved_limit'])
    policy = known.get((p, 'policy'))
    if policy is None:
        # Price 2 weakly dominates every complete unknown-policy route.
        return action(p, 'resolved_limit' if env.task['service_price'] == 2 else 'policy')
    if policy == 'local':
        route = [(p, 'local_limit', 1)]
    elif policy == 'partner':
        v = known.get((p, 'partner'))
        route = [(p, 'partner', 2)] if v is None else [(v, 'spending_limit', 1)]
    elif policy == 'team':
        e = known.get((p, 'owner'))
        d = known.get((e, 'department')) if e is not None else None
        c = known.get((d, 'cost_center')) if d is not None else None
        route = ([(p, 'owner', 4)] if e is None else
                 [(e, 'department', 3)] if d is None else
                 [(d, 'cost_center', 2)] if c is None else [(c, 'spending_limit', 1)])
    else:
        raise ValueError('invalid publicly observed policy')
    e, f, distance = route[0]
    if (e, f) in known:
        return answer(known[e, f])
    if env.task['service_price'] <= distance:
        return action(p, 'resolved_limit')
    return action(e, f)


def completion(task, bootstrap=None):
    env = BranchEnvironment(task)
    for a in task['bootstrap_calls'] if bootstrap is None else bootstrap:
        env.lookup(**a)
    actions = []
    while True:
        a = next_action(env)
        if a['action'] == 'answer':
            assert verify(a['value'], task['expected'])
            return actions, env, a['value']
        call = dict(entity=a['entity'], field=a['field'])
        env.lookup(**call)
        actions.append(call)


def grounded(task, env, answer):
    if not verify(answer, task['expected']):
        return False
    p = task['root_entity']
    if env.known.get((p, 'resolved_limit')) == task['expected']:
        return True
    policy = env.known.get((p, 'policy'))
    if policy == 'local':
        goal = (p, 'local_limit')
    elif policy == 'partner':
        goal = (env.known.get((p, 'partner')), 'spending_limit')
    elif policy == 'team':
        e = env.known.get((p, 'owner')); d = env.known.get((e, 'department'))
        goal = (env.known.get((d, 'cost_center')), 'spending_limit')
    else:
        return False
    return env.known.get(goal) == task['expected']


def oracle(task):
    calls, env, value = completion(task, bootstrap=[])
    assert calls == task['oracle_calls']
    assert len(calls) == task['minimum_calls'] and env.total_cost == task['minimum_cost']
    assert grounded(task, env, value)
    if task['bootstrap_calls']:
        calls, env, value = completion(task)
        assert calls == task['remaining_oracle_calls']
        assert len(env.calls) <= MAX_CALLS and grounded(task, env, value)
    return value

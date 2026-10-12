"""Deterministic multi-hop tools with recoverable faults and explicit call cost."""
from .environment import verify

MAX_CALLS=6
MAX_TURNS=7
FIELDS={'project':['budget','owner'],'employee':['department'],
        'department':['cost_center'],'cost_center':['spending_limit']}


class StressEnvironment:
    def __init__(self,task,max_calls=MAX_CALLS):
        self.task=task
        self.max_calls=max_calls
        self.calls=[]
        self.events=[]
        self.failed_once=set()
        self.pending_retry=set()
        self.pending_redirect=set()
        self.known={(r['entity'],r['field']):r['value'] for r in task['provided_facts']}

    def lookup(self,entity,field):
        if len(self.calls)>=self.max_calls:
            raise ValueError('tool budget exhausted')
        key=(entity,field)
        seen=key in self.calls
        self.calls.append(key)
        needed_retry=key in self.pending_retry
        redirected=key in self.pending_redirect
        redundant=key in self.known
        if entity in self.task['aliases']:
            canonical=self.task['aliases'][entity]
            result={'error':'entity_moved','entity':entity,'field':field,'canonical_entity':canonical}
            self.pending_redirect.add((canonical,field))
        elif key in {tuple(x) for x in self.task['transient_once']} and key not in self.failed_once:
            self.failed_once.add(key)
            self.pending_retry.add(key)
            result={'error':'temporary_unavailable','entity':entity,'field':field,'retryable':True}
        elif entity not in self.task['database']:
            result={'error':'unknown_entity','entity':entity,'field':field}
        elif field not in self.task['database'][entity]:
            result={'error':'unknown_field','entity':entity,'field':field,
                    'allowed_fields':sorted(self.task['database'][entity])}
        else:
            result={'entity':entity,'field':field,'value':self.task['database'][entity][field]}
            self.known[key]=result['value']
            self.pending_retry.discard(key)
            self.pending_redirect.discard(key)
        recoverable=result.get('error') in ('temporary_unavailable','entity_moved')
        self.events.append(dict(entity=entity,field=field,observation=result,
                                necessary_retry=needed_retry and 'error' not in result,
                                followed_redirect=redirected and 'error' not in result,
                                redundant_known_lookup=redundant,
                                unnecessary_repeat=seen and not needed_retry,
                                recoverable_error=recoverable,
                                invalid_query='error' in result and not recoverable))
        return result


def oracle(task):
    env=StressEnvironment(task)
    observations=[env.lookup(a['entity'],a['field']) for a in task['oracle_calls']]
    result=task['provided_value'] if task['kind']=='given_answer' else observations[-1]['value']
    assert verify(result,task['expected'])
    assert len(env.calls)==task['minimum_calls']
    return result,observations

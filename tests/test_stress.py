import copy
import json
import unittest
from collections import Counter

from toolrl.stress_data import make_splits,KINDS
from toolrl.stress_environment import StressEnvironment,oracle


class StressTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.splits=make_splits()

    def task(self,kind):
        return next(t for t in self.splits['test'] if t['kind']==kind)

    def test_deterministic_balanced_and_disjoint(self):
        self.assertEqual(self.splits,make_splits())
        self.assertEqual(Counter(t['kind'] for t in self.splits['test']),{k:8 for k in KINDS})
        a={e for t in self.splits['dev'] for e in t['database']}
        b={e for t in self.splits['test'] for e in t['database']}
        self.assertFalse(a&b)
        self.assertEqual(len({t['fact_group'] for t in self.splits['test']}),24)

    def test_all_oracles_cost_and_correctness(self):
        for rows in self.splits.values():
            for task in rows:
                answer,observations=oracle(task)
                self.assertEqual(answer,task['expected'])
                self.assertEqual(len(observations),task['minimum_calls'])
        self.assertEqual(self.task('chain_three')['minimum_calls'],3)
        self.assertEqual(self.task('chain_four')['minimum_calls'],4)

    def test_temporary_failure_is_once_and_retry_is_not_waste(self):
        task=self.task('retry_two')
        env=StressEnvironment(task)
        actions=task['oracle_calls']
        env.lookup(**actions[0])
        failed=env.lookup(**actions[1])
        success=env.lookup(**actions[2])
        self.assertEqual(failed['error'],'temporary_unavailable')
        self.assertEqual(success['value'],task['expected'])
        self.assertTrue(env.events[-1]['necessary_retry'])
        self.assertFalse(env.events[-1]['unnecessary_repeat'])
        env.lookup(**actions[2])
        self.assertTrue(env.events[-1]['redundant_known_lookup'])
        self.assertTrue(env.events[-1]['unnecessary_repeat'])
        self.assertFalse(env.events[-1]['necessary_retry'])
        self.assertEqual(StressEnvironment(task).lookup(**actions[1])['error'],'temporary_unavailable')

    def test_redirect_requires_actual_canonical_id(self):
        task=self.task('redirect_two')
        env=StressEnvironment(task)
        first=env.lookup(**task['oracle_calls'][0])
        error=env.lookup(first['value'],'department')
        final=env.lookup(error['canonical_entity'],'department')
        self.assertEqual(error['error'],'entity_moved')
        self.assertNotEqual(first['value'],error['canonical_entity'])
        self.assertEqual(final['value'],task['expected'])
        self.assertTrue(env.events[-1]['followed_redirect'])

    def test_provided_fact_query_is_redundant_but_success_can_be_correct(self):
        task=self.task('skip_owner')
        supplied=task['provided_facts'][0]
        env=StressEnvironment(task)
        env.lookup(supplied['entity'],supplied['field'])
        self.assertTrue(env.events[0]['redundant_known_lookup'])
        self.assertEqual(env.lookup(**task['oracle_calls'][0])['value'],task['expected'])
        self.assertEqual(len(env.calls)-task['minimum_calls'],1)
        self.assertEqual(self.task('skip_prefix')['minimum_calls'],1)
        self.assertEqual(self.task('given_answer')['minimum_calls'],0)

    def test_fault_pairs_share_question_facts_and_sampling_seed(self):
        anchor=self.task('anchor_two')
        pair=[t for t in self.splits['test'] if t['fact_group']==anchor['fact_group']]
        retry=next(t for t in pair if t['kind']=='retry_two')
        self.assertEqual(anchor['prompt'],retry['prompt'])
        self.assertEqual(anchor['database'],retry['database'])
        self.assertEqual(anchor['sampling_index'],retry['sampling_index'])
        self.assertEqual(anchor['expected'],retry['expected'])

    def test_invalid_queries_and_recoverable_errors_consume_budget(self):
        env=StressEnvironment(self.task('retry_two'),2)
        env.lookup('missing','department')
        self.assertTrue(env.events[-1]['invalid_query'])
        env.lookup(**self.task('retry_two')['oracle_calls'][1])
        self.assertTrue(env.events[-1]['recoverable_error'])
        with self.assertRaisesRegex(ValueError,'budget'):
            env.lookup(**self.task('retry_two')['oracle_calls'][1])

    def scripted(self,task,actions):
        from toolrl.stress_runner import run_episode
        scripted=iter(actions)
        # A unit-test stub only. The CUDA runner never uses this generator.
        def generate(messages,seed):
            return dict(text=json.dumps(next(scripted)),input_ids=[],output_ids=[],truncated=False,generation_seconds=0.)
        return run_episode(task,generate,'unit test only',7)

    def test_seven_turn_budget_allows_four_hops_plus_two_extra_queries(self):
        task=self.task('chain_four')
        calls=task['oracle_calls']
        actions=[{'action':'lookup',**a} for a in [calls[0],calls[0],calls[1],calls[1],calls[2],calls[3]]]
        row=self.scripted(task,actions+[{'action':'answer','value':task['expected']}])
        self.assertTrue(row['success'])
        self.assertTrue(row['grounded_success'])
        self.assertEqual(row['calls'],6)
        self.assertEqual(len(row['requests']),7)
        self.assertEqual(row['correct_extra_calls'],2)

    def test_early_private_guess_is_not_grounded_or_cost_evidence(self):
        task=self.task('chain_four')
        row=self.scripted(task,[{'action':'answer','value':'not_the_answer'}])
        self.assertFalse(row['success'])
        self.assertIsNone(row['correct_extra_calls'])
        self.assertEqual(row['reward_cost_gated'],0)
        row=self.scripted(task,[{'action':'answer','value':task['expected']}])
        self.assertTrue(row['success'])
        self.assertFalse(row['grounded_success'])


if __name__=='__main__':
    unittest.main()

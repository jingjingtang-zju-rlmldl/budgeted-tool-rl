import copy
import json
import unittest
from collections import Counter

from toolrl.sft4_data import create, harvest, request_examples, edit_type
from toolrl.sft4_environment import BranchEnvironment, completion, next_action, grounded
from toolrl.sft4_episode import run_episode
from toolrl.sft4_prompts import PROTOCOL


class BranchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.splits=create()
    def task(self,kind):return copy.deepcopy(next(t for t in self.splits['test'] if t['kind']==kind))
    def test_no_private_policy_peek(self):
        a=self.task('policy_team');b=copy.deepcopy(a)
        b['database'][b['root_entity']]['policy']='local'; b['expected']=-1
        aa=next_action(BranchEnvironment(a));bb=next_action(BranchEnvironment(b))
        self.assertEqual(aa,bb);self.assertEqual(aa['field'],'policy')
    def test_public_price_changes_legal_best_path(self):
        t=self.task('policy_team'); t['service_price']=2
        self.assertEqual(next_action(BranchEnvironment(t))['field'],'resolved_limit')
        t['service_price']=6
        self.assertEqual(next_action(BranchEnvironment(t))['field'],'policy')
    def test_error_and_repeated_service_charge_real_price(self):
        t=self.task('service_team');e=BranchEnvironment(t)
        o=e.lookup(t['root_entity']+'_bad','resolved_limit')
        self.assertEqual(o['error'],'unknown_entity'); self.assertEqual(o['query_cost'],2)
        e.lookup(t['root_entity'],'resolved_limit');e.lookup(t['root_entity'],'resolved_limit')
        self.assertEqual(e.total_cost,6);self.assertTrue(e.events[-1]['redundant_known_lookup'])
        self.assertTrue(grounded(t,e,str(t['expected'])))
    def test_branch_value_needs_public_policy_or_service(self):
        t=self.task('policy_team');e=BranchEnvironment(t)
        c=next(a['entity'] for a in t['oracle_calls'] if a['field']=='spending_limit')
        e.lookup(c,'spending_limit')
        self.assertFalse(grounded(t,e,t['expected']))
        e.lookup(t['root_entity'],'policy');e.lookup(t['root_entity'],'owner')
        employee=e.known[t['root_entity'],'owner'];e.lookup(employee,'department')
        department=e.known[employee,'department'];e.lookup(department,'cost_center')
        self.assertTrue(grounded(t,e,t['expected']))
    def test_recovery_crosses_edits_positions_wording(self):
        rs=self.splits['recovery']
        self.assertEqual(len({(t['correction_mode'],t['correction_stage'],t['template_id']) for t in rs}),32)
        for t in rs:
            _,e,v=completion(t)
            self.assertEqual(len(e.calls),6);self.assertEqual(e.total_cost,6)
            self.assertTrue(grounded(t,e,v))
    def test_actual_prefix_harvest_preserves_text_and_masks_via_examples(self):
        t=self.task('service_team');gold=t['root_entity'];bad=gold[:-1]
        actual='  '+json.dumps(dict(action='lookup',entity=bad,field='resolved_limit'))+'  '
        e=BranchEnvironment(t);e.lookup(bad,'resolved_limit')
        r=dict(call_events=e.events,requests=[dict(text=actual)],seed=27)
        derived=harvest(t,r);self.assertEqual(derived['natural_copy_type'],'deletion')
        self.assertEqual(derived['correct_copied_entity'],gold)
        examples=request_examples(derived,PROTOCOL)
        self.assertEqual(examples[0]['messages'][2]['content'],actual)
        self.assertEqual(json.loads(examples[0]['target'])['entity'],gold)
        self.assertNotEqual(bad,json.loads(examples[0]['target'])['entity'])
    def test_expensive_direct_route_reward_uses_cost_not_calls(self):
        t=self.task('policy_team');calls=[dict(action='lookup',entity=t['root_entity'],field='resolved_limit'),dict(action='answer',value=t['expected'])]
        def gen(messages,seed):
            a=calls.pop(0);return dict(text=json.dumps(a),truncated=False,output_ids=[],generation_seconds=0.)
        r=run_episode(t,gen,PROTOCOL,17)
        self.assertTrue(r['grounded_success']);self.assertEqual(r['calls'],1);self.assertEqual(r['total_cost'],6)
        self.assertAlmostEqual(r['reward_cost_gated'],.88)
        self.assertEqual(r['correct_extra_cost'],1)
    def test_actual_edit_types_and_no_role_literal_false_positive(self):
        self.assertEqual(edit_type('project_abc','project_abcd'),'deletion')
        self.assertEqual(edit_type('project_abxc','project_abc'),'insertion')
        self.assertEqual(edit_type('project_adc','project_abc'),'substitution')
        self.assertEqual(edit_type('project_acb','project_abc'),'transposition')
        self.assertIsNone(edit_type('project','project_abc'))

if __name__=='__main__':unittest.main()

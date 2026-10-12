import copy
import json
import unittest
from collections import Counter
from toolrl.sft3_data import create,request_examples
from toolrl.stress_environment import StressEnvironment
from toolrl.stress_prompts import PROTOCOL
from toolrl.environment import parse_action
from toolrl.sft3_episode import run_episode

class CorrectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.data=create()

    def test_pair_changes_only_question_and_labels(self):
        for i in range(0,64,2):
            a,b=self.data['pair'][i:i+2];skip={'task_id','template_id','prompt'}
            self.assertEqual({k:v for k,v in a.items() if k not in skip},{k:v for k,v in b.items() if k not in skip})
            self.assertNotEqual(a['prompt'],b['prompt'])

    def test_error_prefix_is_executed_and_not_a_target(self):
        counts=Counter()
        for t in self.data['train']:
            examples=request_examples(t,PROTOCOL)
            if not t['bootstrap_calls']:continue
            first=examples[0];o=json.loads(first['messages'][-1]['content']);counts[o['error']]+=1
            self.assertEqual(o['error'],t['correction_type'])
            correct=parse_action(first['target']);self.assertEqual(correct,dict(action='lookup',**t['oracle_calls'][t['correction_stage']]))
            env=StressEnvironment(t)
            for c in t['bootstrap_calls']:env.lookup(**c)
            for x in examples:
                a=parse_action(x['target'])
                if a['action']=='lookup':self.assertNotIn('error',env.lookup(a['entity'],a['field']))
                else:self.assertEqual(a['value'],t['expected'])
        self.assertEqual(counts,dict(unknown_field=80,unknown_entity=80))

    def test_injected_call_is_charged_and_limits_are_preserved(self):
        for t in self.data['recovery']:
            examples=iter(request_examples(t,PROTOCOL))
            def generate(messages,seed):
                x=next(examples);self.assertEqual(messages,x['messages'])
                return dict(text=x['target'],truncated=False,input_ids=[],output_ids=[],generation_seconds=0)
            r=run_episode(t,generate,PROTOCOL,7)
            self.assertTrue(r['success']);self.assertTrue(r['grounded_success']);self.assertEqual(r['calls'],5)
            self.assertEqual(r['reward_cost_gated'],.85);self.assertEqual(r['invalid_queries'],1)
            self.assertEqual(len(r['requests'])+len(t['bootstrap_calls']),6)
            self.assertEqual(r['correct_extra_calls'],1)

if __name__=='__main__':unittest.main()

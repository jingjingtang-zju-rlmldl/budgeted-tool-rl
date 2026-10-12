import json
import unittest
from collections import Counter
from toolrl.sft_data import make_splits, audit_splits, request_examples, encode_example
from toolrl.environment import oracle, verify


class FakeTokenizer:
    eos_token_id=999
    def apply_chat_template(self,messages,**kwargs):
        return json.dumps(messages)
    def __call__(self,text,**kwargs):
        return {'input_ids':list(text.encode())}


class SFTDataTests(unittest.TestCase):
    def test_splits_disjoint_facts_and_independent_owner_ids(self):
        splits=make_splits()
        self.assertEqual(splits,make_splits())
        audit=audit_splits(splits)
        self.assertFalse(audit['entity_overlap'])
        self.assertFalse(audit['private_value_overlap'])
        self.assertFalse(audit['owner_suffix_relation'])
        self.assertTrue(audit['held_out_test_templates'])
        self.assertEqual({k:len(v) for k,v in splits.items()},{'train':300,'dev':30,'test':60})
        self.assertEqual(Counter(t['kind'] for t in splits['train']),dict(direct=100,one_lookup=100,two_lookup=100))
        for rows in splits.values():
            for task in rows:
                self.assertTrue(verify(oracle(task)[0],task['expected']))

    def test_goal_pairs_do_not_cross_split(self):
        tasks=make_splits()['train']
        counts=Counter(t['fact_group'] for t in tasks)
        self.assertEqual(sum(n==2 for n in counts.values()),40)
        self.assertTrue(all(n in (1,2) for n in counts.values()))

    def test_second_lookup_supervision_sees_actual_owner_not_final_department(self):
        task=next(t for t in make_splits()['train'] if t['kind']=='two_lookup')
        rows=request_examples(task,'protocol')
        self.assertEqual(len(rows),3)
        self.assertEqual(len(rows[0]['messages']),2)
        second=rows[1]
        observed=json.loads(second['messages'][-1]['content'])
        target=json.loads(second['target'])
        self.assertEqual(target['entity'],observed['value'])
        self.assertEqual(target['field'],'department')
        self.assertNotIn(task['expected'],json.dumps(second['messages']))
        self.assertEqual(json.loads(rows[2]['messages'][-1]['content'])['value'],task['expected'])

    def test_six_hundred_masked_targets_and_eos(self):
        rows=[r for t in make_splits()['train'] for r in request_examples(t,'protocol')]
        self.assertEqual(len(rows),600)
        row=encode_example(rows[-1],FakeTokenizer(),max_context=4096)
        self.assertTrue(all(x==-100 for x in row['labels'][:row['prefix_tokens']]))
        self.assertEqual(row['labels'][row['prefix_tokens']:],row['input_ids'][row['prefix_tokens']:])
        self.assertEqual(row['labels'][-1],999)
        with self.assertRaises(ValueError):
            encode_example(rows[-1],FakeTokenizer(),max_context=1)


if __name__=='__main__':
    unittest.main()

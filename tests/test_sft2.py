import json
import unittest
from collections import Counter

from toolrl.sft2_data import build,request_examples,sets,TRAIN_COUNTS
from toolrl.stress_environment import oracle
from toolrl.environment import parse_action
from toolrl.stress_prompts import PROTOCOL


class SFT2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.data=build()

    def test_counts_reproducibility_and_fact_contrasts(self):
        self.assertEqual(self.data,build())
        self.assertEqual(Counter(t['kind'] for t in self.data['train']),TRAIN_COUNTS)
        self.assertEqual(len({t['fact_group'] for t in self.data['train']}),260)

    def test_all_splits_and_historical_exclusions(self):
        for name,rs in self.data.items():
            e,v=sets(rs)
            for other,os in self.data.items():
                if name!=other:
                    oe,ov=sets(os);self.assertFalse(e&oe);self.assertFalse(v&ov)
        e,v=sets(self.data['test'])
        regenerated=build(e,v)
        for rs in regenerated.values():
            ne,nv=sets(rs);self.assertFalse(e&ne);self.assertFalse(v&nv)

    def test_oracle_targets_single_actions_and_error_states(self):
        targets=[];error_targets=0
        for t in self.data['train']:
            oracle(t)
            for x in request_examples(t,PROTOCOL):
                a=parse_action(x['target']);targets.append(x)
                self.assertNotIn('Tool:',x['target'])
                if x['messages'][-1]['role']=='tool' and 'error' in json.loads(x['messages'][-1]['content']):
                    self.assertEqual(a['action'],'lookup');error_targets+=1
                self.assertFalse(any(m['role']=='assistant' and 'Tool:' in m['content'] for m in x['messages']))
        self.assertEqual(len(targets),990)
        self.assertEqual(error_targets,80)

    def test_signal_is_independent_composition_and_resettable(self):
        rs=self.data['signal'];self.assertEqual(len(rs),32)
        self.assertEqual(set(Counter(t['kind'] for t in rs).values()),{4})
        for t in rs:
            a,obs=oracle(t);b,again=oracle(t)
            self.assertEqual((a,obs),(b,again))
            self.assertLessEqual(t['minimum_calls'],5)
            if 'retry' in t['kind'] or 'redirect' in t['kind']:
                self.assertTrue(any('error' in o for o in obs))


if __name__=='__main__':unittest.main()

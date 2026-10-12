import unittest
from toolrl.prompt_study import NEUTRAL, PATH_SENTENCE, PROMPTS, paired, diagnostics
from toolrl.environment import SYSTEM_PROMPT
from toolrl.dataset import make_tasks


class PromptStudyTests(unittest.TestCase):
    def test_neutral_is_single_sentence_ablation(self):
        self.assertEqual(PROMPTS['original'],SYSTEM_PROMPT)
        self.assertIn(PATH_SENTENCE,SYSTEM_PROMPT)
        self.assertEqual(NEUTRAL,SYSTEM_PROMPT.replace(PATH_SENTENCE,'An owner is an employee ID.'))
        self.assertNotIn(PATH_SENTENCE,NEUTRAL)

    def test_balanced_demonstrations_contain_no_evaluation_entities(self):
        tasks=make_tasks(884201,(20,20,20),'p01')
        for task in tasks:
            for entity in task['database']:
                self.assertNotIn(entity,PROMPTS['balanced_examples'])
        self.assertTrue(PROMPTS['balanced_examples'].startswith(NEUTRAL))

    def test_pairing_uses_task_and_seed_not_position(self):
        def row(task,seed,success):
            return dict(task_id=task,seed=seed,success=success)
        left=[row('a',7,True),row('a',10007,False),row('b',7,False)]
        right=[row('b',7,False),row('a',10007,True),row('a',7,False)]
        self.assertEqual(paired(left,right),dict(paired_attempts=3,both_correct=0,gained=1,lost=1,both_wrong=1))
        with self.assertRaises(ValueError):
            paired(left,right[:2])

    def test_first_lookup_metric_rejects_wrong_entity(self):
        task=next(t for t in make_tasks() if t['kind']=='one_lookup')
        row=dict(task_id=task['task_id'],kind='one_lookup',template_id=0,minimum_calls=1,
                 success=False,strict_success=False,calls=1,actions=[{'action':'lookup','entity':'wrong','field':'budget'}],
                 requests=[{}],strict_valid_requests=1,stop_reason='answered')
        self.assertEqual(diagnostics([row],{task['task_id']:task})['correct_first_lookup'],0)


if __name__=='__main__':
    unittest.main()

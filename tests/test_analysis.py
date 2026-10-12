import unittest
from toolrl.analyze import summarize, failure_class


def episode(identifier, success, calls, minimum):
    return dict(task_id=identifier, success=success, strict_success=success, calls=calls, minimum_calls=minimum,
                generation_seconds=1., wall_seconds=1., generated_tokens=10, requests=[{}],
                stop_reason='answered', invalid_tool_calls=0, repeated_calls=0, kind='one_lookup')


class AnalysisTests(unittest.TestCase):
    def test_correctness_conditioned_cost_does_not_reward_early_failure(self):
        result=summarize([episode('a',True,1,1),episode('b',False,0,1)])
        self.assertEqual(result['success_rate'],.5)
        self.assertEqual(result['mean_calls'],.5)
        self.assertEqual(result['correct_mean_calls'],1.)
        self.assertEqual(result['correct_extra_calls'],0.)
        self.assertEqual(result['failure_classes'],{'answer_without_required_lookup':1})

    def test_duplicates_are_not_extra_trials(self):
        row=episode('same',True,1,1)
        with self.assertRaises(ValueError):
            summarize([row,row])


if __name__=='__main__':
    unittest.main()

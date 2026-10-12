import json
import unittest
import hashlib
from pathlib import Path
import tempfile

from toolrl.baseline import run_episode
from toolrl.dataset import make_tasks
from toolrl.environment import LookupEnvironment, oracle, parse_action, verify


class EnvironmentTests(unittest.TestCase):
    def test_written_dataset_byte_hash_matches_manifest_on_windows(self):
        from toolrl.dataset import write_tasks
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'tasks.jsonl'
            write_tasks(path,make_tasks(9,(1,1,1),'test'),9)
            manifest=json.loads(path.with_suffix('.manifest.json').read_text())
            self.assertEqual(manifest['sha256'],hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertNotIn(b'\r\n',path.read_bytes())

    def test_reproducible_and_balanced(self):
        from collections import Counter
        tasks = make_tasks()
        self.assertEqual(tasks, make_tasks())
        self.assertEqual(len({t['task_id'] for t in tasks}), 100)
        self.assertEqual(Counter(t['kind'] for t in tasks), {'direct': 34, 'one_lookup': 33, 'two_lookup': 33})

    def test_oracle_solves_all_tasks(self):
        for task in make_tasks():
            answer, observations = oracle(task)
            self.assertTrue(verify(answer, task['expected']))
            self.assertEqual(len(observations), task['minimum_calls'])

    def test_no_type_cues_in_entity_ids(self):
        for task in make_tasks():
            for entity in task['database']:
                self.assertNotIn('direct', entity)
                self.assertNotIn('one_lookup', entity)
                self.assertNotIn('two_lookup', entity)

    def test_distinct_calibration_entities(self):
        dev = {e for t in make_tasks() for e in t['database']}
        calibration = {e for t in make_tasks(707, (3, 3, 3), 'calibration') for e in t['database']}
        self.assertFalse(dev & calibration)

    def test_episode_isolation(self):
        first, second = make_tasks()[:2]
        entity = next(iter(first['database']))
        self.assertIn('error', LookupEnvironment(second).lookup(entity, 'budget'))

    def test_unknown_queries_count_toward_budget(self):
        env = LookupEnvironment(make_tasks()[0], 2)
        self.assertIn('error', env.lookup('missing', 'budget'))
        self.assertIn('error', env.lookup('missing', 'budget'))
        with self.assertRaisesRegex(ValueError, 'budget'):
            env.lookup('missing', 'budget')

    def test_verifier_numeric_and_string(self):
        self.assertTrue(verify('120.0', 120))
        self.assertFalse(verify('120 dollars', 120))
        self.assertFalse(verify(float('nan'), 120))
        self.assertFalse(verify(float('inf'), 120))
        self.assertFalse(verify('employee_a', 'employee_b'))
        self.assertTrue(verify(' employee_a ', 'employee_a'))

    def test_parser_rejects_invalid_protocol(self):
        for text in ('hello', '[]', '{"action":"answer"}', '{"action":"lookup","entity":"x"}',
                     '{"action":"answer","value":true}', '{"action":"answer","value":"x","extra":0}',
                     '```json\n{"action":"answer","value":1}\n```'):
            with self.assertRaises(ValueError):
                parse_action(text)

    def test_fence_normalization_is_narrow(self):
        fenced = '```json\n{"action":"answer","value":120}\n```'
        self.assertEqual(parse_action(fenced, allow_fence=True)['value'], 120)
        for malformed in ('Explanation\n'+fenced, fenced+'\nExplanation', '```json\n{bad}\n```'):
            with self.assertRaises(ValueError):
                parse_action(malformed, allow_fence=True)

    def test_normalized_success_does_not_count_as_strict_success(self):
        task=next(t for t in make_tasks() if t['kind']=='direct')
        def generate(messages, seed):
            return dict(text='```json\n'+json.dumps({'action':'answer','value':task['expected']})+'\n```',
                        output_ids=[1], input_ids=[2], truncated=False, generation_seconds=0.)
        result=run_episode(task, generate, 7, allow_fence=True)
        self.assertTrue(result['success'])
        self.assertFalse(result['strict_success'])
        self.assertEqual(result['fence_normalized_requests'],1)

    def test_successful_two_turn_chain(self):
        task = next(t for t in make_tasks() if t['kind'] == 'two_lookup')
        scripted = [{'action': 'lookup', **call} for call in task['oracle_calls']]
        scripted.append({'action': 'answer', 'value': task['expected']})
        def generate(messages, seed):
            action = scripted.pop(0)
            # This fake generator is a unit-test stub, never used by the model runner.
            return dict(text=json.dumps(action), output_ids=[1], input_ids=[2], truncated=False, generation_seconds=0.)
        result = run_episode(task, generate, 7)
        self.assertTrue(result['success'])
        self.assertEqual(result['calls'], 2)
        self.assertEqual(len(result['requests']), 3)
        self.assertEqual(sum(m['role'] == 'tool' for m in result['messages']), 2)

    def test_invalid_action_stops_without_tool_execution(self):
        task = make_tasks()[0]
        def generate(messages, seed):
            return dict(text='garbage', output_ids=[1], input_ids=[2], truncated=False, generation_seconds=0.)
        result = run_episode(task, generate, 7)
        self.assertFalse(result['success'])
        self.assertEqual(result['stop_reason'], 'invalid_action')
        self.assertEqual(result['calls'], 0)

    def test_incomplete_json_due_to_limit_is_truncation(self):
        task = make_tasks()[0]
        def generate(messages, seed):
            return dict(text='{', output_ids=[1], input_ids=[2], truncated=True, generation_seconds=0.)
        self.assertEqual(run_episode(task, generate, 7)['stop_reason'], 'generation_truncated')


if __name__ == '__main__':
    unittest.main()

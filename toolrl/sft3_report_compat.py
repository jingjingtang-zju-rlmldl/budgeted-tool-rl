"""Report-only compatibility fix; frozen training and inference stay unchanged."""
import argparse
import ast
import inspect
import json
from pathlib import Path

from . import sft2_report, sft3_report
from .sft3_experiment import validate, write, sha256, rows, read


def install_fix(p):
    original = sft2_report.metrics
    tree = ast.parse(inspect.getsource(original))
    changes = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'max':
            assert not node.keywords
            node.keywords.append(ast.keyword(arg='default', value=ast.Constant(value=0)))
            changes += 1
    assert changes == 1
    ast.fix_missing_locations(tree)
    namespace = dict(original.__globals__)
    exec(compile(tree, str(Path(__file__).resolve()), 'exec'), namespace)
    fixed = namespace['metrics']
    empty = fixed([])
    assert empty['attempts'] == empty['requests'] == empty['max_input_tokens'] == 0
    checked = 0
    for path in (Path(p['root']) / 'eval').glob('*/episodes.jsonl'):
        rs = rows(path)
        if rs:
            assert fixed(rs) == original(rs)
            checked += len(rs)
    sft3_report.metrics = fixed
    root = Path(p['root'])
    amendment = root / 'report-only-amendment.json'
    if amendment.exists():
        previous = read(amendment)
        assert previous['compat_source_sha256'] == sha256(Path(__file__))
    else:
        assert not list((root / 'eval').glob('signal-*/episodes.jsonl'))
        write(amendment, dict(
            recorded_at=sft3_report.now(), plan_sha256=sha256(root / 'plan.json'),
            compat_source_sha256=sha256(Path(__file__)),
            reason='Frozen metrics max() had no default for an empty injected-error subgroup in free-start evaluation.',
            change='Only report-time max_input_tokens for an empty subgroup defaults to zero; nonempty metrics are identical.',
            nonempty_episodes_checked=checked, empty_group_metrics=empty,
            training_inference_rewards_datasets_prompts_unchanged=True))
    return checked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['interim', 'report'])
    parser.add_argument('--root', type=Path, default=Path('results/sft-v3'))
    parser.add_argument('--model-info', type=Path, required=True)
    args = parser.parse_args()
    p = validate(args.root / 'plan.json')
    checked = install_fix(p)
    print(json.dumps(dict(event='report_only_compatibility', nonempty_episodes_checked=checked)), flush=True)
    if args.stage == 'interim':
        sft3_report.interim(p)
    else:
        sft3_report.report(p, args.model_info)


if __name__ == '__main__':
    main()

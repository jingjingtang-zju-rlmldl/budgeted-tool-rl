"""Refresh the portfolio README from the completed audited stress summary."""
import json
from pathlib import Path

root = Path(__file__).resolve().parent.parent
project = root / 'outputs' / 'budgeted-tool-rl'
result = json.loads((project / 'results/stress-v1/summary.json').read_text(encoding='utf-8'))
variants = result['aggregate']['test']
keys = list(variants)
names = {'base/examples': 'Base / dev-best examples',
         'base/protocol': 'Base / dev-best protocol',
         'sft/protocol': 'Existing SFT / short protocol',
         'sft/examples': 'Existing SFT / same examples'}
families = {'anchor_two': 'Two-hop anchor', 'chain_three': 'Three-hop',
            'chain_four': 'Four-hop', 'retry_two': 'Transient error + retry',
            'redirect_two': 'Stale ID + redirect', 'skip_owner': 'Owner provided',
            'skip_prefix': 'Chain prefix provided', 'given_answer': 'Answer provided'}
table = ['| Family | Minimum calls | ' + ' | '.join(names[k] for k in keys) + ' |',
         '|---|---:|' + '---:|' * len(keys)]
for family, label in families.items():
    denominator = variants[keys[0]]['by_kind'][family]['attempts']
    lower = variants[keys[0]]['by_kind'][family]['minimum_calls']
    table.append('| ' + label + f' | {lower} | ' + ' | '.join(
        f"{variants[k]['by_kind'][family]['successes']}/{denominator}" for k in keys) + ' |')
table.append('| **Overall** | — | ' + ' | '.join(
    f"**{variants[k]['successes']}/{variants[k]['attempts']} ({100*variants[k]['successes']/variants[k]['attempts']:.1f}%)**"
    for k in keys) + ' |')
section = f'''## Difficulty upgrade: frozen-model transfer test

The existing adapter and base model were evaluated locally on the RTX 5060,
with **no new training**. New tasks add three/four-hop queries, deterministic
retry and canonical-ID recovery, and supplied facts that make queries redundant.
Both models receive the full tool contract. The base prompt was selected from
two fixed candidates on a separate 24-scenario development set before testing.

{chr(10).join(table)}

Each policy has **64 test scenarios × three seeds = 192 attempts**. Scenarios
come from 24 fact groups with matched normal/fault/supplied-fact variants;
neither the scenarios nor repeated seeds are independent samples. Entities and
private values are disjoint from development and previous SFT data. Test wording
is held out from this experiment's development templates.

The matched example prompt makes the frozen SFT model substantially more
reliable than the short protocol. Remaining failures include simulated tool
transcripts; short-protocol failures also include null answers and unsupported
reverse-lookup arguments. Successful low-call trajectories are separated from
early failures, and required recovery is not counted as redundant repetition.
Prompt dependence should be addressed before claiming robust agent behavior.

**{result['total_verified_episodes']} real episodes / {result['total_verified_requests']} model requests** were
independently replayed and checked against recorded input/output tokens.
Peak PyTorch allocation was **{result['peak_allocated_mib']/1024:.2f} GiB**,
with **{result['peak_reserved_mib']/1024:.2f} GiB reserved**; these are inference measurements.
The original adapter files still match the frozen hashes. RL remains unrun;
the report makes a data-based decision about the next experiment.

Read the [Chinese stress-test analysis and RL decision](results/stress-v1/analysis.zh-CN.md),
[audited metrics and paired comparisons](results/stress-v1/summary.json),
[dev-only prompt selection](results/stress-v1/selection.json),
[actual diagnostic trajectories](results/stress-v1/diagnostic-cases.jsonl),
and [reproduction instructions](results/stress-v1/reproduce.md).

'''
path = project / 'README.md'
original = path.read_text(encoding='utf-8')
assert '## Difficulty upgrade' not in original
original = original.replace('## Local LoRA SFT\n', section + '## Previous local LoRA SFT\n', 1)
original = original.replace(
    'The next useful experiment is a fresh stress test before RL, because SFT neutral saturates the current test.',
    'This saturation motivated the separate difficulty upgrade reported above.')
original = original.replace(
    'It does not measure unseen tools, long planning or GRPO feasibility. Further stages should be chosen from new stress-test failures.',
    'The extended stress test adds controlled schema transfer, four-hop chains and recoverable faults. It still does not measure general external tools, open-ended planning or GRPO training feasibility.')
path.write_text(original, encoding='utf-8', newline='\n')
print('Updated README from the completed audited stress summary.')

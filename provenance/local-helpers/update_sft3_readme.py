from pathlib import Path
import json
root=Path(__file__).resolve().parents[1]/'outputs/budgeted-tool-rl'
s=json.loads((root/'results/sft-v3/summary.json').read_text(encoding='utf-8'))
assert s['total_verified_episodes']==1086 and not s['rl_optimization_performed']
before=s['paired_before'];after=s['paired_after'];t=s['aggregate'];g=s['signal_groups'];m=s['training']
decision=('Formal GRPO is deferred: the held-out diagnostic has sparse effective '
          'groups. The next step is fresh training facts with observation-dependent '
          'branching and legal query-cost choices, followed by another independent '
          'signal audit.' if g['nonconstant_reward_groups']<=8 else
          'These candidate groups require independent training-task signal and memory '
          'checks before an RL experiment.')
breakdown=json.loads((root/'results/sft-v3/descriptive-breakdown.json').read_text(encoding='utf-8'))
assert breakdown['paired_test_comparisons']['old/examples']['gained']==4
assert breakdown['paired_test_comparisons']['old/examples']['lost']==2
table=['| Fixed condition | Free-start test (72 tasks x 2) | Injected-error recovery (32 tasks x 2) |','|---|---:|---:|']
for c in t['test']:
    x=t['test'][c];y=t['recovery'][c]
    table.append(f"| {c} | {x['successes']}/{x['attempts']} ({100*x['successes']/x['attempts']:.1f}%) | {y['successes']}/{y['attempts']} ({100*y['successes']/y['attempts']:.1f}%) |")
section=f'''## Wording control and invalid-query correction SFT

A real **32-fact, two-wording, two-seed paired experiment** holds database,
faults, target, system prompt, budget and within-pair generation seeds fixed.
Only question wording and logging labels differ. With the v2 adapter,
wording A/B succeeds {before['by_wording']['0']['successes']}/64 and
{before['by_wording']['1']['successes']}/64 times; first-action project.employee
misuse occurs {before['first_project_employee_field']['0']}/64 and
{before['first_project_employee_field']['1']}/64 times respectively. The pair
analysis was saved before training. Repeated seeds are correlated; the
independent fact count is 32. Supplied-prefix tasks serve as controls.

Then v2 was continued on **300 new fact-disjoint trajectories / 1,095
assistant-only next-action targets**: 80 executed unknown_field prefixes,
80 executed unknown_entity prefixes, and 140 clean/chain/recoverable-error/
known-information/retained-budget examples. Bad prefixes are scripted,
actually executed, fully masked inputs; they are not supervised targets or
claims of naturally sampled model failures. Correction SFT also broadens
wording coverage, so its training components have not been separately ablated.

{chr(10).join(table)}

The free-start test holds out complete wording and facts. The injected-error
test starts from recorded tool state and charges the injected error to the
normal six-call/seven-turn budget; it measures conditional recovery rather
than spontaneous mistakes. Post-training A/B paired successes are
{after['by_wording']['0']['successes']}/64 and {after['by_wording']['1']['successes']}/64;
that is a diagnostic regression check, since those wording families are
included in training on different facts.

One new free-start fact fails under both roots because the model drops an
internal character while copying the project ID, then repeats that invalid
ID. This is a real residual regression, beyond the scripted type-word and
suffix-error prototypes. Relative to v2/examples, four matched attempts are
gained and two are lost; conditional recovery at 64/64 does not establish
universal unknown_entity correction. Recovery error types also use different
wording templates, so their score differences cannot isolate error-type
difficulty.

After the controls were analyzed, **32 entirely new tasks x four actual
rollouts** succeeded {s['signal']['successes']}/128 times. The unchanged gated
correctness/call-cost reward has **{g['nonconstant_reward_groups']}/32
nonconstant groups**: `{json.dumps(g['counts'])}`. This is reward diagnosis;
**no RL optimization or invented old-policy log probabilities** are involved.
Different-fact signal rounds are not a paired causal comparison.

{decision}

All **{s['total_verified_episodes']} episodes / {s['total_verified_model_requests']}
generation requests** were replayed and checked against actual tokens,
observations, answers, costs and frozen provenance. One epoch uses
{m['optimizer_steps']} optimizer steps with checkpointing and per-update
empty_cache; peak allocated/reserved memory is
{m['cuda_peak_allocated_mib']/1024:.2f}/{m['cuda_peak_reserved_mib']/1024:.2f} GiB.
These process metrics do not establish exclusive physical-VRAM residency or
general agent/RSI capability. Existing evidence and adapters remain unchanged.

Read the [Chinese analysis and next-step decision](results/sft-v3/analysis.zh-CN.md),
[paired-before-training evidence](results/sft-v3/paired-before-training.json),
[audited metrics](results/sft-v3/summary.json),
[fresh per-task rewards](results/sft-v3/reward-signal.json),
[training audit](results/sft-v3/training-audit.json),
[report-only empty-subgroup fix](results/sft-v3/report-only-amendment.json),
[per-family and error-position breakdown](results/sft-v3/descriptive-breakdown.zh-CN.md),
[reproduction](results/sft-v3/reproduce.md), and
[adapter model card](results/sft-v3/model-card.md).

![Wording control, corrections and fresh reward groups](results/sft-v3/sft3-results.png)

'''
p=root/'README.md';text=p.read_text(encoding='utf-8');anchor='## SFT continuation and fresh reward diagnostics';start='## Wording control and invalid-query correction SFT'
if start in text:text=text[:text.index(start)]+text[text.index(anchor):]
assert anchor in text
p.write_text(text.replace(anchor,section+anchor,1),encoding='utf-8',newline='\n')
print(json.dumps({'readme_updated':True,'episodes':s['total_verified_episodes'],'signal_groups':g['counts']}))

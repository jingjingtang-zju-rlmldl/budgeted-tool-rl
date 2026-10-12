"""Publish completed local metrics to the repository README, preserving history."""
from pathlib import Path
import json
root=Path(__file__).resolve().parents[1]/'outputs/budgeted-tool-rl'
r=json.loads((root/'results/sft-v2/summary.json').read_text(encoding='utf-8'))
assert r['total_verified_episodes']==1100 and not r['rl_optimization_performed']
t=r['aggregate']['test'];g=r['signal_groups'];s=r['aggregate']['signal']['new/protocol'];m=r['training']
names={'base/examples':'Base / fixed examples','old/protocol':'Old SFT / short protocol',
       'old/examples':'Old SFT / same examples','new/protocol':'Continued SFT / short protocol'}
table=['| Condition | Correct / attempts | Success | Strict JSON + correct | Extra calls on correct attempts |',
       '|---|---:|---:|---:|---:|']
for k,v in t.items():
    table.append(f"| {names[k]} | {v['successes']}/{v['attempts']} | {100*v['successes']/v['attempts']:.1f}% | {v['strict_successes']}/{v['attempts']} | {v['correct_extra_calls']} |")
section=f'''## SFT continuation and fresh reward diagnostics

The existing adapter was continued locally on **300 new executed oracle
trajectories / 990 assistant-only next-action targets**, including 80 states
immediately after actual recoverable errors. A fixed one-epoch continuation
used **124 optimizer steps**; the initial adapter state exactly matches v1.
Train/dev/test/signal and historical facts are disjoint. Test wording is held
out; training reuses earlier stress-test wording on new facts.

{chr(10).join(table)}

Each condition repeats **72 independent test tasks × three seeds = 216
attempts**. Repeated attempts are correlated. New-model all-three-correct
tasks: **{r['test_task_consistency']['new/protocol']['all_three_correct']}/72**.
Task families include three/four-hop chains, retries, canonical-ID redirects,
supplied facts and retained one-step budget queries. No action repair or
forced decoding is used. The base example prompt remains fixed from the
previous development selection.

After completing and analyzing all controls, the continued model sampled
**32 fresh tasks × four real rollouts = 128 attempts** under the short
protocol. These include long-chain/fault and supplied-prefix/fault compositions.
Answer success was **{s['successes']}/128**; **{g['nonconstant_reward_groups']}/32
groups** had nonconstant gated correctness/call-cost rewards. The four-reward
classification is `{json.dumps(g['counts'])}`. Constant-reward groups provide
no within-group reward ranking. **No RL optimization was performed.**

The fresh probe changes wording as well as fault composition. In 47 of the
48 failed attempts, the first action queries an illegal project field
`employee` instead of `owner`, then often fails to use the returned
`allowed_fields`. All 64 attempts that actually reach a configured recoverable
error succeed. Long-chain success separates by wording template, but the
strata use different facts, so this is an association rather than a paired
causal comparison. Before GRPO, run same-fact wording controls and add
executed invalid-query states with valid field/entity corrections to SFT;
preserve this test and signal set as historical evidence and generate new
training and final-evaluation tasks.

Final training used non-reentrant gradient checkpointing. Peak PyTorch
allocation was **{m['cuda_peak_allocated_mib']/1024:.2f} GiB**, with
**{m['cuda_peak_reserved_mib']/1024:.2f} GiB reserved**. The optimization loop
took {m['optimization_wall_seconds']:.2f} seconds under concurrent GPU load.
Three discarded resource attempts and a memory-only plan amendment are
preserved; no partial weights were reused. Reserved memory exceeded nominal
VRAM, so completion does not establish exclusive physical-VRAM residency or
RL training-memory feasibility on other systems.

All **{r['total_verified_episodes']} evaluation/diagnostic episodes /
{r['total_verified_model_requests']} model requests** were independently replayed,
regraded and checked against input/output tokens. Exact teacher labels,
single-epoch coverage, saved LoRA tensors and frozen provenance were audited.
The old checkpoint bytes remain unchanged. Results concern this controlled
synthetic English schema, not general agent or RSI capability.

Read the [Chinese continuation analysis and next-step decision](results/sft-v2/analysis.zh-CN.md),
[audited metrics and paired comparisons](results/sft-v2/summary.json),
[32 groups with actual rewards](results/sft-v2/reward-signal.json),
[training audit](results/sft-v2/training-audit.json),
[resource-plan amendment](results/sft-v2/resource-amendment.json),
[reproduction](results/sft-v2/reproduce.md), and
[adapter model card](results/sft-v2/model-card.md).

![Continuation SFT and reward-signal groups](results/sft-v2/sft2-results.png)

'''
path=root/'README.md';text=path.read_text(encoding='utf-8')
old='This repository implements the environment, actual Qwen3-0.6B CUDA baselines, paired prompt controls and a measured local LoRA SFT experiment. The environment is synthetic; model responses and optimizer updates are real. RL remains a future stage.'
new='This repository implements the environment, actual Qwen3-0.6B CUDA baselines, paired prompt controls, measured local LoRA SFT and continuation, and fresh rollout reward diagnostics. The environment is synthetic; model responses and optimizer updates are real. RL optimization remains a future stage.'
text=text.replace(old,new)
anchor='## Difficulty upgrade: frozen-model transfer test'
start='## SFT continuation and fresh reward diagnostics'
if start in text:
    text=text[:text.index(start)]+text[text.index(anchor):]
assert anchor in text
text=text.replace(anchor,section+anchor,1)
text=text.replace('It still does not measure general external tools, open-ended planning or GRPO training feasibility.',
    'The continued SFT and fresh reward probe add controlled compositional checks and within-task reward diagnostics. These still do not measure general external tools, open-ended planning or GRPO optimization-memory feasibility.')
path.write_text(text,encoding='utf-8',newline='\n')
print(json.dumps({'readme':str(path),'episodes':r['total_verified_episodes'],'signal_groups':g['counts']}))

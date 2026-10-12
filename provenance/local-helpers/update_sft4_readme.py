from pathlib import Path
import json

root=Path(__file__).resolve().parents[1]/'outputs/budgeted-tool-rl'
s=json.loads((root/'results/sft-v4/summary.json').read_text(encoding='utf-8'))
assert not s['rl_optimization_performed']
t=s['aggregate'];m=s['training'];d=s['training_set'];g=s['signal_groups']
table=['| Fixed condition | Grounded free-start (64 facts x 2) | Correct AND minimum cost | Injected ID recovery (32 x 2) |','|---|---:|---:|---:|']
for c,x in t['test'].items():
    r=t['recovery'][c];table.append(f"| {c} | {x['grounded_correct']}/{x['attempts']} | {x['optimal_weighted_correct']}/{x['attempts']} | {r['grounded_correct']}/{r['attempts']} |")
decision=('The predeclared practical candidate screen passes. Validate sampling on new training facts and measure actual RL memory before an optimizer run; the screen does not guarantee improvement.' if s['rl_candidate_screen_passed'] else
          'The predeclared practical candidate screen does not pass. Formal GRPO is deferred; use the per-family evidence to design fresh training tasks instead of tuning on these held-out diagnostic facts.')
amended=''
if 'copy_probe_amendment_sha256' in s:
    amended='''The original four normal-prompt collector cohorts (512 actual episodes)
yielded no eligible single-character copy failures. A separately registered
conditional training-only probe uses reserved training facts, a public
next-field hint, and temperature 1.2 / top-p .95. Its actual sampled copy
errors are supervised only after source replay; they are not a normal-prompt
deployment failure-rate estimate. Normal tests and final signal retain the
frozen .7 / .8 sampling. See the [collection amendment](results/sft-v4/copy-probe-amendment.json).
'''
section=f'''## Observable branches, query prices and student ID-copy recovery

The extended local environment reveals project.policy before ordinary
local/team/partner routes. A legal resolved_limit service returns the same
effective value directly, priced at the question-quoted 2 or 6 units;
ordinary queries cost one. Errors and repeats consume both calls and fees.
Supplied facts can shorten a route or remove every query.

The oracle reads only supplied facts and real observations, never private
policy or expected answers to choose actions. Correctness is source-grounded:
an unselected branch or lucky numeric guess cannot earn the primary reward.
The new reward is `grounded_correct * (1 - .02 * weighted_query_cost)` with
six calls / seven turns. Old call-count reward is an auxiliary. This is a
new benchmark, not a causal score comparison with v3.

{amended}
Continuation starts exactly from the saved v3 LoRA. The training set has
240 fresh clean trajectories plus **{d['natural_training_tasks']} distinct
actual student copy-error prefixes**, {d['decision_targets']} assistant-only
next-action targets. Copy-error types: `{json.dumps(d['natural_copy_types'])}`.
Bad assistant text is preserved verbatim and fully masked with observations;
only legal public-state continuation JSON and EOS are supervised.
All private facts are separated from development/test/recovery/signal and
historical data. Synthetic recovery probes are explicitly labeled, fully
crossing four edit types, four positions and two wordings.

{chr(10).join(table)}

The old model is v3, not the untrained base. Same-task/seed matched comparisons
are `{json.dumps(s['matched_test_comparisons'])}`. Only one training seed and
one final fixed epoch are used; task learning and error-state training have
not been separately ablated.

After controls were analyzed, fresh **32 facts x four real rollouts** achieved
{s['signal']['grounded_correct']}/128 grounded correctness and
{s['signal']['optimal_weighted_correct']}/128 optimal-cost correctness.
Reward groups: `{json.dumps(g['counts'])}`; **{g['nonconstant_reward_groups']}/32**
have nonconstant reward, spanning `{json.dumps(g['effective_kinds'])}`.
The practical screen was fixed before inference: at least eight effective
groups across at least three task families. {decision}
There is no RL optimizer or fabricated old-policy log probability.

Native RTX 5060 Laptop 8 GB, no cloud accelerator/API: {m['optimizer_steps']}
updates, {m['optimization_wall_seconds']/60:.2f} optimization minutes,
allocated/reserved peaks {m['cuda_peak_allocated_mib']/1024:.2f}/{m['cuda_peak_reserved_mib']/1024:.2f} GiB.
These process metrics are not exclusive physical-VRAM residency or throughput.
All **{s['total_verified_episodes']} actual episodes / {s['total_verified_model_requests']}
generation requests** pass token, tool, price, source-grounding and frozen
provenance replay; supervision, actual student prefixes and saved tensors
are independently audited.

[Chinese analysis](results/sft-v4/analysis.zh-CN.md) ·
[audited summary](results/sft-v4/summary.json) ·
[per-task rewards](results/sft-v4/reward-signal.json) ·
[training data provenance](results/sft-v4/training-set-manifest.json) ·
[training audit](results/sft-v4/training-audit.json) ·
[reproduction](results/sft-v4/reproduce.md) ·
[model card](results/sft-v4/model-card.md)

![Branch decisions, weighted cost and fresh reward groups](results/sft-v4/sft4-results.png)

'''
p=root/'README.md';text=p.read_text(encoding='utf-8')
start='## Observable branches, query prices and student ID-copy recovery';anchor='## Wording control and invalid-query correction SFT'
if start in text:text=text[:text.index(start)]+text[text.index(anchor):]
assert anchor in text
p.write_text(text.replace(anchor,section+anchor,1),encoding='utf-8',newline='\n')
card=f'''# Qwen3-0.6B branch / query-price continuation LoRA

Base Qwen/Qwen3-0.6B revision c1899de289a04d12100db370d81485cdf75e47ca.
Load the v4 adapter once onto the BF16 base; do not stack it with v3.
Frozen base, FP32 q/v LoRA rank8/alpha16/dropout.05. One final fixed epoch,
seed42, AdamW lr1e-4, microbatch1, accumulation8, non-reentrant checkpointing,
empty_cache after each optimizer update. No RL or external model API.

240 clean new facts and {d['natural_training_tasks']} selected actual student
copy-error prefixes form {d['decision_targets']} assistant-only targets.
All error-prefix assistant text and observations are masked input; only
legal public-state next-action JSON and EOS are supervised.

{amended.replace('results/sft-v4/copy-probe-amendment.json','copy-probe-amendment.json')}
This is a deterministic English company-record schema with observable
local/team/partner routes and query price quotes 2/6 versus ordinary unit1.
Grounded correctness and weighted query regret are different metrics.
Actual failures, paired regressions and prototype recovery must be considered
together; high synthetic recovery scores do not imply universal ID repair.
One training seed, finite fresh samples, no isolated training-component
ablation, no general agent or RSI capability claim.

Completed {m['optimizer_steps']} updates, longest sequence {m['max_sequence_tokens']}
tokens, max2048 without silent truncation. Allocated/reserved peaks
{m['cuda_peak_allocated_mib']/1024:.2f}/{m['cuda_peak_reserved_mib']/1024:.2f} GiB;
process metrics do not establish exclusive physical-VRAM residency.
Saved adapter_model.safetensors SHA-256:
{m['adapter_files_sha256']['adapter_model.safetensors']}

See analysis.zh-CN.md, summary.json, diagnostic-cases.jsonl,
training-set-manifest.json, training-audit.json and reward-signal.json.
'''
(root/'results/sft-v4/model-card.md').write_text(card,encoding='utf-8',newline='\n')
print(json.dumps(dict(readme_updated=True,episodes=s['total_verified_episodes'],reward_groups=g['counts'])))

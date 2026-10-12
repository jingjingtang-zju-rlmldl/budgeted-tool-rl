# Reproduce the frozen difficulty upgrade

Use the same local CUDA environment and pinned Qwen3-0.6B snapshot as the
SFT experiment. PEFT 0.17.1, Transformers 4.57.3 and PyTorch 2.9.1+cu128
remain unchanged. This experiment performs inference only: it loads the
existing 75-step adapter and freezes every model parameter. No API, cloud
GPU or additional training is used.

From the repository root:

```powershell
python -m unittest discover -s tests -v
python -m toolrl.stress_experiment prepare --root results/my-stress --data-root data/my-stress --model-info models/model-location.json
python -m toolrl.stress_experiment dev --root results/my-stress --model-info models/model-location.json
python -m toolrl.stress_experiment test --root results/my-stress --model-info models/model-location.json
python -m toolrl.stress_action_audit --root results/my-stress --model-info models/model-location.json
python -m toolrl.stress_report --root results/my-stress
```

Use the CUDA environment's Python for these commands. In the recorded local
run this executable was `work/sft-venv/Scripts/python.exe` outside the repo.
Model downloading and portable adapter loading are described in the
[SFT reproduction instructions](../sft-v1/reproduce.md) and
[adapter model card](../sft-v1/model-card.md). The adapter config's old local
backbone path is provenance; the runner explicitly loads the snapshot from
the model-location JSON before attaching the adapter.

Choose new output/data directories; existing plans and results must not be
overwritten. `prepare` freezes dataset, prompt, source, adapter and historical
data hashes. Historical exclusions affect generated entity/value draws, so
the audited historical inputs must also be preserved for exact data replay.
Recorded dev/test datasets remain the authoritative fixed evaluation sets.

## Task contract

The JSON action format is unchanged. The tool schema now additionally has
`department.cost_center` and `cost_center.spending_limit`. Both models receive
the complete new schema and the same error rules. A turn emits exactly one
lookup or answer; no forced decoding or action/argument repair is applied.
Only a full single Markdown code wrapper may be stripped, with strict raw
JSON validity separately recorded.

Each episode allows six tool calls and seven output turns. Every executed
query, including an error, consumes a call. Sampling remains temperature
0.7, top-p 0.8 and top-k 20, thinking off, batch size one, 64 new tokens per
action, four CPU threads and BF16 CUDA inference. The input ceiling is raised
to 4096 for the extended contract/examples; there is no silent truncation.
Oracle context preflight measured at most 698 tokens for the short protocol
and 1615 for the example prompt. Actual generated contexts are separately
recorded, and may be longer if the policy repeats or goes off path.

Eight families have three dev scenarios and eight test scenarios each:

| Family | Behavior being tested | Minimum calls |
|---|---|---:|
| anchor_two | project owner → employee department | 2 |
| chain_three | owner → department → cost_center | 3 |
| chain_four | owner → department → cost_center → spending_limit | 4 |
| retry_two | same two-hop goal; department query fails once, then succeeds | 3 |
| redirect_two | same goal; stale employee ID yields canonical_entity | 3 |
| skip_owner | owner already provided, only department is private | 1 |
| skip_prefix | department and cost center provided, only limit is private | 1 |
| given_answer | budget and distracting owner provided; answer directly | 0 |

Temporary failures are keyed to the designated query and occur once per
episode, not probabilistically or according to clock time. Alias lookups
always return `entity_moved`; querying the canonical ID reveals the actual
field. Replay therefore reproduces exactly the faults experienced by the
policy. An error that the model never reaches is not counted as an observed
recovery opportunity.

Required retry and redirect actions are tracked independently from
unnecessary repeated queries and queries of already provided facts. The
oracle minimum includes the unavoidable error call. Extra-call costs are
reported conditional on correctness; early failure with few calls is not
an efficiency gain. Exact answer correctness and tool-grounded correctness
are both recorded, so an accidental private-value guess is detectable.

## Pairing and selection

Anchor/retry/redirect/skip-owner/given-answer scenarios share one fact group;
four-hop/skip-prefix scenarios share another. Three-hop uses a third group.
All reuse is within a split. Thus dev has 24 scenarios from nine fact groups;
test has 64 scenarios from 24 groups. Dev, test and historical SFT data have
no overlapping entities/private values. Real and decoy IDs follow the same
formats; linked IDs have independent suffixes. Test wording is held out from
dev, while tool schema and task compositions are controlled.

Matched fact-group scenarios share `sampling_index`. The seed is root seed
plus 100 times that index, then plus turn. Model conditions use exactly the
same scenario/turn seeds. Test root seeds are 7, 10007 and 20007. These are
192 attempts per model/prompt over 64 scenarios, not 192 independent tasks;
the matched fact groups add further correlation.

Two base prompt candidates are frozen: the extended short protocol and
that same protocol with seven complete illustrative episodes. Their
demonstrations cover long chains, both errors and all supplied-fact cases.
The strongest base candidate is selected on dev by exact answer successes,
then strict successes, fewer extra calls in correct episodes, then shorter
prompt. The selected prompt and dev trace/verification hashes are saved in
`selection.json`. Test generation is blocked until that file exists and
matches the frozen plan. Test scores never select or revise a prompt.

Both SFT prompts are also diagnosed on dev, but do not determine base prompt
selection or update weights. Final test compares base with its dev-selected
prompt against existing SFT with the extended short protocol. If base selects
examples, SFT with those same examples is an additional matched-prompt
control. If base selects protocol, that matched control is already the
primary SFT condition and is not duplicated.

## Evidence and limits

Every condition stores exact model input/output token IDs, raw output,
actions, actual errors/results, known-fact redundancy and recovery accounting.
`stress_verify` independently rebuilds the chat inputs, decodes output IDs,
replays tools and regrades answer/cost/error metrics. This verifies the
record's consistency and deterministic execution; it does not independently
rerun generation to prove who produced the recorded tokens.

The final report audits all conditions, prompt selection, data/source hashes
and unchanged adapter files. This is a synthetic English transfer diagnosis
with new schema fields, not a general agent benchmark. Longer chains,
new fields, supplied facts and errors are distinct changes; family controls
help diagnose them but cannot isolate every causal factor or prove a model
capacity ceiling. Any later training/RL experiment needs separate train/dev
data and a fresh held-out final test, rather than repeatedly tuning against
this already inspected test set.

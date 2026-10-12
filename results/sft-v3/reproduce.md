# Reproduce the wording-control and correction experiment

Runtime: native Windows, RTX 5060 Laptop 8 GB, Python 3.13.5, PyTorch
2.9.1+cu128, Transformers 4.57.3, PEFT 0.17.1, Accelerate 1.10.1.
Qwen/Qwen3-0.6B revision c1899de289a04d12100db370d81485cdf75e47ca.
Use the local model snapshot described by model-location.json. No API or
cloud GPU is used. Base weights and the virtual environment are excluded
from the archive. Run commands from the repository directory.

Use new --root and --data-root directories for preparation; existing frozen
outputs must not be overwritten. Exact actual config, data/source/prompt and
old adapter hashes are in plan.json. Train/dev/test/recovery/pair/signal facts
are mutually disjoint and excluded from all historical datasets.

```powershell
python -m unittest discover -s tests -v
python -m toolrl.sft3_experiment prepare --model-info path/to/model-location.json
python -m toolrl.sft3_experiment paired --model-info path/to/model-location.json
$env:PYTORCH_CUDA_ALLOC_CONF='backend:native,garbage_collection_threshold:0.5'
python -m toolrl.sft3_train smoke --checkpointing --model-info path/to/model-location.json --out results/sft-v3/memory-probe
python -m toolrl.sft3_train train --checkpointing --model-info path/to/model-location.json --out results/sft-v3/training
Remove-Item Env:PYTORCH_CUDA_ALLOC_CONF
python -m toolrl.sft3_report audit --model-info path/to/model-location.json
python -m toolrl.sft3_experiment evaluate --model-info path/to/model-location.json
python -m toolrl.sft3_report_compat interim --model-info path/to/model-location.json
python -m toolrl.sft3_experiment signal --model-info path/to/model-location.json
python -m toolrl.sft3_report_compat report --model-info path/to/model-location.json
```

The frozen `evaluate` command completes and verifies all control cohorts,
then its original summary raises `ValueError: max() iterable argument is empty`
for the free-start test's empty injected-error subgroup. Continue with the
report-only `sft3_report_compat interim` command above. It adds a zero default
only to the empty subgroup's maximum input length, checks every nonempty
cohort against the original metrics, and records its source hash before
signal generation in `report-only-amendment.json`. Training, inference,
reward, data and all frozen execution source files remain unchanged.
Existing complete cohorts are reused, so this does not repeat GPU generation.

The paired stage must finish and save its analysis before either smoke or
training. The smoke updates are discarded; full training restarts from the
v2 adapter with a new optimizer. Non-reentrant gradient checkpointing and
per-update empty_cache are frozen choices based on the previous memory study.
Only legal next-action JSON and EOS have supervised labels; erroneous
assistant prefixes and all observations are masked. The erroneous queries
are script-constructed counterfactuals actually executed by the same tool,
not natural model rollouts.

The paired stage has 32 facts x two question wordings x two roots = 128
real episodes. Only the question and logging labels differ; sampling_index,
per-turn seeds, database, faults, given information, targets and system
prompt match within a pair. The second root reverses wording order.
After training the same pair set is evaluated again for regression, while
new complete wordings and facts are used for independent final tests.

Three fixed conditions (v2/protocol, v2/examples, v3/protocol) each run
26 development tasks once, 72 test tasks twice and 32 injected-error recovery
tasks twice. The recovery task initializes real tool state using recorded
assistant calls; injected errors count toward the six-call/seven-turn budget.
It measures response to a given error state, not spontaneous error frequency.

All final controls are analyzed before opening 32 new signal tasks with
four independent rollout roots. The old gated reward is preserved. There
are no policy-gradient updates or fabricated old-policy log probabilities.
Input/output tokens, actual tool observations, actions, costs and seeds are
logged and independently replayed. Repeated rollouts are not independent
fact tasks. This is a controlled synthetic English schema, not a benchmark
of general agent or RSI capability.

After the fixed report, optional descriptive per-family and error-position
breakdown (no model generation):

```powershell
python -m toolrl.sft3_breakdown
```

That source is a separate post-hoc descriptive producer; it records its own
SHA-256 and the already audited raw episode hashes. It is not part of the
frozen execution plan or checkpoint-selection logic.

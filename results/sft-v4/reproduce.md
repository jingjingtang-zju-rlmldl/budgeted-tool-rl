# Reproduce the branch / weighted-query-cost experiment

> Publication status: interrupted by the user before v4 training. These commands
> describe the planned workflow, not completed training or signal evidence.
> See [current status](status.zh-CN.md) and [stop record](interruption/user-stop.json).

Native Windows, RTX 5060 Laptop 8 GB. Actual runtime is recorded in each
manifest (Python 3.13.5, torch 2.9.1+cu128, Transformers 4.57.3, PEFT
0.17.1, Accelerate 1.10.1). Base model Qwen/Qwen3-0.6B, revision
c1899de289a04d12100db370d81485cdf75e47ca, local cache only.
No external model API or cloud GPU is used. Base weights and venv are not
packaged. Commands below run from the repository directory.

The delivered output is immutable evidence. For a rerun use new data/output
directories with `--data-root`/`--root` (and the corresponding `--plan` for
training/run). Do not overwrite historical datasets, plan or checkpoints.
Model-info JSON points to your own pinned local snapshot.

```powershell
python -m unittest discover -s tests -v
python -m toolrl.sft4_experiment prepare --model-info path/to/model-location.json
python -m toolrl.sft4_experiment collect --model-info path/to/model-location.json
python -m toolrl.sft4_experiment baseline --model-info path/to/model-location.json
python -m toolrl.sft4_experiment freeze-train --model-info path/to/model-location.json
$env:PYTORCH_CUDA_ALLOC_CONF='backend:native,garbage_collection_threshold:0.5'
python -m toolrl.sft4_train smoke --checkpointing --model-info path/to/model-location.json --out results/sft-v4/memory-probe
python -m toolrl.sft4_train train --checkpointing --model-info path/to/model-location.json --out results/sft-v4/training
Remove-Item Env:PYTORCH_CUDA_ALLOC_CONF
python -m toolrl.sft4_report audit --model-info path/to/model-location.json
python -m toolrl.sft4_experiment evaluate --model-info path/to/model-location.json
python -m toolrl.sft4_experiment signal --model-info path/to/model-location.json
python -m toolrl.sft4_report report --model-info path/to/model-location.json
```

The collector has a predeclared stop rule and maximum: primary 128 tasks,
then another cohort if fewer than eight distinct eligible actual ID-copy
failures, at most four cohorts on two disjoint 128-task collections. Order,
root seeds and extraction rule are fixed in plan.json. Only one eligible
prefix per fact, at most 80 selected. Final training set is 240 clean facts
plus the selected actual student-copy prefixes, frozen and separately hashed
before smoke or training. If there is no eligible actual copy error, the
pipeline stops instead of fabricating natural-error training data.

Bad assistant texts retain the exact original generated text. Actual tool
observations and error prefixes are masked input, not assistant targets.
Only legal next-action JSON and EOS are supervised. Labels come from the
public-state planner, which cannot access private fields or expected values
to choose actions. The real lookup environment alone reveals private values.

Branch correctness and query price are distinct. Ordinary queries cost one;
resolved_limit queries, including errors, cost the publicly quoted 2 or 6.
Every lookup consumes one of six calls, including errors/repeats; output
budget is seven turns including the answer. Successful source-grounded
reward is 1 - .02 * weighted cost, zero otherwise. Maximum cost 36 keeps
reward nonnegative. Old .03 * call-count reward is stored only as an auxiliary.
Different calls can have equal weighted cost; reward groups compare actual
weighted costs, not call counts. This benchmark differs from v3.

Baseline v3/protocol and v3/examples are evaluated before training on dev16
once, test64 twice and recovery32 twice. The new adapter/protocol uses the
same controls after one fixed final epoch. Recovery is a separate scripted
single-character test, fully crossing four edit types, four positions and
two phrasings; it does not measure spontaneous mistake frequency.

Only after all controls are analyzed is the fresh signal evaluated: 32
disjoint facts, novel complete phrasing, four actual roots. The predeclared
practical RL-candidate screen asks for >=8 nonconstant groups and >=3
distinct task families; this is a screening choice, not a universal sample
size threshold or a guarantee that RL will improve success.

Conditional training-only collection amendment is separately registered in
`copy-probe-amendment.json` before baseline evaluation. If all four original
collector cohorts contain zero eligible actual single-character copy facts,
the original freeze-train refuses to label synthetic data as natural.
Use the reserved training facts with a public next-field hint and temperature
1.2 / top-p .95 at roots 40027 and, if fewer than eight unique eligible facts,
50027. Test/signal settings remain .7 / .8. These probe samples cannot estimate
normal-prompt deployment error frequency. Source, data, original plan hashes
and trigger/stop rules are in the amendment. Original source files are intact.

For this additional layer, use a separate fresh repository/work directory
containing historical datasets/adapters and source, but not this round's
existing `data/sft-v4` / `results/sft-v4`. The layer uses these fixed default
paths; do not delete or overwrite the delivered evidence to rerun it.
Register the conditional rule after prepare/collection and **before**
baseline using `python -m toolrl.sft4_copy_rules prepare`. Then complete
baseline and, only for zero original eligible copy facts, run the commands
below instead of the ordinary freeze/train/evaluate/signal/report stages.

```powershell
python -m toolrl.sft4_copy_probe --model-info path/to/model-location.json --model old --prompt protocol --split copy_probe --seed 40027
# Run root 50027 only if the first probe did not reach eight eligible facts.
python -m toolrl.sft4_copy_probe --model-info path/to/model-location.json --model old --prompt protocol --split copy_probe --seed 50027
python -m toolrl.sft4_copy_rules dispatch toolrl.sft4_experiment freeze-train --model-info path/to/model-location.json
$env:PYTORCH_CUDA_ALLOC_CONF='backend:native,garbage_collection_threshold:0.5'
python -m toolrl.sft4_copy_rules dispatch toolrl.sft4_train smoke --checkpointing --model-info path/to/model-location.json --out results/sft-v4/memory-probe
python -m toolrl.sft4_copy_rules dispatch toolrl.sft4_train train --checkpointing --model-info path/to/model-location.json --out results/sft-v4/training
Remove-Item Env:PYTORCH_CUDA_ALLOC_CONF
python -m toolrl.sft4_copy_rules dispatch toolrl.sft4_report audit --model-info path/to/model-location.json
python -m toolrl.sft4_copy_rules dispatch toolrl.sft4_experiment evaluate --model-info path/to/model-location.json
python -m toolrl.sft4_copy_rules dispatch toolrl.sft4_experiment signal --model-info path/to/model-location.json
python -m toolrl.sft4_copy_rules dispatch toolrl.sft4_report report --model-info path/to/model-location.json
```

The dispatch layer reconstructs selected prefixes from all declared actual
collector sources and adds independent token/tool/cost replay for the probe
cohorts. It does not change normal evaluation model generation or reward.

Independent replay reconstructs inputs, decodes actual generated tokens,
checks model EOS/truncation, real lookup states/prices and grounded answers.
Training audit reconstructs the selected actual student prefixes and exact
supervised tokens/masks/EOS, coverage, gradients, starting/saved LoRA hashes.
There is no RL optimizer update and no invented old-policy log probability.

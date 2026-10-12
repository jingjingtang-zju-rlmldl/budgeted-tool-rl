# Reproduce SFT continuation and the fresh 32 × 4 signal probe

Run from the repository root. Preserve the included evidence directories; use
new roots for a rerun. This experiment uses native Windows Python 3.13.5,
PyTorch 2.9.1+cu128, Transformers 4.57.3, huggingface-hub 0.36.2,
PEFT 0.17.1, Accelerate 1.10.1 and matplotlib 3.10.0. The GPU is an
RTX 5060 Laptop with 8 GB nominal VRAM. No cloud GPU or model API is required.

## Runtime

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python -m pip install transformers==4.57.3 huggingface-hub==0.36.2 peft==0.17.1 accelerate==1.10.1 matplotlib==3.10.0
.venv\Scripts\python -m unittest discover -s tests -v
.venv\Scripts\python -m toolrl.download --cache models --out models/model-location.json
```

The backbone revision is `c1899de289a04d12100db370d81485cdf75e47ca`
of Qwen/Qwen3-0.6B. Base weights and the virtual environment are excluded.
The checked local invocation used the already configured
`../../work/sft-venv/Scripts/python.exe` and `../../work/model-location.json`.
Another machine should use its own Python and model-location file.

## Stages

```powershell
.venv\Scripts\python -m toolrl.sft2_experiment prepare --root results/my-sft2 --data-root data/my-sft2 --model-info models/model-location.json
$env:PYTORCH_CUDA_ALLOC_CONF = 'backend:native,garbage_collection_threshold:0.5'
.venv\Scripts\python -m toolrl.sft2_train smoke --checkpointing --plan results/my-sft2/plan.json --out results/my-sft2/memory-probe --model-info models/model-location.json
.venv\Scripts\python -m toolrl.sft2_train train --checkpointing --plan results/my-sft2/plan.json --out results/my-sft2/training --model-info models/model-location.json
$env:PYTORCH_CUDA_ALLOC_CONF = ''
.venv\Scripts\python -m toolrl.sft2_experiment evaluate --root results/my-sft2 --model-info models/model-location.json
.venv\Scripts\python -m toolrl.sft2_experiment signal --root results/my-sft2 --model-info models/model-location.json
.venv\Scripts\python -m toolrl.sft2_action_audit --root results/my-sft2 --model-info models/model-location.json
.venv\Scripts\python -m toolrl.sft2_report --root results/my-sft2 --model-info models/model-location.json
```

Preparation requires the included v1 checkpoint, its manifest, and the previous
stress prompt-selection record. All continuation runs initialize from that
same old adapter, not a newly initialized LoRA. The eight-step smoke probe
uses the longest contexts and is discarded; full training reloads the old
adapter and a fresh optimizer. These commands choose the final measured
non-reentrant checkpointing configuration directly for a new run. The original
preparation logic records an OOM-only fallback; the checked run explicitly
expanded that resource trigger before evaluation, as detailed below. Never
replace completed evidence. Editing a frozen source, prompt or dataset
requires a new plan.

The checked run initially used the default native CUDA cache. Under concurrent
GPU workloads this retained about 7.1 GiB and slowed training. That attempt
was actively stopped after 11 logged updates, preserved in
`training-attempt-default-cache`, and produced no saved adapter. Full training
then restarted with unused-cache collection. This second attempt was stopped
after 37 logged updates (`training-attempt-gc-cache`). A third attempt with
maximum split size 64 MB was stopped after two logged updates because reserved
memory grew to 15.39 GiB (`training-attempt-split-cache`). No partial attempt
saved an adapter, and no partial weights or optimizer state were reused.

The resource fallback was then expanded from only CUDA OOM to OOM or severe
memory pressure / WDDM paging. `plan-original.json` retains the exact original
plan; only `memory_test` fields changed in `plan.json`. The amendment, all
three discarded attempts, timestamps, hashes and final allocator environment
are documented in `resource-amendment.json`, `resource-retry.json` and
`resource-retry-first.json`. No evaluation had started. A checkpointing probe
completed eight actual updates in `memory-probe-checkpointing`; its weights
were discarded. Full training then freshly reloaded v1 with `--checkpointing`
and native garbage collection threshold 0.5, without a split-size cap.
Sources, data, prompts, seeds, learning rate, total steps and fixed final
checkpoint rule were unchanged. The audit compares original and current
plans and rejects any non-memory change. Other GPU processes were left
running. This is a documented resource-plan deviation, not an entirely
unchanged preregistered run. Performance also depends on changing concurrent
load; do not attribute all timing changes to allocator settings. See the
[PyTorch 2.9 allocator documentation](https://docs.pytorch.org/docs/2.9/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf).

The generator deterministically creates 300 train, 27 dev, 72 test and 32
signal tasks. It excludes entities, alias identifiers and private answers from
all prior task files, and isolates all four new splits. Forty paired goals
occur within train only (260 train fact groups); dev, test and signal each
use independent facts. The task-generator seed is fixed. A new preparation
after this run treats the included sft-v2 tasks as historical and therefore
uses new facts; it does not reproduce their exact bytes. Use the included
sealed datasets to audit this run's recorded data.

Train contains 100 three/four-hop tasks, 80 two-hop recovery tasks, 60
supplied-information tasks and 60 older goal tasks. Oracle trajectories
execute the real environment. The 990 targets each contain one next JSON
action followed by EOS, conditioned on the preceding actual observations.
Only those next-action tokens and EOS have labels; every prefix token is
masked to -100. There is no fabricated tool transcript, packing or silent
truncation. The longest actual sequence is 715 tokens; the ceiling is 1024.
There are 80 next states immediately after actual errors.

One epoch uses all 990 decision states exactly once, with seed 42, microbatch
1, accumulation 8 and 124 optimizer steps (the final group has six states).
The objective averages token cross entropy within each assistant action,
then averages across the accumulation group. Long trajectories contribute
more decision states. AdamW lr=1e-4, weight decay=0, clip=1; BF16 frozen
backbone and 1,146,880 FP32 q/v LoRA parameters, r=8, alpha=16, dropout=0.05.
Only the fixed final checkpoint is evaluated. Teacher-forced dev loss does
not select a checkpoint or replace free-rollout evaluation.

## Paired evaluation and diagnosis

Four conditions are fixed: base/examples, old/protocol, old/examples and
new/protocol. The base example prompt was selected on the previous stress
dev set, not on these test results. Each runs 27 dev tasks with seed 7, then
72 test tasks with seeds 7, 10007, 20007 (216 attempts per condition).
Condition order reverses for the middle seed. All test generation is gated
on the saved final adapter, even for base controls.

Only after all 12 test cohorts are complete and verified does the signal
stage sample the new model with the short protocol: 32 fresh tasks, roots
30007, 40007, 50007, 60007, exactly four rollouts per task. Eight families
have four independent tasks each: three/four-hop normal chains,
three/four-hop retry, three/four-hop redirect, prefix+retry and prefix+redirect.
Fault-plus-long-chain compositions were not explicit training families.
Signal tasks were generated and sealed before training; outcomes do not
change their count or difficulty. No policy optimization takes place.

All inference uses BF16 SDPA, batch 1, thinking off, temperature 0.7,
top-p 0.8, top-k 20, max input 4096, max output 64 tokens/action, six tool
calls and seven output turns. Per-turn seed = root + 100 × sampling_index
+ turn, matched across test conditions. Narrow whole-output code-fence
normalization is allowed; raw JSON compliance is graded separately. No
forced decoding, action repair or transcript extraction is performed.

The diagnostic reward remains `exact_correct * (1 - 0.03 * calls)`.
Failures score zero; error calls count toward cost. Correctness grounded in
provided facts or real tool observations is audited separately. A group with
four failures or four equal-cost successes has constant reward; it is not
evidence of a useful within-group ranking. Group variances from G=4 are
finite-sample observations. No old-policy log-probabilities are fabricated
and no gradient update is made from these inference traces.

## Evidence

`plan.json` freezes model revision, old adapter bytes and state hash,
historical and new datasets, prompts, source files, seeds and budgets.
`training/supervision.jsonl` stores exact tokens and labels;
`training/steps.jsonl` stores all request IDs, losses, gradient norms and
memory; `training/manifest.json` stores dtypes, initial/final states and
saved checkpoint hashes. `training-audit.json` independently rebuilds the
990 labels and verifies single-epoch coverage and continuation provenance.

Each inference cohort retains raw outputs, exact input/output tokens,
actions, actual observations, timing and deterministic grading. A separate
verifier replays tools, rebuilds inputs, decodes outputs, regrades rewards,
and checks the configured EOS against truncation. This checks record
consistency, not an independent re-generation of model output. Completed
cohorts can be reused; incomplete ones resume only with matching provenance.
The report validates frozen hashes across all cohorts and the final adapter.

`summary.json` contains paired comparisons and all per-family counts;
`reward-signal.json` contains all 32 groups with four rewards, call counts
and population variances. `diagnostic-cases.jsonl` retains failures and
extra-cost cases, including raw outputs. See the Chinese analysis for the
actual outcomes and next-step decision. Only one training seed was used,
and these controlled synthetic tool results do not establish general agent
or RSI capability. Kernel/runtime differences may change sampled outputs.

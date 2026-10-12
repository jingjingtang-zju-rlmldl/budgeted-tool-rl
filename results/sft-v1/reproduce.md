# Reproduce the local LoRA SFT experiment

Run commands from the repository root. Recorded results are immutable evidence;
choose new output and data directories for a new run. The checked runtime was
native Windows, Python 3.13.5, CUDA 12.8 / PyTorch 2.9.1+cu128, Transformers
4.57.3, PEFT 0.17.1 and Accelerate 1.10.1. The measured machine had an RTX 5060
Laptop GPU with 8 GiB nominal VRAM. Only local CUDA was used.

## Runtime

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python -m pip install transformers==4.57.3 huggingface-hub==0.36.2 peft==0.17.1 accelerate==1.10.1 matplotlib==3.10.0
.venv\Scripts\python -m unittest discover -s tests -v
.venv\Scripts\python -m toolrl.download --cache models --out models/model-location.json
```

The actual local run created `work/sft-venv` with `--system-site-packages` to
reuse the already validated CUDA runtime and installed PEFT only in that venv.
The model snapshot is Qwen/Qwen3-0.6B revision
`c1899de289a04d12100db370d81485cdf75e47ca`. The model-location JSON records its
local snapshot path; base weights are deliberately excluded from the package.

## Stages

```powershell
.venv\Scripts\python -m toolrl.sft_experiment prepare --root results/my-sft --data-root data/my-sft --model-info models/model-location.json
.venv\Scripts\python -m toolrl.sft_experiment baseline-dev --root results/my-sft --model-info models/model-location.json
.venv\Scripts\python -m toolrl.sft_experiment train --root results/my-sft --model-info models/model-location.json
.venv\Scripts\python -m toolrl.sft_experiment evaluate-final --root results/my-sft --model-info models/model-location.json
.venv\Scripts\python -m toolrl.sft_report --root results/my-sft --model-info models/model-location.json
```

Preparation writes 300 train, 30 dev and 60 test tasks with disjoint entities
and private values. Test uses held-out English paraphrases of the same task
families. Forty within-train project fact groups occur under both budget and
owner-department goals; this reuse never crosses a split. All three task
families have equal episode counts within a split.

Source/data/prompt hashes are frozen before baseline or training. Editing a
frozen implementation requires a new experiment, rather than replacing the
checked results. No source amendment was needed in the recorded run.

`baseline-dev` reruns both frozen prompts on the new 30-task development set.
`train` first performs eight real optimizer steps on the eight longest train
decision contexts, each repeated eight times, and discards that adapter.
Only CUDA OOM permits an automatic retry with non-reentrant gradient
checkpointing. Full training then reloads the base model and initializes a
fresh adapter. No smoke weight or optimizer state is reused.

The 300 oracle trajectories produce 600 separately conditioned next-action
targets (100 / 200 / 300 for direct / one-hop / two-hop). Each input includes
only the question and preceding actual oracle tool observations. Labels mask
the entire prefix with `-100`; only the next compact JSON action and EOS are
supervised. There is no packing or silent truncation. Actual maximum training
sequence length was 406 tokens, despite a configured ceiling of 768.

The fixed training configuration is BF16 frozen backbone, FP32 q/v LoRA
parameters, rank 8, alpha 16, dropout 0.05, microbatch 1, gradient accumulation
8, AdamW learning rate 1e-4, no weight decay, gradient clipping 1.0 and one
epoch. The loss is mean assistant-token cross entropy per decision request,
averaged across the accumulation group. This weights decision states equally;
longer trajectories contribute more states. All 600 states are shuffled once
with seed 42: 75 optimizer steps, no checkpoint selection from test scores.

`evaluate-final` evaluates SFT on dev, then evaluates base/SFT × neutral/system
examples on 60 test tasks with seeds 7, 10007 and 20007. Test generation is
gated until the final adapter has been saved and hashed, including for base
controls. The unchanged inference engine uses temperature 0.7, top-p 0.8,
top-k 20, thinking disabled, maximum 64 new tokens/action, 1024 input tokens,
three lookup calls and four turns. Task/turn seeds match across conditions.
No forced tool decoding, JSON repair or missing-action completion is used.

Evaluation can resume an incomplete condition with matching provenance.
Training has no silent resume/overwrite. Kernel differences may change
sampling; a rerun does not promise bitwise-identical outcomes.

## Evidence

`training/supervision.jsonl` records the 600 exact masked token sequences.
`training/steps.jsonl` records request IDs, loss, gradient norm and CUDA peaks
for all 75 steps. `training/manifest.json` records actual adapter changes,
parameter dtypes, memory, runtime, versions and saved checkpoint hashes.
`training-audit.json` reconstructs the labels, verifies complete exposure and
matches the saved safetensors to the post-training adapter-state hash.

Every evaluation condition preserves raw output, input/output tokens,
actions, observations and deterministic grading. Its `verification.json`
independently replays tools, regrades answers, rebuilds input token IDs and
decodes output IDs. This verifies recorded provenance and environment
execution; it does not independently rerun generation to prove the model
produced the recorded tokens. The cross-condition report also checks all
configuration, dataset, prompt, source and adapter hashes.

Per-condition `summary.json` comes from the retained legacy inference
analyzer. Its generic caveats describe the inference process and should not
be read as saying the SFT model was never trained or the new test is a dev
set. Use the experiment-level summary and Chinese report for the actual
training, split and repeated-seed interpretation.

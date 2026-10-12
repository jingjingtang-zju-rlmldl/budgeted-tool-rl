# Qwen3-0.6B invalid-query correction LoRA

This adapter continues the saved sft-v2 adapter on the frozen sft-v3 data.
It is an experimental local portfolio artifact for a deterministic synthetic
English company-record lookup environment, not a generally validated tool
agent or RSI system.

Base: Qwen/Qwen3-0.6B, revision
c1899de289a04d12100db370d81485cdf75e47ca. The BF16 base is frozen; only
FP32 q_proj/v_proj LoRA tensors are trainable (rank 8, alpha 16, dropout 0.05).
Load the v3 adapter once onto the base model; do not stack v2 and v3 adapters.

Training uses 300 new fact-disjoint trajectories: 80 executed unknown_field
prefixes, 80 executed unknown_entity prefixes, and 140 clean/chain/recoverable
error/known-information/retained-budget trajectories. Error queries are
script-generated and executed by the actual deterministic tool. They are
fully masked inputs, not targets or claims of spontaneous model errors.
Only legal next-action JSON and EOS are supervised. Training additionally
covers wording families identified in the paired diagnostic. Their effects
are not isolated from error-state supervision by a training ablation.

One fixed epoch uses 1,095 assistant targets, microbatch 1, accumulation 8,
AdamW lr 1e-4, seed 42, clipping 1, no weight decay, SDPA BF16 autocast,
non-reentrant gradient checkpointing and per-update empty_cache. The final
checkpoint is fixed in advance, not chosen on test or fresh signal scores.
See training/manifest.json and training-audit.json for measured resources,
coverage, starting/saved tensor hashes, loss and update evidence.

Evaluation includes same-fact wording pairs, new-fact/new-phrasing free-start
tests, explicitly injected illegal-query state tests, and 32 fresh reward
signal tasks with four real rollouts. These measure different behaviors and
must not be merged into one benchmark score. Repeated seeds are correlated
within each fact task. There is one training seed and no policy-gradient
training, external model API, or cloud accelerator in this run.

Performance and actual failure cases are in analysis.zh-CN.md, summary.json,
diagnostic-cases.jsonl and reward-signal.json. The existing gated correctness
reward and fixed six-call/seven-turn budget are preserved. Token/source/data/
prompt/adapter hashes and replay checks establish record consistency, not
universal capability or safety.

Known limitation: one free-start test fact fails under both evaluation roots
because the model deletes an internal character in the project ID and repeats
that unknown entity. The scripted type-word/suffix error tests do not cover
all natural ID-copying failures. In the recovery set, unknown_field and
unknown_entity also use different wording templates; between-error score
differences do not isolate error-type difficulty. See
descriptive-breakdown.zh-CN.md for matched gains/losses and actual failures.

Report-only empty-subgroup compatibility is documented in
report-only-amendment.json; frozen training and inference source stay intact.

Measured completed training: 137 optimizer steps; peak allocated/reserved 2.55/7.35 GiB.

Saved adapter_model.safetensors SHA-256:
978026c255400dd0bdaaeeeb8a44fed5bb92e6f5040a913851e7d46911eac820

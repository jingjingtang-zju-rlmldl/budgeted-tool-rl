# Real-model baseline results

Model: `Qwen/Qwen3-0.6B` at `c1899de289a04d12100db370d81485cdf75e47ca`.
GPU: NVIDIA GeForce RTX 5060 Laptop GPU; dtype: BF16; thinking disabled; batch size 1.

**Actual CUDA inference on synthetic development tasks. No SFT or RL was run.**

| Task family | Correct / total | Success | Strict success | Mean calls | Calls on correct answers | Correct extra calls |
|---|---:|---:|---:|---:|---:|---:|
| overall | 58 / 60 | 96.7% | 96.7% | 0.967 | 1.000 | 0.000 |
| direct | 20 / 20 | 100.0% | 100.0% | 0.000 | 0.000 | 0.000 |
| one_lookup | 18 / 20 | 90.0% | 90.0% | 0.900 | 1.000 | 0.000 |
| two_lookup | 20 / 20 | 100.0% | 100.0% | 2.000 | 2.000 | 0.000 |

Parser: `fenced`. Strict-valid requests: 118 / 118; fenced wrappers normalized: 0.
Fence normalization removes only a single complete Markdown wrapper; it never fixes JSON, arguments or answers.

## Hardware and throughput

- Model parameters in memory: 1141.2 MiB.
- Peak CUDA allocated: 1342.4 MiB; peak reserved: 1492.0 MiB.
- Generated tokens: 2318; summed generation time: 134.44 seconds.
- Generation throughput: 17.24 tokens/s, including prefill and sampling, excluding model load.
- Mean task wall time: 2.25 seconds.

## Stops

```json
{
  "answered": 60
}
```

## Representative failures

### test_0028 (one_lookup)

Question: What number is saved in the budget field of project_e515ae37d16f?
Expected: `6276`; answer: `51578437`; stop: `answered`.

```json
{
  "actions": [
    {
      "action": "answer",
      "value": 51578437
    }
  ],
  "observations": [],
  "raw_outputs": [
    "{\"action\":\"answer\",\"value\":51578437}"
  ]
}
```

### test_0025 (one_lookup)

Question: What number is saved in the budget field of project_36317d4e5363?
Expected: `1226`; answer: `875`; stop: `answered`.

```json
{
  "actions": [
    {
      "action": "answer",
      "value": 875
    }
  ],
  "observations": [],
  "raw_outputs": [
    "{\"action\":\"answer\",\"value\":875}"
  ]
}
```

## Interpretation limits

- Real CUDA model inference on synthetic English tasks; no training performed.
- One sampled attempt per task; observed success is not a repeated-trial pass@k study.
- Development tasks used for analysis; future final test must be independently generated.
- Wilson intervals are descriptive under binomial assumptions; controlled templates are not IID real-world tasks.
- PyTorch memory measures this process; nvidia-smi includes other apps and WDDM accounting.
- Inference feasibility does not establish backward/optimizer or GRPO training feasibility.

# Real-model baseline results

Model: `Qwen/Qwen3-0.6B` at `c1899de289a04d12100db370d81485cdf75e47ca`.
GPU: NVIDIA GeForce RTX 5060 Laptop GPU; dtype: BF16; thinking disabled; batch size 1.

**Actual CUDA inference on synthetic development tasks. No SFT or RL was run.**

| Task family | Correct / total | Success | Strict success | Mean calls | Calls on correct answers | Correct extra calls |
|---|---:|---:|---:|---:|---:|---:|
| overall | 56 / 60 | 93.3% | 93.3% | 0.933 | 1.000 | 0.000 |
| direct | 20 / 20 | 100.0% | 100.0% | 0.000 | 0.000 | 0.000 |
| one_lookup | 16 / 20 | 80.0% | 80.0% | 0.800 | 1.000 | 0.000 |
| two_lookup | 20 / 20 | 100.0% | 100.0% | 2.000 | 2.000 | 0.000 |

Parser: `fenced`. Strict-valid requests: 116 / 116; fenced wrappers normalized: 0.
Fence normalization removes only a single complete Markdown wrapper; it never fixes JSON, arguments or answers.

## Hardware and throughput

- Model parameters in memory: 1141.2 MiB.
- Peak CUDA allocated: 1342.4 MiB; peak reserved: 1492.0 MiB.
- Generated tokens: 2276; summed generation time: 113.68 seconds.
- Generation throughput: 20.02 tokens/s, including prefill and sampling, excluding model load.
- Mean task wall time: 1.90 seconds.

## Stops

```json
{
  "answered": 60
}
```

## Representative failures

### test_0027 (one_lookup)

Question: Using the company records, report the budget for project_646253650f41.
Expected: `2288`; answer: `1234567890`; stop: `answered`.

```json
{
  "actions": [
    {
      "action": "answer",
      "value": 1234567890
    }
  ],
  "observations": [],
  "raw_outputs": [
    "{\"action\":\"answer\",\"value\":1234567890}"
  ]
}
```

### test_0039 (one_lookup)

Question: Using the company records, report the budget for project_8f92a1491e0e.
Expected: `1690`; answer: `875`; stop: `answered`.

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

### test_0028 (one_lookup)

Question: What number is saved in the budget field of project_e515ae37d16f?
Expected: `6276`; answer: `51570815`; stop: `answered`.

```json
{
  "actions": [
    {
      "action": "answer",
      "value": 51570815
    }
  ],
  "observations": [],
  "raw_outputs": [
    "{\"action\":\"answer\",\"value\":51570815}"
  ]
}
```

### test_0025 (one_lookup)

Question: What number is saved in the budget field of project_36317d4e5363?
Expected: `1226`; answer: `1234567890`; stop: `answered`.

```json
{
  "actions": [
    {
      "action": "answer",
      "value": 1234567890
    }
  ],
  "observations": [],
  "raw_outputs": [
    "{\"action\":\"answer\",\"value\":1234567890}"
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

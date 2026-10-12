# Real-model baseline results

Model: `Qwen/Qwen3-0.6B` at `c1899de289a04d12100db370d81485cdf75e47ca`.
GPU: NVIDIA GeForce RTX 5060 Laptop GPU; dtype: BF16; thinking disabled; batch size 1.

**Actual CUDA inference on synthetic development tasks. No SFT or RL was run.**

| Task family | Correct / total | Success | Strict success | Mean calls | Calls on correct answers | Correct extra calls |
|---|---:|---:|---:|---:|---:|---:|
| overall | 20 / 60 | 33.3% | 33.3% | 0.000 | 0.000 | 0.000 |
| direct | 20 / 20 | 100.0% | 100.0% | 0.000 | 0.000 | 0.000 |
| one_lookup | 0 / 20 | 0.0% | 0.0% | 0.000 | n/a | n/a |
| two_lookup | 0 / 20 | 0.0% | 0.0% | 0.000 | n/a | n/a |

Parser: `fenced`. Strict-valid requests: 53 / 60; fenced wrappers normalized: 0.
Fence normalization removes only a single complete Markdown wrapper; it never fixes JSON, arguments or answers.

## Hardware and throughput

- Model parameters in memory: 1136.9 MiB.
- Peak CUDA allocated: 1288.0 MiB; peak reserved: 1348.0 MiB.
- Generated tokens: 906; summed generation time: 60.41 seconds.
- Generation throughput: 15.00 tokens/s, including prefill and sampling, excluding model load.
- Mean task wall time: 1.01 seconds.

## Stops

```json
{
  "answered": 53,
  "invalid_action": 7
}
```

## Representative failures

### p01_862ed1cd68fd (two_lookup)

Question: Return the department of the owner recorded for project_p01_862ed1cd68fd.
Expected: `department_43033`; answer: `department_48216`; stop: `answered`.

```json
{
  "actions": [
    {
      "action": "answer",
      "value": "department_48216"
    }
  ],
  "observations": [],
  "raw_outputs": [
    "{\"action\":\"answer\",\"value\":\"department_48216\"}"
  ]
}
```

### p01_947a2dbd5be0 (one_lookup)

Question: What is the budget of project_p01_947a2dbd5be0?
Expected: `5640`; answer: `None`; stop: `invalid_action`.

```json
{
  "actions": [],
  "observations": [],
  "raw_outputs": [
    "{\"action\":\"answer\",\"value\":947a2dbd5be0}"
  ]
}
```

### p01_3c17f86a584f (two_lookup)

Question: Return the department of the owner recorded for project_p01_3c17f86a584f.
Expected: `department_12534`; answer: `department_48216`; stop: `answered`.

```json
{
  "actions": [
    {
      "action": "answer",
      "value": "department_48216"
    }
  ],
  "observations": [],
  "raw_outputs": [
    "{\"action\":\"answer\",\"value\":\"department_48216\"}"
  ]
}
```

### p01_46ac3959559e (one_lookup)

Question: Return the budget of project_p01_46ac3959559e from the private database.
Expected: `3388`; answer: `463955955955955955`; stop: `answered`.

```json
{
  "actions": [
    {
      "action": "answer",
      "value": 463955955955955955
    }
  ],
  "observations": [],
  "raw_outputs": [
    "{\"action\":\"answer\",\"value\":463955955955955955}"
  ]
}
```

### p01_1ba40973ed9c (two_lookup)

Question: Find the department of the employee who owns project_p01_1ba40973ed9c.
Expected: `department_28631`; answer: `department_48216`; stop: `answered`.

```json
{
  "actions": [
    {
      "action": "answer",
      "value": "department_48216"
    }
  ],
  "observations": [],
  "raw_outputs": [
    "{\"action\":\"answer\",\"value\":\"department_48216\"}"
  ]
}
```

### p01_c7272160951b (one_lookup)

Question: Return the budget of project_p01_c7272160951b from the private database.
Expected: `4229`; answer: `321`; stop: `answered`.

```json
{
  "actions": [
    {
      "action": "answer",
      "value": 321
    }
  ],
  "observations": [],
  "raw_outputs": [
    "{\"action\":\"answer\",\"value\":321}"
  ]
}
```

### p01_ba35224d7589 (one_lookup)

Question: Find the budget recorded for project_p01_ba35224d7589.
Expected: `4159`; answer: `875`; stop: `answered`.

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

### p01_50085ef63d34 (two_lookup)

Question: Find the department of the employee who owns project_p01_50085ef63d34.
Expected: `department_68148`; answer: `department_48216`; stop: `answered`.

```json
{
  "actions": [
    {
      "action": "answer",
      "value": "department_48216"
    }
  ],
  "observations": [],
  "raw_outputs": [
    "{\"action\":\"answer\",\"value\":\"department_48216\"}"
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

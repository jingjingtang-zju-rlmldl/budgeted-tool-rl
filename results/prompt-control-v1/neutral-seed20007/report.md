# Real-model baseline results

Model: `Qwen/Qwen3-0.6B` at `c1899de289a04d12100db370d81485cdf75e47ca`.
GPU: NVIDIA GeForce RTX 5060 Laptop GPU; dtype: BF16; thinking disabled; batch size 1.

**Actual CUDA inference on synthetic development tasks. No SFT or RL was run.**

| Task family | Correct / total | Success | Strict success | Mean calls | Calls on correct answers | Correct extra calls |
|---|---:|---:|---:|---:|---:|---:|
| overall | 33 / 60 | 55.0% | 10.0% | 0.217 | 0.394 | 0.000 |
| direct | 20 / 20 | 100.0% | 10.0% | 0.000 | 0.000 | 0.000 |
| one_lookup | 13 / 20 | 65.0% | 20.0% | 0.650 | 1.000 | 0.000 |
| two_lookup | 0 / 20 | 0.0% | 0.0% | 0.000 | n/a | n/a |

Parser: `fenced`. Strict-valid requests: 23 / 73; fenced wrappers normalized: 49.
Fence normalization removes only a single complete Markdown wrapper; it never fixes JSON, arguments or answers.

## Hardware and throughput

- Model parameters in memory: 1136.9 MiB.
- Peak CUDA allocated: 1212.0 MiB; peak reserved: 1224.0 MiB.
- Generated tokens: 1662; summed generation time: 80.49 seconds.
- Generation throughput: 20.65 tokens/s, including prefill and sampling, excluding model load.
- Mean task wall time: 1.35 seconds.

## Stops

```json
{
  "answered": 59,
  "invalid_action": 1
}
```

## Representative failures

### p01_862ed1cd68fd (two_lookup)

Question: Return the department of the owner recorded for project_p01_862ed1cd68fd.
Expected: `department_43033`; answer: `department`; stop: `answered`.

```json
{
  "actions": [
    {
      "action": "answer",
      "value": "department"
    }
  ],
  "observations": [],
  "raw_outputs": [
    "```json\n{\"action\": \"answer\", \"value\": \"department\"}\n```"
  ]
}
```

### p01_3c17f86a584f (two_lookup)

Question: Return the department of the owner recorded for project_p01_3c17f86a584f.
Expected: `department_12534`; answer: `department`; stop: `answered`.

```json
{
  "actions": [
    {
      "action": "answer",
      "value": "department"
    }
  ],
  "observations": [],
  "raw_outputs": [
    "```json\n{\"action\": \"answer\", \"value\": \"department\"}\n```"
  ]
}
```

### p01_1ba40973ed9c (two_lookup)

Question: Find the department of the employee who owns project_p01_1ba40973ed9c.
Expected: `department_28631`; answer: `department`; stop: `answered`.

```json
{
  "actions": [
    {
      "action": "answer",
      "value": "department"
    }
  ],
  "observations": [],
  "raw_outputs": [
    "```json\n{\"action\": \"answer\", \"value\": \"department\"}\n```"
  ]
}
```

### p01_ba35224d7589 (one_lookup)

Question: Find the budget recorded for project_p01_ba35224d7589.
Expected: `4159`; answer: `1000000`; stop: `answered`.

```json
{
  "actions": [
    {
      "action": "answer",
      "value": "1000000"
    }
  ],
  "observations": [],
  "raw_outputs": [
    "```json\n{\"action\": \"answer\", \"value\": \"1000000\"}\n```"
  ]
}
```

### p01_50085ef63d34 (two_lookup)

Question: Find the department of the employee who owns project_p01_50085ef63d34.
Expected: `department_68148`; answer: `department`; stop: `answered`.

```json
{
  "actions": [
    {
      "action": "answer",
      "value": "department"
    }
  ],
  "observations": [],
  "raw_outputs": [
    "{\"action\": \"answer\", \"value\": \"department\"}"
  ]
}
```

### p01_972580b9502c (one_lookup)

Question: What is the budget of project_p01_972580b9502c?
Expected: `3416`; answer: `The budget of project_p01_972580b9502c is $500,000.`; stop: `answered`.

```json
{
  "actions": [
    {
      "action": "answer",
      "value": "The budget of project_p01_972580b9502c is $500,000."
    }
  ],
  "observations": [],
  "raw_outputs": [
    "```json\n{\"action\": \"answer\", \"value\": \"The budget of project_p01_972580b9502c is $500,000.\"}\n```"
  ]
}
```

### p01_911b29ee131a (one_lookup)

Question: Find the budget recorded for project_p01_911b29ee131a.
Expected: `2807`; answer: `The budget recorded for project_p01_911b29ee131a is $500,000.`; stop: `answered`.

```json
{
  "actions": [
    {
      "action": "answer",
      "value": "The budget recorded for project_p01_911b29ee131a is $500,000."
    }
  ],
  "observations": [],
  "raw_outputs": [
    "```json\n{\"action\": \"answer\", \"value\": \"The budget recorded for project_p01_911b29ee131a is $500,000.\"}\n```"
  ]
}
```

### p01_2ff4cdaf2fd4 (two_lookup)

Question: Return the department of the owner recorded for project_p01_2ff4cdaf2fd4.
Expected: `department_90207`; answer: `department`; stop: `answered`.

```json
{
  "actions": [
    {
      "action": "answer",
      "value": "department"
    }
  ],
  "observations": [],
  "raw_outputs": [
    "```json\n{\"action\": \"answer\", \"value\": \"department\"}\n```"
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

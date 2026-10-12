# Real-model baseline results

Model: `Qwen/Qwen3-0.6B` at `c1899de289a04d12100db370d81485cdf75e47ca`.
GPU: NVIDIA GeForce RTX 5060 Laptop GPU; dtype: BF16; thinking disabled; batch size 1.

**Actual CUDA inference on synthetic development tasks. No SFT or RL was run.**

| Task family | Correct / total | Success | Strict success | Mean calls | Calls on correct answers | Correct extra calls |
|---|---:|---:|---:|---:|---:|---:|
| overall | 18 / 30 | 60.0% | 13.3% | 0.300 | 0.444 | 0.000 |
| direct | 10 / 10 | 100.0% | 20.0% | 0.000 | 0.000 | 0.000 |
| one_lookup | 8 / 10 | 80.0% | 20.0% | 0.900 | 1.000 | 0.000 |
| two_lookup | 0 / 10 | 0.0% | 0.0% | 0.000 | n/a | n/a |

Parser: `fenced`. Strict-valid requests: 13 / 39; fenced wrappers normalized: 25.
Fence normalization removes only a single complete Markdown wrapper; it never fixes JSON, arguments or answers.

## Hardware and throughput

- Model parameters in memory: 1136.9 MiB.
- Peak CUDA allocated: 1208.8 MiB; peak reserved: 1224.0 MiB.
- Generated tokens: 849; summed generation time: 38.35 seconds.
- Generation throughput: 22.14 tokens/s, including prefill and sampling, excluding model load.
- Mean task wall time: 1.28 seconds.

## Stops

```json
{
  "answered": 29,
  "invalid_action": 1
}
```

## Representative failures

### dev_0024 (two_lookup)

Question: What department does the owner of project_c79e8f64343e belong to?
Expected: `department_76565`; answer: `department`; stop: `answered`.

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

### dev_0020 (two_lookup)

Question: What department does the owner of project_52721b108e89 belong to?
Expected: `department_66897`; answer: `department`; stop: `answered`.

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

### dev_0028 (two_lookup)

Question: What department does the owner of project_1c520eccafee belong to?
Expected: `department_62709`; answer: `department`; stop: `answered`.

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

### dev_0022 (two_lookup)

Question: Return the department of the owner recorded for project_c989fd5e14ef.
Expected: `department_15112`; answer: `department`; stop: `answered`.

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

### dev_0029 (two_lookup)

Question: Find the department of the employee who owns project_ee2a40f77341.
Expected: `department_77019`; answer: `department`; stop: `answered`.

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

### dev_0015 (one_lookup)

Question: Find the budget recorded for project_5c2820a0f92b.
Expected: `4856`; answer: `Budget recorded for project_5c2820a0f92b: $1,200,000.`; stop: `answered`.

```json
{
  "actions": [
    {
      "action": "answer",
      "value": "Budget recorded for project_5c2820a0f92b: $1,200,000."
    }
  ],
  "observations": [],
  "raw_outputs": [
    "```json\n{\"action\": \"answer\", \"value\": \"Budget recorded for project_5c2820a0f92b: $1,200,000.\"}\n```"
  ]
}
```

### dev_0025 (two_lookup)

Question: Find the department of the employee who owns project_0d71d3347a8e.
Expected: `department_48618`; answer: `department`; stop: `answered`.

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

### dev_0021 (two_lookup)

Question: Find the department of the employee who owns project_2b415403a7c1.
Expected: `department_31985`; answer: `department`; stop: `answered`.

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
    "```json\n{\n  \"action\": \"answer\",\n  \"value\": \"department\"\n}\n```"
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

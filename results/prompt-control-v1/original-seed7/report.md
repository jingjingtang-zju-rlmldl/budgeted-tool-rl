# Real-model baseline results

Model: `Qwen/Qwen3-0.6B` at `c1899de289a04d12100db370d81485cdf75e47ca`.
GPU: NVIDIA GeForce RTX 5060 Laptop GPU; dtype: BF16; thinking disabled; batch size 1.

**Actual CUDA inference on synthetic development tasks. No SFT or RL was run.**

| Task family | Correct / total | Success | Strict success | Mean calls | Calls on correct answers | Correct extra calls |
|---|---:|---:|---:|---:|---:|---:|
| overall | 38 / 60 | 63.3% | 8.3% | 0.333 | 0.474 | 0.000 |
| direct | 20 / 20 | 100.0% | 15.0% | 0.000 | 0.000 | 0.000 |
| one_lookup | 18 / 20 | 90.0% | 10.0% | 1.000 | 1.000 | 0.000 |
| two_lookup | 0 / 20 | 0.0% | 0.0% | 0.000 | n/a | n/a |

Parser: `fenced`. Strict-valid requests: 22 / 80; fenced wrappers normalized: 56.
Fence normalization removes only a single complete Markdown wrapper; it never fixes JSON, arguments or answers.

## Hardware and throughput

- Model parameters in memory: 1136.9 MiB.
- Peak CUDA allocated: 1215.7 MiB; peak reserved: 1262.0 MiB.
- Generated tokens: 1879; summed generation time: 138.90 seconds.
- Generation throughput: 13.53 tokens/s, including prefill and sampling, excluding model load.
- Mean task wall time: 2.32 seconds.

## Stops

```json
{
  "answered": 58,
  "invalid_action": 2
}
```

## Representative failures

### p01_862ed1cd68fd (two_lookup)

Question: Return the department of the owner recorded for project_p01_862ed1cd68fd.
Expected: `department_43033`; answer: `Department: Engineering`; stop: `answered`.

```json
{
  "actions": [
    {
      "action": "answer",
      "value": "Department: Engineering"
    }
  ],
  "observations": [],
  "raw_outputs": [
    "{\"action\": \"answer\", \"value\": \"Department: Engineering\"}"
  ]
}
```

### p01_3c17f86a584f (two_lookup)

Question: Return the department of the owner recorded for project_p01_3c17f86a584f.
Expected: `department_12534`; answer: `Department: Engineering`; stop: `answered`.

```json
{
  "actions": [
    {
      "action": "answer",
      "value": "Department: Engineering"
    }
  ],
  "observations": [],
  "raw_outputs": [
    "```json\n{\n  \"action\": \"answer\",\n  \"value\": \"Department: Engineering\"\n}\n```"
  ]
}
```

### p01_1ba40973ed9c (two_lookup)

Question: Find the department of the employee who owns project_p01_1ba40973ed9c.
Expected: `department_28631`; answer: `Department: Engineering`; stop: `answered`.

```json
{
  "actions": [
    {
      "action": "answer",
      "value": "Department: Engineering"
    }
  ],
  "observations": [],
  "raw_outputs": [
    "```json\n{\n  \"action\": \"answer\",\n  \"value\": \"Department: Engineering\"\n}\n```"
  ]
}
```

### p01_50085ef63d34 (two_lookup)

Question: Find the department of the employee who owns project_p01_50085ef63d34.
Expected: `department_68148`; answer: `Department: Department of the employee who owns project_p01_50085ef63d34 is not available.`; stop: `answered`.

```json
{
  "actions": [
    {
      "action": "answer",
      "value": "Department: Department of the employee who owns project_p01_50085ef63d34 is not available."
    }
  ],
  "observations": [],
  "raw_outputs": [
    "```json\n{\"action\": \"answer\", \"value\": \"Department: Department of the employee who owns project_p01_50085ef63d34 is not available.\"}\n```"
  ]
}
```

### p01_972580b9502c (one_lookup)

Question: What is the budget of project_p01_972580b9502c?
Expected: `3416`; answer: `None`; stop: `invalid_action`.

```json
{
  "actions": [
    {
      "action": "lookup",
      "entity": "project_p01_972580b9502c",
      "field": "budget"
    }
  ],
  "observations": [
    {
      "entity": "project_p01_972580b9502c",
      "field": "budget",
      "value": 3416
    }
  ],
  "raw_outputs": [
    "```json\n{\"action\": \"lookup\", \"entity\": \"project_p01_972580b9502c\", \"field\": \"budget\"}\n```",
    "```json\n{\"action\": \"answer\", \"value\": \"3416\"}"
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
    "```json\n{\n  \"action\": \"answer\",\n  \"value\": \"department\"\n}\n```"
  ]
}
```

### p01_f554470e7da5 (two_lookup)

Question: What department does the owner of project_p01_f554470e7da5 belong to?
Expected: `department_58479`; answer: `department`; stop: `answered`.

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

### p01_7dcc5b2745af (two_lookup)

Question: What department does the owner of project_p01_7dcc5b2745af belong to?
Expected: `department_78466`; answer: `department`; stop: `answered`.

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

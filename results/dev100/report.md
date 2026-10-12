# Real-model baseline results

Model: `Qwen/Qwen3-0.6B` at `c1899de289a04d12100db370d81485cdf75e47ca`.
GPU: NVIDIA GeForce RTX 5060 Laptop GPU; dtype: BF16; thinking disabled; batch size 1.

**Actual CUDA inference on synthetic development tasks. No SFT or RL was run.**

| Task family | Correct / total | Success | Strict success | Mean calls | Calls on correct answers | Correct extra calls |
|---|---:|---:|---:|---:|---:|---:|
| overall | 59 / 100 | 59.0% | 7.0% | 0.390 | 0.424 | 0.000 |
| direct | 34 / 34 | 100.0% | 14.7% | 0.000 | 0.000 | 0.000 |
| one_lookup | 25 / 33 | 75.8% | 6.1% | 1.182 | 1.000 | 0.000 |
| two_lookup | 0 / 33 | 0.0% | 0.0% | 0.000 | n/a | n/a |

Parser: `fenced`. Strict-valid requests: 39 / 139; fenced wrappers normalized: 98.
Fence normalization removes only a single complete Markdown wrapper; it never fixes JSON, arguments or answers.

## Hardware and throughput

- Model parameters in memory: 1136.9 MiB.
- Peak CUDA allocated: 1239.0 MiB; peak reserved: 1274.0 MiB.
- Generated tokens: 3197; summed generation time: 225.60 seconds.
- Generation throughput: 14.17 tokens/s, including prefill and sampling, excluding model load.
- Mean task wall time: 2.26 seconds.

## Stops

```json
{
  "answered": 98,
  "invalid_action": 2
}
```

## Representative failures

### dev_fc3b4b4507ba (two_lookup)

Question: Find the department of the employee who owns project_dev_fc3b4b4507ba.
Expected: `department_21877`; answer: `Department: Development`; stop: `answered`.

```json
{
  "actions": [
    {
      "action": "answer",
      "value": "Department: Development"
    }
  ],
  "observations": [],
  "raw_outputs": [
    "```json\n{\n  \"action\": \"answer\",\n  \"value\": \"Department: Development\"\n}\n```"
  ]
}
```

### dev_13eb9564aa43 (two_lookup)

Question: What department does the owner of project_dev_13eb9564aa43 belong to?
Expected: `department_97236`; answer: `Department: Development`; stop: `answered`.

```json
{
  "actions": [
    {
      "action": "answer",
      "value": "Department: Development"
    }
  ],
  "observations": [],
  "raw_outputs": [
    "```json\n{\n  \"action\": \"answer\",\n  \"value\": \"Department: Development\"\n}\n```"
  ]
}
```

### dev_b1cc5c5ac92e (two_lookup)

Question: Find the department of the employee who owns project_dev_b1cc5c5ac92e.
Expected: `department_19840`; answer: `Department: Development`; stop: `answered`.

```json
{
  "actions": [
    {
      "action": "answer",
      "value": "Department: Development"
    }
  ],
  "observations": [],
  "raw_outputs": [
    "```json\n{\n  \"action\": \"answer\",\n  \"value\": \"Department: Development\"\n}\n```"
  ]
}
```

### dev_144d2218db74 (one_lookup)

Question: What is the budget of project_dev_144d2218db74?
Expected: `7375`; answer: `None`; stop: `invalid_action`.

```json
{
  "actions": [
    {
      "action": "lookup",
      "entity": "project_dev_144d2218db74",
      "field": "budget"
    }
  ],
  "observations": [
    {
      "entity": "project_dev_144d2218db74",
      "field": "budget",
      "value": 7375
    }
  ],
  "raw_outputs": [
    "```json\n{\"action\": \"lookup\", \"entity\": \"project_dev_144d2218db74\", \"field\": \"budget\"}\n```",
    "```json\n{\"action\": \"answer\", \"value\": \"7375\"}"
  ]
}
```

### dev_2b93cd966743 (two_lookup)

Question: Find the department of the employee who owns project_dev_2b93cd966743.
Expected: `department_82199`; answer: `department`; stop: `answered`.

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

### dev_b8038bcc0e13 (one_lookup)

Question: Return the budget of project_dev_b8038bcc0e13 from the private database.
Expected: `1598`; answer: `0`; stop: `answered`.

```json
{
  "actions": [
    {
      "action": "lookup",
      "entity": "project_dev_b8038bcc0e13",
      "field": "owner"
    },
    {
      "action": "lookup",
      "entity": "employee_dev_b8038bcc0e13",
      "field": "budget"
    },
    {
      "action": "answer",
      "value": "0"
    }
  ],
  "observations": [
    {
      "entity": "project_dev_b8038bcc0e13",
      "field": "owner",
      "value": "employee_dev_b8038bcc0e13"
    },
    {
      "error": "unknown field",
      "entity": "employee_dev_b8038bcc0e13",
      "field": "budget"
    }
  ],
  "raw_outputs": [
    "```json\n{\"action\": \"lookup\", \"entity\": \"project_dev_b8038bcc0e13\", \"field\": \"owner\"}\n```",
    "{\"action\": \"lookup\", \"entity\": \"employee_dev_b8038bcc0e13\", \"field\": \"budget\"}",
    "{\"action\": \"answer\", \"value\": \"0\"}"
  ]
}
```

### dev_f6d60ffcf486 (two_lookup)

Question: What department does the owner of project_dev_f6d60ffcf486 belong to?
Expected: `department_11801`; answer: `department`; stop: `answered`.

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

### dev_d6b3dd961b0f (two_lookup)

Question: Find the department of the employee who owns project_dev_d6b3dd961b0f.
Expected: `department_20102`; answer: `department`; stop: `answered`.

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

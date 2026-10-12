# Real-model baseline results

Model: `Qwen/Qwen3-0.6B` at `c1899de289a04d12100db370d81485cdf75e47ca`.
GPU: NVIDIA GeForce RTX 5060 Laptop GPU; dtype: BF16; thinking disabled; batch size 1.

**Actual CUDA inference on synthetic development tasks. No SFT or RL was run.**

| Task family | Correct / total | Success | Strict success | Mean calls | Calls on correct answers | Correct extra calls |
|---|---:|---:|---:|---:|---:|---:|
| overall | 20 / 30 | 66.7% | 66.7% | 0.567 | 0.500 | 0.000 |
| direct | 10 / 10 | 100.0% | 100.0% | 0.000 | 0.000 | 0.000 |
| one_lookup | 10 / 10 | 100.0% | 100.0% | 1.000 | 1.000 | 0.000 |
| two_lookup | 0 / 10 | 0.0% | 0.0% | 0.700 | n/a | n/a |

Parser: `fenced`. Strict-valid requests: 47 / 47; fenced wrappers normalized: 0.
Fence normalization removes only a single complete Markdown wrapper; it never fixes JSON, arguments or answers.

## Hardware and throughput

- Model parameters in memory: 1136.9 MiB.
- Peak CUDA allocated: 1310.2 MiB; peak reserved: 1360.0 MiB.
- Generated tokens: 874; summed generation time: 39.28 seconds.
- Generation throughput: 22.25 tokens/s, including prefill and sampling, excluding model load.
- Mean task wall time: 1.32 seconds.

## Stops

```json
{
  "answered": 30
}
```

## Representative failures

### dev_0024 (two_lookup)

Question: What department does the owner of project_c79e8f64343e belong to?
Expected: `department_76565`; answer: `employee_8d58895adf07`; stop: `answered`.

```json
{
  "actions": [
    {
      "action": "lookup",
      "entity": "project_c79e8f64343e",
      "field": "owner"
    },
    {
      "action": "answer",
      "value": "employee_8d58895adf07"
    }
  ],
  "observations": [
    {
      "entity": "project_c79e8f64343e",
      "field": "owner",
      "value": "employee_8d58895adf07"
    }
  ],
  "raw_outputs": [
    "{\"action\":\"lookup\",\"entity\":\"project_c79e8f64343e\",\"field\":\"owner\"}",
    "{\"action\":\"answer\",\"value\":\"employee_8d58895adf07\"}"
  ]
}
```

### dev_0020 (two_lookup)

Question: What department does the owner of project_52721b108e89 belong to?
Expected: `department_66897`; answer: `employee_4616d858bc13`; stop: `answered`.

```json
{
  "actions": [
    {
      "action": "lookup",
      "entity": "project_52721b108e89",
      "field": "owner"
    },
    {
      "action": "answer",
      "value": "employee_4616d858bc13"
    }
  ],
  "observations": [
    {
      "entity": "project_52721b108e89",
      "field": "owner",
      "value": "employee_4616d858bc13"
    }
  ],
  "raw_outputs": [
    "{\"action\":\"lookup\",\"entity\":\"project_52721b108e89\",\"field\":\"owner\"}",
    "{\"action\":\"answer\",\"value\":\"employee_4616d858bc13\"}"
  ]
}
```

### dev_0028 (two_lookup)

Question: What department does the owner of project_1c520eccafee belong to?
Expected: `department_62709`; answer: `employee_b5614ae45974`; stop: `answered`.

```json
{
  "actions": [
    {
      "action": "lookup",
      "entity": "project_1c520eccafee",
      "field": "owner"
    },
    {
      "action": "answer",
      "value": "employee_b5614ae45974"
    }
  ],
  "observations": [
    {
      "entity": "project_1c520eccafee",
      "field": "owner",
      "value": "employee_b5614ae45974"
    }
  ],
  "raw_outputs": [
    "{\"action\":\"lookup\",\"entity\":\"project_1c520eccafee\",\"field\":\"owner\"}",
    "{\"action\":\"answer\",\"value\":\"employee_b5614ae45974\"}"
  ]
}
```

### dev_0022 (two_lookup)

Question: Return the department of the owner recorded for project_c989fd5e14ef.
Expected: `department_15112`; answer: `employee_d4dd1caeb9f4`; stop: `answered`.

```json
{
  "actions": [
    {
      "action": "lookup",
      "entity": "project_c989fd5e14ef",
      "field": "owner"
    },
    {
      "action": "answer",
      "value": "employee_d4dd1caeb9f4"
    }
  ],
  "observations": [
    {
      "entity": "project_c989fd5e14ef",
      "field": "owner",
      "value": "employee_d4dd1caeb9f4"
    }
  ],
  "raw_outputs": [
    "{\"action\":\"lookup\",\"entity\":\"project_c989fd5e14ef\",\"field\":\"owner\"}",
    "{\"action\":\"answer\",\"value\":\"employee_d4dd1caeb9f4\"}"
  ]
}
```

### dev_0029 (two_lookup)

Question: Find the department of the employee who owns project_ee2a40f77341.
Expected: `department_77019`; answer: `employee_ea6b9d427d4a`; stop: `answered`.

```json
{
  "actions": [
    {
      "action": "lookup",
      "entity": "project_ee2a40f77341",
      "field": "owner"
    },
    {
      "action": "answer",
      "value": "employee_ea6b9d427d4a"
    }
  ],
  "observations": [
    {
      "entity": "project_ee2a40f77341",
      "field": "owner",
      "value": "employee_ea6b9d427d4a"
    }
  ],
  "raw_outputs": [
    "{\"action\":\"lookup\",\"entity\":\"project_ee2a40f77341\",\"field\":\"owner\"}",
    "{\"action\":\"answer\",\"value\":\"employee_ea6b9d427d4a\"}"
  ]
}
```

### dev_0025 (two_lookup)

Question: Find the department of the employee who owns project_0d71d3347a8e.
Expected: `department_48618`; answer: `department_48216`; stop: `answered`.

```json
{
  "actions": [
    {
      "action": "lookup",
      "entity": "project_0d71d3347a8e",
      "field": "owner"
    },
    {
      "action": "answer",
      "value": "department_48216"
    }
  ],
  "observations": [
    {
      "entity": "project_0d71d3347a8e",
      "field": "owner",
      "value": "employee_aaea7b207cc8"
    }
  ],
  "raw_outputs": [
    "{\"action\":\"lookup\",\"entity\":\"project_0d71d3347a8e\",\"field\":\"owner\"}",
    "{\"action\":\"answer\",\"value\":\"department_48216\"}"
  ]
}
```

### dev_0021 (two_lookup)

Question: Find the department of the employee who owns project_2b415403a7c1.
Expected: `department_31985`; answer: `department_48216`; stop: `answered`.

```json
{
  "actions": [
    {
      "action": "lookup",
      "entity": "project_2b415403a7c1",
      "field": "owner"
    },
    {
      "action": "answer",
      "value": "department_48216"
    }
  ],
  "observations": [
    {
      "entity": "project_2b415403a7c1",
      "field": "owner",
      "value": "employee_b81b27a8db59"
    }
  ],
  "raw_outputs": [
    "{\"action\":\"lookup\",\"entity\":\"project_2b415403a7c1\",\"field\":\"owner\"}",
    "{\"action\":\"answer\",\"value\":\"department_48216\"}"
  ]
}
```

### dev_0023 (two_lookup)

Question: Which department does the employee listed as the owner of project_c9921e7dc229 work in?
Expected: `department_60147`; answer: `department_48216`; stop: `answered`.

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

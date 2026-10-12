# Real-model baseline results

Model: `Qwen/Qwen3-0.6B` at `c1899de289a04d12100db370d81485cdf75e47ca`.
GPU: NVIDIA GeForce RTX 5060 Laptop GPU; dtype: BF16; thinking disabled; batch size 1.

**Actual CUDA inference on synthetic development tasks. No SFT or RL was run.**

| Task family | Correct / total | Success | Strict success | Mean calls | Calls on correct answers | Correct extra calls |
|---|---:|---:|---:|---:|---:|---:|
| overall | 40 / 60 | 66.7% | 66.7% | 0.550 | 0.500 | 0.000 |
| direct | 20 / 20 | 100.0% | 100.0% | 0.000 | 0.000 | 0.000 |
| one_lookup | 20 / 20 | 100.0% | 100.0% | 1.000 | 1.000 | 0.000 |
| two_lookup | 0 / 20 | 0.0% | 0.0% | 0.650 | n/a | n/a |

Parser: `fenced`. Strict-valid requests: 93 / 93; fenced wrappers normalized: 0.
Fence normalization removes only a single complete Markdown wrapper; it never fixes JSON, arguments or answers.

## Hardware and throughput

- Model parameters in memory: 1136.9 MiB.
- Peak CUDA allocated: 1315.8 MiB; peak reserved: 1412.0 MiB.
- Generated tokens: 1834; summed generation time: 124.40 seconds.
- Generation throughput: 14.74 tokens/s, including prefill and sampling, excluding model load.
- Mean task wall time: 2.08 seconds.

## Stops

```json
{
  "answered": 60
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

### p01_1ba40973ed9c (two_lookup)

Question: Find the department of the employee who owns project_p01_1ba40973ed9c.
Expected: `department_28631`; answer: `employee_p01_1ba40973ed9c`; stop: `answered`.

```json
{
  "actions": [
    {
      "action": "lookup",
      "entity": "project_p01_1ba40973ed9c",
      "field": "owner"
    },
    {
      "action": "answer",
      "value": "employee_p01_1ba40973ed9c"
    }
  ],
  "observations": [
    {
      "entity": "project_p01_1ba40973ed9c",
      "field": "owner",
      "value": "employee_p01_1ba40973ed9c"
    }
  ],
  "raw_outputs": [
    "{\"action\":\"lookup\",\"entity\":\"project_p01_1ba40973ed9c\",\"field\":\"owner\"}",
    "{\"action\":\"answer\",\"value\":\"employee_p01_1ba40973ed9c\"}"
  ]
}
```

### p01_50085ef63d34 (two_lookup)

Question: Find the department of the employee who owns project_p01_50085ef63d34.
Expected: `department_68148`; answer: `employee_p01_50085ef63d34`; stop: `answered`.

```json
{
  "actions": [
    {
      "action": "lookup",
      "entity": "project_p01_50085ef63d34",
      "field": "owner"
    },
    {
      "action": "answer",
      "value": "employee_p01_50085ef63d34"
    }
  ],
  "observations": [
    {
      "entity": "project_p01_50085ef63d34",
      "field": "owner",
      "value": "employee_p01_50085ef63d34"
    }
  ],
  "raw_outputs": [
    "{\"action\":\"lookup\",\"entity\":\"project_p01_50085ef63d34\",\"field\":\"owner\"}",
    "{\"action\":\"answer\",\"value\":\"employee_p01_50085ef63d34\"}"
  ]
}
```

### p01_2ff4cdaf2fd4 (two_lookup)

Question: Return the department of the owner recorded for project_p01_2ff4cdaf2fd4.
Expected: `department_90207`; answer: `employee_p01_2ff4cdaf2fd4`; stop: `answered`.

```json
{
  "actions": [
    {
      "action": "lookup",
      "entity": "project_p01_2ff4cdaf2fd4",
      "field": "owner"
    },
    {
      "action": "answer",
      "value": "employee_p01_2ff4cdaf2fd4"
    }
  ],
  "observations": [
    {
      "entity": "project_p01_2ff4cdaf2fd4",
      "field": "owner",
      "value": "employee_p01_2ff4cdaf2fd4"
    }
  ],
  "raw_outputs": [
    "{\"action\":\"lookup\",\"entity\":\"project_p01_2ff4cdaf2fd4\",\"field\":\"owner\"}",
    "{\"action\":\"answer\",\"value\":\"employee_p01_2ff4cdaf2fd4\"}"
  ]
}
```

### p01_f554470e7da5 (two_lookup)

Question: What department does the owner of project_p01_f554470e7da5 belong to?
Expected: `department_58479`; answer: `employee_p01_f554470e7da5`; stop: `answered`.

```json
{
  "actions": [
    {
      "action": "lookup",
      "entity": "project_p01_f554470e7da5",
      "field": "owner"
    },
    {
      "action": "answer",
      "value": "employee_p01_f554470e7da5"
    }
  ],
  "observations": [
    {
      "entity": "project_p01_f554470e7da5",
      "field": "owner",
      "value": "employee_p01_f554470e7da5"
    }
  ],
  "raw_outputs": [
    "{\"action\":\"lookup\",\"entity\":\"project_p01_f554470e7da5\",\"field\":\"owner\"}",
    "{\"action\":\"answer\",\"value\":\"employee_p01_f554470e7da5\"}"
  ]
}
```

### p01_7dcc5b2745af (two_lookup)

Question: What department does the owner of project_p01_7dcc5b2745af belong to?
Expected: `department_78466`; answer: `employee_p01_7dcc5b2745af`; stop: `answered`.

```json
{
  "actions": [
    {
      "action": "lookup",
      "entity": "project_p01_7dcc5b2745af",
      "field": "owner"
    },
    {
      "action": "answer",
      "value": "employee_p01_7dcc5b2745af"
    }
  ],
  "observations": [
    {
      "entity": "project_p01_7dcc5b2745af",
      "field": "owner",
      "value": "employee_p01_7dcc5b2745af"
    }
  ],
  "raw_outputs": [
    "{\"action\":\"lookup\",\"entity\":\"project_p01_7dcc5b2745af\",\"field\":\"owner\"}",
    "{\"action\":\"answer\",\"value\":\"employee_p01_7dcc5b2745af\"}"
  ]
}
```

### p01_b3b5d2899c93 (two_lookup)

Question: What department does the owner of project_p01_b3b5d2899c93 belong to?
Expected: `department_70526`; answer: `department_48216`; stop: `answered`.

```json
{
  "actions": [
    {
      "action": "lookup",
      "entity": "project_p01_b3b5d2899c93",
      "field": "owner"
    },
    {
      "action": "answer",
      "value": "department_48216"
    }
  ],
  "observations": [
    {
      "entity": "project_p01_b3b5d2899c93",
      "field": "owner",
      "value": "employee_p01_b3b5d2899c93"
    }
  ],
  "raw_outputs": [
    "{\"action\":\"lookup\",\"entity\":\"project_p01_b3b5d2899c93\",\"field\":\"owner\"}",
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

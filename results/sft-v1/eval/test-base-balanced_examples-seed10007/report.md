# Real-model baseline results

Model: `Qwen/Qwen3-0.6B` at `c1899de289a04d12100db370d81485cdf75e47ca`.
GPU: NVIDIA GeForce RTX 5060 Laptop GPU; dtype: BF16; thinking disabled; batch size 1.

**Actual CUDA inference on synthetic development tasks. No SFT or RL was run.**

| Task family | Correct / total | Success | Strict success | Mean calls | Calls on correct answers | Correct extra calls |
|---|---:|---:|---:|---:|---:|---:|
| overall | 27 / 60 | 45.0% | 45.0% | 0.250 | 0.259 | 0.000 |
| direct | 20 / 20 | 100.0% | 100.0% | 0.000 | 0.000 | 0.000 |
| one_lookup | 7 / 20 | 35.0% | 35.0% | 0.350 | 1.000 | 0.000 |
| two_lookup | 0 / 20 | 0.0% | 0.0% | 0.400 | n/a | n/a |

Parser: `fenced`. Strict-valid requests: 73 / 75; fenced wrappers normalized: 0.
Fence normalization removes only a single complete Markdown wrapper; it never fixes JSON, arguments or answers.

## Hardware and throughput

- Model parameters in memory: 1136.9 MiB.
- Peak CUDA allocated: 1314.5 MiB; peak reserved: 1360.0 MiB.
- Generated tokens: 1296; summed generation time: 61.72 seconds.
- Generation throughput: 21.00 tokens/s, including prefill and sampling, excluding model load.
- Mean task wall time: 1.03 seconds.

## Stops

```json
{
  "answered": 58,
  "invalid_action": 2
}
```

## Representative failures

### test_0057 (two_lookup)

Question: From company records, give the department of the person marked as the owner of project_8c13141d6bdb.
Expected: `department_18498`; answer: `department_48216`; stop: `answered`.

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

### test_0054 (two_lookup)

Question: From company records, give the department of the person marked as the owner of project_4181d4c35e60.
Expected: `department_73056`; answer: `department_ed62600b1797`; stop: `answered`.

```json
{
  "actions": [
    {
      "action": "lookup",
      "entity": "project_4181d4c35e60",
      "field": "owner"
    },
    {
      "action": "answer",
      "value": "department_ed62600b1797"
    }
  ],
  "observations": [
    {
      "entity": "project_4181d4c35e60",
      "field": "owner",
      "value": "employee_ed62600b1797"
    }
  ],
  "raw_outputs": [
    "{\"action\":\"lookup\",\"entity\":\"project_4181d4c35e60\",\"field\":\"owner\"}",
    "{\"action\":\"answer\",\"value\":\"department_ed62600b1797\"}"
  ]
}
```

### test_0047 (two_lookup)

Question: Identify the owner of project_1e7e08cb1834 and report that person's department.
Expected: `department_11358`; answer: `department_48216`; stop: `answered`.

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

### test_0034 (one_lookup)

Question: What number is saved in the budget field of project_2a8db5930b02?
Expected: `2812`; answer: `1234567890`; stop: `answered`.

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

### test_0036 (one_lookup)

Question: Using the company records, report the budget for project_5b23fb92f2be.
Expected: `967`; answer: `321`; stop: `answered`.

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

### test_0046 (two_lookup)

Question: Which department is associated with the employee listed as the owner of project_2ca224b36adb?
Expected: `department_91227`; answer: `employee_47dbd64602fe`; stop: `answered`.

```json
{
  "actions": [
    {
      "action": "lookup",
      "entity": "project_2ca224b36adb",
      "field": "owner"
    },
    {
      "action": "answer",
      "value": "employee_47dbd64602fe"
    }
  ],
  "observations": [
    {
      "entity": "project_2ca224b36adb",
      "field": "owner",
      "value": "employee_47dbd64602fe"
    }
  ],
  "raw_outputs": [
    "{\"action\":\"lookup\",\"entity\":\"project_2ca224b36adb\",\"field\":\"owner\"}",
    "{\"action\":\"answer\",\"value\":\"employee_47dbd64602fe\"}"
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

## Interpretation limits

- Real CUDA model inference on synthetic English tasks; no training performed.
- One sampled attempt per task; observed success is not a repeated-trial pass@k study.
- Development tasks used for analysis; future final test must be independently generated.
- Wilson intervals are descriptive under binomial assumptions; controlled templates are not IID real-world tasks.
- PyTorch memory measures this process; nvidia-smi includes other apps and WDDM accounting.
- Inference feasibility does not establish backward/optimizer or GRPO training feasibility.

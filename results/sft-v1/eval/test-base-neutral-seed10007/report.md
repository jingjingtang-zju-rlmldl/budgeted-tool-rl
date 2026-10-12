# Real-model baseline results

Model: `Qwen/Qwen3-0.6B` at `c1899de289a04d12100db370d81485cdf75e47ca`.
GPU: NVIDIA GeForce RTX 5060 Laptop GPU; dtype: BF16; thinking disabled; batch size 1.

**Actual CUDA inference on synthetic development tasks. No SFT or RL was run.**

| Task family | Correct / total | Success | Strict success | Mean calls | Calls on correct answers | Correct extra calls |
|---|---:|---:|---:|---:|---:|---:|
| overall | 28 / 60 | 46.7% | 20.0% | 0.133 | 0.286 | 0.000 |
| direct | 20 / 20 | 100.0% | 40.0% | 0.000 | 0.000 | 0.000 |
| one_lookup | 8 / 20 | 40.0% | 20.0% | 0.400 | 1.000 | 0.000 |
| two_lookup | 0 / 20 | 0.0% | 0.0% | 0.000 | n/a | n/a |

Parser: `fenced`. Strict-valid requests: 26 / 68; fenced wrappers normalized: 42.
Fence normalization removes only a single complete Markdown wrapper; it never fixes JSON, arguments or answers.

## Hardware and throughput

- Model parameters in memory: 1136.9 MiB.
- Peak CUDA allocated: 1209.5 MiB; peak reserved: 1224.0 MiB.
- Generated tokens: 1586; summed generation time: 68.92 seconds.
- Generation throughput: 23.01 tokens/s, including prefill and sampling, excluding model load.
- Mean task wall time: 1.15 seconds.

## Stops

```json
{
  "answered": 60
}
```

## Representative failures

### test_0057 (two_lookup)

Question: From company records, give the department of the person marked as the owner of project_8c13141d6bdb.
Expected: `department_18498`; answer: `Department: Engineering`; stop: `answered`.

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
    "```json\n{\"action\": \"answer\", \"value\": \"Department: Engineering\"}\n```"
  ]
}
```

### test_0054 (two_lookup)

Question: From company records, give the department of the person marked as the owner of project_4181d4c35e60.
Expected: `department_73056`; answer: `Department: Engineering`; stop: `answered`.

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

### test_0047 (two_lookup)

Question: Identify the owner of project_1e7e08cb1834 and report that person's department.
Expected: `department_11358`; answer: `Owner: Project_1e7e08cb1834, Department: Engineering`; stop: `answered`.

```json
{
  "actions": [
    {
      "action": "answer",
      "value": "Owner: Project_1e7e08cb1834, Department: Engineering"
    }
  ],
  "observations": [],
  "raw_outputs": [
    "```json\n{\"action\": \"answer\", \"value\": \"Owner: Project_1e7e08cb1834, Department: Engineering\"}\n```"
  ]
}
```

### test_0027 (one_lookup)

Question: Using the company records, report the budget for project_646253650f41.
Expected: `2288`; answer: `The budget for project_646253650f41 is $500,000.`; stop: `answered`.

```json
{
  "actions": [
    {
      "action": "answer",
      "value": "The budget for project_646253650f41 is $500,000."
    }
  ],
  "observations": [],
  "raw_outputs": [
    "```json\n{\"action\": \"answer\", \"value\": \"The budget for project_646253650f41 is $500,000.\"}\n```"
  ]
}
```

### test_0034 (one_lookup)

Question: What number is saved in the budget field of project_2a8db5930b02?
Expected: `2812`; answer: `2450000000`; stop: `answered`.

```json
{
  "actions": [
    {
      "action": "answer",
      "value": "2450000000"
    }
  ],
  "observations": [],
  "raw_outputs": [
    "```json\n{\"action\": \"answer\", \"value\": \"2450000000\"}\n```"
  ]
}
```

### test_0036 (one_lookup)

Question: Using the company records, report the budget for project_5b23fb92f2be.
Expected: `967`; answer: `The budget for project_5b23fb92f2be is $1.2 million.`; stop: `answered`.

```json
{
  "actions": [
    {
      "action": "answer",
      "value": "The budget for project_5b23fb92f2be is $1.2 million."
    }
  ],
  "observations": [],
  "raw_outputs": [
    "```json\n{\"action\": \"answer\", \"value\": \"The budget for project_5b23fb92f2be is $1.2 million.\"}\n```"
  ]
}
```

### test_0046 (two_lookup)

Question: Which department is associated with the employee listed as the owner of project_2ca224b36adb?
Expected: `department_91227`; answer: `department`; stop: `answered`.

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

### test_0039 (one_lookup)

Question: Using the company records, report the budget for project_8f92a1491e0e.
Expected: `1690`; answer: `The budget for project_8f92a1491e0e is not available in the provided company records.`; stop: `answered`.

```json
{
  "actions": [
    {
      "action": "answer",
      "value": "The budget for project_8f92a1491e0e is not available in the provided company records."
    }
  ],
  "observations": [],
  "raw_outputs": [
    "```json\n{\"action\": \"answer\", \"value\": \"The budget for project_8f92a1491e0e is not available in the provided company records.\"}\n```"
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

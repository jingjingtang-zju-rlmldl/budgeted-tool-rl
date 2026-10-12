# Real-model baseline results

Model: `Qwen/Qwen3-0.6B` at `c1899de289a04d12100db370d81485cdf75e47ca`.
GPU: NVIDIA GeForce RTX 5060 Laptop GPU; dtype: BF16; thinking disabled; batch size 1.

**Actual CUDA inference on synthetic development tasks. No SFT or RL was run.**

| Task family | Correct / total | Success | Strict success | Mean calls | Calls on correct answers | Correct extra calls |
|---|---:|---:|---:|---:|---:|---:|
| overall | 29 / 60 | 48.3% | 13.3% | 0.150 | 0.310 | 0.000 |
| direct | 20 / 20 | 100.0% | 25.0% | 0.000 | 0.000 | 0.000 |
| one_lookup | 9 / 20 | 45.0% | 15.0% | 0.450 | 1.000 | 0.000 |
| two_lookup | 0 / 20 | 0.0% | 0.0% | 0.000 | n/a | n/a |

Parser: `fenced`. Strict-valid requests: 22 / 69; fenced wrappers normalized: 47.
Fence normalization removes only a single complete Markdown wrapper; it never fixes JSON, arguments or answers.

## Hardware and throughput

- Model parameters in memory: 1136.9 MiB.
- Peak CUDA allocated: 1209.5 MiB; peak reserved: 1224.0 MiB.
- Generated tokens: 1597; summed generation time: 70.53 seconds.
- Generation throughput: 22.64 tokens/s, including prefill and sampling, excluding model load.
- Mean task wall time: 1.18 seconds.

## Stops

```json
{
  "answered": 60
}
```

## Representative failures

### test_0057 (two_lookup)

Question: From company records, give the department of the person marked as the owner of project_8c13141d6bdb.
Expected: `department_18498`; answer: `Department`; stop: `answered`.

```json
{
  "actions": [
    {
      "action": "answer",
      "value": "Department"
    }
  ],
  "observations": [],
  "raw_outputs": [
    "{\"action\": \"answer\", \"value\": \"Department\"}"
  ]
}
```

### test_0054 (two_lookup)

Question: From company records, give the department of the person marked as the owner of project_4181d4c35e60.
Expected: `department_73056`; answer: `department`; stop: `answered`.

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

### test_0047 (two_lookup)

Question: Identify the owner of project_1e7e08cb1834 and report that person's department.
Expected: `department_11358`; answer: `Owner: John Doe, Department: Engineering`; stop: `answered`.

```json
{
  "actions": [
    {
      "action": "answer",
      "value": "Owner: John Doe, Department: Engineering"
    }
  ],
  "observations": [],
  "raw_outputs": [
    "```json\n{\n  \"action\": \"answer\",\n  \"value\": \"Owner: John Doe, Department: Engineering\"\n}\n```"
  ]
}
```

### test_0027 (one_lookup)

Question: Using the company records, report the budget for project_646253650f41.
Expected: `2288`; answer: `The budget for project_646253650f41 is $2.5 million.`; stop: `answered`.

```json
{
  "actions": [
    {
      "action": "answer",
      "value": "The budget for project_646253650f41 is $2.5 million."
    }
  ],
  "observations": [],
  "raw_outputs": [
    "```json\n{\"action\": \"answer\", \"value\": \"The budget for project_646253650f41 is $2.5 million.\"}\n```"
  ]
}
```

### test_0034 (one_lookup)

Question: What number is saved in the budget field of project_2a8db5930b02?
Expected: `2812`; answer: `4500000`; stop: `answered`.

```json
{
  "actions": [
    {
      "action": "answer",
      "value": "4500000"
    }
  ],
  "observations": [],
  "raw_outputs": [
    "```json\n{\"action\": \"answer\", \"value\": \"4500000\"}\n```"
  ]
}
```

### test_0036 (one_lookup)

Question: Using the company records, report the budget for project_5b23fb92f2be.
Expected: `967`; answer: `The budget for project_5b23fb92f2be is $500,000.`; stop: `answered`.

```json
{
  "actions": [
    {
      "action": "answer",
      "value": "The budget for project_5b23fb92f2be is $500,000."
    }
  ],
  "observations": [],
  "raw_outputs": [
    "```json\n{\"action\": \"answer\", \"value\": \"The budget for project_5b23fb92f2be is $500,000.\"}\n```"
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
    "```json\n{\"action\": \"answer\", \"value\": \"department\"}\n```"
  ]
}
```

### test_0039 (one_lookup)

Question: Using the company records, report the budget for project_8f92a1491e0e.
Expected: `1690`; answer: `The budget for project_8f92a1491e0e is not available in the provided data.`; stop: `answered`.

```json
{
  "actions": [
    {
      "action": "answer",
      "value": "The budget for project_8f92a1491e0e is not available in the provided data."
    }
  ],
  "observations": [],
  "raw_outputs": [
    "```json\n{\"action\": \"answer\", \"value\": \"The budget for project_8f92a1491e0e is not available in the provided data.\"}\n```"
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

# Real-model baseline results

Model: `Qwen/Qwen3-0.6B` at `c1899de289a04d12100db370d81485cdf75e47ca`.
GPU: NVIDIA GeForce RTX 5060 Laptop GPU; dtype: BF16; thinking disabled; batch size 1.

**Actual CUDA inference on synthetic development tasks. No SFT or RL was run.**

| Task family | Correct / total | Success | Strict success | Mean calls | Calls on correct answers | Correct extra calls |
|---|---:|---:|---:|---:|---:|---:|
| overall | 4 / 9 | 44.4% | 0.0% | 0.222 | 0.250 | 0.000 |
| direct | 3 / 3 | 100.0% | 0.0% | 0.000 | 0.000 | 0.000 |
| one_lookup | 1 / 3 | 33.3% | 0.0% | 0.667 | 1.000 | 0.000 |
| two_lookup | 0 / 3 | 0.0% | 0.0% | 0.000 | n/a | n/a |

Parser: `fenced`. Strict-valid requests: 1 / 11; fenced wrappers normalized: 9.
Fence normalization removes only a single complete Markdown wrapper; it never fixes JSON, arguments or answers.

## Hardware and throughput

- Model parameters in memory: 1136.9 MiB.
- Peak CUDA allocated: 1215.0 MiB; peak reserved: 1262.0 MiB.
- Generated tokens: 291; summed generation time: 22.64 seconds.
- Generation throughput: 12.86 tokens/s, including prefill and sampling, excluding model load.
- Mean task wall time: 2.53 seconds.

## Stops

```json
{
  "answered": 8,
  "invalid_action": 1
}
```

## Representative failures

### calibration_4a3f99f9229c (two_lookup)

Question: Find the department of the employee who owns project_calibration_4a3f99f9229c.
Expected: `department_55515`; answer: `The department of the employee who owns project_calibration_4a3f99f9229c is not available.`; stop: `answered`.

```json
{
  "actions": [
    {
      "action": "answer",
      "value": "The department of the employee who owns project_calibration_4a3f99f9229c is not available."
    }
  ],
  "observations": [],
  "raw_outputs": [
    "```json\n{\n  \"action\": \"answer\",\n  \"value\": \"The department of the employee who owns project_calibration_4a3f99f9229c is not available.\"\n}\n```"
  ]
}
```

### calibration_667df7b38b80 (one_lookup)

Question: Find the budget recorded for project_calibration_667df7b38b80.
Expected: `6144`; answer: `The budget recorded for project_calibration_667df7b38b80 is $500,000.`; stop: `answered`.

```json
{
  "actions": [
    {
      "action": "answer",
      "value": "The budget recorded for project_calibration_667df7b38b80 is $500,000."
    }
  ],
  "observations": [],
  "raw_outputs": [
    "```json\n{\"action\": \"answer\", \"value\": \"The budget recorded for project_calibration_667df7b38b80 is $500,000.\"}\n```"
  ]
}
```

### calibration_801cdfc64228 (two_lookup)

Question: What department does the owner of project_calibration_801cdfc64228 belong to?
Expected: `department_11191`; answer: `department`; stop: `answered`.

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

### calibration_f34e467dda17 (one_lookup)

Question: What is the budget of project_calibration_f34e467dda17?
Expected: `7288`; answer: `None`; stop: `invalid_action`.

```json
{
  "actions": [
    {
      "action": "lookup",
      "entity": "project_calibration_f34e467dda17",
      "field": "budget"
    }
  ],
  "observations": [
    {
      "entity": "project_calibration_f34e467dda17",
      "field": "budget",
      "value": 7288
    }
  ],
  "raw_outputs": [
    "```json\n{\"action\": \"lookup\", \"entity\": \"project_calibration_f34e467dda17\", \"field\": \"budget\"}\n```",
    "```json\n{\"action\": \"answer\", \"value\": \"7288\"}"
  ]
}
```

### calibration_6a590ef4b903 (two_lookup)

Question: Return the department of the owner recorded for project_calibration_6a590ef4b903.
Expected: `department_26581`; answer: `Department: Engineering`; stop: `answered`.

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

## Interpretation limits

- Real CUDA model inference on synthetic English tasks; no training performed.
- One sampled attempt per task; observed success is not a repeated-trial pass@k study.
- Development tasks used for analysis; future final test must be independently generated.
- Wilson intervals are descriptive under binomial assumptions; controlled templates are not IID real-world tasks.
- PyTorch memory measures this process; nvidia-smi includes other apps and WDDM accounting.
- Inference feasibility does not establish backward/optimizer or GRPO training feasibility.

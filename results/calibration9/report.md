# Real-model baseline results

Model: `Qwen/Qwen3-0.6B` at `c1899de289a04d12100db370d81485cdf75e47ca`.
GPU: NVIDIA GeForce RTX 5060 Laptop GPU; dtype: BF16; thinking disabled; batch size 1.

**Actual CUDA inference on synthetic development tasks. No SFT or RL was run.**

| Task family | Correct / total | Success | Mean calls | Calls on correct answers | Correct extra calls |
|---|---:|---:|---:|---:|---:|
| overall | 3 / 9 | 33.3% | 0.222 | 0.000 | 0.000 |
| direct | 3 / 3 | 100.0% | 0.000 | 0.000 | 0.000 |
| one_lookup | 0 / 3 | 0.0% | 0.667 | n/a | n/a |
| two_lookup | 0 / 3 | 0.0% | 0.000 | n/a | n/a |

## Hardware and throughput

- Model parameters in memory: 1136.9 MiB.
- Peak CUDA allocated: 1193.8 MiB; peak reserved: 1222.0 MiB.
- Generated tokens: 191; summed generation time: 25.39 seconds.
- Generation throughput: 7.52 tokens/s, including prefill and sampling, excluding model load.
- Mean task wall time: 2.84 seconds.

## Stops

```json
{
  "answered": 5,
  "invalid_action": 4
}
```

## Representative failures

### calibration_9e9f69a74b94 (one_lookup)

Question: Return the budget of project_calibration_9e9f69a74b94 from the private database.
Expected: `7546`; answer: `no data available`; stop: `answered`.

```json
{
  "actions": [
    {
      "action": "lookup",
      "entity": "project_123",
      "field": "budget"
    },
    {
      "action": "answer",
      "value": "no data available"
    }
  ],
  "observations": [
    {
      "error": "unknown entity",
      "entity": "project_123"
    }
  ],
  "raw_outputs": [
    "{\"action\":\"lookup\",\"entity\":\"project_123\",\"field\":\"budget\"}",
    "{\"action\":\"answer\",\"value\":\"no data available\"}"
  ]
}
```

### calibration_4a3f99f9229c (two_lookup)

Question: Find the department of the employee who owns project_calibration_4a3f99f9229c.
Expected: `department_55515`; answer: `None`; stop: `invalid_action`.

```json
{
  "actions": [],
  "observations": [],
  "raw_outputs": [
    "```json\n{\"action\": \"lookup\", \"entity\": \"project_123\", \"field\": \"owner\"}\n```"
  ]
}
```

### calibration_667df7b38b80 (one_lookup)

Question: Find the budget recorded for project_calibration_667df7b38b80.
Expected: `6144`; answer: `unknown`; stop: `answered`.

```json
{
  "actions": [
    {
      "action": "lookup",
      "entity": "project_123",
      "field": "budget"
    },
    {
      "action": "answer",
      "value": "unknown"
    }
  ],
  "observations": [
    {
      "error": "unknown entity",
      "entity": "project_123"
    }
  ],
  "raw_outputs": [
    "{\"action\":\"lookup\",\"entity\":\"project_123\",\"field\":\"budget\"}",
    "{\"action\":\"answer\",\"value\":\"unknown\"}"
  ]
}
```

### calibration_801cdfc64228 (two_lookup)

Question: What department does the owner of project_calibration_801cdfc64228 belong to?
Expected: `department_11191`; answer: `None`; stop: `invalid_action`.

```json
{
  "actions": [],
  "observations": [],
  "raw_outputs": [
    "```json\n{\"action\":\"lookup\",\"entity\":\"project_123\",\"field\":\"owner\"}\n```"
  ]
}
```

### calibration_f34e467dda17 (one_lookup)

Question: What is the budget of project_calibration_f34e467dda17?
Expected: `7288`; answer: `None`; stop: `invalid_action`.

```json
{
  "actions": [],
  "observations": [],
  "raw_outputs": [
    "```json\n{\"action\": \"lookup\", \"entity\": \"project_123\", \"field\": \"budget\"}\n```"
  ]
}
```

### calibration_6a590ef4b903 (two_lookup)

Question: Return the department of the owner recorded for project_calibration_6a590ef4b903.
Expected: `department_26581`; answer: `None`; stop: `invalid_action`.

```json
{
  "actions": [],
  "observations": [],
  "raw_outputs": [
    "```json\n{\"action\":\"answer\",\"value\":\"employee_456\"}\n```"
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

# Real-model baseline results

Model: `Qwen/Qwen3-0.6B` at `c1899de289a04d12100db370d81485cdf75e47ca`.
GPU: NVIDIA GeForce RTX 5060 Laptop GPU; dtype: BF16; thinking disabled; batch size 1.

**Actual CUDA inference on synthetic development tasks. No SFT or RL was run.**

| Task family | Correct / total | Success | Strict success | Mean calls | Calls on correct answers | Correct extra calls |
|---|---:|---:|---:|---:|---:|---:|
| overall | 60 / 60 | 100.0% | 100.0% | 1.000 | 1.000 | 0.000 |
| direct | 20 / 20 | 100.0% | 100.0% | 0.000 | 0.000 | 0.000 |
| one_lookup | 20 / 20 | 100.0% | 100.0% | 1.000 | 1.000 | 0.000 |
| two_lookup | 20 / 20 | 100.0% | 100.0% | 2.000 | 2.000 | 0.000 |

Parser: `fenced`. Strict-valid requests: 120 / 120; fenced wrappers normalized: 0.
Fence normalization removes only a single complete Markdown wrapper; it never fixes JSON, arguments or answers.

## Hardware and throughput

- Model parameters in memory: 1141.2 MiB.
- Peak CUDA allocated: 1234.7 MiB; peak reserved: 1266.0 MiB.
- Generated tokens: 2367; summed generation time: 112.05 seconds.
- Generation throughput: 21.12 tokens/s, including prefill and sampling, excluding model load.
- Mean task wall time: 1.87 seconds.

## Stops

```json
{
  "answered": 60
}
```

## Representative failures

## Interpretation limits

- Real CUDA model inference on synthetic English tasks; no training performed.
- One sampled attempt per task; observed success is not a repeated-trial pass@k study.
- Development tasks used for analysis; future final test must be independently generated.
- Wilson intervals are descriptive under binomial assumptions; controlled templates are not IID real-world tasks.
- PyTorch memory measures this process; nvidia-smi includes other apps and WDDM accounting.
- Inference feasibility does not establish backward/optimizer or GRPO training feasibility.

import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import fmean, median


def failure_class(row):
    if row['success']:
        return 'correct'
    if row['stop_reason'] != 'answered':
        return row['stop_reason']
    if row['invalid_tool_calls']:
        return 'invalid_lookup_argument'
    if row['minimum_calls'] > 0 and row['calls'] == 0:
        return 'answer_without_required_lookup'
    if row['calls'] < row['minimum_calls']:
        return 'premature_answer'
    return 'wrong_lookup_or_answer'


def wilson(successes, n):
    if not n:
        return None
    z = 1.959963984540054
    p = successes / n
    center = (p + z*z/(2*n)) / (1 + z*z/n)
    half = z * ((p*(1-p)/n + z*z/(4*n*n)) ** .5) / (1 + z*z/n)
    return [center-half, center+half]


def summarize(rows):
    if not rows or len({r['task_id'] for r in rows}) != len(rows):
        raise ValueError('nonempty unique task outcomes required')
    successes = sum(r['success'] for r in rows)
    generation_seconds = sum(r['generation_seconds'] for r in rows)
    correct = [r for r in rows if r['success']]
    failed = [r for r in rows if not r['success']]
    return dict(tasks=len(rows), successes=successes, success_rate=successes/len(rows),
                strict_successes=sum(r.get('strict_success',False) for r in rows),
                strict_success_rate=sum(r.get('strict_success',False) for r in rows)/len(rows),
                requests=sum(len(r['requests']) for r in rows),
                strict_valid_requests=sum(r.get('strict_valid_requests',0) for r in rows),
                fence_normalized_requests=sum(r.get('fence_normalized_requests',0) for r in rows),
                descriptive_wilson_95=wilson(successes, len(rows)),
                mean_calls=fmean(r['calls'] for r in rows), mean_minimum_calls=fmean(r['minimum_calls'] for r in rows),
                correct_mean_calls=fmean(r['calls'] for r in correct) if correct else None,
                correct_extra_calls=fmean(r['calls']-r['minimum_calls'] for r in correct) if correct else None,
                generated_tokens=sum(r['generated_tokens'] for r in rows), generation_seconds=generation_seconds,
                generation_tokens_per_second=sum(r['generated_tokens'] for r in rows)/generation_seconds if generation_seconds else None,
                mean_task_wall_seconds=fmean(r['wall_seconds'] for r in rows),
                median_task_wall_seconds=median(r['wall_seconds'] for r in rows),
                repeated_calls=sum(r['repeated_calls'] for r in rows), invalid_tool_calls=sum(r['invalid_tool_calls'] for r in rows),
                stop_reasons=dict(Counter(r['stop_reason'] for r in rows)),
                failed_answered=sum(r['stop_reason']=='answered' for r in failed),
                direct_unnecessary_lookup_tasks=sum(r['kind']=='direct' and r['calls']>0 for r in rows),
                failure_classes=dict(Counter(failure_class(r) for r in failed)),
                peak_cuda_allocated_mib=max(r.get('cuda_peak_allocated_mib',0) for r in rows))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    args=parser.parse_args()
    manifest=json.loads((args.run/'manifest.json').read_text(encoding='utf-8'))
    rows=[json.loads(line) for line in (args.run/'episodes.jsonl').read_text(encoding='utf-8').splitlines()]
    overall=summarize(rows)
    kinds={kind:summarize([r for r in rows if r['kind']==kind]) for kind in ('direct','one_lookup','two_lookup') if any(r['kind']==kind for r in rows)}
    summary=dict(overall=overall, by_kind=kinds, manifest=manifest,
                 caveats=['Real CUDA model inference on synthetic English tasks; no training performed.',
                          'One sampled attempt per task; observed success is not a repeated-trial pass@k study.',
                          'Development tasks used for analysis; future final test must be independently generated.',
                          'Wilson intervals are descriptive under binomial assumptions; controlled templates are not IID real-world tasks.',
                          'PyTorch memory measures this process; nvidia-smi includes other apps and WDDM accounting.',
                          'Inference feasibility does not establish backward/optimizer or GRPO training feasibility.'])
    (args.run/'summary.json').write_text(json.dumps(summary, indent=2)+'\n',encoding='utf-8')
    lines=['# Real-model baseline results','',f"Model: `{manifest['config']['model_id']}` at `{manifest['config']['model_revision']}`.",
           f"GPU: {manifest['gpu']}; dtype: BF16; thinking disabled; batch size 1.",'',
           '**Actual CUDA inference on synthetic development tasks. No SFT or RL was run.**','',
           '| Task family | Correct / total | Success | Strict success | Mean calls | Calls on correct answers | Correct extra calls |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for kind,value in [('overall',overall),*kinds.items()]:
        fmt=lambda x:'n/a' if x is None else f'{x:.3f}'
        lines.append(f"| {kind} | {value['successes']} / {value['tasks']} | {value['success_rate']:.1%} | {value['strict_success_rate']:.1%} | {value['mean_calls']:.3f} | {fmt(value['correct_mean_calls'])} | {fmt(value['correct_extra_calls'])} |")
    lines+=['',f"Parser: `{manifest['config'].get('parser','strict')}`. Strict-valid requests: {overall['strict_valid_requests']} / {overall['requests']}; fenced wrappers normalized: {overall['fence_normalized_requests']}.",
            'Fence normalization removes only a single complete Markdown wrapper; it never fixes JSON, arguments or answers.',
            '','## Hardware and throughput','',f"- Model parameters in memory: {manifest.get('parameter_mib',0):.1f} MiB.",
            f"- Peak CUDA allocated: {manifest.get('cuda_peak_allocated_mib',0):.1f} MiB; peak reserved: {manifest.get('cuda_peak_reserved_mib',0):.1f} MiB.",
            f"- Generated tokens: {overall['generated_tokens']}; summed generation time: {overall['generation_seconds']:.2f} seconds.",
            f"- Generation throughput: {overall['generation_tokens_per_second']:.2f} tokens/s, including prefill and sampling, excluding model load.",
            f"- Mean task wall time: {overall['mean_task_wall_seconds']:.2f} seconds.",'','## Stops','',
            '```json',json.dumps(overall['stop_reasons'],indent=2),'```','','## Representative failures','']
    for row in [r for r in rows if not r['success']][:8]:
        lines += [f"### {row['task_id']} ({row['kind']})",'',f"Question: {row['messages'][row.get('question_index',1)]['content']}",
                  f"Expected: `{row['expected']}`; answer: `{row['answer']}`; stop: `{row['stop_reason']}`.",'',
                  '```json',json.dumps(dict(actions=row['actions'],observations=row['observations'],
                                          raw_outputs=[r['text'] for r in row['requests']]),indent=2),'```','']
    lines+=['## Interpretation limits','',*['- '+x for x in summary['caveats']]]
    (args.run/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'overall':overall,'by_kind':kinds},indent=2))


if __name__=='__main__':
    main()

"""Package local evidence and the small LoRA adapter without base weights."""
from pathlib import Path
import hashlib
import json
import zipfile

root = Path(__file__).resolve().parent.parent
project = root / 'outputs' / 'budgeted-tool-rl'
target = root / 'outputs' / 'budgeted-tool-rl.zip'
excluded = {'__pycache__', '.git', '.venv', 'models', 'build', 'dist'}
files = sorted(p for p in project.rglob('*') if p.is_file()
               and not any(part in excluded or part.endswith('.egg-info') for part in p.relative_to(project).parts)
               and p.suffix not in {'.pyc', '.pyo'}
               and not p.name.startswith('.env'))
with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
    for path in files:
        archive.write(path, path.relative_to(project.parent).as_posix())
with zipfile.ZipFile(target) as archive:
    assert archive.testzip() is None
    names = archive.namelist()
    for required in ['README.md', 'toolrl/baseline.py', '.github/workflows/environment.yml',
                     'results/dev100/episodes.jsonl', 'results/dev100/analysis.zh-CN.md',
                     'results/dev100/verification.json', 'data/dev100.jsonl',
                     'results/sft-v1/analysis.zh-CN.md', 'results/sft-v1/summary.json',
                     'results/sft-v1/training-audit.json',
                     'toolrl/stress_runner.py', 'tests/test_stress.py',
                     'data/stress-v1/dev.jsonl', 'data/stress-v1/test.jsonl',
                     'results/stress-v1/analysis.zh-CN.md', 'results/stress-v1/summary.json',
                     'results/stress-v1/selection.json', 'results/stress-v1/action-budget-audit.json',
                     'toolrl/sft2_experiment.py', 'tests/test_sft2.py',
                     'data/sft-v2/train.jsonl', 'data/sft-v2/test.jsonl', 'data/sft-v2/signal.jsonl',
                     'results/sft-v2/analysis.zh-CN.md', 'results/sft-v2/summary.json',
                     'results/sft-v2/reward-signal.json', 'results/sft-v2/training-audit.json',
                     'results/sft-v2/model-card.md', 'results/sft-v2/reproduce.md',
                     'results/sft-v2/plan-original.json', 'results/sft-v2/resource-amendment.json',
                     'results/sft-v2/action-budget-audit.json', 'results/sft-v2/sft2-results.png',
                     'results/sft-v2/test-analysis-before-signal.json',
                     'results/sft-v2/training/adapter/adapter_model.safetensors',
                     'toolrl/sft3_data.py', 'toolrl/sft3_episode.py', 'toolrl/sft3_verify.py',
                     'toolrl/sft3_experiment.py', 'tests/test_sft3.py', 'data/sft-v3/train.jsonl',
                     'data/sft-v3/pair.jsonl', 'data/sft-v3/recovery.jsonl', 'data/sft-v3/signal.jsonl',
                     'results/sft-v3/plan.json', 'results/sft-v3/analysis.zh-CN.md',
                     'results/sft-v3/summary.json', 'results/sft-v3/reward-signal.json',
                     'toolrl/sft3_report_compat.py', 'toolrl/sft3_breakdown.py',
                     'results/sft-v3/report-only-amendment.json',
                     'results/sft-v3/descriptive-breakdown.json',
                     'results/sft-v3/descriptive-breakdown.zh-CN.md',
                     'results/sft-v3/paired-before-training.json', 'results/sft-v3/paired-after-training.json',
                     'results/sft-v3/test-analysis-before-signal.json', 'results/sft-v3/training-audit.json',
                     'results/sft-v3/reproduce.md', 'results/sft-v3/model-card.md', 'results/sft-v3/sft3-results.png',
                     'results/sft-v3/training/adapter/adapter_model.safetensors',
                     'results/sft-v1/training/adapter/adapter_model.safetensors']:
        assert 'budgeted-tool-rl/' + required in names
    assert not any('__pycache__' in name for name in names)
    assert [n for n in names if n.endswith('.safetensors')]==[
        'budgeted-tool-rl/results/sft-v1/training/adapter/adapter_model.safetensors',
        'budgeted-tool-rl/results/sft-v2/training/adapter/adapter_model.safetensors',
        'budgeted-tool-rl/results/sft-v3/training/adapter/adapter_model.safetensors']
    for seed in (30007, 40007, 50007, 60007):
        cohort = project / 'results/sft-v2/eval' / f'signal-new-protocol-seed{seed}'
        manifest = json.loads((cohort / 'manifest.json').read_text(encoding='utf-8'))
        verification = json.loads((cohort / 'verification.json').read_text(encoding='utf-8'))
        assert manifest['status'] == 'complete' and verification['tasks_verified'] == 32
    for seed in (30017, 40017, 50017, 60017):
        cohort = project / 'results/sft-v3/eval' / f'signal-new-protocol-seed{seed}'
        manifest = json.loads((cohort / 'manifest.json').read_text(encoding='utf-8'))
        verification = json.loads((cohort / 'verification.json').read_text(encoding='utf-8'))
        assert manifest['status'] == 'complete' and verification['tasks_verified'] == 32
    assert all(project_file.stat().st_size<30_000_000 for project_file in files)
print(json.dumps({'archive': str(target), 'files': len(files), 'bytes': target.stat().st_size,
                  'sha256': hashlib.sha256(target.read_bytes()).hexdigest()}, indent=2))

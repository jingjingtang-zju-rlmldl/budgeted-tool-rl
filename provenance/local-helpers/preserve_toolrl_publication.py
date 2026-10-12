"""Archive the user-requested stop; never generates model outputs or updates weights."""
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

workspace = Path(__file__).resolve().parents[1]
project = workspace / 'outputs/budgeted-tool-rl'
sys.path.insert(0, str(project))
from toolrl.sft4_data import harvest
from toolrl.sft4_experiment import validate, build_train

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n', encoding='utf-8', newline='\n')

import os
os.chdir(project)
root = project / 'results/sft-v4'
plan = validate(root/'plan.json')
assert not build_train(plan)[1]
assert not (root/'training/manifest.json').exists()
assert not (root/'training-set-manifest.json').exists()
assert not list((root/'eval').glob('signal-*'))
stop = root/'interruption'
stop.mkdir(exist_ok=True)
cohorts = []
eligible = set()
for manifest_path in sorted(list((root/'eval').glob('*/manifest.json')) + list((root/'copy-probe/eval').glob('*/manifest.json'))):
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    raw_path = manifest_path.parent/'episodes.jsonl'
    raw = raw_path.read_bytes()
    decoded = []
    malformed = []
    for i, line in enumerate(raw.splitlines()):
        try: decoded.append(json.loads(line))
        except (ValueError, UnicodeDecodeError): malformed.append(i+1)
    split = manifest['config']['split']
    if split in ('collect', 'reserve', 'copy_probe'):
        dataset = (root/'copy-probe/data/copy_probe.jsonl') if split == 'copy_probe' else project/'data/sft-v4'/f'{split}.jsonl'
        tasks = {t['task_id']:t for t in map(json.loads, dataset.read_text(encoding='utf-8').splitlines())}
        for row in decoded:
            found = harvest(tasks[row['task_id']], row)
            if found: eligible.add(found['fact_group'])
    verification = manifest_path.parent/'verification.json'
    if manifest['status'] == 'complete':
        v = json.loads(verification.read_text(encoding='utf-8'))
        assert v['episodes_sha256'] == sha(raw_path)
        assert v['tasks_verified'] == len(decoded) == manifest['config']['task_count']
        assert not malformed
    else:
        shutil.copyfile(manifest_path, stop/(manifest_path.parent.name+'.manifest-at-stop.json'))
    cohorts.append(dict(path=manifest_path.parent.relative_to(project).as_posix(),
        recorded_manifest_status=manifest['status'], publication_status='complete' if manifest['status']=='complete' else 'interrupted_by_user',
        complete_json_records=len(decoded), planned_records=manifest['config']['task_count'],
        malformed_lines=malformed, ends_with_newline=raw.endswith(b'\n'),
        manifest_sha256=sha(manifest_path), episodes_sha256=sha(raw_path),
        complete_cohort_verification_present=verification.exists(),
        grounded_correct=sum(x['grounded_success'] for x in decoded)))
assert len(eligible)==0
helpers = project/'provenance/local-helpers'
helpers.mkdir(parents=True, exist_ok=True)
exclude = {'publish.py','push_profile_edit.py','prepare_repos.py','finalize.py','download_qwen3.py'}
snapshots = {}
for path in sorted((workspace/'work').glob('*.py')):
    if path.name in exclude: continue
    shutil.copyfile(path, helpers/path.name)
    snapshots[path.name] = sha(path)
write(helpers/'snapshot-manifest.json', dict(recorded_at=datetime.now(timezone.utc).isoformat(),
    purpose='Exact historical helper snapshots; paths assume the original workspace. Do not execute these snapshots automatically.',
    files_sha256=snapshots))
(helpers/'README.md').write_text('''# Historical local helper snapshots

These are byte-preserved preparation, monitoring, packaging and driver scripts
from the original local workspace. Their workspace-relative paths are historical;
they are not a new execution entry point. Reproduction instructions live in each
`results/*/reproduce.md`. Some helpers were prepared but never run, notably the
v4 completion README/report helpers. The v4 amended driver was stopped by the user.
Do not run generator or maintenance snapshots against archived evidence.
No base model cache, virtual environment or credentials are included.
''', encoding='utf-8', newline='\n')
status = dict(recorded_at=datetime.now(timezone.utc).isoformat(), status='interrupted_by_user',
    user_instruction='先不要继续做实验；上传本地代码到 GitHub，保留记录。',
    timestamp_note='recorded_at is the preservation time, not a reconstructed exact process termination timestamp.',
    plan_sha256=sha(root/'plan.json'), original_manifests_unchanged=True,
    completed_cohorts=sum(c['publication_status']=='complete' for c in cohorts),
    complete_json_records=sum(c['complete_json_records'] for c in cohorts),
    natural_copy_facts=0, v4_training_performed=False, v4_training_memory_probe_performed=False,
    v4_new_model_evaluation_performed=False, v4_fresh_32x4_signal_performed=False,
    original_freeze_gate=dict(outcome='refused_empty_natural_copy_set',
        assertion='No actual eligible copy error; cannot claim correction SFT',
        provenance_note='Observed previous driver failure; this record does not rerun the failed pipeline.'),
    cohorts=cohorts)
write(stop/'user-stop.json', status)
baseline = json.loads((root/'baseline-before-training.json').read_text(encoding='utf-8'))['aggregate']
incomplete = [c for c in cohorts if c['publication_status']!='complete']
partial_count = sum(c['complete_json_records'] for c in incomplete)
report = f'''# 分支决策与查询成本实验：发布时状态

2026-10-12，应用户要求停止实验并归档。**本轮没有进行新 SFT、训练显存探测或全新 32×4 rollout 奖励诊断。** 已完成的 SFT v1/v2/v3 权重与证据原样保留。

## 已完成的真实模型基线

本轮使用已有 v3 adapter；“old”指该 adapter，不是未经训练的基座模型。
环境为确定性合成数据库，模型输出来自本地 RTX 5060 Laptop 的实际推理。
新的主指标要求答案具有已查询/已提供事实支持，并按工具收费计成本。
这改变了任务及奖励定义，不能直接把它与历轮分数作因果对照。

| 条件 | 自由起步测试（64 题×2 seed） | 注入 ID 错误恢复（32 题×2 seed） |
|---|---:|---:|
| v3 / 短协议 | {baseline['test']['old/protocol']['grounded_correct']}/128 | {baseline['recovery']['old/protocol']['grounded_correct']}/64 |
| v3 / 示例提示 | {baseline['test']['old/examples']['grounded_correct']}/128 | {baseline['recovery']['old/examples']['grounded_correct']}/64 |

示例提示的自由起步答案全部正确，但成功轨迹仍有 **{baseline['test']['old/examples']['successful_cost_regret']} 个合成费用单位**的总超额查询成本。错误恢复条件从脚本注入的真实工具状态开始，包含已收费错误；它不代表自然错误发生率。重复 seed 不是独立事实，两个测试分别只有 64 与 32 个独立事实。

## 采样与停止原因

原始训练采集完成四组、共 512 条真实 episode，在 256 个独立事实上没有发现满足预注册规则的自然单字符 ID 复制错误。类型词、占位符或任意错误 ID 不算单字符复制错误。冻结训练集的代码因此拒绝继续，未执行纠错训练。

按事先登记的条件规则，随后开展仅用于训练事实的字段提示、高温补充采样。该组原定 128 题，停止时仅保存 **{partial_count}/128** 条完整 JSON episode，仍没有合格自然单字符错误。它不具有正常部署错误频率的解释。第二个补充 seed 没有运行。

共保留 **{status['completed_cohorts']} 组完成采样 / {status['complete_json_records']} 条完整 JSON episode**，其中 928 条来自原始已完成采集和基线，{partial_count} 条来自中途停止的补充组。完整组原有 verification 记录可用；补充组未完成整组验证，本次仅解析已落盘记录、计算字节哈希和按原规则检查错误资格。被中断但未落盘的推理不计入这些数字。

## 证据与限制

- [停止记录与逐组哈希](interruption/user-stop.json) 明确标记 incomplete；原始 manifest 的 `running` 状态原样保留，并附停止时快照，不能理解为任务仍在运行。
- [训练前基线汇总](baseline-before-training.json)、`eval/*/episodes.jsonl`、各组 manifest/verification、冻结计划与数据、条件采样登记、资源维护前缀与变更记录全部保留。
- 当前没有 v4 adapter，也没有 v4 训练收益、RL 奖励方差或 RL 更新结果。用于后续训练/诊断的实现与命令属于待运行方案。
- 环境和奖励只支持这个可控实验；不构成通用 agent 或 RSI 能力证明。

基线表明示例提示能改善分支推进，成本控制和注入错误恢复仍存在问题。自然复制错误的数据采集未获得预定训练证据，不能据此宣称纠错 SFT 有效。实验已停止，发布阶段只整理记录。
'''
(root/'status.zh-CN.md').write_text(report, encoding='utf-8', newline='\n')
readme = project/'README.md'
text = readme.read_text(encoding='utf-8')
marker = '## Branch decisions and weighted tool costs (paused before new SFT)'
assert marker not in text
section = f'''{marker}

The latest experiment was stopped at the user's request on 2026-10-12.
It adds fresh facts, observation-dependent branches and priced query choices.
With the existing v3 adapter, free-start grounded success is **37/128** under
the short protocol and **128/128** with examples; injected ID-error recovery
is **10/64** and **17/64** respectively. Strong-prompt successful free-start
trajectories still incur **218 synthetic fee units** of total cost regret.
These new tasks and weighted rewards differ from earlier benchmarks.

Four completed training-collection cohorts found **zero eligible natural
single-character ID-copy failures** across 512 episodes. The correction-training
gate refused that empty set. A preregistered, training-only hinted/high-temperature
probe was interrupted after **{partial_count}/128** records. **No v4 SFT, new
training-memory measurement or fresh v4 32×4 reward diagnosis was performed.**
Completed and partial raw records are retained; the interrupted manifest is
preserved with an explicit stop sidecar, rather than relabeled complete.

Read the [current Chinese status and baseline analysis](results/sft-v4/status.zh-CN.md),
[stop record](results/sft-v4/interruption/user-stop.json),
[baseline metrics](results/sft-v4/baseline-before-training.json), and
[prospective reproduction protocol](results/sft-v4/reproduce.md).

'''
text = text.replace('## Wording control and invalid-query correction SFT', section+'## Wording control and invalid-query correction SFT', 1)
readme.write_text(text, encoding='utf-8', newline='\n')
reproduce = root/'reproduce.md'
old = reproduce.read_text(encoding='utf-8')
reproduce.write_text(old.replace('\n\n', '\n\n> Publication status: interrupted by the user before v4 training. These commands\n> describe the planned workflow, not completed training or signal evidence.\n> See [current status](status.zh-CN.md) and [stop record](interruption/user-stop.json).\n\n', 1), encoding='utf-8', newline='\n')
print(json.dumps({k:status[k] for k in ['status','completed_cohorts','complete_json_records','natural_copy_facts','v4_training_performed']}, ensure_ascii=False))

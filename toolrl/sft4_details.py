"""Post-hoc source-audited error/cost descriptions; never used for model selection."""
from collections import Counter,defaultdict
import json
from pathlib import Path
from .sft4_experiment import read,rows,sha256,write
from .sft4_data import edit_type
from .sft4_report import metrics


def main():
    root=Path('results/sft-v4');s=read(root/'summary.json');p=read(root/'plan.json');raw={}
    for name,c in s['cohorts'].items():
        directory=root/'copy-probe/eval'/name.split('/',1)[1] if name.startswith('copy-probe/') else root/'eval'/name
        assert sha256(directory/'episodes.jsonl')==c['episodes_sha256'];raw[name]=rows(directory/'episodes.jsonl')
    def control(split,c):
        model,prompt=c.split('/')
        return [r for name,rs in raw.items() if name.startswith(f'{split}-{model}-{prompt}-seed') for r in rs]
    cost_comparisons={}
    new={(r['task_id'],r['seed']):r for r in control('test','new/protocol')}
    for c in ('old/protocol','old/examples'):
        old={(r['task_id'],r['seed']):r for r in control('test',c)};assert old.keys()==new.keys()
        diffs=[old[k]['total_cost']-new[k]['total_cost'] for k in old if old[k]['grounded_success'] and new[k]['grounded_success']]
        cost_comparisons[c]=dict(both_grounded=len(diffs),new_cheaper=sum(x>0 for x in diffs),equal_cost=sum(x==0 for x in diffs),
                                new_more_expensive=sum(x<0 for x in diffs),net_cost_saved_on_both_grounded=sum(diffs))
    tasks={t['task_id']:t for t in rows(Path(p['data_root'])/'recovery.jsonl')};recovery={}
    for c in p['eval_conditions']:
        by=defaultdict(list)
        for r in control('recovery',c):
            t=tasks[r['task_id']];by[t['correction_mode'],t['correction_stage']].append(r)
        recovery[c]={f'{typ}/stage{stage}':metrics(rs,tasks) for (typ,stage),rs in by.items()}
    train=rows(Path(p['data_root'])/'derived_train.jsonl');natural=[]
    for t in train:
        if not t.get('natural_copy_type'):continue
        bad=t['bootstrap_calls'][-1]['entity'];good=t['correct_copied_entity']
        assert edit_type(bad,good)==t['natural_copy_type']
        source=next(r for r in raw[t['source_cohort']] if r['task_id']==t['source_task_id'] and r['seed']==t['source_seed'])
        assert source['requests'][t['source_request_index']]['text']==t['bootstrap_texts'][-1]
        natural.append(dict(training_task_id=t['task_id'],copy_type=t['natural_copy_type'],source=t['source_cohort'],
                            source_task_id=t['source_task_id'],seed=t['source_seed'],bad_entity=bad,correct_known_entity=good,
                            actual_bad_text=t['bootstrap_texts'][-1],supervised_next_action=t['remaining_oracle_calls'][0]))
    test={t['task_id']:t for t in rows(Path(p['data_root'])/'test.jsonl')};failed=[]
    for r in new.values():
        if r['grounded_success']:continue
        failed.append(dict(task_id=r['task_id'],seed=r['seed'],kind=r['kind'],question=test[r['task_id']]['prompt'],
                           stop=r['stop_reason'],exact_correct=r['success'],first_action=r['actions'][0] if r['actions'] else None,
                           actions=r['actions'],observations=r['observations']))
    effective=[g for g in s['signal_groups']['details'] if g['population_reward_variance']>0]
    out=dict(post_hoc_descriptive_only=True,source_sha256=sha256(Path(__file__)),
             episodes_sha256={n:c['episodes_sha256'] for n,c in s['cohorts'].items()},
             same_task_cost_comparisons=cost_comparisons,recovery_by_edit_and_stage=recovery,
             actual_natural_prefixes=natural,new_test_failed_cases=failed,nonconstant_signal_groups=effective)
    write(root/'descriptive-details.json',out)
    text=['# 费用配对、自然复制错误与剩余失败','',
          '这是固定实验完成后的描述性分解；未用于选 checkpoint、改任务、调采样或回填测试事实。','',
          '## 同题／同种子费用比较','',
          '| 旧条件 | 两边可溯源正确 | 新模型更便宜 | 等费用 | 新模型更贵 | 两边正确样本净节省费用 |',
          '|---|---:|---:|---:|---:|---:|']
    for c,x in cost_comparisons.items():text.append(f"| {c} | {x['both_grounded']} | {x['new_cheaper']} | {x['equal_cost']} | {x['new_more_expensive']} | {x['net_cost_saved_on_both_grounded']} |")
    text += ['', '此表条件于两边都正确，不能取代全样本奖励和成功率；成功/失败转换见主报告配对统计。费用是合成单位，不是实际 API 金额。','',
             '## 实际学生复制前缀','',f'独立训练事实数 {len(natural)}；类型 {dict(Counter(t["copy_type"] for t in natural))}。所有来源原文和哈希已复核。']
    if 'copy_probe_amendment_sha256' in s:text += ['', '普通提示原定512条没有符合规则的字符错误。下表前缀来自已登记的训练用途字段提示／温度1.2探针，不能作为普通提示或部署错误率；不是脚本扰动。测试、奖励诊断始终保持原0.7采样。']
    for t in natural[:8]:
        text += ['',f"- {t['source_task_id']}，种子 {t['seed']}，{t['copy_type']}：实际 `{t['bad_entity']}`；此前可见正确 ID `{t['correct_known_entity']}`。"]
    text += ['', '## 注入错误恢复按类型与位置','',
             '| 扰动／位置 | 条件 | 可溯源正确 | 同状态最优费用且正确 | 正确轨迹费用遗憾 |',
             '|---|---|---:|---:|---:|']
    for key in sorted(recovery['new/protocol']):
        for c in p['eval_conditions']:
            x=recovery[c][key];text.append(f"| {key} | {c} | {x['grounded_correct']}/{x['attempts']} | {x['optimal_weighted_correct']}/{x['attempts']} | {x['successful_cost_regret']} |")
    text += ['', '每个错误／位置含两种措辞×两个种子，四次尝试。事实仍只有两题。位置0/1/2/3分别是 project.policy、project.owner、employee.department、department.cost_center。错误类型与措辞交叉，固定注入错误照常计入预算。','',
             '## 新模型自由测试失败','',f'失败尝试 {len(failed)}；不同事实 {len({r["task_id"] for r in failed})}。']
    for r in failed[:12]:text += ['',f"- {r['task_id']}，种子{r['seed']}，{r['stop']}，首动作 `{json.dumps(r['first_action'])}`。"]
    text += ['', '## 全新奖励的有效组','',f'非恒定组 {len(effective)}/32，覆盖 {s["signal_groups"]["effective_kinds"]}。']
    for g in effective:text += ['',f"- {g['task_id']}：{g['classification']}；费用 {g['weighted_costs']}，奖励 {g['rewards']}，总体方差 {g['population_reward_variance']:.6g}。"]
    text += ['', 'G=4 的有限采样不能估计所有失败概率或保证 policy-gradient 改善。全错组没有可比较的正确样本；全对等费用组也没有当前组内优势。不要把本轮诊断题回填训练。',
             '', '[机器可读细节与来源哈希](descriptive-details.json)。','']
    (root/'descriptive-details.zh-CN.md').write_text('\n'.join(text),encoding='utf-8',newline='\n')
    print(json.dumps(dict(event='descriptive_details',natural_prefixes=len(natural),new_test_failed_attempts=len(failed),cost_comparisons=cost_comparisons)))

if __name__=='__main__':main()

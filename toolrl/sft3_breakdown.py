"""Post-hoc descriptive breakdown; does not generate, select, or train models."""
from collections import Counter,defaultdict
import json
from pathlib import Path
from .sft3_experiment import read,rows,sha256,write
from .sft2_report import metrics

def main():
    root=Path('results/sft-v3');summary=read(root/'summary.json');plan=read(root/'plan.json')
    raw={}
    for name,v in summary['cohorts'].items():
        p=root/'eval'/name/'episodes.jsonl';assert sha256(p)==v['episodes_sha256'];raw[name]=rows(p)
    def cohort(split,c):
        model,prompt=c.split('/')
        return [r for name,rs in raw.items() if name.startswith(f'{split}-{model}-{prompt}-seed') for r in rs]
    conditions=plan['eval_conditions'];test={c:cohort('test',c) for c in conditions}
    by_kind={k:{c:metrics([r for r in rs if r['kind']==k]) for c,rs in test.items()} for k in sorted({r['kind'] for r in test['new/protocol']})}
    def matched(old,new):
        a={(r['task_id'],r['seed']):r for r in old};b={(r['task_id'],r['seed']):r for r in new};assert a.keys()==b.keys()
        return dict(gained=sum(not a[k]['success'] and b[k]['success'] for k in a),lost=sum(a[k]['success'] and not b[k]['success'] for k in a),
                    both_correct=sum(a[k]['success'] and b[k]['success'] for k in a),both_failed=sum(not a[k]['success'] and not b[k]['success'] for k in a))
    comparisons={c:matched(test[c],test['new/protocol']) for c in conditions if c!='new/protocol'}
    def first_bad_fields(rs):
        return dict(Counter(r['call_events'][0]['field'] for r in rs if r['call_events'] and r['call_events'][0]['observation'].get('error')=='unknown_field'))
    first_field_misuse={c:first_bad_fields(rs) for c,rs in test.items()}
    test_tasks={t['task_id']:t for t in rows(Path(plan['data_root'])/'test.jsonl')}
    new_failed_cases=[]
    for r in test['new/protocol']:
        if r['success']:continue
        t=test_tasks[r['task_id']];a=r['actions'][0] if r['actions'] else {};gold=t['oracle_calls'][0]['entity'] if t['oracle_calls'] else None
        bad=a.get('entity');deletion=bool(gold and bad and len(bad)==len(gold)-1 and any(gold[:i]+gold[i+1:]==bad for i in range(len(gold))))
        new_failed_cases.append(dict(task_id=r['task_id'],seed=r['seed'],kind=r['kind'],question=t['prompt'],
                                     first_action=a,first_oracle_entity=gold,single_character_deletion=deletion,stop=r['stop_reason']))
    tasks={t['task_id']:t for t in rows(Path(plan['data_root'])/'recovery.jsonl')};recovery={}
    for c in conditions:
        groups=defaultdict(list)
        for r in cohort('recovery',c):
            t=tasks[r['task_id']];groups[(t['correction_type'],t['correction_stage'])].append(r)
        recovery[c]={}
        for (typ,stage),rs in groups.items():
            def correct_first(r):
                t=tasks[r['task_id']];return bool(r['actions']) and r['actions'][0]==dict(action='lookup',**t['remaining_oracle_calls'][0])
            recovery[c][f'{typ}/stage{stage}']=dict(attempts=len(rs),first_action_matches_minimum_cost_repair=sum(correct_first(r) for r in rs),
                final_grounded_success=sum(r['grounded_success'] for r in rs),
                generated_invalid_queries=sum(e['invalid_query'] for r in rs for e in r['call_events'][len(r['bootstrap_calls']):]),
                extra_calls_after_fixed_injected_error=sum(r['calls']-r['minimum_calls']-1 for r in rs if r['success']))
    signal=cohort('signal','new/protocol')
    paired_by_kind={model:{k:{str(tid):metrics([r for r in cohort('pair',model+'/protocol') if r['kind']==k and r['template_id']==tid]) for tid in (0,1)}
                          for k in sorted({r['kind'] for r in signal})} for model in ('old','new')}
    out=dict(post_hoc_descriptive_only=True,source_sha256=sha256(Path(__file__)),episodes_sha256={k:v['episodes_sha256'] for k,v in summary['cohorts'].items()},
        test_by_kind=by_kind,paired_test_comparisons=comparisons,recovery_by_error_and_stage=recovery,paired_wording_by_kind=paired_by_kind,
        first_field_misuse_in_free_test=first_field_misuse,first_field_misuse_in_signal=first_bad_fields(signal),
        new_test_failed_cases=new_failed_cases,
        recovery_error_type_template_counts={typ:dict(Counter(str(t['template_id']) for t in tasks.values() if t['correction_type']==typ)) for typ in ('unknown_field','unknown_entity')},
        signal_by_kind={k:metrics([r for r in signal if r['kind']==k]) for k in sorted({r['kind'] for r in signal})},
        new_test_failure_stops=dict(Counter(r['stop_reason'] for r in test['new/protocol'] if not r['success'])),
        new_signal_failure_stops=dict(Counter(r['stop_reason'] for r in signal if not r['success'])))
    write(root/'descriptive-breakdown.json',out)
    text=['# 逐类结果与错误位置分解','',
          '这是固定实验完成后的描述性统计，没有用于挑选 checkpoint、修改任务或采样配置。重复种子仍属于同一个事实任务。','',
          '| 同事实配对类别 | v2 措辞A | v2 措辞B | v3 措辞A | v3 措辞B |','|---|---:|---:|---:|---:|']
    for k in paired_by_kind['old']:
        text.append('| '+k+' | '+' | '.join(f"{paired_by_kind[model][k][str(tid)]['successes']}/8" for model in ('old','new') for tid in (0,1))+' |')
    text += ['', '| 自由起始任务 | v2 简洁 | v2 强提示 | v3 简洁 |','|---|---:|---:|---:|']
    for k,cs in by_kind.items():text.append('| '+k+' | '+' | '.join(f"{cs[c]['successes']}/{cs[c]['attempts']}" for c in conditions)+' |')
    text += ['',f'新模型对原模型的同任务／同种子配对结果：{json.dumps(comparisons,ensure_ascii=False)}。',
             '',f'自由测试首个工具调用的非法字段频次：{json.dumps(first_field_misuse,ensure_ascii=False)}。',
             '',f"新 signal 首个工具调用的非法字段频次：{out['first_field_misuse_in_signal']}。",'',
             '| 注入错误／位置 | 条件 | 首动作匹配最少调用纠错目标 | 最终可溯源成功 | 后续非法查询 | 正确轨迹额外调用（扣除固定注入错误） |',
             '|---|---|---:|---:|---:|---:|']
    for key in sorted(recovery['new/protocol']):
        for c in conditions:
            x=recovery[c][key];text.append(f"| {key} | {c} | {x['first_action_matches_minimum_cost_repair']}/{x['attempts']} | {x['final_grounded_success']}/{x['attempts']} | {x['generated_invalid_queries']} | {x['extra_calls_after_fixed_injected_error']} |")
    text += ['', '位置 0/1/2/3 分别是 project、employee、department、cost_center。注入错误由脚本执行，不是模型自行产生；首动作匹配与最终任务成功分别统计。',
             '', '恢复集有一个设计限制：unknown_field 使用完整措辞模板 0，unknown_entity 使用模板 1，错误类型与措辞耦合，且两类使用不同事实。模型条件间比较仍保持相同问题和状态；不能把两类错误的分数差直接解释为错误类型自身的难度差。后续若专门比较错误类型，需要同事实交叉两类错误和两种措辞。','',
             '| 全新奖励诊断类别 | 答对 | 可溯源答对 | 额外查询 | 非法查询 |','|---|---:|---:|---:|---:|']
    for k,x in out['signal_by_kind'].items():text.append(f"| {k} | {x['successes']}/{x['attempts']} | {x['grounded_successes']}/{x['attempts']} | {x['correct_extra_calls']} | {x['invalid_queries']} |")
    text += ['',f"新自由测试失败停止原因：{out['new_test_failure_stops']}。新 signal 失败停止原因：{out['new_signal_failure_stops']}。",'',
             '## 新模型剩余失败','']
    for case in new_failed_cases:
        text.append(f"- {case['task_id']}，种子 {case['seed']}：应复制 `{case['first_oracle_entity']}`，首动作实际为 `{json.dumps(case['first_action'])}`，字符遗漏={case['single_character_deletion']}，停止={case['stop']}。")
    if any(c['single_character_deletion'] for c in new_failed_cases):
        text += ['', '这类真实复制失败与本轮注入的类型词／_typo 后缀错误不同。注入错误测试高分不能证明所有 unknown_entity 都能恢复。后续若补数据，应另生成事实，在新训练样本中采集学生失败前缀或加入字符遗漏／替换／插入等不同扰动；当前测试事实保留为历史证据，不回填训练。','']
    groups=summary['signal_groups'];effective=groups['nonconstant_reward_groups']
    text += ['', '## 奖励诊断决策', '',
             f"当前只有 {effective}/32 个任务在四次采样中产生非恒定奖励（{100*effective/32:.1f}%）。类别为 {json.dumps(groups['counts'],ensure_ascii=False)}。"]
    if summary['signal']['successes']==128:
        text += ['', '本轮没有成功／失败混合组。若有有效组，其差异来自答对所需的调用成本，不能解释为模型已展现可供成功率学习的组内差异。全对等成本组即使成本偏高，也没有当前组内相对优势。']
    if effective<=8:
        text += ['', '不建议现在直接对这批诊断题启动正式 GRPO：有效组比例较低，而且这是留出的诊断集。下一步优先在全新训练事实中增加需要根据观察选择分支、利用已知信息跳过查询、权衡不同合法查询成本的任务，再独立检查奖励正确性和有效组覆盖率。自然 ID 复制错误应先用不同事实的学生错误前缀处理，不能把本轮失败测试题回填训练。温度扫描如要做，应提前固定并使用另一个开发集，不能追着当前 32 题调参。']
    text += ['',
             '[逐项机器可读统计与原始证据哈希](descriptive-breakdown.json)。统计口径与主报告一致，新增分层属于事后分析。','']
    (root/'descriptive-breakdown.zh-CN.md').write_text('\n'.join(text),encoding='utf-8',newline='\n')
    print(json.dumps(dict(event='descriptive_breakdown',comparisons=comparisons,signal_failures=out['new_signal_failure_stops'])))

if __name__=='__main__':main()

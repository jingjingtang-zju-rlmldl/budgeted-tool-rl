"""Audited paired transfer, recovery and correctness-conditioned cost analysis."""
import argparse
from collections import Counter,defaultdict
from datetime import datetime
import json
from pathlib import Path
import statistics

from .baseline import sha256
from .stress_data import KINDS,audit,prior_sets
from .stress_experiment import read,rows,write,validate,allowed_test_conditions

LABELS={'anchor_two':'正常两步锚点','chain_three':'三步查询','chain_four':'四步查询',
        'retry_two':'临时错误重试','redirect_two':'失效 ID 跳转','skip_owner':'已给出 owner，省一步',
        'skip_prefix':'已给出链前缀，省三步','given_answer':'已给出答案，零查询'}


def object_count(text):
    """Diagnose consecutive objects without repairing an executed action."""
    decoder=json.JSONDecoder()
    count=0
    rest=text.strip()
    while rest:
        try:
            value,end=decoder.raw_decode(rest)
        except ValueError:
            break
        if not isinstance(value,dict):
            break
        count+=1
        rest=rest[end:].lstrip()
    return count


def metrics(values,tasks):
    correct=[r for r in values if r['success']]
    failures=Counter()
    progress=Counter()
    invalid_outputs=Counter()
    parser_errors=Counter()
    for r in values:
        for request in r['requests']:
            if 'parse_error' not in request:
                continue
            parser_errors[request['parse_error']]+=1
            try:
                obj=json.loads(request['text'])
            except ValueError:
                continue
            if isinstance(obj,dict) and obj.get('action')=='answer' and obj.get('value','missing') is None:
                invalid_outputs['answer_value_null']+=1
            if isinstance(obj,dict) and obj.get('action')=='lookup' and set(obj)-{'action','entity','field'}:
                invalid_outputs['lookup_extra_arguments']+=1
    for r in values:
        task=tasks[r['task_id']]
        if r['success']:
            continue
        if r['stop_reason']!='answered':
            label=r['stop_reason']
        elif r['calls']<r['minimum_calls']:
            label='premature_answer'
        else:
            label='wrong_final_answer'
        failures[label]+=1
        if task['kind'] in ('chain_three','chain_four'):
            intermediate={record[field] for record in task['database'].values() for field in ('owner','department','cost_center') if field in record}
            if str(r['answer']) in intermediate and r['answer']!=r['expected']:
                progress['answered_intermediate_entity']+=1
    by_kind={}
    for kind in KINDS:
        subset=[r for r in values if r['kind']==kind]
        ok=[r for r in subset if r['success']]
        exposed=[r for r in subset if r['recoverable_errors']>0]
        goal_reached=sum(any(o.get('entity')==tasks[r['task_id']]['oracle_calls'][-1]['entity']
                                and o.get('field')==tasks[r['task_id']]['oracle_calls'][-1]['field'] and 'value' in o
                                for o in r['observations']) for r in subset) if kind!='given_answer' else len(subset)
        by_kind[kind]=dict(attempts=len(subset),successes=len(ok),grounded_successes=sum(r['grounded_success'] for r in subset),
                          strict_successes=sum(r['strict_success'] for r in subset),
                          minimum_calls=tasks[subset[0]['task_id']]['minimum_calls'],
                          mean_calls=statistics.mean(r['calls'] for r in subset),
                          correct_mean_calls=statistics.mean(r['calls'] for r in ok) if ok else None,
                          correct_extra_calls=sum(r['correct_extra_calls'] for r in ok),
                          optimal_correct=sum(r['success'] and r['calls']==r['minimum_calls'] for r in subset),
                          correct_with_extra_calls=sum(r['success'] and r['calls']>r['minimum_calls'] for r in subset),
                          observed_recoverable_errors=len(exposed),
                          success_after_error=sum(r['success'] for r in exposed),
                          necessary_retry_actions=sum(r['necessary_retries'] for r in subset),
                          followed_redirect_actions=sum(r['followed_redirects'] for r in subset),
                          reached_goal_field=goal_reached,
                          redundant_known_lookups=sum(r['redundant_known_lookups'] for r in subset),
                          invalid_queries=sum(r['invalid_queries'] for r in subset),
                          stops=dict(Counter(r['stop_reason'] for r in subset)),
                          truncated_requests=sum(q.get('truncated',False) for r in subset for q in r['requests']),
                          multiple_object_requests=sum(object_count(q.get('text',''))>=2 for r in subset for q in r['requests']))
    stability=defaultdict(list)
    rollout_groups=defaultdict(list)
    for r in values:
        stability[r['task_id']].append(r['success'])
        rollout_groups[r['task_id']].append(r)
    reward_groups=Counter()
    for group in rollout_groups.values():
        if not any(r['success'] for r in group):
            reward_groups['all_failed']+=1
        elif not all(r['success'] for r in group):
            reward_groups['mixed_correctness']+=1
        elif len({r['calls'] for r in group})>1:
            reward_groups['all_correct_different_cost']+=1
        else:
            reward_groups['all_correct_same_cost']+=1
        if len({r['reward_cost_gated'] for r in group})>1:
            reward_groups['nonconstant_cost_gated_reward']+=1
    requests=sum(len(r['requests']) for r in values)
    templates={kind:{str(t):dict(attempts=sum(r['kind']==kind and r['template_id']==t for r in values),
                       successes=sum(r['success'] for r in values if r['kind']==kind and r['template_id']==t))
                       for t in sorted({r['template_id'] for r in values if r['kind']==kind})} for kind in KINDS}
    return dict(attempts=len(values),unique_scenarios=len(stability),fact_groups=len({r['fact_group'] for r in values}),
                successes=len(correct),grounded_successes=sum(r['grounded_success'] for r in values),
                strict_successes=sum(r['strict_success'] for r in values),by_kind=by_kind,by_kind_template=templates,
                requests=requests,strict_valid_requests=sum(r['strict_valid_requests'] for r in values),
                fence_normalized_requests=sum(r['fence_normalized_requests'] for r in values),
                max_input_tokens=max(len(q['input_ids']) for r in values for q in r['requests']),
                context_failures=sum(r['stop_reason']=='context_limit' for r in values),
                truncated_requests=sum(q.get('truncated',False) for r in values for q in r['requests']),
                multiple_object_requests=sum(object_count(q.get('text',''))>=2 for r in values for q in r['requests']),
                fabricated_tool_transcript_requests=sum('Tool:' in q.get('text','') for r in values for q in r['requests']),
                failures=dict(failures),chain_failure_diagnostics=dict(progress),
                invalid_output_diagnostics=dict(invalid_outputs),
                parser_error_counts=dict(parser_errors),
                correct_mean_calls=statistics.mean(r['calls'] for r in correct) if correct else None,
                correct_extra_calls=sum(r['correct_extra_calls'] for r in correct),
                recoverable_error_events=sum(r['recoverable_errors'] for r in values),
                necessary_retry_actions=sum(r['necessary_retries'] for r in values),
                followed_redirect_actions=sum(r['followed_redirects'] for r in values),
                unnecessary_repeat_actions=sum(r['unnecessary_repeats'] for r in values),
                redundant_known_lookups=sum(r['redundant_known_lookups'] for r in values),
                invalid_queries=sum(r['invalid_queries'] for r in values),
                generation_seconds=sum(r['generation_seconds'] for r in values),
                generated_tokens=sum(r['generated_tokens'] for r in values),
                exploratory_three_seed_reward_groups=dict(reward_groups),
                stability=dict(always_correct=sum(all(v) for v in stability.values()),
                               always_wrong=sum(not any(v) for v in stability.values()),
                               mixed=sum(any(v) and not all(v) for v in stability.values())))


def paired(left,right):
    a={(r['task_id'],r['seed']):r for r in left}
    b={(r['task_id'],r['seed']):r for r in right}
    assert a.keys()==b.keys() and len(a)==len(left) and len(b)==len(right)
    return dict(attempts=len(a),both_correct=sum(a[k]['success'] and b[k]['success'] for k in a),
                gained=sum(not a[k]['success'] and b[k]['success'] for k in a),
                lost=sum(a[k]['success'] and not b[k]['success'] for k in a),
                both_wrong=sum(not a[k]['success'] and not b[k]['success'] for k in a),
                by_kind={kind:dict(gained=sum(not a[k]['success'] and b[k]['success'] for k in a if a[k]['kind']==kind),
                                  lost=sum(a[k]['success'] and not b[k]['success'] for k in a if a[k]['kind']==kind)) for kind in KINDS})


def scenario_pairs(values):
    grouped=defaultdict(dict)
    for r in values:
        key=(r['fact_group'],r['seed'])
        assert r['kind'] not in grouped[key]
        grouped[key][r['kind']]=r
    result={}
    for first,second in [('anchor_two','retry_two'),('anchor_two','redirect_two'),('anchor_two','skip_owner'),('chain_four','skip_prefix')]:
        matches=[(g[first],g[second]) for g in grouped.values() if first in g and second in g]
        result[first+' -> '+second]=dict(pairs=len(matches),both_correct=sum(a['success'] and b['success'] for a,b in matches),
            normal_correct=sum(a['success'] for a,b in matches),changed_correct=sum(b['success'] for a,b in matches),
            lost=sum(a['success'] and not b['success'] for a,b in matches),
            gained=sum(not a['success'] and b['success'] for a,b in matches),
            changed_error_observed=sum(b['recoverable_errors']>0 for a,b in matches))
    return result


def aggregate(root):
    plan_path=root/'plan.json'
    plan=validate(plan_path)
    selection=read(root/'selection.json')
    expected_conditions=allowed_test_conditions(plan,plan_path)
    data_root=Path(plan['data_root'])
    splits={s:rows(data_root/(s+'.jsonl')) for s in ('dev','test')}
    prior_e,prior_v=prior_sets([Path(p) for p in plan['historical_files_sha256']])
    audit(splits,prior_e,prior_v)
    tasks={s:{r['task_id']:r for r in rs} for s,rs in splits.items()}
    conditions=[]
    groups={'dev':{},'test':{}}
    verified_tasks=verified_requests=0
    peak=reserved=0.
    for split in ('dev','test'):
        combinations=[tuple(c.split('/')) for c in plan['dev_conditions']] if split=='dev' else expected_conditions
        for model,prompt in combinations:
            all_values=[]
            for seed in plan['evaluation_seeds'][split]:
                path=root/'eval'/f'{split}-{model}-{prompt}-seed{seed}'
                manifest=read(path/'manifest.json')
                config=manifest['config']
                verification=read(path/'verification.json')
                expected=dict(model_id=plan['model_id'],model_revision=plan['model_revision'],
                    dataset_sha256=plan['dataset_sha256'][split],task_count=plan['data_counts'][split],
                    model=model,prompt=prompt,seed=seed,adapter_sha256=plan['adapter_sha256'] if model=='sft' else None,
                    plan_sha256=sha256(plan_path),source_sha256=plan['source_sha256'],
                    system_prompt_sha256=plan['prompt_sha256'][prompt],**plan['eval_fixed'])
                assert config==expected and manifest['status']=='complete'
                assert manifest['trainable_parameters']==0 and not manifest['new_training_performed']
                assert manifest['uses_previous_sft_adapter']==(model=='sft')
                assert verification['episodes_sha256']==sha256(path/'episodes.jsonl')
                assert all(verification[k] for k in ('tool_replay_matches','answer_and_cost_regrading_matches',
                    'input_token_provenance_matches','output_token_decode_matches','recovery_and_redundancy_accounting_matches'))
                values=rows(path/'episodes.jsonl')
                assert len(values)==verification['tasks_verified']==plan['data_counts'][split]
                assert {r['task_id'] for r in values}==tasks[split].keys()
                assert all(r['messages'][0]['content']==Path(plan['prompt_files'][prompt]).read_text(encoding='utf-8') for r in values)
                if split=='test':
                    assert datetime.fromisoformat(manifest['started_at'])>datetime.fromisoformat(selection['selected_at'])
                verified_tasks+=len(values)
                verified_requests+=verification['requests_verified']
                peak=max(peak,manifest['cuda_peak_allocated_mib'])
                reserved=max(reserved,manifest['cuda_peak_reserved_mib'])
                all_values.extend(values)
                conditions.append(dict(split=split,model=model,prompt=prompt,seed=seed,
                                       metrics=metrics(values,tasks[split]),episodes_sha256=sha256(path/'episodes.jsonl'),
                                       verification_sha256=sha256(path/'verification.json'),
                                       peak_allocated_mib=manifest['cuda_peak_allocated_mib']))
            groups[split][model+'/'+prompt]=all_values
    overall={s:{key:metrics(rs,tasks[s]) for key,rs in variants.items()} for s,variants in groups.items()}
    base_key='base/'+selection['selected_prompt']
    comparisons={base_key+' -> sft/protocol':paired(groups['test'][base_key],groups['test']['sft/protocol'])}
    if selection['selected_prompt']!='protocol':
        comparisons[base_key+' -> sft/'+selection['selected_prompt']]=paired(groups['test'][base_key],groups['test']['sft/'+selection['selected_prompt']])
        comparisons['sft/protocol -> sft/'+selection['selected_prompt']]=paired(groups['test']['sft/protocol'],groups['test']['sft/'+selection['selected_prompt']])
    result=dict(generated_at=datetime.now().astimezone().isoformat(),report_source_sha256=sha256(Path(__file__)),
                plan_sha256=sha256(plan_path),selection=selection,aggregate=overall,conditions=conditions,
                paired_comparisons=comparisons,scenario_pairs={k:scenario_pairs(v) for k,v in groups['test'].items()},
                total_verified_episodes=verified_tasks,total_verified_requests=verified_requests,
                peak_allocated_mib=peak,peak_reserved_mib=reserved,existing_adapter_sha256=plan['adapter_sha256'],
                weights_updated=False,caveats=plan['limitations'])
    write(root/'summary.json',result)
    examples=[]
    for key,rs in groups['test'].items():
        for r in rs:
            if r['kind'] in ('chain_three','chain_four','retry_two','redirect_two','skip_owner','skip_prefix'):
                examples.append(dict(condition=key,task_id=r['task_id'],task_seed=r['seed'],kind=r['kind'],
                    question=tasks['test'][r['task_id']]['prompt'],success=r['success'],expected=r['expected'],answer=r['answer'],
                    calls=r['calls'],minimum_calls=r['minimum_calls'],stop_reason=r['stop_reason'],
                    actions=r['actions'],observations=r['observations'],call_events=r['call_events'],
                    raw_outputs=[q.get('text','') for q in r['requests']]))
    (root/'diagnostic-cases.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in examples),encoding='utf-8',newline='\n')
    return result


def plot(root,result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    variants=result['aggregate']['test']
    fig,axes=plt.subplots(2,1,figsize=(12,8),layout='constrained')
    colors=['#64748b','#14b8a6','#38bdf8']
    names={'base/protocol':'Base / dev-best protocol','base/examples':'Base / dev-best examples',
           'sft/protocol':'Existing SFT / protocol','sft/examples':'Existing SFT / same examples'}
    width=.24 if len(variants)==3 else .32
    for j,(key,m) in enumerate(variants.items()):
        offset=(j-(len(variants)-1)/2)*width
        rates=[100*m['by_kind'][k]['successes']/m['by_kind'][k]['attempts'] for k in KINDS]
        bars=axes[0].bar([i+offset for i in range(8)],rates,width=width*.94,label=names[key],color=colors[j])
        axes[0].bar_label(bars,fmt='%.0f',fontsize=8,padding=2)
    axes[0].set_xticks(range(8),['2-hop anchor','3-hop','4-hop','Retry','Redirect','Skip owner','Skip prefix','Given answer'],rotation=12)
    axes[0].set_ylim(0,112)
    axes[0].set_ylabel('Answer success (%)')
    axes[0].set_title('8 scenarios per family, three seeds; paired scenarios share facts')
    axes[0].legend(loc='lower center',bbox_to_anchor=(.5,1.12),ncol=3,fontsize=8)
    kinds=['skip_owner','skip_prefix','given_answer']
    for j,(key,m) in enumerate(variants.items()):
        xs=[i+(j-(len(variants)-1)/2)*width for i in range(3)]
        values=[]
        for k in kinds:
            family=m['by_kind'][k]
            values.append(family['correct_extra_calls']/family['successes'] if family['successes'] else 0.)
        bars=axes[1].bar(xs,values,width=width*.94,label=names[key],color=colors[j])
        for bar,k in zip(bars,kinds):
            family=m['by_kind'][k]
            label=f"{bar.get_height():.2f}\n(n={family['successes']})" if family['successes'] else 'N/A\n(n=0)'
            axes[1].text(bar.get_x()+bar.get_width()/2,bar.get_height()+.04,label,ha='center',fontsize=8)
    axes[1].set_xticks(range(3),['Owner provided (min 1)','Chain prefix provided (min 1)','Answer provided (min 0)'])
    axes[1].set_ylim(0,max(.7,axes[1].get_ylim()[1]+.3))
    axes[1].set_ylabel('Mean extra calls, correct trials only')
    axes[1].set_title('Correctness-conditioned cost; n = correct attempts')
    fig.savefig(root/'stress-results.png',dpi=160)
    plt.close(fig)


def write_report(root,result):
    test=result['aggregate']['test']
    selection=result['selection']
    base_key='base/'+selection['selected_prompt']
    labels={base_key:'未训练模型＋开发集最强提示','sft/protocol':'现有 SFT＋扩展短协议','sft/examples':'现有 SFT＋同一强示例提示'}
    keys=list(test)
    base=test[base_key]
    sft=test['sft/protocol']
    matched_key='sft/'+selection['selected_prompt']
    matched=test[matched_key]
    action_audit=read(root/'action-budget-audit.json')
    assert action_audit['plan_sha256']==result['plan_sha256']
    assert action_audit['audit_source_sha256']==sha256(Path(__file__).with_name('stress_action_audit.py'))
    assert action_audit['all_checked_actions_fit']
    def pct(a,b):
        return f'{100*a/b:.1f}%' if b else '无观测机会'
    def ratio(m):
        return f"{m['successes']}/{m['attempts']}（{pct(m['successes'],m['attempts'])}）"
    main=['| 场景 | 最少调用 | '+' | '.join(labels[k] for k in keys)+' |',
          '|---|---:|'+'---:|'*len(keys)]
    for kind in KINDS:
        main.append('| '+LABELS[kind]+f" | {sft['by_kind'][kind]['minimum_calls']} | "+
                    ' | '.join(ratio(test[k]['by_kind'][kind]) for k in keys)+' |')
    main.append('| **合计** | — | '+' | '.join('**'+ratio(test[k])+'**' for k in keys)+' |')
    seeds=['| 测试采样种子 | '+' | '.join(labels[k] for k in keys)+' |',
           '|---|'+'---:|'*len(keys)]
    for seed in (7,10007,20007):
        cells=[]
        for key in keys:
            model,prompt=key.split('/')
            m=next(c['metrics'] for c in result['conditions'] if c['split']=='test' and c['seed']==seed and c['model']==model and c['prompt']==prompt)
            cells.append(ratio(m))
        seeds.append('| '+str(seed)+' | '+' | '.join(cells)+' |')
    dev=['| 开发集条件 | 答对 / 24 | 严格 JSON 且答对 | 答对轨迹的额外查询总数 |',
         '|---|---:|---:|---:|']
    for key,m in result['aggregate']['dev'].items():
        dev.append(f"| {key} | {m['successes']}/24 | {m['strict_successes']}/24 | {m['correct_extra_calls']} |")
    recovery=['| 条件 / 场景 | 场景尝试 | 真正观察到错误 | 观察错误后答对 | 成功执行恢复动作 |',
              '|---|---:|---:|---:|---:|']
    for key in keys:
        for kind in ('retry_two','redirect_two'):
            m=test[key]['by_kind'][kind]
            actions=m['necessary_retry_actions'] if kind=='retry_two' else m['followed_redirect_actions']
            recovery.append(f"| {labels[key]} / {LABELS[kind]} | {m['attempts']} | {m['observed_recoverable_errors']} | {m['success_after_error']}/{m['observed_recoverable_errors']} | {actions} |")
    costs=['| 条件 / 场景 | 答对 | 答对且最少调用 | 答对但有额外调用 | 正确轨迹平均额外调用 |',
           '|---|---:|---:|---:|---:|']
    for key in keys:
        for kind in ('skip_owner','skip_prefix','given_answer'):
            m=test[key]['by_kind'][kind]
            extra=f"{m['correct_extra_calls']/m['successes']:.2f}" if m['successes'] else '无正确轨迹'
            costs.append(f"| {labels[key]} / {LABELS[kind]} | {m['successes']}/{m['attempts']} | {m['optimal_correct']} | {m['correct_with_extra_calls']} | {extra} |")
    primary=result['paired_comparisons'][base_key+' -> sft/protocol']
    same_prompt=result['paired_comparisons'].get(base_key+' -> '+matched_key,primary)
    prompt_pair=result['paired_comparisons'].get('sft/protocol -> '+matched_key)
    prompt_tradeoff=(f"同一 SFT 从短协议换成强示例提示，由错转对 {prompt_pair['gained']} 次、由对转错 {prompt_pair['lost']} 次；"
                    f"其中三步查询的退步为 {prompt_pair['by_kind']['chain_three']['lost']} 次。"
                    '这些退步对应 assistant 续写虚构 Tool: 返回的失败，强提示的总体提升并不意味着每类任务都更稳定。'
                    if prompt_pair else '')
    dev_prefix=result['aggregate']['dev']['sft/protocol']['by_kind']['skip_prefix']
    long_attempts=sum(sft['by_kind'][k]['attempts'] for k in ('chain_three','chain_four'))
    short_long=sum(sft['by_kind'][k]['successes'] for k in ('chain_three','chain_four'))
    matched_long=sum(matched['by_kind'][k]['successes'] for k in ('chain_three','chain_four'))
    cases=rows(root/'diagnostic-cases.jsonl')
    example_blocks=[]
    predicates=[('长链停止或走偏',lambda c:c['condition']=='sft/protocol' and c['kind']=='chain_four' and not c['success']),
                ('尚未读取目标数值就回答',lambda c:c['condition']=='sft/protocol' and c['kind']=='chain_four' and not c['success'] and c['calls']==3 and isinstance(c['answer'],(int,float))),
                ('观察错误后的行为',lambda c:c['condition']=='sft/protocol' and c['kind'] in ('retry_two','redirect_two') and any('error' in o for o in c['observations'])),
                ('接口拒绝 null 答案',lambda c:c['condition']=='sft/protocol' and c['stop_reason']=='invalid_action' and any('"value":null' in raw or '"value": null' in raw for raw in c['raw_outputs'])),
                ('逆向查询违反当前参数 schema',lambda c:c['condition']=='sft/protocol' and c['stop_reason']=='invalid_action' and any('"entity":"employee"' in raw and '"value":' in raw for raw in c['raw_outputs'])),
                ('答对仍有冗余查询',lambda c:c['condition']=='sft/protocol' and c['kind']=='skip_owner' and c['success'] and c['calls']>c['minimum_calls']),
                ('强提示下的三四步成功',lambda c:c['condition']==matched_key and c['kind'] in ('chain_three','chain_four') and c['success'])]
    predicates.append(('强提示下模拟工具返回',lambda c:c['condition']==matched_key and not c['success'] and any('Tool:' in raw for raw in c['raw_outputs'])))
    for title,predicate in predicates:
        case=next((c for c in cases if predicate(c)),None)
        if case:
            payload={k:case[k] for k in ('task_id','task_seed','kind','question','success','expected','answer','calls','minimum_calls','stop_reason','actions','observations','raw_outputs')}
            example_blocks.append('### '+title+'\n\n```json\n'+json.dumps(payload,indent=2)+'\n```')
    if matched_long<max(2,long_attempts*.1):
        decision=('三四步成功轨迹很少，当前直接做纯结果奖励的 GRPO 会遇到稀疏正反馈。'
                  '建议先补少量覆盖新字段和长链停止条件的 oracle SFT，并混入旧两步与已知事实任务保持目标选择；'
                  '另加入报错后的下一步状态监督。等开发集上能稳定产生成功长链，再试小规模 RL。')
    elif matched_long>short_long:
        decision=('强提示能让同一 adapter 产生更多长链成功轨迹，说明 rollout 上下文是重要变量。'
                  '建议下一轮优先用新生成的正确轨迹做少量 SFT，把长链推进、报错后的下一步、已知事实和单动作边界学进简洁提示条件，再考虑 RL。'
                  '训练轨迹中只把真实执行得到的观察放在 tool 角色，assistant 每回合只监督一个动作，绝不监督伪造的 Tool: 续写。'
                  '不能把短协议下的失败直接解释为模型容量不够。')
    else:
        decision=('现有模型已经能产生一定数量的长链成功轨迹，具备小规模 RL 试验的部分前提。'
                  '下一轮用新训练环境构建同任务多采样组，先检查成功率和组内奖励方差，再决定是否开始 GRPO。')
    waste=sft['by_kind']['skip_owner']['correct_extra_calls']
    efficiency=(f"已给出 owner 的正确轨迹仍有 {waste} 次额外查询，这是可量化的成本改进空间。"
                if waste else '短协议在已给出 owner 的正确轨迹中没有额外调用；这部分暂时没有成本优化信号。')
    max_context=max(m['max_input_tokens'] for m in test.values())
    template_table=['| 场景 / 测试措辞编号 | '+' | '.join(labels[k] for k in keys)+' |',
                    '|---|'+'---:|'*len(keys)]
    for kind in ('anchor_two','chain_three','chain_four'):
        for t in ('0','1'):
            template_table.append('| '+LABELS[kind]+' / '+t+' | '+
                ' | '.join(ratio(test[k]['by_kind_template'][kind][t]) for k in keys)+' |')
    group_table=['| 条件 | 三次全错的任务 | 成败混合 | 三次全对但成本不同 | 三次全对且成本相同 |',
                 '|---|---:|---:|---:|---:|']
    for key in keys:
        g=test[key]['exploratory_three_seed_reward_groups']
        group_table.append('| '+labels[key]+' | '+' | '.join(str(g.get(k,0)) for k in
            ('all_failed','mixed_correctness','all_correct_different_cost','all_correct_same_cost'))+' |')
    report=f'''# 难度升级：冻结 SFT 的长链、错误恢复与冗余查询对照

本轮只评测现有权重。未训练模型使用开发集从两套固定候选中选出的最强提示；SFT 使用扩展短协议，并保留同强提示的对照（若不同）。新测试不再只检验旧的两步 schema。

相同强提示下，SFT 为 **{ratio(matched)}**，base 为 **{ratio(base)}**；SFT 短协议为 **{ratio(sft)}**。现有 SFT 已具备部分新链路与恢复能力，但提示依赖和单动作边界仍是主要问题。建议先补新状态监督，再在新任务上小规模检查 RL 信号。

## 测试结果

{chr(10).join(main)}

{chr(10).join(seeds)}

每个条件是 **64 个场景 × 3 个采样种子 = 192 次尝试**。每类 8 个场景重复三次，所以表中分母为 24。场景来自 **24 组事实**：正常/错误/已知信息条件刻意共享事实，形成可配对的诊断。它们不是 192 个独立题目，也不适合直接按 IID 样本做显著性推断。

现有 SFT 短协议相对强基线的配对结果：由错转对 {primary['gained']} 次，由对转错 {primary['lost']} 次，两者均对 {primary['both_correct']} 次，两者均错 {primary['both_wrong']} 次。短协议 SFT 的三四步合计 {short_long}/{long_attempts}，同强提示 SFT 为 {matched_long}/{long_attempts}。完整按类配对与事实场景配对见 [summary.json](summary.json)。

使用相同强提示时，SFT 相对 base 的同任务同种子配对为由错转对 {same_prompt['gained']} 次、由对转错 {same_prompt['lost']} 次。强提示并未让未训练模型取得相同成绩，因此提升不能只归功于提示；另一方面，同一 adapter 的短协议与强提示差距说明已有训练效果仍依赖推理上下文。二者是需要同时呈现的实验结果。

## 强提示如何选择

{chr(10).join(dev)}

开发集每类三场景，共 24 场景、9 组事实。候选一是完整新 schema 和恢复规则的短协议；候选二是在同一协议中加入七个完整示例，覆盖三四步、重试、ID 跳转和三类已知信息。按答对数、严格答对数、正确轨迹额外调用、提示长度依次决胜，选中 `{selection['selected_prompt']}`。这是**本轮两候选中的开发集最强**，不是宣称搜索了所有可能提示。

选择写入 [selection.json](selection.json) 并记录开发集轨迹/复核哈希，之后才开放测试生成。SFT 的开发集结果没有参与 base 提示选择；没有依据测试修改提示、模型、任务或采样参数。示例内容与长度共同改变，不单独做因果归因。

## 三四步链：多发查询不等于完成目标

三步需 project.owner → employee.department → department.cost_center；四步再读取 cost_center.spending_limit。两个新字段对两种模型都在工具契约中完整说明。部门仍使用与旧训练相同的 department 数字 ID 格式，项目、员工、成本中心 ID 的后缀独立随机，无法从项目 ID 推导 owner。

SFT 短协议在三步任务真正到达目标字段 {sft['by_kind']['chain_three']['reached_goal_field']}/24 次，四步为 {sft['by_kind']['chain_four']['reached_goal_field']}/24 次。长链失败中，最终回答某个中间实体的次数为 {sft['chain_failure_diagnostics'].get('answered_intermediate_entity',0)}。目标字段已查询仍答错、提前把 owner/department 当答案、查询错误实体或字段，是不同失败，需要分别看原始轨迹。

短协议 SFT 有 {sft['invalid_output_diagnostics'].get('answer_value_null',0)} 个请求输出 `answer` 且 `value: null`。JSON 语法成立，但 null 违反当前答案必须是字符串/数字的契约，因此接口拒绝执行；不能将其当作一个正常的错误标量答案。行为标记、完整解析原因和停止分类分别保存在 summary.json 的 `invalid_output_diagnostics`、`parser_error_counts` 与 `failures` 字段。

另有 {sft['invalid_output_diagnostics'].get('lookup_extra_arguments',0)} 个无效 lookup 请求带额外参数；实际片段中出现以 `employee` 类型为 entity、以项目作 value 的逆向查询写法。当前接口只接受实体 ID 和单字段读取，无法执行这种查询。这同时涉及任务到动作的映射，不能一概称为 JSON 语法差。

{chr(10).join(template_table)}

每种测试措辞对应四个事实实例、三个种子，共 12 次尝试。两种措辞分配不同事实实例，这张表只描述分组差异，不能把差异单独归因于措辞；它提示后续开发集需要更丰富且事实配对的释义检查。

强示例 SFT 有 {matched['fabricated_tool_transcript_requests']} 个生成请求包含字面 `Tool:`，即模型在 assistant 输出中模拟工具对话；这不是实际工具返回，接口不会执行它。原始失败片段中还出现将项目 ID 的后缀复制成员工 ID。真实链路的后缀独立，模型必须通过查询取得 owner，不能按这种模式推导。示例提示含对话展示，该格式可能影响续写，但本轮没有格式消融，不能认定它就是唯一原因。

{prompt_tradeoff}

两步锚点短协议 SFT 为 {sft['by_kind']['anchor_two']['successes']}/24。它使用新工具说明和新措辞，因此与上一轮旧 schema 的 60/60 不能直接当作同一题集的分数变化；锚点用于观察新契约下旧技能是否仍能保持。新字段、提示长度、措辞与链深度都有变化，本轮不能单独隔离每一因素。

## 报错恢复：先数真正遇到错误的机会

{chr(10).join(recovery)}

临时错误在指定 employee.department 第一次被查询时发生，下一次相同请求成功；失效 ID 返回 canonical_entity，需要查询该 ID 的同一字段。故障按查询键确定，重放能得到同样结果，错误也消耗一次调用。工具绝不在错误里泄露最终答案。

没有走到指定查询的失败不算「已观察错误却不会恢复」。恢复动作成功也不等于最终答对，表里分开报告。必要重试不是冗余：正常两步最少两次，注入错误的版本最少三次，成本下界已包含不可避免的错误调用。

## 冗余查询：在正确性条件下计算成本

{chr(10).join(costs)}

已知 owner 时可以直接查 employee.department；已知成本中心时只需查 spending_limit；预算答案已给出时零查询。重新查询问题已提供的字段、重复已成功查询的字段单独记录。{efficiency}失败策略过早回答造成的少调用不算效率提升。精确正确与工具可溯源正确同时报告，当前 SFT 短协议分别为 {sft['successes']} 和 {sft['grounded_successes']} 次。

开发集短协议 SFT 在「已给出链前缀」任务中出现输出截断 {dev_prefix['truncated_requests']} 次，其中含两个或更多完整连续 JSON 对象的请求 {dev_prefix['multiple_object_requests']} 次；该类测试中相应数值为 {sft['by_kind']['skip_prefix']['truncated_requests']} 和 {sft['by_kind']['skip_prefix']['multiple_object_requests']}。开发集的失败形式未直接重现到该类测试，不能沿用开发集诊断解释所有测试失败。多对象输出违反一个回合一个动作的接口；本轮不执行截断输出中的第一个对象，也不自动补全后续动作。各类完整停止原因见 summary.json。

事后 tokenizer 检查了开发/测试 oracle 动作的紧凑与默认 JSON 写法，共 {action_audit['action_spellings_checked']} 个实例，最长含一个终止 token 为 {action_audit['max_oracle_action_tokens_with_one_eos']} tokens，小于 64。该检查用于确认这些标准单动作可以容纳，不代表所有合法写法都不可能超长，也没有据此修改评测。记录见 [action-budget-audit.json](action-budget-audit.json)。

## 实际轨迹片段

{chr(10).join(example_blocks) if example_blocks else '本轮没有满足以上诊断条件的轨迹片段。'}

全部相关类别的真实动作、观察、错误及原始输出保存在 [diagnostic-cases.jsonl](diagnostic-cases.jsonl)。这些片段来自模型实际生成，未用 oracle 补齐动作。

## 环境、显存与证据

本轮共 **{result['total_verified_episodes']} 个 episode、{result['total_verified_requests']} 个模型请求**完成独立复核。工具重放、答案及成本重判、输入 token 重建、输出 token 解码、错误恢复和冗余计数一致；所有条件的数据、提示、源文件及 adapter 哈希均匹配冻结计划。复核验证记录一致性，不是重新生成模型输出。

本地 RTX 5060 Laptop 8GB，Qwen3-0.6B 固定 revision，BF16 主干、FP32 旧 LoRA，所有参数冻结。PyTorch 最大 allocated 为 {result['peak_allocated_mib']/1024:.2f} GiB，reserved 为 {result['peak_reserved_mib']/1024:.2f} GiB；这些是进程内存，桌面还有额外占用。模型没有新增权重更新，旧 adapter SHA-256 仍为 `{result['existing_adapter_sha256']}`。

采样保持 temperature 0.7、top-p 0.8、top-k 20，关闭 thinking，每动作 64 新 tokens、最多六次查询和七个输出回合，batch 1。输入上限为 4096，实际测试最大 {max_context} tokens；上下文超限合计 {sum(m['context_failures'] for m in test.values())} 次，输出截断 {sum(m['truncated_requests'] for m in test.values())} 次。没有自动 JSON 修复或强制工具调用解码，窄代码围栏兼容与严格 JSON 指标分开。

新开发与测试互不重叠，也与历史 SFT/基线数据无实体和私有值交集。测试措辞与开发分开，全部 oracle 路径实际执行过并验证可解。可复现步骤和场景定义见 [reproduce.md](reproduce.md)，完整冻结配置见 [plan.json](plan.json)。这是合成英语 schema 的迁移诊断，不能宣称一般 agent、真实外部工具或通用 RSI 能力。

## 对 RL 的决定

{decision}

{chr(10).join(group_table)}

以上把每个测试任务的三个种子结果作为描述性小组，共 64 组/条件。它能提示奖励是否已有变化，但不是预定的 GRPO rollout 组，也不能代表训练分布上的组内奖励方差；每组只有三次采样，不能据此给出可靠的探索概率。测试任务和这些轨迹不回流到训练。

同强提示 SFT 的 {matched['successes']} 条正确轨迹总共只有 {matched['correct_extra_calls']} 次额外查询；64 个任务小组中，{matched['exploratory_three_seed_reward_groups'].get('nonconstant_cost_gated_reward',0)} 组出现非恒定的当前结果＋成本奖励。若新训练分布也出现这种接近饱和且成本一致的情况，直接扩大同类 rollout 并不自动创造有用的优化信号。这个判断是对当前有限采样的分析，仍需下一阶段的新训练分布诊断。

[DeepSeekMath 的 GRPO 原论文 §4.1.2](https://arxiv.org/html/2402.03300v3#S4.SS1.SSS2) 用同一问题的组内奖励均值和标准差计算相对优势。由此可推知：若整组奖励相同，组内没有可用于偏好排序的差异；实现需要明确处理零方差，不能直接除以零。这个算法依据支持先测新训练分布上的组内信号，并不证明当前测试表现可以预测 RL 增益。

奖励设计先保留最终正确性为门槛，再对正确结果惩罚真实调用成本；错误调用也算成本，但必要恢复不要按「重复字符串」额外误罚。若后续加入过程奖励，应围绕可验证的目标字段推进、正确读取错误信息与未知事实查询，避免把查询次数本身当成能力。

后续训练集应新生成，按事实组切分，同时保留旧目标、长链、恢复和可省调用的混合任务。当前测试已经被查看，只用作已完成的诊断记录；RL 的最终比较应另留新的测试集，并先实测 rollout + backward + optimizer 的显存，不能由本轮推理显存推断。

可执行的下一阶段建议（本轮未执行）：先预留全新的开发/测试事实组，再生成约 300 条正确轨迹作为小规模候选：三/四步共 100 条、两类恢复共 80 条、已知信息三类共 60 条、旧 direct/one/two 目标共 60 条。增加训练释义并检查恢复状态后的监督，所有 teacher 观察均来自环境执行。这个配比是起步方案，是否训练、训练量与最终采用哪种提示，应由新开发集和训练显存探针决定。

随后仅在新训练分布做 32 个任务 × 每任务 4 条 rollout 的小预算采样诊断。分别记录全错、全对等成本、成败混合、全对不同成本的组；先确认有成功探索和非恒定奖励，再决定小规模 GRPO。若当前强提示在新分布仍近乎饱和，可考察「长链＋故障」「已知中间事实＋故障」的组合迁移，而不是强制插入多余查询制造成本信号。进入优化时记录真实旧策略 log-prob，只给 assistant 动作 tokens 加策略损失，tool 观察作为环境上下文；本轮推理记录没有这些旧策略概率，不能伪装为 RL 训练记录。

![难度升级正确率与正确轨迹成本](stress-results.png)
'''
    (root/'analysis.zh-CN.md').write_text(report,encoding='utf-8',newline='\n')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path('results/stress-v1'))
    args=parser.parse_args()
    result=aggregate(args.root)
    plot(args.root,result)
    write_report(args.root,result)
    print(json.dumps(dict(episodes=result['total_verified_episodes'],requests=result['total_verified_requests'],
                         test={k:v['successes'] for k,v in result['aggregate']['test'].items()}),indent=2))


if __name__=='__main__':
    main()

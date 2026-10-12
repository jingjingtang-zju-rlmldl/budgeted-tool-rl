"""Audit frozen SFT evidence and aggregate paired, repeated-seed rollouts."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import statistics

from .environment import parse_action
from .prompt_study import diagnostics, paired
from .sft_experiment import validate


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def lines(path):
    return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(rows, tasks):
    result=diagnostics(rows,tasks)
    result['by_template']={f'{kind}:{i}':dict(attempts=sum(r['kind']==kind and r['template_id']==i for r in rows),
                                            successes=sum(r['success'] for r in rows if r['kind']==kind and r['template_id']==i))
                           for kind,i in sorted({(r['kind'],r['template_id']) for r in rows})}
    two=[r for r in rows if r['kind']=='two_lookup']
    correct=[r for r in rows if r['success']]
    def lookups(row):
        return [a for a in row['actions'] if a['action']=='lookup'][:row['calls']]
    def required(row):
        return [{'action':'lookup',**a} for a in tasks[row['task_id']]['oracle_calls']]
    result.update(
        unique_tasks=len({r['task_id'] for r in rows}),
        correct_mean_calls=statistics.mean(r['calls'] for r in correct) if correct else None,
        correct_extra_calls=sum(r['calls']-r['minimum_calls'] for r in correct),
        two_correct_first_lookup=sum(lookups(r)[:1]==required(r)[:1] for r in two),
        two_correct_chain_prefix=sum(lookups(r)[:2]==required(r) for r in two),
        two_second_lookup_to_actual_owner=sum(required(r)[1] in lookups(r) for r in two),
        two_copied_example_department=sum(r['answer']=='department_48216' for r in two),
        two_answered_owner_id=sum(r['answer']==tasks[r['task_id']]['oracle_calls'][1]['entity'] for r in two),
        invalid_tool_calls=sum(r['invalid_tool_calls'] for r in rows),
        repeated_calls=sum(r['repeated_calls'] for r in rows),
        output_truncations=sum(q.get('truncated',False) for r in rows for q in r['requests']),
        context_failures=sum(r['stop_reason']=='context_limit' for r in rows),
        max_input_tokens=max(len(q['input_ids']) for r in rows for q in r['requests']),
        generation_seconds=sum(r['generation_seconds'] for r in rows),
        generated_tokens=sum(r['generated_tokens'] for r in rows))
    stability={}
    for kind in ('direct','one_lookup','two_lookup','all'):
        grouped=defaultdict(list)
        for row in rows:
            if kind=='all' or row['kind']==kind:
                grouped[row['task_id']].append(row['success'])
        stability[kind]=dict(unique_tasks=len(grouped),all_attempts_correct=sum(all(v) for v in grouped.values()),
                             all_attempts_wrong=sum(not any(v) for v in grouped.values()),
                             mixed=sum(any(v) and not all(v) for v in grouped.values()))
    result['task_stability']=stability
    failures=Counter()
    for row in rows:
        if row['success']:
            continue
        if row['stop_reason']!='answered':
            category=row['stop_reason']
        elif row['calls']<row['minimum_calls']:
            category='answer_before_required_lookups'
        else:
            category='wrong_answer_after_lookups'
        failures[category]+=1
    result['failure_classes']=dict(failures)
    return result


def audit_training(root,plan,info):
    """Rebuild labels from preceding observations; check the saved adapter changed."""
    import torch
    from transformers import AutoTokenizer
    from safetensors.torch import load_file
    from .sft_train import tensor_hash
    from .sft_data import request_examples, encode_example, audit_splits
    splits={s:lines(Path(plan['data_root'])/(s+'.jsonl')) for s in ('train','dev','test')}
    audit_splits(splits)
    manifest=read(root/'training/manifest.json')
    smoke=read(root/('smoke-checkpointed' if manifest['gradient_checkpointing'] else 'smoke')/'manifest.json')
    assert smoke['status']=='complete' and smoke['optimizer_steps']==8 and smoke['microbatches']==64
    assert manifest['status']=='complete' and manifest['training_performed']
    assert manifest['plan_sha256']==sha(root/'plan.json') and manifest['config']==plan['training']
    assert manifest['model_id']==plan['model_id'] and manifest['model_revision']==plan['model_revision']
    assert manifest['train_sha256']==plan['dataset_sha256']['train'] and manifest['dev_sha256']==plan['dataset_sha256']['dev']
    assert manifest['source_sha256']==plan['source_sha256']['sft_train.py']
    assert manifest['optimizer_steps']==75 and manifest['distinct_decision_targets_used']==600
    assert manifest['distinct_tasks_used']==300 and manifest['microbatches']==600
    steps=lines(root/'training/steps.jsonl')
    ids=[i for step in steps for i in step['request_ids']]
    assert len(steps)==75 and len(ids)==600 and len(set(ids))==600
    assert all(step['gradient_norm_before_clip']>0 for step in steps)
    assert manifest['initial_adapter_sha256']!=manifest['final_adapter_state_sha256']
    assert manifest['initial_adapter_sha256']==smoke['initial_adapter_sha256']
    assert manifest['final_adapter_state_sha256']!=smoke['final_adapter_state_sha256']
    assert manifest['frozen_base_gradients_absent']
    for name,value in manifest['adapter_files_sha256'].items():
        assert sha(root/'training/adapter'/name)==value
    state=load_file(str(root/'training/adapter/adapter_model.safetensors'))
    assert all('lora_' in name for name in state)
    assert sum(p.numel() for p in state.values())==manifest['trainable_parameters']
    assert all(p.dtype==torch.float32 for p in state.values())
    assert tensor_hash(state)==manifest['final_adapter_state_sha256']
    tokenizer=AutoTokenizer.from_pretrained(info['snapshot'],local_files_only=True)
    prompt=Path(plan['prompt_files']['neutral']).read_text(encoding='utf-8')
    expected=[encode_example(r,tokenizer,plan['training']['max_context'])
              for t in splits['train'] for r in request_examples(t,prompt)]
    supervision=root/'training/supervision.jsonl'
    assert sha(supervision)==manifest['supervision_sha256'] and lines(supervision)==expected
    assert set(ids)=={r['task_id']+':'+str(r['turn']) for r in expected}
    result=dict(training_manifest_sha256=sha(root/'training/manifest.json'),
                training_steps_sha256=sha(root/'training/steps.jsonl'),
                supervision_sha256=sha(supervision),adapter_files_sha256=manifest['adapter_files_sha256'],
                rebuilt_masked_requests=600,unique_training_episodes=300,optimizer_steps=75,
                second_lookup_states=100,split_overlap_absent=True,all_training_targets_seen_once=True,
                training_and_smoke_same_initial_adapter=True,training_did_not_reuse_smoke_adapter=True,
                saved_adapter_state_hash_matches=True,saved_adapter_parameters=manifest['trainable_parameters'],
                saved_adapter_dtype='float32',only_lora_tensors_saved=True,
                limitation='Base tensors were frozen and excluded from optimizer; replay verifies recorded token/tool provenance, not a rerun of model generation.')
    (root/'training-audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    return manifest,smoke,result


def aggregate(root,info_path):
    plan=validate(root/'plan.json')
    training,smoke,audit=audit_training(root,plan,read(info_path))
    task_sets={s:{t['task_id']:t for t in lines(Path(plan['data_root'])/(s+'.jsonl'))} for s in ('dev','test')}
    conditions=[]
    groups={}
    verified_episodes=verified_requests=0
    for split,seeds in plan['evaluation_seeds'].items():
        groups[split]={}
        for model in ('base','sft'):
            for prompt in ('neutral','balanced_examples'):
                combined=[]
                for seed in seeds:
                    path=root/'eval'/f'{split}-{model}-{prompt}-seed{seed}'
                    manifest=read(path/'manifest.json')
                    condition=read(path/'condition.json')
                    verification=read(path/'verification.json')
                    rows=lines(path/'episodes.jsonl')
                    config=manifest['config']
                    expected=dict(model_id=plan['model_id'],model_revision=plan['model_revision'],
                                  dataset_sha256=plan['dataset_sha256'][split],task_count=len(task_sets[split]),
                                  seed=seed,do_sample=True,temperature=.7,top_p=.8,top_k=20,enable_thinking=False,
                                  dtype='bfloat16',device='cuda:0',batch_size=1,max_new_tokens=64,max_context=1024,
                                  max_turns=4,max_calls=3,parser='fenced',cpu_threads=4,
                                  system_prompt_sha256=plan['prompt_sha256'][prompt],
                                  baseline_source_sha256=plan['source_sha256']['baseline.py'],
                                  environment_source_sha256=plan['source_sha256']['environment.py'])
                    assert config==expected and manifest['status']=='complete'
                    assert condition['plan_sha256']==sha(root/'plan.json')
                    assert condition['source_sha256']==plan['source_sha256']['sft_eval.py']
                    assert condition['model']==model and condition['prompt']==prompt and condition['split']==split and condition['seed']==seed
                    assert condition['adapter_sha256']==(training['adapter_files_sha256']['adapter_model.safetensors'] if model=='sft' else None)
                    assert {r['task_id'] for r in rows}==task_sets[split].keys() and len(rows)==len(task_sets[split])
                    assert verification['episodes_sha256']==sha(path/'episodes.jsonl')
                    assert all(verification[k] for k in ('tool_replay_matches','answer_regrading_matches','input_token_provenance_matches','output_token_decode_matches'))
                    assert verification['tasks_verified']==len(rows)
                    for index,row in enumerate(rows):
                        assert row['seed']==seed+100*index
                        assert all(row[k]==task_sets[split][row['task_id']][k] for k in ('kind','template_id','minimum_calls','split'))
                        assert hashlib.sha256(row['messages'][0]['content'].encode()).hexdigest()==plan['prompt_sha256'][prompt]
                        valid=[]
                        for request in row['requests']:
                            try:
                                parse_action(request.get('text',''))
                                valid.append(True)
                            except ValueError:
                                valid.append(False)
                        assert sum(valid)==row['strict_valid_requests']
                        assert row['strict_success']==(row['success'] and all(valid))
                    if split=='test':
                        train_end=datetime.fromisoformat(training['started_at'])+timedelta(seconds=training['process_wall_seconds'])
                        assert datetime.fromisoformat(manifest['started_at'])>train_end
                    verified_episodes+=len(rows)
                    verified_requests+=verification['requests_verified']
                    combined.extend(rows)
                    conditions.append(dict(split=split,model=model,prompt=prompt,seed=seed,
                                           metrics=metrics(rows,task_sets[split]),
                                           episodes_sha256=verification['episodes_sha256'],
                                           verification_sha256=sha(path/'verification.json'),
                                           peak_allocated_mib=manifest['cuda_peak_allocated_mib']))
                groups[split][model+'/'+prompt]=combined
    results={split:{key:metrics(rows,task_sets[split]) for key,rows in variants.items()} for split,variants in groups.items()}
    comparisons={}
    for split,variants in groups.items():
        comparisons[split]={}
        for left,right in [('base/balanced_examples','sft/neutral'),('base/neutral','sft/neutral'),
                           ('base/balanced_examples','sft/balanced_examples'),('sft/neutral','sft/balanced_examples')]:
            comparisons[split][left+' -> '+right]=paired(variants[left],variants[right])
    summary=dict(plan_sha256=sha(root/'plan.json'),report_source_sha256=sha(Path(__file__)),
                 generated_at=datetime.now().astimezone().isoformat(),training=training,smoke=smoke,training_audit=audit,
                 conditions=conditions,aggregate=results,paired_comparisons=comparisons,
                 total_verified_episodes=verified_episodes,total_verified_requests=verified_requests,
                 caveats=['60 unique test tasks per condition, each repeated under three sampling seeds; 180 attempts are correlated.',
                          'Held-out paraphrases and entities, same English tool schema and two-hop composition; not a real-world agent benchmark.',
                          'One LoRA training seed and one fixed final checkpoint; no test-selected hyperparameters.',
                          'CUDA allocation/reservation are process metrics; desktop GPU usage is additional.',
                          'Generic per-condition analyzer caveats describe the legacy inference engine; this report records real LoRA training separately.'])
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8',newline='\n')
    examples=[]
    left={(r['task_id'],r['seed']):r for r in groups['test']['base/balanced_examples']}
    for r in groups['test']['sft/neutral']:
        a=left[r['task_id'],r['seed']]
        if (not a['success'] and r['success']) or not r['success']:
            examples.append(dict(task_id=r['task_id'],task_seed=r['seed'],kind=r['kind'],
                                 question=task_sets['test'][r['task_id']]['prompt'],
                                 baseline_success=a['success'],sft_success=r['success'],
                                 baseline_actions=a['actions'],baseline_observations=a['observations'],
                                 sft_actions=r['actions'],sft_observations=r['observations'],
                                 expected=task_sets['test'][r['task_id']]['expected']))
    (root/'case-comparisons.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in examples),encoding='utf-8',newline='\n')
    neutral={(r['task_id'],r['seed']):r for r in groups['test']['sft/neutral']}
    interference=[]
    for r in groups['test']['sft/balanced_examples']:
        a=neutral[r['task_id'],r['seed']]
        if a['success']!=r['success']:
            interference.append(dict(task_id=r['task_id'],task_seed=r['seed'],kind=r['kind'],
                                     question=task_sets['test'][r['task_id']]['prompt'],
                                     neutral_success=a['success'],examples_success=r['success'],
                                     neutral_actions=a['actions'],neutral_observations=a['observations'],
                                     example_actions=r['actions'],example_observations=r['observations'],
                                     example_raw_outputs=[q.get('text','') for q in r['requests']],
                                     expected=task_sets['test'][r['task_id']]['expected']))
    (root/'prompt-interference-cases.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in interference),encoding='utf-8',newline='\n')
    return summary


def plot(root,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=['Base / neutral','Base / examples','SFT / neutral','SFT / examples']
    keys=['base/neutral','base/balanced_examples','sft/neutral','sft/balanced_examples']
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
    positions=range(4)
    for offset,kind,color in [(-.23,'direct','#94a3b8'),(0,'one_lookup','#38bdf8'),(.23,'two_lookup','#14b8a6')]:
        values=[100*summary['aggregate']['test'][key]['by_kind'][kind]['successes']/60 for key in keys]
        bars=axes[0].bar([p+offset for p in positions],values,width=.22,label=kind,color=color)
        axes[0].bar_label(bars,fmt='%.0f',padding=3,fontsize=8)
    axes[0].set_xticks(list(positions),labels,rotation=15,ha='right')
    axes[0].set_ylim(0,117)
    axes[0].set_ylabel('Test success (%)')
    axes[0].set_title('20 unique tasks / family, three sampling seeds')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=3,frameon=False,fontsize=9)
    records=lines(root/'training/steps.jsonl')
    axes[1].plot([r['step'] for r in records],[r['loss'] for r in records],color='#64748b',alpha=.55,label='8-request step mean')
    averages=[statistics.mean(r['loss'] for r in records[max(0,i-4):i+1]) for i in range(len(records))]
    axes[1].plot([r['step'] for r in records],averages,color='#0d9488',label='5-step moving mean')
    axes[1].set_yscale('log')
    axes[1].set_xlabel('Optimizer step (one epoch = 75)')
    axes[1].set_ylabel('Assistant target cross entropy')
    axes[1].set_title('Training loss is not rollout success')
    axes[1].legend(fontsize=8)
    fig.savefig(root/'sft-results.png',dpi=170)
    plt.close(fig)


def write_report(root,summary):
    """Keep all reported denominators and comparisons tied to audited traces."""
    def pct(count,total):
        return f'{100*count/total:.1f}%'
    labels={'base/neutral':'原模型 + 短协议',
            'base/balanced_examples':'原模型 + 此前最强示例提示',
            'sft/neutral':'SFT + 短协议',
            'sft/balanced_examples':'SFT + 同一示例提示'}
    table=['| 条件 | 直接回答 | 一步查询 | 两步查询 | 总正确率 | 严格 JSON 且答对 |',
           '|---|---:|---:|---:|---:|---:|']
    for key,label in labels.items():
        m=summary['aggregate']['test'][key]
        cells=[f"{m['by_kind'][kind]['successes']}/60" for kind in ('direct','one_lookup','two_lookup')]
        table.append('| '+label+' | '+' | '.join(cells)+f" | {m['successes']}/180 ({pct(m['successes'],180)}) | {pct(m['strict_successes'],180)} |")
    seed_table=['| 采样种子 | 原模型短协议 | 原模型示例提示 | SFT短协议 | SFT示例提示 |',
                '|---|---:|---:|---:|---:|']
    for seed in (7,10007,20007):
        values=[]
        for key in labels:
            model,prompt=key.split('/')
            c=next(c for c in summary['conditions'] if c['split']=='test' and c['seed']==seed and c['model']==model and c['prompt']==prompt)
            values.append(str(c['metrics']['successes'])+'/60')
        seed_table.append('| '+str(seed)+' | '+' | '.join(values)+' |')
    train=summary['training']
    smoke=summary['smoke']
    dev=summary['aggregate']['dev']
    base=summary['aggregate']['test']['base/balanced_examples']
    sft=summary['aggregate']['test']['sft/neutral']
    comparison=summary['paired_comparisons']['test']['base/balanced_examples -> sft/neutral']
    best_base_key=max(('base/neutral','base/balanced_examples'),key=lambda k:summary['aggregate']['test'][k]['successes'])
    best_base=summary['aggregate']['test'][best_base_key]
    best_gain=100*(sft['successes']-best_base['successes'])/180
    gain=(sft['successes']-base['successes'])/180*100
    verdict=(f'本次固定配置 SFT 比此前最强示例提示提高 {gain:.1f} 个百分点。' if gain>0
             else f'本次固定配置 SFT 没有超过此前最强示例提示，差值为 {gain:.1f} 个百分点。')
    verdict+=f"本轮两组未训练基线中得分较高的是「{labels[best_base_key]}」，{best_base['successes']}/180；与它相比，SFT 短协议差值为 {best_gain:+.1f} 个百分点。"
    cases=lines(root/'case-comparisons.jsonl')
    case=next((c for c in cases if c['kind']=='two_lookup' and c['sft_success'] and not c['baseline_success']),None)
    case_text='没有观察到两步任务由基线错误转为 SFT 正确的案例。'
    if case:
        case_text=(f"任务 `{case['task_id']}`，实际任务种子 `{case['task_seed']}`：\n\n"
                   f"> {case['question']}\n\n"
                   '最强提示基线的实际动作：\n\n```json\n'+json.dumps(case['baseline_actions'],indent=2)+'\n```\n\n'
                   'SFT 短协议的实际动作：\n\n```json\n'+json.dumps(case['sft_actions'],indent=2)+'\n```\n\n'
                   'SFT 实际收到的工具结果：\n\n```json\n'+json.dumps(case['sft_observations'],indent=2)+'\n```\n\n'
                   f"最终验证答案为 `{case['expected']}`。动作和观察来自实际生成记录，没有用 oracle 补齐缺失步骤。")
    decision=('短协议在当前测试上已经饱和，继续增加同类 SFT 轮数或直接做 GRPO 的信息增益有限。下一轮先建立新的压力测试：'
              '三至四步链、可省略的查询、不同信息已给出的情况、工具报错与恢复，以及新工具字段。'
              '把开发集与最终测试重新划分，先复测这次固定 adapter，再依据失败类型决定补 SFT 还是开始 RL。'
              if sft['successes']==180 else
              '下一轮先用开发集定位剩余失败：首步目标选择、实际 owner ID 的复制、查完后的答案提取、协议错误。'
              '再针对开发集决定数据补充或训练配置；本次测试集保持封存，不用它反复选轮数或学习率。')
    interference=summary['paired_comparisons']['test']['sft/neutral -> sft/balanced_examples']
    sft_examples=summary['aggregate']['test']['sft/balanced_examples']
    interference_text=(f"同一 adapter 加入示例后，从短协议正确变成错误 {interference['lost']} 次，从错误变成正确 {interference['gained']} 次。"
                       f"SFT 加示例的总分是 {sft_examples['successes']}/180；更长的示例提示没有自动带来更好结果。"
                       '训练使用短协议，而示例条件加入额外的固定实体和值，因此这个对照同时改变了提示长度与示例内容，不能单独归因于某一因素。')
    regressions=lines(root/'prompt-interference-cases.jsonl')
    regression=next((c for c in regressions if c['neutral_success'] and not c['examples_success']),None)
    if regression:
        interference_text+=(f"例如 `{regression['task_id']}` 的同一任务种子 `{regression['task_seed']}`，问题是「{regression['question']}」。"
                            f"加入示例后的动作是 `{json.dumps(regression['example_actions'],separators=(',',':'))}`，"
                            f"实际答案应为 `{regression['expected']}`。完整配对记录见 [prompt-interference-cases.jsonl](prompt-interference-cases.jsonl)。")
    template_table=['| 一步查询的测试措辞 | 原模型短协议 | 原模型示例提示 | SFT短协议 | SFT示例提示 |',
                    '|---|---:|---:|---:|---:|']
    test_templates=read(Path(read(root/'plan.json')['data_root'])/'audit.json')['test_templates']
    for i,template in enumerate(test_templates['one_lookup']):
        cells=[]
        for key in labels:
            m=summary['aggregate']['test'][key]['by_template'][f'one_lookup:{i}']
            cells.append(f"{m['successes']}/{m['attempts']}")
        template_table.append('| '+template+' | '+' | '.join(cells)+' |')
    report=f'''# 300 条轨迹 LoRA SFT：本地训练与最强提示基线对比

{verdict}5060 Laptop 8GB 完成了实际反向传播、优化器更新和真实工具交互评测。这里的结论限定在这套合成英语查询任务内。

## 独立测试结果

{chr(10).join(table)}

每个条件有 **60 个独立测试任务 × 3 个采样种子 = 180 次尝试**。每类是 20 个任务重复三次，不能写成 180 个独立问题。四个条件合计 720 次测试尝试；开发集另有 120 次。实体、私有值和测试措辞均与训练隔离，但工具 schema 和两步组合模式相同。

{chr(10).join(seed_table)}

预先固定的主要部署对比是「SFT 短协议 vs 原模型此前最强示例提示」，同时保留同提示的 base/SFT 对照。主要配对对比中，由错转对 **{comparison['gained']}** 次，由对转错 **{comparison['lost']}** 次，两者均对 **{comparison['both_correct']}** 次，两者均错 **{comparison['both_wrong']}** 次。另报告本轮较强的未训练基线，避免只与较弱条件比较；该较强基线是测试后描述性对比，并未用于重新训练或选模型。三个种子复用了同一批问题，不能把这些计数当作独立样本做显著性推断。

开发集在训练前重跑：原模型短协议 {dev['base/neutral']['successes']}/30、此前最强示例提示 {dev['base/balanced_examples']['successes']}/30，两步分别为 {dev['base/neutral']['by_kind']['two_lookup']['successes']}/10 和 {dev['base/balanced_examples']['by_kind']['two_lookup']['successes']}/10；SFT 短协议 {dev['sft/neutral']['successes']}/30，SFT 示例提示 {dev['sft/balanced_examples']['successes']}/30。开发集用于诊断，最终 adapter 固定为一轮的最后一次更新。所有测试生成发生在 adapter 保存并哈希锁定之后，没有依据测试分数选择 checkpoint、学习率或轮数。

## 提升发生在哪里

此前示例提示在开发集上最强，换成保留措辞后不能假设这个排序仍然成立。一步任务按测试措辞展开如下，可检查格式正确但没有读取私有值的失败：

{chr(10).join(template_table)}

最强提示基线的两步尝试中，开始查询 {base['two_lookup_started']}/60 次，首步正确查询 owner {base['two_correct_first_lookup']}/60 次，完成正确 owner → department 查询前缀 {base['two_correct_chain_prefix']}/60 次，最后答对 {base['by_kind']['two_lookup']['successes']}/60 次。

SFT 短协议相应为 {sft['two_lookup_started']}/60、{sft['two_correct_first_lookup']}/60、{sft['two_correct_chain_prefix']}/60、{sft['by_kind']['two_lookup']['successes']}/60。它向实际工具返回的 employee ID 查询 department {sft['two_second_lookup_to_actual_owner']}/60 次。最强提示基线复制示例 department 的次数是 {base['two_copied_example_department']}，直接把实际 owner ID 当最终答案的次数是 {base['two_answered_owner_id']}；SFT 短协议分别是 {sft['two_copied_example_department']} 和 {sft['two_answered_owner_id']}。

原模型最强提示原始 JSON 有效率为 {base['strict_valid_requests']}/{base['requests']}（{pct(base['strict_valid_requests'],base['requests'])}），SFT 短协议为 {sft['strict_valid_requests']}/{sft['requests']}（{pct(sft['strict_valid_requests'],sft['requests'])}）。查询链指标与最终答案一起看，才能区分格式改善和决策改善。

SFT 短协议有 {sft['task_stability']['all']['all_attempts_correct']}/60 个问题在三个种子下都正确，{sft['task_stability']['all']['mixed']} 个问题结果不稳定。正确轨迹额外查询总数为 {sft['correct_extra_calls']}，无效工具调用 {sft['invalid_tool_calls']} 次；生成截断 {sft['output_truncations']} 次、上下文超限 {sft['context_failures']} 次。调用成本必须以正确性为前提比较，基线过早回答造成的低调用数不能视作效率收益。

{interference_text}

## 数据修正与监督方式

项目和员工使用独立随机的 12 位十六进制 ID，不再共享后缀。真实实体与干扰实体使用同一 ID 格式。训练、开发、测试的实体和私有值交集均为零，并排除提示中固定的示例答案，避免复制示例偶然命中。

划分为训练 300 条（每类 100）、开发 30 条（每类 10）、测试 60 条（每类 20）。训练和开发使用四种措辞，测试使用每类三种保留措辞。训练内有 40 对同一项目、不同目标的轨迹：预算目标应查 budget，部门目标应查 owner 再查 department。共 260 组训练事实，配对复用只发生在训练内部。

300 条正确 oracle 轨迹生成 **600 个下一步决策目标**：直接回答 100、一步轨迹 200、两步轨迹 300，其中包含 100 个关键的「已经拿到 owner，下一步继续查询」状态。oracle 先实际执行工具并验证，再导出目标；训练输入只包含问题、之前的动作与工具结果，不含未来观察或 expected 字段。

每个决策独立应用模型原生 chat template，关闭 thinking；prefix 标签全部为 -100，只监督下一条紧凑 JSON 和 EOS。每个决策的 assistant token loss 先取均值，再对累积组取均值；两步轨迹贡献三个决策状态。没有 packing、没有静默截断。完整 [supervision.jsonl](training/supervision.jsonl) 保存实际 token IDs 和 loss mask。

## 5060 的实测训练显存

| 测量 | 真实优化器步数 | 实际 microbatch 数 | PyTorch 峰值 allocated | 峰值 reserved | 优化循环时间 |
|---|---:|---:|---:|---:|---:|
| 显存探针，权重丢弃 | 8 | 64 | {smoke['cuda_peak_allocated_mib']/1024:.2f} GiB | {smoke['cuda_peak_reserved_mib']/1024:.2f} GiB | {smoke['optimization_wall_seconds']:.2f} 秒 |
| 正式 SFT，重新初始化 | 75 | 600 | {train['cuda_peak_allocated_mib']/1024:.2f} GiB | {train['cuda_peak_reserved_mib']/1024:.2f} GiB | {train['optimization_wall_seconds']:.2f} 秒 |

训练器记录的加载、开发集 loss 评估、优化与保存共 {train['process_wall_seconds']:.2f} 秒，不含 Python 启动和库导入。显存探针重复八个最长决策上下文，只用于检验显存；它不是完整训练，也未用于最终评测。正式训练与探针具有相同初始 adapter 哈希、不同最终哈希，确认没有沿用探针权重。

配置为 Qwen3-0.6B（固定 revision）、BF16 冻结主干、FP32 LoRA、q_proj/v_proj、rank 8、alpha 16、dropout 0.05，microbatch 1、累积 8，AdamW 学习率 1e-4、weight decay 0、gradient clip 1.0、seed 42、一轮。实际可训练参数 {train['trainable_parameters']:,}。没有量化或 CPU offload，gradient checkpointing {'开启' if train['gradient_checkpointing'] else '关闭'}。配置上下文上限 768，**实测最长序列仅 {train['max_sequence_tokens']} tokens**；长上下文训练和 GRPO 的显存不能由此推断。

开发集 teacher-forced loss 从 {train['dev_loss_before']:.6f} 降到 {train['dev_loss_after']:.8f}。它只衡量给定正确历史时的下一步预测；上面的真实多轮 rollout 才检验模型能否自己走到后续状态。PyTorch allocated/reserved 是该进程指标，桌面应用还会占用额外显存。

## 一条实际配对轨迹

{case_text}

短协议相对于此前示例基线的全部由错转对案例，以及短协议 SFT 失败案例（若有），保存在 [case-comparisons.jsonl](case-comparisons.jsonl)；加示例后的退步另存于 [prompt-interference-cases.jsonl](prompt-interference-cases.jsonl)。可回到各条件的 episodes.jsonl 检查原始字符串和精确 token。

## 证据与解释范围

本轮共 **{summary['total_verified_episodes']} 个真实模型 episode、{summary['total_verified_requests']} 个模型请求**完成独立复核。复核重放工具、重新判答案、重建输入 token 和解码输出 token，同时核验条件配置、数据、提示、源代码和 adapter 哈希。复核验证记录的一致性与执行来源，不是重新运行模型生成。

[训练审计](training-audit.json) 确认 600 个目标恰好各暴露一次，完整 loss mask 可重建，保存的 safetensors 与训练后权重哈希一致，文件中只含 LoRA 张量。主干冻结且不进入优化器，训练器逐步检查主干没有梯度。没有 API 调用、云 GPU 或 RL。全部复现命令见 [reproduce.md](reproduce.md)，模型加载与适用范围见 [model-card.md](model-card.md)。

旧的 100 题和提示实验保留原始字节，不与本轮新划分分数直接配对或混算。旧通用 analyzer 的 caveat 描述的是 inference 进程；本轮实际训练和测试划分以本报告及 [summary.json](summary.json) 为准。

一次训练种子、三次共享任务的采样重复，仍不足以证明通用 agent 能力。测试保留的是措辞、实体和事实，并未保留全新工具或规划结构。这次结果支持小模型能够在此 schema 上通过少量监督学会或激活所需策略；不能据此断言此前失败代表模型容量上限，也不能宣称 RL 或 RSI 已完成。

## 下一步判断

{decision}

面试材料可以如实描述为：构建有数据审计和独立轨迹复核的工具决策环境，在 8GB 本地 GPU 上用 300 条 oracle 轨迹进行 LoRA SFT，与固定最强提示基线做配对比较，并分析查询链、协议有效率和调用成本。把「合成环境、60 个独立测试问题、三次采样、单训练种子」与结果一并说明。

![测试正确率与训练曲线](sft-results.png)
'''
    (root/'analysis.zh-CN.md').write_text(report,encoding='utf-8',newline='\n')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path('results/sft-v1'))
    parser.add_argument('--model-info',type=Path,required=True)
    args=parser.parse_args()
    summary=aggregate(args.root,args.model_info)
    plot(args.root,summary)
    write_report(args.root,summary)
    print(json.dumps(dict(verified_episodes=summary['total_verified_episodes'],verified_requests=summary['total_verified_requests'],
                         test={k:v['successes'] for k,v in summary['aggregate']['test'].items()}),indent=2))


if __name__=='__main__':
    main()

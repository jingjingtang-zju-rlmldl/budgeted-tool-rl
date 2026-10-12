"""Audit real continuation training, paired evaluation and 32x4 signal groups."""
from collections import Counter,defaultdict
from datetime import datetime
import json
import math
from pathlib import Path
import statistics

from .sft2_experiment import read,rows,write,validate,checkpoint,sha256,CONDITIONS
from .sft2_data import KINDS,audit,request_examples,encode_example
from .stress_data import prior_sets

LABELS={'base/examples':'未训练模型＋固定强提示','old/protocol':'补训前 SFT＋简洁提示',
        'old/examples':'补训前 SFT＋强提示','new/protocol':'补训后 SFT＋简洁提示'}
FAMILIES=dict(anchor_two='两步锚点',chain_three='三步查询',chain_four='四步查询',
              retry_two='临时错误重试',redirect_two='失效 ID 恢复',skip_owner='已知 owner',
              skip_prefix='已知链前缀',given_answer='已知答案',one_lookup='旧单步预算查询')


def metrics(rs):
    good=[r for r in rs if r['success']]
    errors=[r for r in rs if r['recoverable_errors']]
    return dict(attempts=len(rs),successes=len(good),grounded_successes=sum(r['grounded_success'] for r in rs),
        strict_successes=sum(r['strict_success'] for r in rs),requests=sum(len(r['requests']) for r in rs),
        correct_extra_calls=sum(r['correct_extra_calls'] for r in good),
        optimal_correct=sum(r['calls']==r['minimum_calls'] for r in good),
        correct_mean_calls=statistics.mean(r['calls'] for r in good) if good else None,
        failed_stops=dict(Counter(r['stop_reason'] for r in rs if not r['success'])),
        failed_without_required_lookup=sum(not r['success'] and r['minimum_calls']>0 and r['calls']==0 for r in rs),
        failed_after_partial_call_budget=sum(not r['success'] and 0<r['calls']<r['minimum_calls'] for r in rs),
        failed_call_counts={str(k):v for k,v in Counter(r['calls'] for r in rs if not r['success']).items()},
        observed_error_episodes=len(errors),correct_after_error=sum(r['success'] for r in errors),
        necessary_retries=sum(r['necessary_retries'] for r in rs),followed_redirects=sum(r['followed_redirects'] for r in rs),
        invalid_queries=sum(r['invalid_queries'] for r in rs),
        unknown_field_errors=sum(o.get('error')=='unknown_field' for r in rs for o in r['observations']),
        unknown_entity_errors=sum(o.get('error')=='unknown_entity' for r in rs for o in r['observations']),
        project_employee_field_actions=sum(a.get('action')=='lookup' and str(a.get('entity','')).startswith('project_') and a.get('field')=='employee' for r in rs for a in r['actions']),
        simulated_tool_requests=sum('Tool:' in q['text'] for r in rs for q in r['requests']),
        multiple_action_marker_requests=sum(q['text'].count('"action"')>1 for r in rs for q in r['requests']),
        truncated_requests=sum(q['truncated'] for r in rs for q in r['requests']),
        max_input_tokens=max(len(q['input_ids']) for r in rs for q in r['requests']))


def grouped(rs):
    return metrics(rs)|dict(by_kind={k:metrics([r for r in rs if r['kind']==k]) for k in sorted({r['kind'] for r in rs})})


def paired(a,b):
    aa={(r['task_id'],r['seed']):r for r in a};bb={(r['task_id'],r['seed']):r for r in b}
    assert aa.keys()==bb.keys() and len(aa)==len(a) and len(bb)==len(b)
    return dict(both_correct=sum(aa[k]['success'] and bb[k]['success'] for k in aa),
        gained=sum(not aa[k]['success'] and bb[k]['success'] for k in aa),
        lost=sum(aa[k]['success'] and not bb[k]['success'] for k in aa),
        both_wrong=sum(not aa[k]['success'] and not bb[k]['success'] for k in aa))


def test_consistency(rs):
    groups=defaultdict(list)
    for r in rs:groups[r['task_id']].append(r)
    assert len(groups)==72 and all(len(g)==3 for g in groups.values())
    counts=Counter(sum(r['success'] for r in g) for g in groups.values())
    return dict(unique_tasks=72,attempts_per_task=3,
                success_counts_histogram={str(i):counts[i] for i in range(4)},
                all_three_correct=counts[3],all_three_failed=counts[0],mixed=counts[1]+counts[2])


def signal_groups(rs):
    groups=defaultdict(list)
    for r in rs:groups[r['task_id']].append(r)
    assert len(groups)==32 and all(len(g)==4 for g in groups.values())
    result=[]
    for task_id,g in groups.items():
        if not any(r['success'] for r in g):label='all_failed'
        elif not all(r['success'] for r in g):label='mixed_correctness'
        elif len({r['calls'] for r in g})>1:label='all_correct_different_cost'
        else:label='all_correct_same_cost'
        rewards=[r['reward_cost_gated'] for r in g]
        result.append(dict(task_id=task_id,kind=g[0]['kind'],classification=label,
            successes=sum(r['success'] for r in g),seeds=[r['seed'] for r in g],
            calls=[r['calls'] for r in g],minimum_calls=g[0]['minimum_calls'],rewards=rewards,
            population_reward_variance=statistics.pvariance(rewards),
            stop_reasons=[r['stop_reason'] for r in g]))
    return dict(groups=len(result),rollouts=len(rs),counts=dict(Counter(g['classification'] for g in result)),
        nonconstant_reward_groups=sum(g['population_reward_variance']>0 for g in result),
        details=result)


def signal_wording(rs,tasks):
    taskmap={t['task_id']:t for t in tasks}
    return dict(by_kind_and_template={kind:{str(tid):metrics([r for r in rs if r['kind']==kind and taskmap[r['task_id']]['template_id']==tid]) for tid in (0,1)} for kind in sorted({r['kind'] for r in rs})},
        first_project_employee_field_failures=sum(not r['success'] and bool(r['actions']) and r['actions'][0].get('action')=='lookup' and str(r['actions'][0].get('entity','')).startswith('project_') and r['actions'][0].get('field')=='employee' for r in rs),
        failed_before_recoverable_error=sum(not r['success'] and not r['recoverable_errors'] for r in rs),
        caveat='Signal changes wording and fault composition together; wording strata use different facts, not paired counterfactuals. Associations are descriptive, not isolated causal effects.')


def training_audit(root,plan,info_path):
    from transformers import AutoTokenizer
    from safetensors.torch import load_file
    from .sft2_train import tensor_hash
    tok=AutoTokenizer.from_pretrained(read(info_path)['snapshot'],local_files_only=True)
    train=rows(Path(plan['data_root'])/'train.jsonl')
    prompt=Path(plan['prompt_files']['protocol']).read_text(encoding='utf-8')
    expected=[encode_example(x,tok,plan['training']['max_context']) for t in train for x in request_examples(t,prompt)]
    saved=rows(root/'training'/'supervision.jsonl');assert expected==saved
    assert sha256(root/'training'/'supervision.jsonl')==read(root/'training'/'manifest.json')['supervision_sha256']
    steps=rows(root/'training'/'steps.jsonl')
    ids=[i for s in steps for i in s['request_ids']]
    targets={x['task_id']+':'+str(x['turn']) for x in expected}
    assert set(ids)==targets and len(ids)==len(targets)==990 and len(steps)==math.ceil(990/8)
    assert [s['step'] for s in steps]==list(range(1,len(steps)+1))
    assert all(math.isfinite(s['loss']) and math.isfinite(s['gradient_norm_before_clip']) and s['gradient_norm_before_clip']>0 for s in steps)
    m=read(root/'training'/'manifest.json')
    assert m['initial_adapter_sha256']==plan['old_adapter_state_sha256']
    assert m['adapter_changed'] and m['frozen_base_gradients_absent'] and m['training_performed']
    assert m['final_adapter_state_sha256']!=m['initial_adapter_sha256']
    state=load_file(str(root/'training'/'adapter'/'adapter_model.safetensors'))
    assert all('lora_' in k for k in state)
    assert all(str(v.dtype)=='torch.float32' for v in state.values())
    assert sum(v.numel() for v in state.values())==m['trainable_parameters']==1146880
    assert tensor_hash(state)==m['final_adapter_state_sha256']
    result=dict(supervision_matches_rebuilt_oracle=True,assistant_only_masks_and_eos_match=True,
                unique_decision_targets=990,trajectories=300,optimizer_steps=len(steps),one_epoch_exact_coverage=True,
                continuation_initial_weights_match=True,adapter_changed=True,base_has_no_gradients=True,
                saved_lora_tensors_match_post_training_state=True,
                training_steps_sha256=sha256(root/'training'/'steps.jsonl'),
                supervision_sha256=sha256(root/'training'/'supervision.jsonl'),
                adapter_files_sha256=m['adapter_files_sha256'],
                supervised_error_types=dict(Counter(json.loads(x['messages'][-1]['content'])['error'] for x in expected if x['messages'][-1]['role']=='tool' and 'error' in json.loads(x['messages'][-1]['content']))),
                recovery_next_state_targets=sum(x['messages'][-1]['role']=='tool' and 'error' in json.loads(x['messages'][-1]['content']) for x in expected),
                training_manifest_sha256=sha256(root/'training'/'manifest.json'))
    write(root/'training-audit.json',result);return result


def aggregate(root,info):
    path=root/'plan.json';p=validate(path);training=checkpoint(p,path)
    historical=list(p['historical_files_sha256'])
    old_e,old_v=prior_sets([Path(s) for s in historical])
    for file in historical:
        for r in rows(file):old_e.update(r.get('aliases',{}))
    splits={s:rows(Path(p['data_root'])/(s+'.jsonl')) for s in p['dataset_sha256']}
    audit(splits,old_e,old_v)
    data={};cohorts=[];episodes=requests=0;peak=reserved=0.
    for split in ('dev','test','signal'):
        data[split]={}
        combinations=CONDITIONS if split!='signal' else [('new','protocol')]
        for model,prompt in combinations:
            combined=[]
            for seed in p['evaluation_seeds'][split]:
                out=root/'eval'/f'{split}-{model}-{prompt}-seed{seed}'
                m=read(out/'manifest.json');v=read(out/'verification.json');rs=rows(out/'episodes.jsonl')
                config=m['config'];assert m['status']=='complete' and m['trainable_parameters']==0 and not m['new_training_performed']
                assert config['plan_sha256']==sha256(path) and config['source_sha256']==p['source_sha256']
                assert config['dataset_sha256']==p['dataset_sha256'][split] and config['system_prompt_sha256']==p['prompt_sha256'][prompt]
                assert config['model']==model and config['prompt']==prompt and config['seed']==seed
                assert config['adapter_files_sha256']==(p['old_adapter_files_sha256'] if model=='old' else training['adapter_files_sha256'] if model=='new' else None)
                assert all(config[k]==val for k,val in p['eval_fixed'].items())
                assert v['episodes_sha256']==sha256(out/'episodes.jsonl') and v['tasks_verified']==len(rs)==p['data_counts'][split]
                assert {r['task_id'] for r in rs}=={r['task_id'] for r in splits[split]}
                assert all(val is True for key,val in v.items() if key.endswith('_matches') or key.endswith('_match'))
                assert m['checkpoint_manifest_sha256']==sha256(root/'training'/'manifest.json')
                assert datetime.fromisoformat(m['started_at'])>datetime.fromisoformat(training['started_at'])
                combined+=rs;episodes+=len(rs);requests+=v['requests_verified']
                peak=max(peak,m['cuda_peak_allocated_mib']);reserved=max(reserved,m['cuda_peak_reserved_mib'])
                cohorts.append(dict(split=split,model=model,prompt=prompt,seed=seed,metrics=grouped(rs),
                    episodes_sha256=sha256(out/'episodes.jsonl'),verification_sha256=sha256(out/'verification.json')))
            data[split][model+'/'+prompt]=combined
    aggregates={s:{k:grouped(rs) for k,rs in ds.items()} for s,ds in data.items()}
    comparisons={k+' -> new/protocol':paired(v,data['test']['new/protocol']) for k,v in data['test'].items() if k!='new/protocol'}
    probe=signal_groups(data['signal']['new/protocol'])
    result=dict(generated_at=datetime.now().astimezone().isoformat(),plan_sha256=sha256(path),
        report_source_sha256=sha256(Path(__file__)),aggregate=aggregates,cohorts=cohorts,
        comparisons=comparisons,signal_groups=probe,training=training,
        test_task_consistency={key:test_consistency(rs) for key,rs in data['test'].items()},
        signal_wording_diagnostic=signal_wording(data['signal']['new/protocol'],splits['signal']),
        memory_probe=read(root/'memory-probe'/'manifest.json'),training_audit=training_audit(root,p,info),
        total_verified_episodes=episodes,total_verified_model_requests=requests,
        inference_peak_allocated_mib=peak,inference_peak_reserved_mib=reserved,
        rl_optimization_performed=False)
    if (root/'test-analysis-before-signal.json').exists():
        interim=read(root/'test-analysis-before-signal.json')
        assert interim['plan_sha256']==sha256(path) and interim['signal_not_opened'] is True
        def check_recorded_fields(before,after):
            for key,value in before.items():
                assert key in after
                if isinstance(value,dict):check_recorded_fields(value,after[key])
                else:assert value==after[key],key
        # Extra descriptive error metrics were added after opening signal;
        # retain the original timestamped analysis and check all recorded fields.
        check_recorded_fields(interim['metrics'],aggregates['test'])
        assert all(interim['comparisons'][key]==comparisons[key+' -> new/protocol'] for key in interim['comparisons'])
        for seed in p['evaluation_seeds']['signal']:
            m=read(root/'eval'/f'signal-new-protocol-seed{seed}'/'manifest.json')
            assert datetime.fromisoformat(interim['analyzed_at'])<datetime.fromisoformat(m['started_at'])
        result['test_analysis_before_signal_sha256']=sha256(root/'test-analysis-before-signal.json')
    if (root/'action-budget-audit.json').exists():
        aa=read(root/'action-budget-audit.json')
        assert aa['plan_sha256']==sha256(path) and aa['dataset_sha256']==p['dataset_sha256']
        assert aa['source_sha256']==sha256(Path(__file__).with_name('sft2_action_audit.py'))
        assert aa['all_checked_spellings_fit'] and aa['max_tokens_including_eos']<=p['eval_fixed']['max_new_tokens']
        result['action_budget_audit']=aa
    if (root/'resource-amendment.json').exists():
        amended=read(root/'resource-amendment.json');original=read(root/'plan-original.json')
        assert sha256(root/'plan-original.json')==amended['original_plan_sha256']
        restored=dict(p);restored['memory_test']=original['memory_test'];assert restored==original
        assert training['gradient_checkpointing'] is True
        assert datetime.fromisoformat(amended['amended_at'])<datetime.fromisoformat(training['started_at'])
        attempt=root/amended['additional_discarded_attempt']
        assert sha256(attempt/'steps.jsonl')==amended['additional_steps_sha256']
        assert len(rows(attempt/'steps.jsonl'))==amended['additional_discarded_optimizer_steps']
        assert not (attempt/'adapter').exists()
        cp=read(root/'memory-probe-checkpointing'/'manifest.json')
        assert cp['status']=='complete' and cp['gradient_checkpointing'] and cp['optimizer_steps']==8
        assert cp['initial_adapter_sha256']==p['old_adapter_state_sha256']
        result['resource_amendment']=amended;result['checkpointing_memory_probe']=cp
    if (root/'resource-retry.json').exists():
        retry=read(root/'resource-retry.json')
        attempt=root/retry['partial_attempt']
        assert read(attempt/'manifest.json')['status']=='aborted_resource_contention'
        assert sha256(attempt/'steps.jsonl')==retry['partial_steps_sha256']
        assert len(rows(attempt/'steps.jsonl'))==retry['discarded_optimizer_steps']
        assert not (attempt/'adapter').exists()
        result['resource_retry']=retry
        if retry.get('previous_retry_record'):
            first=read(root/retry['previous_retry_record'])
            first_attempt=root/first['partial_attempt']
            assert read(first_attempt/'manifest.json')['status']=='aborted_resource_contention'
            assert sha256(first_attempt/'steps.jsonl')==first['partial_steps_sha256']
            assert len(rows(first_attempt/'steps.jsonl'))==first['discarded_optimizer_steps']
            assert not (first_attempt/'adapter').exists()
            result['resource_retry_first']=first
    write(root/'summary.json',result);write(root/'reward-signal.json',probe)
    cases=[]
    for split in ('test','signal'):
        taskmap={t['task_id']:t for t in splits[split]}
        for key,rs in data[split].items():
            for r in rs:
                if not r['success'] or r['calls']>r['minimum_calls']:
                    cases.append(dict(split=split,condition=key,question=taskmap[r['task_id']]['prompt'],
                        **{k:r[k] for k in ('task_id','kind','seed','success','grounded_success','answer','expected',
                                           'calls','minimum_calls','stop_reason','actions','observations','call_events')},
                        raw_outputs=[q['text'] for q in r['requests']]))
    (root/'diagnostic-cases.jsonl').write_text(''.join(json.dumps(c)+'\n' for c in cases),encoding='utf-8',newline='\n')
    return result


def report(root,r):
    test=r['aggregate']['test'];new=test['new/protocol'];old=test['old/protocol']
    keys=list(test)
    def ratio(m):return f"{m['successes']}/{m['attempts']}（{100*m['successes']/m['attempts']:.1f}%）"
    dev=r['aggregate']['dev']
    dev_note='开发集自由 rollout：'+ '；'.join(LABELS[k]+' '+ratio(v) for k,v in dev.items())+'。开发集不用于重新选择提示或 checkpoint；teacher-forced loss 在下面独立报告。'
    interim_note=('完整对照在采样奖励诊断前已汇总到 [test-analysis-before-signal.json](test-analysis-before-signal.json)，最终审计核对其分数和配对结果与最终报告一致，并验证分析时间早于四组 signal 的开始时间。'
                  if r.get('test_analysis_before_signal_sha256') else 'signal 阶段由脚本要求全部测试条件先完成并通过验证。')
    table=['| 场景 | '+' | '.join(LABELS[k] for k in keys)+' |','|---|'+'---:|'*len(keys)]
    for kind in KINDS:
        table.append('| '+FAMILIES[kind]+' | '+' | '.join(ratio(test[k]['by_kind'][kind]) for k in keys)+' |')
    table.append('| **合计** | '+' | '.join('**'+ratio(test[k])+'**' for k in keys)+' |')
    seeds=['| 种子 | '+' | '.join(LABELS[k] for k in keys)+' |','|---|'+'---:|'*len(keys)]
    for s in (7,10007,20007):
        seeds.append('| '+str(s)+' | '+' | '.join(ratio(next(c['metrics'] for c in r['cohorts'] if c['split']=='test' and c['seed']==s and c['model']+'/'+c['prompt']==k)) for k in keys)+' |')
    signal=r['aggregate']['signal']['new/protocol'];g=r['signal_groups']
    wording=r['signal_wording_diagnostic']
    wording_table=['| signal 类别 | 模板 0（2 题 × 4） | 模板 1（2 题 × 4） |','|---|---:|---:|']
    for kind,parts in wording['by_kind_and_template'].items():
        wording_table.append('| '+kind+' | '+ratio(parts['0'])+' | '+ratio(parts['1'])+' |')
    signal_table=['| 组合 | 答对 / 16 | 真正遇错的尝试 | 遇错后答对 | 正确轨迹额外调用 |',
                  '|---|---:|---:|---:|---:|']
    for kind,m in signal['by_kind'].items():
        signal_table.append(f"| {kind} | {m['successes']}/{m['attempts']} | {m['observed_error_episodes']} | {m['correct_after_error']} | {m['correct_extra_calls']} |")
    tm=r['training'];sm=r['memory_probe']
    memory_limit_note=('本次缓存保留峰值超过名义 8GB 物理显存；Windows WDDM 可涉及共享显存/分页，因此完成运行不等于全程驻留独占 VRAM。未逐页测量迁移量，不能给出纯物理显存吞吐或据此保证其他环境也能运行。'
                       if tm['cuda_peak_reserved_mib']>8151 else '')
    oldpair=r['comparisons']['old/protocol -> new/protocol']
    aa=r.get('action_budget_audit')
    action_note=(f"事后审计 {aa['target_strings']} 个已执行 oracle 的紧凑下一动作 JSON 拼写（含 EOS），最长 {aa['max_tokens_including_eos']} tokens，均小于 64-token 上限，见 [action-budget-audit.json](action-budget-audit.json)。这不覆盖所有合法空白/键顺序拼写，但排除了标准单动作本身就无法放入预算的问题；审计没有生成模型输出或改变 checkpoint。" if aa else '')
    retry=r.get('resource_retry')
    retry_note=(f"默认缓存尝试在 {retry.get('previous_discarded_steps',0)} 个已记录更新后主动中止，仅使用缓存回收的尝试又在 {retry['discarded_optimizer_steps']} 步后中止，两者均未保存 adapter；"
        "变长样本的缓存保留量继续增长，第二次中止前超过名义整卡显存。记录保存在 training-attempt-default-cache、training-attempt-gc-cache。"
        f"进一步使用 `{retry['environment']['PYTORCH_CUDA_ALLOC_CONF']}` 的缓存尝试也未能解决问题。"
        "中止权重没有用于后续评测。缓存尝试记录见 [resource-retry.json](resource-retry.json)；"
        "[PyTorch 2.9 缓存文档](https://docs.pytorch.org/docs/2.9/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf)说明回收和限制大块拆分设置。"
        if retry else '')
    if r.get('resource_amendment'):
        am=r['resource_amendment'];cp=r['checkpointing_memory_probe']
        retry_note+=f"\n\n大块拆分尝试在 {am['additional_discarded_optimizer_steps']} 个已记录更新后也主动中止，reserved 达 {am['peak_reserved_mib']/1024:.2f} GiB；记录在 training-attempt-split-cache。未出现 CUDA OOM，但缓存保留量超过名义显存且训练严重变慢。因此在任何评测输出产生前，把资源回退触发条件从仅 OOM 扩为 OOM 或严重显存压力，并启用已有的非重入 checkpointing。原计划保存在 [plan-original.json](plan-original.json)，仅 memory_test 字段调整为当前 [plan.json](plan.json)，审计验证其他字段完全相同，见 [resource-amendment.json](resource-amendment.json)。数据、提示、种子、源码、学习率、更新次数和最终 checkpoint 规则不变。checkpointing 八步探针 allocated {cp['cuda_peak_allocated_mib']/1024:.2f} GiB、reserved {cp['cuda_peak_reserved_mib']/1024:.2f} GiB；探针再次丢弃。正式训练从 v1 adapter 与新 optimizer 重新开始，使用 `PYTORCH_CUDA_ALLOC_CONF={am['final_environment']['PYTORCH_CUDA_ALLOC_CONF']}` 与 `--checkpointing`。这属于明确记录的资源计划调整，而非完全未偏离原计划的运行。"
    decision=('当前奖励大多恒定，暂不进入 GRPO 优化。先在新的训练分布确认哪些组合能同时产生成功和失败，必要时补恢复组合监督；'
              '全错组没有成功排序信号，全对等成本组也没有相对奖励差异。优先做同事实、仅换措辞的对照，再补实体类型/字段名绑定与 unknown_field 之后的合法纠错状态；不要把所有工具错误都按暂时错误机械重试。'
              if g['nonconstant_reward_groups']<8 else
              '本次至少四分之一任务组出现非恒定奖励，具备继续做小规模 RL 显存/优化探针的信号前提；'
              '仍需区分格式失败与路径失败，避免仅从四次采样推断长期收益。')
    text=f'''# 第二轮 SFT：单动作边界、长链与报错恢复

已在 RTX 5060 Laptop 8GB 完成旧 adapter 的继续训练与独立对照。补训后简洁提示为 **{ratio(new)}**，补训前同提示为 **{ratio(old)}**。本轮最终 checkpoint 由预定一轮训练确定，没有用测试或奖励诊断选模型。RL 部分只采样和诊断，没有策略梯度更新。

## 新测试上的对照

{dev_note}

{chr(10).join(table)}

{chr(10).join(seeds)}

每个条件重复 72 个测试任务、三个种子，共 216 次尝试；不是 216 个独立任务。测试九类各八任务，所有测试事实与新训练、开发、奖励诊断及历史数据隔离。训练复用上一轮已查看的开发/测试措辞，但只用新事实；本轮开发和测试采用另外的新措辞。数字取值范围、工具 schema、提示、采样和预算保持一致。事实组内复用只发生在训练：40 对目标对照；开发、测试和 signal 分别使用独立事实。

补训前后同提示配对：由错转对 {oldpair['gained']} 次，由对转错 {oldpair['lost']} 次，均对 {oldpair['both_correct']} 次，均错 {oldpair['both_wrong']} 次。固定强提示来自上一轮开发集选择，本轮不重新按测试选择。

补训后 {r['test_task_consistency']['new/protocol']['all_three_correct']}/72 个任务在三个种子下都正确，{r['test_task_consistency']['new/protocol']['mixed']} 个任务结果随种子变化，{r['test_task_consistency']['new/protocol']['all_three_failed']} 个任务三次全错。各条件逐题重复采样的一致性见 summary.json 的 test_task_consistency；不把重复尝试当作独立样本计算置信区间。

## 学到了什么，仍在哪里失败

补训后严格 JSON 且答对 {new['strict_successes']}/{new['attempts']}，工具可溯源答对 {new['grounded_successes']}/{new['attempts']}。正确轨迹共 {new['correct_extra_calls']} 次额外查询，{new['optimal_correct']} 条达到最少调用。补训前同提示含 `Tool:` 的请求 {old['simulated_tool_requests']} 次，补训后 {new['simulated_tool_requests']} 次；含多个 `"action"` 标记的请求分别为 {old['multiple_action_marker_requests']}、{new['multiple_action_marker_requests']}，这是原始文本诊断而非额外解析/修复规则；截断请求分别为 {old['truncated_requests']}、{new['truncated_requests']}。

新模型失败停止原因：`{json.dumps(new['failed_stops'],ensure_ascii=False)}`。每类真实错误暴露、恢复动作和成本见 [summary.json](summary.json)。原始错误、动作与输出见 [diagnostic-cases.jsonl](diagnostic-cases.jsonl)。这里的动作可靠性和成功率仅对应控制合成环境，不能推广为通用 agent 或 RSI 能力。

{action_note}

## 真实训练与显存

300 条轨迹，990 个 assistant 下一动作目标，其中 80 个目标紧接实际错误观察。目标只有一个合法 JSON 和 EOS；system/user/tool/past assistant 全部 mask 为 -100，不监督伪造工具结果。新训练按长链 100、恢复 80、已知信息 60、旧目标 60 的方案分配，给定答案类合计 40，其中 20 保留旧直接预算目标。

继续训练已保存的 v1 q/v LoRA（r=8、alpha=16、dropout=0.05），主干 BF16 冻结，LoRA FP32；AdamW lr=1e-4，microbatch=1、累积 8、seed=42，一轮 **{tm['optimizer_steps']} 个 optimizer steps**。每个训练目标恰好使用一次，初始 state 哈希与 v1 完全一致，最终 adapter 已改变，主干没有梯度。监督 token 对齐、EOS、步骤覆盖与旧权重哈希已独立审计，见 [training-audit.json](training-audit.json)。

最长实际序列 {tm['max_sequence_tokens']} tokens，上限 1024，无截断训练。八步最长上下文探针成功，allocated {sm['cuda_peak_allocated_mib']/1024:.2f} GiB、reserved {sm['cuda_peak_reserved_mib']/1024:.2f} GiB；探针权重丢弃。正式训练重新从旧 adapter 开始，allocated **{tm['cuda_peak_allocated_mib']/1024:.2f} GiB**、reserved **{tm['cuda_peak_reserved_mib']/1024:.2f} GiB**，优化循环 {tm['optimization_wall_seconds']:.2f} 秒，gradient checkpointing={tm['gradient_checkpointing']}。运行期间同卡还有其他训练/评测及桌面负载，耗时不能当作独占 GPU 的吞吐基准。allocated/reserved 是本进程 PyTorch 指标，分别代表张量分配峰值与缓存保留峰值，不等于整卡可用空间或所有进程的总显存。这适用于该小模型 LoRA 和实际序列长度。

开发集 teacher-forced 平均损失 {tm['dev_loss_before']:.6g} → {tm['dev_loss_after']:.6g}。它是给定正确历史状态的下一动作损失，不等于自由 rollout 成功率，二者分开报告。

{memory_limit_note}

{retry_note}

## 全新 32 题 × 4 条 rollout：奖励信号

新模型使用简洁提示；每题四个独立采样根种子 30007、40007、50007、60007，共 128 个真实 episode。32 个任务八种预定组合各四题；schema 相同，但「长链＋故障」「已知前缀＋故障」组合没有作为训练类别使用。任务在训练前冻结并与所有划分隔离，只有完成对照评测后才开放采样。

{chr(10).join(signal_table)}

合计 **{ratio(signal)}**，工具可溯源成功 {signal['grounded_successes']}，正确轨迹额外查询 {signal['correct_extra_calls']}。

### signal 中的新失败与措辞关联

{chr(10).join(wording_table)}

长链六类的模板 0 使用 “ownership chain”，模板 1 使用 “employee responsible for”。prefix 两类的两种措辞另行定义在 sft2_data.py。每格只有两个独立事实任务，各重复四次；不同模板没有复用同一事实。signal 同时改变了措辞和部分故障组合，不能把成功率变化全部归因为故障组合，以上是事后描述性分层，不是隔离的措辞因果实验。

失败轨迹中 {wording['first_project_employee_field_failures']} 次第一条动作把 project 的字段写成非法的 `employee`；project 合法字段是 `owner` 和 `budget`。实际 `unknown_field` 错误 {signal['unknown_field_errors']} 次，`unknown_entity` 错误 {signal['unknown_entity_errors']} 次；工具的 unknown_field 返回 allowed_fields，但部分轨迹仍反复查无效字段。共有 {wording['failed_before_recoverable_error']} 次失败尚未遇到预设可恢复错误；真正遇到可恢复错误的 {signal['observed_error_episodes']} 次里，{signal['correct_after_error']} 次答对。标准测试的 216/216 不能推广为这个新分布上的成功率。

当前训练紧接错误的监督类型为 `{json.dumps(r['training_audit']['supervised_error_types'])}`，未覆盖学生自行产生的 unknown_field/unknown_entity 状态。动作边界改善与 schema 语义泛化、非重试错误纠正是不同问题：这里的重复合法 JSON 仍可反复发出非法查询。下一轮可用同事实措辞对照验证关联，再添加真实执行的错误前缀＋合法纠错目标；保持现有测试与 signal 结果封存，未来训练和最终测试另生成。

32 个任务组分类：`{json.dumps(g['counts'],ensure_ascii=False)}`。非恒定奖励 **{g['nonconstant_reward_groups']}/32 组**，逐题奖励、方差、调用数和停止原因见 [reward-signal.json](reward-signal.json)。

若同题四个奖励都是常数，`R_i − mean(R)` 全为零；在跳过零方差组或加 epsilon 的安全实现中，该组没有组内相对奖励优势。不同调用数的失败仍都得到零，这是当前正确性门控奖励的预定行为，不会把更早放弃误奖励为效率提升。恒定组结论仅针对当前四次采样；不能推断其他采样配置下永远没有成功概率。

诊断保持旧奖励 `R = exact_correct × (1 − 0.03 × calls)`，错误调用也计费，必要重试没有额外的重复字符串惩罚。按同一任务比较时其最少调用成本是常数；全错组为零、全对等成本组为常数，不能当作有成本优化信号。exact 与工具可溯源正确分别复核，避免把私有值猜中误认成查询能力。未生成用于 RL 更新的旧策略 log-prob，也未计算策略梯度。

{decision}

[GRPO 原论文 §4.1.2](https://arxiv.org/html/2402.03300v3#S4.SS1.SSS2) 用同题组内相对奖励构造优势；恒定奖励需要处理零方差。本次 G=4 的方差只是有限样本诊断，不证明下一轮 RL 一定增益。未来若开始 RL，训练任务须另生成，保留 assistant-only 策略损失、真实 old-policy log-prob 与独立最终测试，并实测 rollout、backward、optimizer 的联合显存。

## 证据与复现

本轮独立复核 **{r['total_verified_episodes']} 个评测/诊断 episode、{r['total_verified_model_requests']} 个模型生成请求**。输入 token 重建、输出 token 解码、工具重放、答案及成本重判、恢复计数和 EOS 截断标记一致；新旧 adapter、数据、提示和执行源文件与冻结计划相符。复核是记录一致性验证，不是重新生成输出。

{interim_note}

固定 Qwen3-0.6B revision，温度 0.7、top-p 0.8、top-k 20，thinking off，每回合 64 新 tokens，六次查询、七个输出回合，batch 1。推理 allocated 峰值 {r['inference_peak_allocated_mib']/1024:.2f} GiB，reserved {r['inference_peak_reserved_mib']/1024:.2f} GiB。完整配置在 [plan.json](plan.json)，运行步骤在 [reproduce.md](reproduce.md)，新 adapter 在 training/adapter；旧 adapter 字节保持不变。

![第二轮 SFT 与奖励组诊断](sft2-results.png)
'''
    (root/'analysis.zh-CN.md').write_text(text,encoding='utf-8',newline='\n')


def plot(root,r):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    variants=r['aggregate']['test'];fig,ax=plt.subplots(2,1,figsize=(13,8),layout='constrained')
    names=['Base / fixed examples','Old SFT / protocol','Old SFT / examples','New SFT / protocol']
    colors=['#94a3b8','#f59e0b','#38bdf8','#14b8a6'];width=.2
    for j,(key,m) in enumerate(variants.items()):
        bars=ax[0].bar([i+(j-1.5)*width for i in range(9)],
             [100*m['by_kind'][k]['successes']/m['by_kind'][k]['attempts'] for k in KINDS],width*.94,label=names[j],color=colors[j])
        ax[0].bar_label(bars,fmt='%.0f',fontsize=7,padding=2)
    ax[0].set_xticks(range(9),['2-hop','3-hop','4-hop','Retry','Redirect','Owner known','Prefix known','Answer known','1-hop budget'],rotation=12)
    ax[0].set_ylim(0,113);ax[0].set_ylabel('Answer success (%)')
    ax[0].set_title('72 fresh tasks per policy, three seeds; one fixed continuation checkpoint')
    ax[0].legend(loc='lower center',bbox_to_anchor=(.5,1.1),ncol=4,fontsize=8)
    classes=['all_failed','mixed_correctness','all_correct_different_cost','all_correct_same_cost']
    counts=[r['signal_groups']['counts'].get(k,0) for k in classes]
    bars=ax[1].bar(range(4),counts,color=['#ef4444','#14b8a6','#38bdf8','#94a3b8'],width=.55)
    ax[1].bar_label(bars,fmt='%d',padding=3)
    ax[1].set_xticks(range(4),['All four fail','Mixed correctness','All correct, different cost','All correct, same cost'])
    ax[1].set_ylim(0,max(counts)+4);ax[1].set_ylabel('Task groups (32 total)')
    ax[1].set_title('Fresh RL signal probe: four real rollouts per task, no RL updates')
    fig.savefig(root/'sft2-results.png',dpi=160);plt.close(fig)


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path('results/sft-v2'))
    parser.add_argument('--model-info',type=Path,required=True);args=parser.parse_args()
    r=aggregate(args.root,args.model_info);report(args.root,r);plot(args.root,r)
    print(json.dumps(dict(episodes=r['total_verified_episodes'],requests=r['total_verified_model_requests'],
        test={k:v['successes'] for k,v in r['aggregate']['test'].items()},signal=r['signal_groups']['counts']),indent=2))


if __name__=='__main__':main()

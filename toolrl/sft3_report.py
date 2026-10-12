"""Replay-audited paired wording effects, executed corrections and RL signals."""
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
import json
import math
from pathlib import Path

from .sft3_experiment import read,rows,write,sha256,validate,checkpoint,completed,CONDITIONS,run_dir,paired_gate,signal_gate
from .sft2_report import metrics,signal_groups
from .sft3_data import audit,request_examples,encode_example
from .stress_environment import StressEnvironment

def now():return datetime.now(timezone.utc).isoformat()

def aggregate(p,split,model,prompt):
    rs=[]
    for seed in p['evaluation_seeds'][split]:rs+=rows(completed(p,split,model,prompt,seed)/'episodes.jsonl')
    return rs

def paired(p,model):
    rs=aggregate(p,'pair',model,'protocol');by=defaultdict(dict)
    tasks={t['task_id']:t for t in rows(Path(p['data_root'])/'pair.jsonl')}
    for r in rs:by[(r['seed'],tasks[r['task_id']]['pair_id'])][r['template_id']]=r
    assert len(by)==64 and all(set(v)=={0,1} for v in by.values())
    # The same root seed is shared within each fact/wording pair.
    units=[]
    for (seed,pair_id),v in by.items():
        def wrongfield(r):return bool(r['actions']) and r['actions'][0].get('field')=='employee' and r['actions'][0].get('entity','').startswith('project_')
        units.append(dict(seed=seed,pair_id=pair_id,kind=v[0]['kind'],control_success=v[0]['success'],variant_success=v[1]['success'],
                          control_first_employee_field=wrongfield(v[0]),variant_first_employee_field=wrongfield(v[1])))
    return dict(attempts=len(rs),independent_facts=32,repeated_paired_units=len(units),
                by_wording={str(tid):metrics([r for r in rs if r['template_id']==tid]) for tid in (0,1)},
                control_only_correct=sum(x['control_success'] and not x['variant_success'] for x in units),
                variant_only_correct=sum(not x['control_success'] and x['variant_success'] for x in units),
                both_correct=sum(x['control_success'] and x['variant_success'] for x in units),
                both_wrong=sum(not x['control_success'] and not x['variant_success'] for x in units),
                first_project_employee_field={str(tid):sum(bool(r['actions']) and r['actions'][0].get('field')=='employee' and r['actions'][0].get('entity','').startswith('project_') for r in rs if r['template_id']==tid) for tid in (0,1)},
                paired_units=units)

def save_paired(p,model,filename):
    result=paired(p,model)|dict(analyzed_at=now(),plan_sha256=sha256(Path(p['root'])/'plan.json'),
                              episodes_sha256={run_dir(p,'pair',model,'protocol',s).name:sha256(run_dir(p,'pair',model,'protocol',s)/'episodes.jsonl') for s in p['evaluation_seeds']['pair']})
    write(Path(p['root'])/filename,result)
    print(json.dumps(dict(event='paired_analysis',model=model,control_success=result['by_wording']['0']['successes'],
        variant_success=result['by_wording']['1']['successes'],control_only_correct=result['control_only_correct'],first_employee=result['first_project_employee_field'])),flush=True)
    return result

def extra_metrics(rs):
    m=metrics(rs)
    # The injected bad call is already executed and charged in total cost.
    # Free-rollout metrics never subtract model-generated mistakes.
    m['additional_calls_beyond_oracle_and_injected_error']=sum(r['calls']-r['minimum_calls']-sum('error' in o for o in r['bootstrap_observations']) for r in rs if r['success'])
    m['by_injected_error']={typ:metrics([r for r in rs if any(o.get('error')==typ for o in r['bootstrap_observations'])]) for typ in ('unknown_field','unknown_entity')}
    return m

def test_metrics(p):
    return {split:{'/'.join(c):extra_metrics(aggregate(p,split,*c)) for c in CONDITIONS} for split in ('test','recovery')}

def interim(p):
    assert not list((Path(p['root'])/'eval').glob('signal-*/episodes.jsonl'))
    result=dict(analyzed_at=now(),plan_sha256=sha256(Path(p['root'])/'plan.json'),aggregate=test_metrics(p),
        episodes_sha256={run_dir(p,split,*c,s).name:sha256(completed(p,split,*c,s)/'episodes.jsonl') for split in ('test','recovery') for s in p['evaluation_seeds'][split] for c in CONDITIONS},
        paired_after=save_paired(p,'new','paired-after-training.json'))
    write(Path(p['root'])/'test-analysis-before-signal.json',result)
    print(json.dumps(dict(event='interim',successes={s:{c:m['successes'] for c,m in cs.items()} for s,cs in result['aggregate'].items()})),flush=True)

def training_audit(p,info_path):
    from transformers import AutoTokenizer
    from safetensors.torch import load_file
    from .sft2_train import tensor_hash
    tok=AutoTokenizer.from_pretrained(read(info_path)['snapshot'],local_files_only=True)
    root=Path(p['root']);m=checkpoint(p,root/'plan.json');tasks=rows(Path(p['data_root'])/'train.jsonl')
    prompt=Path(p['prompt_files']['protocol']).read_text(encoding='utf-8')
    expected=[encode_example(x,tok,p['training']['max_context']) for t in tasks for x in request_examples(t,prompt)]
    saved=rows(root/'training/supervision.jsonl');assert expected==saved
    assert sha256(root/'training/supervision.jsonl')==m['supervision_sha256']
    error_counts=Counter()
    for task in tasks:
        env=StressEnvironment(task)
        for call in task['bootstrap_calls']:o=env.lookup(**call)
        if task['bootstrap_calls']:
            assert o['error']==task['correction_type'];error_counts[o['error']]+=1
            assert task['remaining_oracle_calls']==task['oracle_calls'][task['correction_stage']:]
        for a in task.get('remaining_oracle_calls',task['oracle_calls']):o=env.lookup(**a)
        assert task['kind']=='given_answer' or o.get('value')==task['expected']
    assert error_counts==dict(unknown_field=80,unknown_entity=80)
    for r in saved:
        assert r['labels'][:r['prefix_tokens']]==[-100]*r['prefix_tokens']
        assert r['labels'][r['prefix_tokens']:]==r['input_ids'][r['prefix_tokens']:]
        assert r['input_ids'][-1]==tok.eos_token_id
        assert tok.decode(r['input_ids'][r['prefix_tokens']:-1],skip_special_tokens=True)==r['target']
    steps=rows(root/'training/steps.jsonl');ids=[i for s in steps for i in s['request_ids']];targets=[r['task_id']+':'+str(r['turn']) for r in expected]
    assert sorted(ids)==sorted(targets) and len(ids)==len(set(ids))==p['decision_targets']
    assert len(steps)==math.ceil(len(ids)/8) and [s['step'] for s in steps]==list(range(1,len(steps)+1))
    assert all(math.isfinite(s['loss']) and math.isfinite(s['gradient_norm_before_clip']) and s['gradient_norm_before_clip']>0 for s in steps)
    assert m['initial_adapter_sha256']==p['old_adapter_state_sha256'] and m['frozen_base_gradients_absent'] and m['gradient_checkpointing']
    assert tensor_hash(load_file(str(root/'training/adapter/adapter_model.safetensors')))==m['final_adapter_state_sha256']!=m['initial_adapter_sha256']
    paired_gate(p);assert read(root/'paired-before-training.json')['analyzed_at']<m['started_at']
    out=dict(targets=len(ids),optimizer_steps=len(steps),exact_tokens_masks_eos_match=True,executed_error_prefix_counts=dict(error_counts),
             bad_queries_are_masked_inputs_not_targets=True,each_target_used_once=True,old_initial_and_new_saved_tensor_hashes_match=True,
             frozen_base_has_no_gradients=True,training_manifest_sha256=sha256(root/'training/manifest.json'),
             steps_sha256=sha256(root/'training/steps.jsonl'),supervision_sha256=sha256(root/'training/supervision.jsonl'))
    write(root/'training-audit.json',out);return out

def verify_all(p,info_path):
    from transformers import AutoTokenizer
    from .sft3_verify import verify_rows
    tok=AutoTokenizer.from_pretrained(read(info_path)['snapshot'],local_files_only=True)
    eos=read(Path(read(info_path)['snapshot'])/'generation_config.json')['eos_token_id'];eos=[eos] if isinstance(eos,int) else eos
    cohort={};all_rows=[];count=0
    for d in sorted((Path(p['root'])/'eval').iterdir()):
        m=read(d/'manifest.json');c=m['config'];v=read(d/'verification.json');rs=rows(d/'episodes.jsonl')
        assert m['status']=='complete' and m['trainable_parameters']==0 and not m['new_training_performed']
        assert c['source_sha256']==p['source_sha256'] and c['plan_sha256']==sha256(Path(p['root'])/'plan.json')
        assert c['dataset_sha256']==p['dataset_sha256'][c['split']] and c['system_prompt_sha256']==p['prompt_sha256'][c['prompt']]
        hashes=p['old_adapter_files_sha256'] if c['model']=='old' else checkpoint(p,Path(p['root'])/'plan.json')['adapter_files_sha256']
        assert c['adapter_files_sha256']==hashes and v['episodes_sha256']==sha256(d/'episodes.jsonl')
        assert all(c[k]==value for k,value in p['eval_fixed'].items())
        tasks={t['task_id']:t for t in rows(Path(p['data_root'])/(c['split']+'.jsonl'))}
        n=verify_rows(rs,tasks,c,tok,Path(p['prompt_files'][c['prompt']]).read_text(encoding='utf-8'))
        assert n==v['requests_verified'] and len(rs)==v['tasks_verified']
        for r in rs:
            for q in r['requests']:
                if not q.get('context_limit'):assert q['truncated']==(len(q['output_ids'])>=c['max_new_tokens'] and q['output_ids'][-1] not in eos)
        cohort[d.name]=dict(episodes=len(rs),requests=n,episodes_sha256=v['episodes_sha256']);all_rows += [dict(cohort=d.name,**r) for r in rs];count+=n
    return cohort,all_rows,count

def report(p,info_path):
    root=Path(p['root']);paired_gate(p);signal_gate(p);ta=training_audit(p,info_path)
    auditdata=read(Path(p['data_root'])/'audit.json');old_e=set();old_v=set()
    from .stress_data import prior_sets
    old_e,old_v=prior_sets([Path(x) for x in p['historical_files_sha256']])
    for path in p['historical_files_sha256']:
        for t in rows(path):old_e.update(t.get('aliases',{}))
    audit({s:rows(Path(p['data_root'])/(s+'.jsonl')) for s in p['dataset_sha256']},old_e,old_v)
    cohorts,all_rs,n=verify_all(p,info_path);before=paired(p,'old');after=paired(p,'new')
    for name,result in [('paired-before-training.json',before),('paired-after-training.json',after)]:
        saved=read(root/name);assert all(saved[k]==v for k,v in result.items())
    tm=test_metrics(p);a=read(root/'test-analysis-before-signal.json');assert a['aggregate']==tm
    for seed in p['evaluation_seeds']['signal']:
        d=completed(p,'signal','new','protocol',seed);assert a['analyzed_at']<read(d/'manifest.json')['started_at']
    signal=aggregate(p,'signal','new','protocol');groups=signal_groups(signal);m=checkpoint(p,root/'plan.json')
    summary=dict(generated_at=now(),plan_sha256=sha256(root/'plan.json'),aggregate=tm,
        paired_before=before,paired_after=after,signal=metrics(signal),signal_groups=groups,training=m,training_audit=ta,
        total_verified_episodes=len(all_rs),total_verified_model_requests=n,cohorts=cohorts,rl_optimization_performed=False,
        signal_by_template={str(i):metrics([r for r in signal if r['template_id']==i]) for i in (0,1)},
        interpretation='Pair changes question wording only. Correction SFT also changes wording coverage and executed error-state coverage; their contributions are not isolated. Synthetic fixed schema only.')
    assert len(all_rs)==len(before['paired_units'])*2*2+3*26+3*2*(72+32)+128==1086
    write(root/'summary.json',summary);write(root/'reward-signal.json',groups)
    cases=[r for r in all_rs if not r['success'] or (r['calls']-r['minimum_calls']-sum('error' in o for o in r['bootstrap_observations']))>0]
    (root/'diagnostic-cases.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in cases),encoding='utf-8',newline='\n')
    table=[]
    for c in p['eval_conditions']:
        x=tm['test'][c];y=tm['recovery'][c]
        table.append(f"| {c} | {x['successes']}/{x['attempts']}（{100*x['successes']/x['attempts']:.1f}%） | {y['successes']}/{y['attempts']}（{100*y['successes']/y['attempts']:.1f}%） |")
    b0,b1=before['by_wording']['0'],before['by_wording']['1'];a0,a1=after['by_wording']['0'],after['by_wording']['1']
    e0,e1=before['first_project_employee_field']['0'],before['first_project_employee_field']['1']
    conclusion=('当前 32 组均为恒定奖励，暂不启动 GRPO。下一步应另建能产生成功/失败或正确成本差异的训练任务，并在新任务上验证采样信号；不可把全错组机械变成成本优化样本。' if groups['nonconstant_reward_groups']==0 else
                f"本轮有 {groups['nonconstant_reward_groups']}/32 组出现非恒定奖励。可在另生成的训练任务上继续验证有效组占比与奖励正确性，再实测 RL 联合显存；这次仅诊断，尚不能说明 GRPO 会提升成功率。")
    text=f'''# 第三轮：措辞配对、非法查询纠错 SFT 与全新奖励诊断

已在 RTX 5060 Laptop 8GB 完成。本轮先分析同事实措辞配对，再继续训练 v2 adapter，最终用未参与训练的新事实、新完整措辞评测，并在对照分析之后开放新的 32 题 × 4 条 rollout。

## 同事实、只换问题措辞

32 个独立事实任务，八类各四题。每个任务复制两份，数据库、故障、给定信息、目标、预算、system 提示、采样配置、sampling_index 完全相同，仅 user 问题措辞不同；task_id 和模板标签仅用于记录。两种措辞分别来自上一轮 signal 的模板 0、1。每对共享根种子与逐回合种子，重复两个根种子，得到 64 个配对单元、128 条真实 rollout。32 个事实才是独立任务，重复采样相关。第二个根种子反转配对内的执行顺序。

| 模型＋简洁提示 | ownership chain 措辞/前缀控制 | employee responsible 措辞/前缀控制 | 首动作错用 project.employee（模板0/1） |
|---|---:|---:|---:|
| 补训前 v2 | {b0['successes']}/64 | {b1['successes']}/64 | {e0}/{e1} |
| 纠错补训后 v3 | {a0['successes']}/64 | {a1['successes']}/64 | {after['first_project_employee_field']['0']}/{after['first_project_employee_field']['1']} |

补训前配对：仅模板 0 正确 {before['control_only_correct']} 组、仅模板 1 正确 {before['variant_only_correct']} 组、均对 {before['both_correct']} 组、均错 {before['both_wrong']} 组。prefix 两类是控制，不使用这两个长链词组；八类逐条结果见 summary.json。该设计消除了上一轮不同模板使用不同事实的混杂，但结论只覆盖这两种措辞、模型和有限采样；不是所有语言表述的泛化结论。

配对分析在任何训练前保存到 [paired-before-training.json](paired-before-training.json)，审计核对时间、哈希和配对输入一致。训练包含这两个已诊断措辞家族及其他训练措辞，因此训练后的同配对复测是诊断性回归检查；另有下面新事实、新完整措辞的最终测试。

## 纠错训练与独立对照

300 条新训练轨迹：160 条有实际执行的非法查询前缀（unknown_field 80、unknown_entity 80），另有 140 条正常目标、三四步链、临时错误/ID 跳转、已知信息和旧预算任务。非法字段覆盖四层实体，非法 ID 包含实体类型词及拼写错误；正确 ID 必须从问题或之前的真实观察复制。这里的错误前缀是脚本构造的反事实动作，经过真实工具执行，不冒充模型自然产生的失败。

错误动作与错误观察只作为输入，全部 mask=-100；只监督合法纠错及之后推进动作、最终答案和 EOS。训练共 {p['decision_targets']} 个下一动作目标，一轮 {m['optimizer_steps']} 个更新。初始 adapter 精确等于 v2，冻结 BF16 主干、FP32 q/v LoRA r=8、alpha=16、dropout=0.05，AdamW lr=1e-4，累积 8、microbatch 1、seed 42。最终一轮 checkpoint 固定，没有按测试或 signal 选模型。

| 条件 | 新措辞自由起始测试（72题×2种子） | 注入错误恢复测试（32题×2种子） |
|---|---:|---:|
{chr(10).join(table)}

错误恢复测试从实际执行的合法前缀＋一次非法查询开始，模型只生成之后的动作；注入动作不是模型自行犯错，不能把它的恢复成功率与自由起始测试直接混合。注入错误照常计入六次调用、七回合预算。正确成本指标同时给出总成本和扣除固定注入错误之后的多余查询数；不扣除模型自行犯的错。unknown_field 返回 allowed_fields，unknown_entity 没有 canonical_entity 提示。字段合法仍不保证实体/目标正确。

train/dev/test/recovery/pair/signal 的私有实体、alias、数值与所有历史事实隔离；只有 pair 的两份措辞共享事实。新测试九类各八题，新恢复测试两类错误×四个链位置×四题。所有划分、提示、源码与配置在生成第一条模型结果前冻结，见 [plan.json](plan.json) 与 data/sft-v3/audit.json。

本轮同时增加了措辞覆盖与错误状态覆盖，不能单独归因于其中一种训练成分。只有一次 SFT 训练种子，控制合成英语工具 schema 保持固定，不代表通用 agent、真实外部工具或 RSI 能力。

## 5060 资源与训练证据

最长监督序列 {m['max_sequence_tokens']} tokens，上限 1536，无静默截断。全程启用非重入 gradient checkpointing，每次 optimizer 更新后 empty_cache；八步最长上下文探针丢弃，再从 v2 与新 optimizer 开始正式训练。allocated 峰值 {m['cuda_peak_allocated_mib']/1024:.2f} GiB，reserved 峰值 {m['cuda_peak_reserved_mib']/1024:.2f} GiB，优化循环 {m['optimization_wall_seconds']/60:.2f} 分钟。PyTorch 进程指标不是整卡显存；WDDM 和其他 GPU 负载使完成运行不能自动证明独占物理显存驻留或独占吞吐。

teacher-forced dev loss {m['dev_loss_before']:.6g} → {m['dev_loss_after']:.6g}，与自由 rollout 成功率分开解释。监督 token/mask/EOS、错误工具前缀、每个目标恰好用一次、有限非零梯度、旧初始与新保存 LoRA tensor 哈希、主干无梯度通过独立审计，见 [training-audit.json](training-audit.json)。

## 全新 32题×4条 rollout 的奖励信号

训练前冻结新 signal，八类各四题，既更换事实又使用未训练的完整措辞。模型 v3＋原简洁提示，根种子 30017/40017/50017/60017，温度0.7、top-p0.8、top-k20、每回合64 tokens、六次查询、七回合。四个种子真实生成，不复制轨迹。

正确 {summary['signal']['successes']}/128（{100*summary['signal']['successes']/128:.1f}%），工具可溯源正确 {summary['signal']['grounded_successes']}/128。组分类 {json.dumps(groups['counts'],ensure_ascii=False)}，非恒定奖励 **{groups['nonconstant_reward_groups']}/32**。完整逐题四个奖励、成本、方差见 [reward-signal.json](reward-signal.json)。

奖励沿用 R=exact_correct×(1−0.03×calls)，错误调用也计费。全错组全为零；全对等成本组同为常数；这两种情况没有组内相对奖励优势。若有混合正确性或正确成本差异，它们只是可用信号的候选，还需检查可溯源答案与奖励漏洞。G=4 是有限样本，不推断成功概率永远为零或保证 RL 能提升。

{conclusion}

**没有 RL optimizer 更新，没有编造 old-policy log-prob。** 配对测试、错误恢复评测与奖励诊断原始失败/多余调用见 [diagnostic-cases.jsonl](diagnostic-cases.jsonl)。旧 signal 的 80/128 与这次使用不同事实、措辞的新 signal 不构成同任务因果提升。

## 复核

共 {len(all_rs)} 个真实评测/诊断 episode、{n} 个实际生成请求。所有工具重放、答案和成本重判、输入 token 重建、输出 token 解码、EOS 截断标记、训练/提示/adapter/source 哈希通过复核。完整对照已在 signal 采样前保存到 [test-analysis-before-signal.json](test-analysis-before-signal.json)，报告验证其数字和开始时间。历史源码与权重保持不变。

![措辞配对、纠错测试与奖励组](sft3-results.png)
'''
    (root/'analysis.zh-CN.md').write_text(text,encoding='utf-8',newline='\n')
    plot(root,summary)
    print(json.dumps(dict(event='report',episodes=len(all_rs),requests=n,test={c:x['successes'] for c,x in tm['test'].items()},
        recovery={c:x['successes'] for c,x in tm['recovery'].items()},signal_success=summary['signal']['successes'],reward_groups=groups['counts'])),flush=True)

def plot(root,s):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,3,figsize=(15,5),layout='constrained')
    colors=['#91a4bb','#29b5e6','#16b39f']
    for i,(label,key) in enumerate([('v2','paired_before'),('v3','paired_after')]):
        ax[0].bar([i*3,i*3+1],[100*s[key]['by_wording'][str(t)]['successes']/64 for t in (0,1)],color=['#91a4bb','#16b39f'])
    ax[0].set_xticks([0,1,3,4],['v2 / A','v2 / B','v3 / A','v3 / B']);ax[0].set_title('Same facts, only question wording changes')
    labels=['v2 / short','v2 / examples','v3 / short']
    for i,c in enumerate(s['aggregate']['test']):
        ax[1].bar([i*3,i*3+1],[100*s['aggregate'][split][c]['successes']/s['aggregate'][split][c]['attempts'] for split in ('test','recovery')],color=['#29b5e6','#16b39f'],label=labels[i])
    ax[1].set_xticks([.5,3.5,6.5],labels,rotation=15);ax[1].set_title('Blue: free start; green: injected-error recovery')
    for a in ax[:2]:a.set_ylim(0,112);a.set_ylabel('Success (%)');a.bar_label(a.containers[0],fmt='%.0f')
    for cont in ax[0].containers[1:]+ax[1].containers[1:]:
        (ax[0] if cont in ax[0].containers else ax[1]).bar_label(cont,fmt='%.0f')
    keys=['all_failed','mixed_correctness','all_correct_different_cost','all_correct_same_cost'];values=[s['signal_groups']['counts'].get(k,0) for k in keys]
    bars=ax[2].bar(range(4),values,color=['#ed4b4b','#16b39f','#29b5e6','#91a4bb']);ax[2].bar_label(bars)
    ax[2].set_xticks(range(4),['All fail','Mixed','Correct,\ncost varies','Correct,\nsame cost'],rotation=10);ax[2].set_ylim(0,35);ax[2].set_ylabel('Tasks (32 total)');ax[2].set_title('Four fresh real rollouts per task')
    fig.savefig(root/'sft3-results.png',dpi=160);plt.close(fig)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['audit','report']);parser.add_argument('--root',type=Path,default=Path('results/sft-v3'));parser.add_argument('--model-info',type=Path,required=True);a=parser.parse_args();p=validate(a.root/'plan.json')
    if a.stage=='audit':print(json.dumps(training_audit(p,a.model_info)),flush=True)
    else:report(p,a.model_info)

if __name__=='__main__':main()

"""Independent replay, actual student-prefix audit, and weighted reward diagnosis."""
import argparse
from collections import Counter, defaultdict
import math
from pathlib import Path
import statistics
import json

from .sft4_experiment import read, rows, write, sha256, now, validate, checkpoint, completed, CONDITIONS, training_gate, baseline_gate, signal_gate
from .sft4_data import request_examples, encode_example, audit, private_sets
from .sft4_environment import completion


def metrics(rs,tasks):
    good=[r for r in rs if r['grounded_success']]
    def conditioned(r):
        return completion(tasks[r['task_id']])[1].total_cost
    return dict(attempts=len(rs),exact_correct=sum(r['success'] for r in rs),grounded_correct=len(good),
        strict_grounded_correct=sum(r['grounded_success'] and r['strict_success'] for r in rs),
        requests=sum(len(r['requests']) for r in rs),optimal_weighted_correct=sum(r['total_cost']==conditioned(r) for r in good),
        successful_total_cost=sum(r['total_cost'] for r in good),successful_cost_regret=sum(r['total_cost']-conditioned(r) for r in good),
        correct_mean_cost=statistics.mean(r['total_cost'] for r in good) if good else None,
        mean_reward=statistics.mean(r['reward_cost_gated'] for r in rs) if rs else None,
        actual_unknown_entity=sum(o.get('error')=='unknown_entity' for r in rs for o in r['observations']),
        actual_unknown_field=sum(o.get('error')=='unknown_field' for r in rs for o in r['observations']),
        failed_stops=dict(Counter(r['stop_reason'] for r in rs if not r['grounded_success'])),
        ungrounded_exact_answers=sum(r['success'] and not r['grounded_success'] for r in rs),
        max_input_tokens=max((len(q['input_ids']) for r in rs for q in r['requests']),default=0))


def aggregate(p,split,c):
    rs=[]
    for seed in p['evaluation_seeds'][split]:rs+=rows(completed(p,split,*c,seed)/'episodes.jsonl')
    return rs


def controls(p,conditions=CONDITIONS):
    out={}
    for split in ('dev','test','recovery'):
        tasks={t['task_id']:t for t in rows(Path(p['data_root'])/(split+'.jsonl'))}
        out[split]={'/'.join(c):metrics(aggregate(p,split,c),tasks) for c in conditions}
    return out


def hashes(p,conditions=CONDITIONS):
    return {d.name:sha256(d/'episodes.jsonl') for split in ('dev','test','recovery') for seed in p['evaluation_seeds'][split]
            for c in conditions for d in [completed(p,split,*c,seed)]}


def save_baseline(p):
    root=Path(p['root']);assert not (root/'training/manifest.json').exists()
    out=dict(analyzed_at=now(),plan_sha256=sha256(root/'plan.json'),aggregate=controls(p,CONDITIONS[:2]),episodes_sha256=hashes(p,CONDITIONS[:2]))
    write(root/'baseline-before-training.json',out)
    print(json.dumps(dict(event='baseline_analysis',test={c:m['grounded_correct'] for c,m in out['aggregate']['test'].items()},
        recovery={c:m['grounded_correct'] for c,m in out['aggregate']['recovery'].items()})),flush=True)


def save_interim(p):
    root=Path(p['root']);assert not list((root/'eval').glob('signal-*/episodes.jsonl'))
    out=dict(analyzed_at=now(),plan_sha256=sha256(root/'plan.json'),aggregate=controls(p),episodes_sha256=hashes(p))
    write(root/'test-analysis-before-signal.json',out)
    print(json.dumps(dict(event='interim',test={c:m['grounded_correct'] for c,m in out['aggregate']['test'].items()},
        recovery={c:m['grounded_correct'] for c,m in out['aggregate']['recovery'].items()})),flush=True)


def training_audit(p,info_path):
    from transformers import AutoTokenizer
    from safetensors.torch import load_file
    from .sft4_train import tensor_hash
    root=Path(p['root']);commit=training_gate(p);m=checkpoint(p,root/'plan.json')
    tokenizer=AutoTokenizer.from_pretrained(read(info_path)['snapshot'],local_files_only=True)
    tasks=rows(Path(p['data_root'])/'derived_train.jsonl')
    prompt=Path(p['prompt_files']['protocol']).read_text(encoding='utf-8')
    expected=[encode_example(x,tokenizer,p['training']['max_context']) for t in tasks for x in request_examples(t,prompt)]
    actual=rows(root/'training/supervision.jsonl');assert actual==expected
    assert sha256(root/'training/supervision.jsonl')==m['supervision_sha256']
    for r in actual:
        assert r['labels'][:r['prefix_tokens']]==[-100]*r['prefix_tokens']
        assert r['labels'][r['prefix_tokens']:]==r['input_ids'][r['prefix_tokens']:]
        assert tokenizer.decode(r['input_ids'][r['prefix_tokens']:-1],skip_special_tokens=True)==r['target']
        assert r['input_ids'][-1]==tokenizer.eos_token_id
    ids=[r['task_id']+':'+str(r['turn']) for r in actual]
    steps=rows(root/'training/steps.jsonl');used=[v for r in steps for v in r['request_ids']]
    assert sorted(used)==sorted(ids) and len(used)==len(set(used))==commit['decision_targets']
    assert len(steps)==math.ceil(len(ids)/8) and [r['step'] for r in steps]==list(range(1,len(steps)+1))
    assert all(math.isfinite(r['loss']) and math.isfinite(r['gradient_norm_before_clip']) and r['gradient_norm_before_clip']>0 for r in steps)
    assert m['initial_adapter_sha256']==p['old_adapter_state_sha256'] and m['frozen_base_gradients_absent'] and m['gradient_checkpointing']
    assert tensor_hash(load_file(str(root/'training/adapter/adapter_model.safetensors')))==m['final_adapter_state_sha256']!=m['initial_adapter_sha256']
    assert read(root/'baseline-before-training.json')['analyzed_at']<commit['frozen_at']<m['started_at']
    out=dict(targets=len(actual),optimizer_steps=len(steps),training_tasks=len(tasks),natural_copy_tasks=commit['natural_training_tasks'],
        exact_tokens_masks_eos_match=True,each_target_used_once=True,actual_student_prefix_reconstruction_matches=True,
        source_episode_hashes_match=True,bad_queries_masked_not_supervised=True,public_state_planner_only=True,
        initial_and_saved_tensor_hashes_match=True,frozen_base_has_no_gradients=True,
        training_set_manifest_sha256=sha256(root/'training-set-manifest.json'),training_manifest_sha256=sha256(root/'training/manifest.json'),
        steps_sha256=sha256(root/'training/steps.jsonl'),supervision_sha256=sha256(root/'training/supervision.jsonl'))
    write(root/'training-audit.json',out);return out


def verify_all(p,info_path):
    from transformers import AutoTokenizer
    from .sft4_verify import verify_rows
    info=read(info_path); tok=AutoTokenizer.from_pretrained(info['snapshot'],local_files_only=True)
    eos=read(Path(info['snapshot'])/'generation_config.json')['eos_token_id'];eos=[eos] if isinstance(eos,int) else eos
    cohorts={};allrows=[];requests=0
    for d in sorted((Path(p['root'])/'eval').iterdir()):
        m=read(d/'manifest.json');c=m['config'];v=read(d/'verification.json');rs=rows(d/'episodes.jsonl')
        assert m['status']=='complete' and m['trainable_parameters']==0 and not m['new_training_performed']
        assert c['source_sha256']==p['source_sha256'] and c['plan_sha256']==sha256(Path(p['root'])/'plan.json')
        assert c['dataset_sha256']==p['dataset_sha256'][c['split']] and c['system_prompt_sha256']==p['prompt_sha256'][c['prompt']]
        h=p['old_adapter_files_sha256'] if c['model']=='old' else checkpoint(p,Path(p['root'])/'plan.json')['adapter_files_sha256']
        assert c['adapter_files_sha256']==h and v['episodes_sha256']==sha256(d/'episodes.jsonl')
        assert all(c[k]==x for k,x in p['eval_fixed'].items())
        tasks={t['task_id']:t for t in rows(Path(p['data_root'])/(c['split']+'.jsonl'))}
        n=verify_rows(rs,tasks,c,tok,Path(p['prompt_files'][c['prompt']]).read_text(encoding='utf-8'))
        assert n==v['requests_verified'] and len(rs)==v['tasks_verified']
        for r in rs:
            assert 0<=r['reward_cost_gated']<=1
            for q in r['requests']:
                if not q.get('context_limit'):assert q['truncated']==(len(q['output_ids'])>=c['max_new_tokens'] and q['output_ids'][-1] not in eos)
        cohorts[d.name]=dict(episodes=len(rs),requests=n,episodes_sha256=v['episodes_sha256'])
        allrows += [dict(cohort=d.name,**r) for r in rs];requests+=n
    return cohorts,allrows,requests


def signal_groups(rs):
    by=defaultdict(list)
    for r in rs:by[r['task_id']].append(r)
    assert len(by)==32 and all(len(g)==4 for g in by.values())
    details=[]
    for tid,g in by.items():
        if not any(r['grounded_success'] for r in g):kind='all_failed'
        elif not all(r['grounded_success'] for r in g):kind='mixed_correctness'
        elif len({r['total_cost'] for r in g})>1:kind='all_correct_different_cost'
        else:kind='all_correct_same_cost'
        rewards=[r['reward_cost_gated'] for r in g]
        details.append(dict(task_id=tid,kind=g[0]['kind'],classification=kind,
            grounded_successes=sum(r['grounded_success'] for r in g),seeds=[r['seed'] for r in g],
            calls=[r['calls'] for r in g],weighted_costs=[r['total_cost'] for r in g],rewards=rewards,
            population_reward_variance=statistics.pvariance(rewards),stop_reasons=[r['stop_reason'] for r in g]))
    active=[d for d in details if d['population_reward_variance']>0]
    return dict(groups=32,rollouts=128,counts=dict(Counter(d['classification'] for d in details)),
                nonconstant_reward_groups=len(active),effective_kinds=sorted({d['kind'] for d in active}),details=details)


def matched(a,b):
    aa={(r['task_id'],r['seed']):r for r in a};bb={(r['task_id'],r['seed']):r for r in b};assert aa.keys()==bb.keys()
    return dict(gained=sum(not aa[k]['grounded_success'] and bb[k]['grounded_success'] for k in aa),
                lost=sum(aa[k]['grounded_success'] and not bb[k]['grounded_success'] for k in aa),
                both_correct=sum(aa[k]['grounded_success'] and bb[k]['grounded_success'] for k in aa),
                both_failed=sum(not aa[k]['grounded_success'] and not bb[k]['grounded_success'] for k in aa))


def report(p,info_path):
    root=Path(p['root']);baseline_gate(p);signal_gate(p);ta=training_audit(p,info_path)
    old_e,old_v=set(),set()
    for f in p['historical_files_sha256']:
        e,v=private_sets(rows(f));old_e.update(e);old_v.update(v)
    audit({s:rows(Path(p['data_root'])/(s+'.jsonl')) for s in p['dataset_sha256']},old_e,old_v)
    cohorts,allrs,n=verify_all(p,info_path);control=controls(p);interim=read(root/'test-analysis-before-signal.json')
    assert interim['aggregate']==control
    before=read(root/'baseline-before-training.json');assert before['aggregate']==controls(p,CONDITIONS[:2])
    for seed in p['evaluation_seeds']['signal']:
        assert interim['analyzed_at']<read(completed(p,'signal','new','protocol',seed)/'manifest.json')['started_at']
    signal=aggregate(p,'signal',('new','protocol')); groups=signal_groups(signal)
    tasks={t['task_id']:t for t in rows(Path(p['data_root'])/'signal.jsonl')}
    sm=metrics(signal,tasks);by_kind={k:metrics([r for r in signal if r['kind']==k],tasks) for k in sorted({r['kind'] for r in signal})}
    comparison={ '/'.join(c):matched(aggregate(p,'test',c),aggregate(p,'test',('new','protocol'))) for c in CONDITIONS[:2]}
    tc={t['task_id']:t for t in rows(Path(p['data_root'])/'test.jsonl')}
    test_by_kind={k:{'/'.join(c):metrics([r for r in aggregate(p,'test',c) if r['kind']==k],tc) for c in CONDITIONS} for k in sorted({r['kind'] for r in tc.values()})}
    commit=training_gate(p);m=checkpoint(p,root/'plan.json')
    gate=p['candidate_rl_gate'];passed=groups['nonconstant_reward_groups']>=gate['min_nonconstant_groups'] and len(groups['effective_kinds'])>=gate['min_distinct_kinds']
    expected_total=624+128+128*len(commit['collection_order_used']);assert len(allrs)==expected_total
    out=dict(generated_at=now(),plan_sha256=sha256(root/'plan.json'),aggregate=control,matched_test_comparisons=comparison,
        test_by_kind=test_by_kind,signal=sm,signal_by_kind=by_kind,signal_groups=groups,training=m,training_set=commit,training_audit=ta,
        total_verified_episodes=len(allrs),total_verified_model_requests=n,cohorts=cohorts,rl_candidate_screen_passed=passed,
        rl_candidate_screen=gate,rl_optimization_performed=False,new_benchmark_reward_not_causal_comparison_to_v3=True)
    write(root/'summary.json',out);write(root/'reward-signal.json',groups)
    cases=[r for r in allrs if not r['grounded_success'] or (r['correct_extra_cost'] or 0)>0]
    (root/'diagnostic-cases.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in cases),encoding='utf-8',newline='\n')
    table=[]
    for c in p['eval_conditions']:
        t=control['test'][c];r=control['recovery'][c]
        table.append(f"| {c} | {t['grounded_correct']}/{t['attempts']} | {t['optimal_weighted_correct']}/{t['attempts']} | {r['grounded_correct']}/{r['attempts']} | {t['successful_cost_regret']} |")
    kindtable=[]
    for k,x in by_kind.items():kindtable.append(f"| {k} | {x['grounded_correct']}/{x['attempts']} | {x['optimal_weighted_correct']}/{x['attempts']} | {x['successful_cost_regret']} |")
    decision=('通过预设候选筛查，但这只支持在另生成的训练任务上验证信号，再测 RL 联合显存；不能保证 RL 提升。' if passed else
              '未通过预设候选筛查，暂不直接启动正式 GRPO。根据逐类失败和费用差异，在新训练事实上改进任务覆盖或纠错；不把这批诊断题回填训练或追着它调温度。')
    text=f'''# 第四轮：可观察分支、查询费用权衡与自然 ID 复制纠错

已在 RTX 5060 Laptop 8GB 完成真实 CUDA 收集、SFT、对照和全新奖励诊断。没有模型 API、云 GPU 或 RL optimizer 更新。

## 任务与奖励

project.policy 的实际观察决定 local/team/partner 分支。local 查项目 local_limit，team 查 owner→department→cost_center→spending_limit，partner 查 partner→vendor.spending_limit。不同分支和预算值互不相等，不能用未选中分支或 budget 冒充答案。已知关系可跳过。

另一条合法路线 project.resolved_limit 直接返回有效值；其每次费用为题目公开报价 2 或 6，其他查询每次费用 1。错误、重复和已知字段重查照常计费，六次调用/七回合上限不变。费用是合成环境的固定单位，不是实际 API 金额。

主奖励为 **grounded_correct×(1−0.02×weighted_cost)**。最多六次查询、每次最多六单位，奖励始终非负；错误答案和无有效来源的碰巧正确答案均为零。旧 exact_correct×(1−0.03×calls) 仅作辅助，不用于这轮分组。这是新 benchmark 和新费用口径，不能与上一轮 128/128 直接构成因果变化比较。

oracle 只用问题给出的事实及已返回观察：不知道 policy 时，报价 2 的解析服务弱支配所有完整路线；报价 6 时先读取 policy，再按观测分支推进。已知 policy/关系时比较剩余路线费用。源码 next_action 不访问私有数据库或 expected，没有偷看分支后挑路线。费用相同时优先较少调用。

## 真实学生复制错误与 SFT

先在独立训练收集事实上让固定 v3 模型＋简洁提示真实采样。预先规定首个可修复的 unknown_entity 字符遗漏/替换/插入/相邻转置、候选正确 ID 必须此前已出现且唯一、后续合法计划必须在原预算内。先运行 128 题，若不足八个独立自然错误事实，按冻结顺序继续，最多四组。不是按测试表现改数据；同一事实只取一次，最多八十个。

实际使用 {len(commit['collection_order_used'])} 组、{128*len(commit['collection_order_used'])} 条学生收集轨迹；找到并训练 {commit['natural_training_tasks']} 个独立自然复制错误前缀，类型 {json.dumps(commit['natural_copy_types'],ensure_ascii=False)}。保留真实模型输出文本、错误观察、种子、来源轨迹哈希；不把脚本造错算为自然错误。

另有 240 条新事实正确轨迹，合计 {commit['training_tasks']} 条训练轨迹、{commit['decision_targets']} 个 assistant-only 下一动作目标。所有错误前缀和工具观察 mask=-100，只监督 public-state planner 的合法下一动作和 EOS。train/collect/reserve/dev/test/recovery/signal 的私有实体与数值互不重叠，也与全部历史事实隔离；derived_train 仅允许引用 clean train 与已声明收集来源。测试和 signal 从未参与标签生成。

一轮固定 final checkpoint，v3 LoRA 起点精确匹配，冻结 BF16 主干，FP32 q/v LoRA r8/alpha16/dropout0.05，AdamW lr1e-4、microbatch1、累积8、seed42。没有按测试/dev/signal 挑 checkpoint，也没有分别消融新任务训练与纠错训练。

## 固定对照

旧模型是 v3，并非未训练基座。old/protocol、old/examples 在训练前固定运行；new/protocol 在训练后运行。自由测试 64 新事实×两个根种子，恢复测试 32 新事实×两个根种子，开发集16题一次。测试/恢复使用新完整措辞，ID 后缀长度16/24/32，重复采样不增加独立事实数。

| 条件 | 可溯源自由测试 | 最优费用且正确 | 注入 ID 错误恢复 | 正确自由轨迹费用遗憾总量 |
|---|---:|---:|---:|---:|
{chr(10).join(table)}

同题同种子的补训前后比较：{json.dumps(comparison,ensure_ascii=False)}。逐类成绩见 summary.json。恢复集是脚本单字符扰动，四种错误×四个链位置×两种措辞完全交叉；与真实学生训练前缀分开。固定注入错误计入调用及费用预算，恢复费用指标减去的是 public-state oracle 在同一错误状态下的最优总费用，不豁免模型后来犯错的成本。

## 5060 训练与复核

八步最长序列探针更新丢弃，正式从旧 v3 和新 optimizer 开始。全程非重入 checkpointing，每次更新后 empty_cache。正式 {m['optimizer_steps']} 步，最长序列 {m['max_sequence_tokens']} tokens、上限2048、无静默截断。优化循环 {m['optimization_wall_seconds']/60:.2f} 分钟，allocated/reserved 峰值 {m['cuda_peak_allocated_mib']/1024:.2f}/{m['cuda_peak_reserved_mib']/1024:.2f} GiB。进程内指标不等于独占整卡物理显存或吞吐。teacher-forced dev loss {m['dev_loss_before']:.6g}→{m['dev_loss_after']:.6g} 与自由采样成功率分开解释。

完整重放 {len(allrs)} 个 episode、{n} 次实际生成请求；token 重建/解码/EOS、工具状态与费用、分支可溯源答案、来源错误前缀、训练 mask/EOS/覆盖/有限梯度、初始与保存权重、私有划分隔离通过审计。训练前保存 [baseline-before-training.json](baseline-before-training.json)，全新 signal 前保存 [test-analysis-before-signal.json](test-analysis-before-signal.json)。

## 全新 32题×4条奖励诊断

signal 的事实与完整问题措辞均在训练前冻结，八类各四题；只在完整对照分析保存之后，使用 v4＋原简洁提示与30017/40017/50017/60017 四个根种子真实生成。温度0.7、top-p0.8、top-k20、每回合96tokens、六调用七回合，未追着诊断结果调采样。

可溯源正确 **{sm['grounded_correct']}/128**，最优费用且正确 **{sm['optimal_weighted_correct']}/128**。分类 {json.dumps(groups['counts'],ensure_ascii=False)}，非恒定奖励 **{groups['nonconstant_reward_groups']}/32**，有效组任务类别 {groups['effective_kinds']}。同答案/不同调用数不等于不同费用，分组比较的是实际 weighted_cost。逐题四个成本、奖励、方差、种子见 [reward-signal.json](reward-signal.json)。

| 诊断类别 | 可溯源正确 | 最优费用且正确 | 正确轨迹费用遗憾 |
|---|---:|---:|---:|
{chr(10).join(kindtable)}

训练前设定的实用候选门槛为至少8/32非恒定组、覆盖至少3类任务；不是理论保证，也不说明成功概率永久为零。**{decision}** 全错组和全对等费用组的当前组内优势为零。只在少数成本组有差异不能宣称成功率 RL 已可行，需看有效组覆盖、源证据、奖励漏洞和新训练集采样结果。

失败和多余费用原始轨迹见 [diagnostic-cases.jsonl](diagnostic-cases.jsonl)。这仍是固定英文合成工具 schema、一次训练种子的实验，不代表通用 agent 或 RSI 能力。

![分支任务、最优费用与奖励组](sft4-results.png)
'''
    (root/'analysis.zh-CN.md').write_text(text,encoding='utf-8',newline='\n');plot(root,out)
    print(json.dumps(dict(event='report',episodes=len(allrs),requests=n,test={c:x['grounded_correct'] for c,x in control['test'].items()},
        recovery={c:x['grounded_correct'] for c,x in control['recovery'].items()},signal=sm,groups=groups['counts'],rl_screen_passed=passed)),flush=True)


def plot(root,s):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,3,figsize=(15,5),layout='constrained')
    labels=['v3 / short','v3 / examples','v4 / short']
    for i,c in enumerate(s['aggregate']['test']):
        t=s['aggregate']['test'][c];r=s['aggregate']['recovery'][c]
        bars=ax[0].bar([i*3,i*3+1],[100*t['grounded_correct']/t['attempts'],100*r['grounded_correct']/r['attempts']],color=['#29b5e6','#16b39f']);ax[0].bar_label(bars,fmt='%.1f')
        bars=ax[1].bar(i,100*t['optimal_weighted_correct']/t['attempts'],color=['#91a4bb','#29b5e6','#16b39f'][i]);ax[1].bar_label(bars,fmt='%.1f')
    ax[0].set_xticks([.5,3.5,6.5],labels,rotation=12);ax[0].set_title('Blue: free start; green: ID recovery')
    ax[1].set_xticks(range(3),labels,rotation=12);ax[1].set_title('Correct AND minimum weighted cost')
    for a in ax[:2]:a.set_ylim(0,112);a.set_ylabel('Episodes (%)')
    names=['all_failed','mixed_correctness','all_correct_different_cost','all_correct_same_cost']
    values=[s['signal_groups']['counts'].get(k,0) for k in names]
    bars=ax[2].bar(range(4),values,color=['#ed4b4b','#16b39f','#29b5e6','#91a4bb']);ax[2].bar_label(bars)
    ax[2].set_xticks(range(4),['All fail','Mixed','Correct,\ncost varies','Correct,\nsame cost'],rotation=10)
    ax[2].set_ylim(0,35);ax[2].set_title('Fresh 32 tasks x 4 actual rollouts');ax[2].set_ylabel('Tasks')
    fig.savefig(root/'sft4-results.png',dpi=160);plt.close(fig)


def main():
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['audit','report']);a.add_argument('--root',type=Path,default=Path('results/sft-v4'));a.add_argument('--model-info',type=Path,required=True)
    args=a.parse_args();p=validate(args.root/'plan.json')
    if args.stage=='audit':print(json.dumps(training_audit(p,args.model_info)),flush=True)
    else:report(p,args.model_info)

if __name__=='__main__':main()

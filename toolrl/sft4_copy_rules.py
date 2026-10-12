"""Conditional training-only collection amendment; original execution files frozen."""
import argparse
import copy
import importlib
import json
from pathlib import Path
import sys
import subprocess

from . import sft4_experiment as experiment
from .sft4_data import harvest, save_rows
from .sft4_environment import BranchEnvironment, next_action

ROOT=Path('results/sft-v4')
ORIGINAL_BUILD=experiment.build_train


def amendment():
    a=experiment.read(ROOT/'copy-probe-amendment.json')
    assert a['original_plan_sha256']==experiment.sha256(ROOT/'plan.json')
    for n,h in a['source_sha256'].items():assert experiment.sha256(Path(__file__).with_name(n))==h
    assert experiment.sha256(ROOT/'copy-probe/data/copy_probe.jsonl')==a['data_sha256']
    return a


def prepare():
    p=experiment.validate(ROOT/'plan.json'); assert not (ROOT/'baseline-before-training.json').exists()
    assert not (ROOT/'training/manifest.json').exists() and not (ROOT/'copy-probe-amendment.json').exists()
    out=[]
    for t in experiment.rows(Path(p['data_root'])/'reserve.jsonl'):
        first=next_action(BranchEnvironment(t));field=first.get('field','resolved_limit')
        v=copy.deepcopy(t);v.update(task_id=t['task_id']+'_copy_probe',split='copy_probe',
            source_reserved_task_id=t['task_id'],
            prompt=t['prompt']+f' Training-only copy probe: emit your next single lookup for field "{field}" at the corresponding entity already supplied in this question, before any answer.')
        out.append(v)
    data=ROOT/'copy-probe/data';data.mkdir(parents=True);save_rows(data/'copy_probe.jsonl',out)
    a=dict(registered_at=experiment.now(),original_plan_sha256=experiment.sha256(ROOT/'plan.json'),
        source_sha256={n:experiment.sha256(Path(__file__).with_name(n)) for n in ['sft4_copy_rules.py','sft4_copy_probe.py']},
        data_sha256=experiment.sha256(data/'copy_probe.jsonl'),task_count=128,seeds=[40027,50027],
        temperature=1.2,top_p=.95,top_k=20,model='original v3 adapter, unchanged',
        reason='Original training collection has no eligible actual single-character copy failure so far; conditional mining rule registered before baseline evaluation.',
        condition='Enable only if all four original training collector cohorts complete with zero eligible copy facts.',
        stop='At least eight distinct eligible actual single-edit copy facts or both fixed probe roots complete; no artificial corruption.',
        distribution='Training-only hinted field and higher sampling entropy; not a deployment error-frequency estimate.',
        unaffected='Original sources/plan/data/prompts/baseline and final signal sampling remain frozen; no test/signal-based tuning.',
        exact_question_source='Reserved training facts only; field hint from original public-state planner; no answer value or hidden policy hint.',
        selection='Original unique-known-ID single-edit eligibility, budget check, one selected prefix per fact, total cap80 unchanged.')
    experiment.write(ROOT/'copy-probe-amendment.json',a);amendment()
    print(json.dumps(dict(event='conditional_copy_probe_registered',seeds=a['seeds'])),flush=True)


def augmented_validate(path):
    p=experiment.validate(path);a=amendment()
    _,natural,_,order=ORIGINAL_BUILD(p)
    assert len(order)==4 and not natural,'Conditional probe requires zero original copy facts after all four cohorts'
    q=copy.deepcopy(p);q['data_root']=str(ROOT/'copy-probe/data')
    q['evaluation_seeds']['copy_probe']=a['seeds'];q['dataset_sha256']['copy_probe']=a['data_sha256'];q['data_counts']['copy_probe']=128
    q['eval_fixed'].update(temperature=a['temperature'],top_p=a['top_p'])
    return q


def extended_build(p):
    clean_and_natural,natural,provenance,order=ORIGINAL_BUILD(p)
    active=[];seen={t['fact_group'] for t in natural};amendment()
    for seed in amendment()['seeds']:
        d=ROOT/'copy-probe/eval'/f'copy_probe-old-protocol-seed{seed}'
        if not (d/'verification.json').exists():break
        v=experiment.read(d/'verification.json');m=experiment.read(d/'manifest.json')
        assert m['status']=='complete' and v['tasks_verified']==128 and v['episodes_sha256']==experiment.sha256(d/'episodes.jsonl')
        name='copy-probe/'+d.name;provenance[name]=v['episodes_sha256'];order.append(['copy_probe',seed])
        raw={r['task_id']:r for r in experiment.rows(d/'episodes.jsonl')}
        for t in sorted(experiment.rows(ROOT/'copy-probe/data/copy_probe.jsonl'),key=lambda t:t['sampling_index']):
            if t['fact_group'] in seen:continue
            derived=harvest(t,raw[t['task_id']])
            if derived:
                derived.update(source_cohort=name,source_episodes_sha256=provenance[name]);active.append(derived);seen.add(t['fact_group'])
        if len(natural)+len(active)>=p['natural_target_facts']:break
    selected=(natural+active)[:p['natural_cap']]
    clean=experiment.rows(Path(p['data_root'])/'train.jsonl')
    return clean+selected,selected,provenance,order


def verify_extra(p,info_path):
    from transformers import AutoTokenizer
    from .sft4_verify import verify_rows
    a=amendment();info=experiment.read(info_path);tok=AutoTokenizer.from_pretrained(info['snapshot'],local_files_only=True)
    eos=experiment.read(Path(info['snapshot'])/'generation_config.json')['eos_token_id'];eos=[eos] if isinstance(eos,int) else eos
    tasks={t['task_id']:t for t in experiment.rows(ROOT/'copy-probe/data/copy_probe.jsonl')}
    # Ensure this amendment changes question hints only, not hidden facts or truth.
    original={t['task_id']:t for t in experiment.rows(Path(p['data_root'])/'reserve.jsonl')}
    for t in tasks.values():
        o=original[t['source_reserved_task_id']]
        assert all(t[k]==o[k] for k in o if k not in ('task_id','split','prompt'))
        assert t['prompt'].startswith(o['prompt']+' Training-only copy probe:')
    cohorts={};allrows=[];count=0
    for d in sorted((ROOT/'copy-probe/eval').glob('*')):
        m=experiment.read(d/'manifest.json');c=m['config'];v=experiment.read(d/'verification.json');rs=experiment.rows(d/'episodes.jsonl')
        assert m['status']=='complete' and not m['new_training_performed'] and m['trainable_parameters']==0
        assert c['plan_sha256']==a['original_plan_sha256'] and c['source_sha256']==p['source_sha256']
        assert c['dataset_sha256']==a['data_sha256'] and c['adapter_files_sha256']==p['old_adapter_files_sha256']
        assert c['copy_probe_amendment_sha256']==experiment.sha256(ROOT/'copy-probe-amendment.json')
        assert c['copy_probe_source_sha256']==a['source_sha256']['sft4_copy_probe.py']
        expected=p['eval_fixed']|dict(temperature=a['temperature'],top_p=a['top_p'])
        assert all(c[k]==value for k,value in expected.items())
        assert c['seed'] in a['seeds'] and v['episodes_sha256']==experiment.sha256(d/'episodes.jsonl')
        n=verify_rows(rs,tasks,c,tok,Path(p['prompt_files']['protocol']).read_text(encoding='utf-8'))
        assert n==v['requests_verified'] and len(rs)==v['tasks_verified']==128
        for row in rs:
            for q in row['requests']:
                if not q.get('context_limit'):assert q['truncated']==(len(q['output_ids'])>=c['max_new_tokens'] and q['output_ids'][-1] not in eos)
        name='copy-probe/'+d.name;cohorts[name]=dict(episodes=len(rs),requests=n,episodes_sha256=v['episodes_sha256'],training_only_amendment=True)
        allrows += [dict(cohort=name,**r) for r in rs];count+=n
    return cohorts,allrows,count


def install():
    amendment();experiment.build_train=extended_build
    def dispatch_evaluate(p,path,info,split,model,prompt,seed):
        subprocess.run([sys.executable,'-m','toolrl.sft4_copy_rules','dispatch','toolrl.sft4_run',
            '--plan',str(path),'--model-info',str(info),'--split',split,'--model',model,'--prompt',prompt,'--seed',str(seed)],check=True)
    experiment.evaluate=dispatch_evaluate
    from . import sft4_report
    original_verify=sft4_report.verify_all
    def joined_verify(p,info):
        c,rs,n=original_verify(p,info);ec,er,en=verify_extra(p,info)
        return c|ec,rs+er,n+en
    sft4_report.verify_all=joined_verify


def main():
    if len(sys.argv)==2 and sys.argv[1]=='prepare':prepare();return
    assert sys.argv[1]=='dispatch'
    module=sys.argv[2];assert module in ('toolrl.sft4_experiment','toolrl.sft4_train','toolrl.sft4_report','toolrl.sft4_run')
    install();sys.argv=[module]+sys.argv[3:];target=importlib.import_module(module);target.main()
    if module=='toolrl.sft4_experiment' and sys.argv[1]=='freeze-train':
        f=ROOT/'training-set-manifest.json';c=experiment.read(f)
        c['copy_probe_amendment_sha256']=experiment.sha256(ROOT/'copy-probe-amendment.json')
        c['additional_collector_source_sha256']=amendment()['source_sha256'];experiment.write(f,c)
    if module=='toolrl.sft4_report' and sys.argv[1]=='report':
        p=ROOT/'analysis.zh-CN.md';text=p.read_text(encoding='utf-8').replace('最多四组。','原定最多四组；零复制错误时追加训练用途采样探针，详见追加记录。')
        text += '\n## 训练采样追加记录\n\n原定四组使用温度0.7，均未产生符合规则的单字符复制错误。追加探针仅使用预留训练事实，给出 public-state planner 的下一步字段提示，温度1.2、top-p0.95，最多两个固定根种子。收集的前缀仍是真实 v3 输出，不是脚本扰动；这一分布不能估计普通提示下的部署错误率。基线、测试和最终 signal 的采样仍为原定0.7/0.8。来源、触发条件、数据/新增源码哈希见 [copy-probe-amendment.json](copy-probe-amendment.json)。报告对全部追加轨迹也重建 token、重放工具、核对模型与费用。\n'
        p.write_text(text,encoding='utf-8',newline='\n')
        f=ROOT/'summary.json';s=experiment.read(f);s['copy_probe_amendment_sha256']=experiment.sha256(ROOT/'copy-probe-amendment.json');experiment.write(f,s)

if __name__=='__main__':main()

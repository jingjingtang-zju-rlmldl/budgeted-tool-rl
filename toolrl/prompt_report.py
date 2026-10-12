"""Render verified prompt-control results; no model execution or weight updates."""
import argparse
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('root',type=Path)
    args=parser.parse_args()
    root=args.root
    summary=json.loads((root/'summary.json').read_text(encoding='utf-8'))
    names={'original':'原提示','neutral':'中性提示','balanced_examples':'平衡示例提示'}
    kinds={'direct':'直接回答','one_lookup':'一次查询','two_lookup':'两次查询'}
    aggregate=summary['aggregate']
    lines=['# 提示对照：真实模型实验结果','',
           '本轮仅改变系统提示；未进行 SFT、RL 或其他权重更新。',
           '60 个新生成的开发任务，0/1/2 次查询各 20 个；每种提示使用相同任务与三个采样种子，因此每组为 180 次任务尝试。',
           '**重复种子共享任务，180 次尝试不是 180 个独立任务。** 三组均使用与前次相同的模板族，只更换实体和私有事实；本轮不测量模板外泛化。','',
           '## 提示变化','',
           '- 原提示：与原 100 题 baseline 的系统提示完全相同。',
           '- 中性提示：仅将具体 owner→department 路径说明替换为“owner 是 employee ID”。其他文字不变。',
           '- 平衡示例：在中性提示后追加三段完整示范，分别涵盖直接回答、预算查询和负责人部门查询。示范使用独立实体和值，没有评测答案。示范采用系统提示内的文本对话，不是新增真实 tool messages。',
           '- 平衡示例同时增加了输入长度和示范信息，因此不能单独归因于长度、格式示范或任务示范中的某一项。','',
           '## 正确性','',
           '| 提示 | 直接回答 | 一次查询 | 两次查询 | 总体 | 全程严格 JSON 且正确 |',
           '|---|---:|---:|---:|---:|---:|']
    for variant,m in aggregate.items():
        cell=lambda k:f"{m['by_kind'][k]['successes']}/{m['by_kind'][k]['attempts']}"
        lines.append(f"| {names[variant]} | {cell('direct')} | {cell('one_lookup')} | {cell('two_lookup')} | {m['successes']}/{m['episodes']} ({m['successes']/m['episodes']:.1%}) | {m['strict_successes']}/{m['episodes']} ({m['strict_successes']/m['episodes']:.1%}) |")
    lines+=['','主要正确性分数沿用 `fenced` 解析：只移除完整单个 Markdown 围栏，不修复 JSON、参数或答案。严格分数要求整条成功轨迹所有请求的原始输出直接符合 JSON 动作协议。','',
            '![提示对照结果](comparison.png)','','## 工具选择与协议','',
            '| 提示 | 私有任务首个查询正确 | 两步任务启动查询 | 预算题首次误查 owner | 请求原始 JSON 合规 |',
            '|---|---:|---:|---:|---:|']
    for variant,m in aggregate.items():
        lines.append(f"| {names[variant]} | {m['correct_first_lookup']}/{m['private_episodes']} | {m['two_lookup_started']}/{m['two_lookup_attempts']} | {m['budget_first_lookup_owner']}/{m['one_lookup_attempts']} | {m['strict_valid_requests']}/{m['requests']} ({m['strict_valid_request_rate']:.1%}) |")
    lines+=['','首个查询正确要求实体 ID 和字段同时等于 oracle 的第一步；启动查询只说明调用过工具，不等于完成了两步或答对。','',
            '## 每个种子的结果','',
            '| 提示 | 种子 | 直接回答 | 一次查询 | 两次查询 | 总体 |',
            '|---|---:|---:|---:|---:|---:|']
    for c in summary['conditions']:
        m=c['metrics']
        cells=' | '.join(f"{m['by_kind'][k]['successes']}/20" for k in kinds)
        lines.append(f"| {names[c['variant']]} | {c['seed']} | {cells} | {m['successes']}/60 |")
    lines+=['','## 相同题目与种子的成对比较','',
            '| 提示变化 | 失败变成功 | 成功变失败 | 两组都成功 | 两组都失败 |',
            '|---|---:|---:|---:|---:|']
    for key,p in summary['paired'].items():
        a,b=key.split('_to_')
        lines.append(f"| {names[a]} → {names[b]} | {p['gained']} | {p['lost']} | {p['both_correct']} | {p['both_wrong']} |")
    lines+=['','种子相同用于配对比较；由于提示改变了输入，后续生成与随机数使用路径仍可能不同。成对比较不意味着两个条件生成过程逐 token 相同。','',
            '## 各措辞模板','',
            '| 任务与模板 | 原提示 | 中性提示 | 平衡示例 |',
            '|---|---:|---:|---:|']
    for k in kinds:
        for i in range(3):
            cells=' | '.join(f"{m['by_template'][f'{k}:{i}']['successes']}/{m['by_template'][f'{k}:{i}']['attempts']}" for m in aggregate.values())
            lines.append(f"| {kinds[k]}，模板 {i} | {cells} |")
    lines+=['','## 资源与复核','',
            f"- 全部完成 {summary['real_episodes']} 次真实 CUDA 任务尝试，独立验证 {summary['verified_requests']} 次请求的工具回放、判分、输入 token 与输出解码。",
            f"- 九个条件累计推理耗时 {sum(c['inference_wall_seconds'] for c in summary['conditions']):.2f} 秒；不含模型加载及验证。",
            f"- 最大 PyTorch CUDA allocated 为 {max(c['peak_cuda_allocated_mib'] for c in summary['conditions']):.2f} MiB；不是整机总显存或训练显存。",
            '- 固定 Qwen3-0.6B snapshot、BF16、SDPA、batch 1、关闭 thinking；temperature 0.7、top-p 0.8、top-k 20。',
            '- 每次最多 1,024 输入 token / 64 输出 token，每题最多 3 次调用 / 4 次生成。没有更换解析器、强制解码、重试修复或调用其他模型。',
            '- 各种子轮换提示顺序，减轻固定执行顺序与桌面负载的混淆。采样未做跨平台逐位确定性保证。','',
            '## 停止原因','',
            '```json',json.dumps({names[v]:m['stops'] for v,m in aggregate.items()},ensure_ascii=False,indent=2),'```','',
            '## 复现与原始记录','',
            '实验前固定的方案、代码和提示哈希见 [plan.json](plan.json)，数值见 [summary.json](summary.json)。每个 `提示-seed种子` 目录保存 manifest、episodes、单轮分析与 verification；[failure_examples.json](failure_examples.json) 汇集部分失败轨迹便于人工阅读。','',
            '在项目根目录，用新的输出目录复现，避免覆盖这次记录：','',
            '```bash',
            'python -m toolrl.prompt_study prepare --root results/my-prompt-control --data data/my-prompt60.jsonl',
            'python -m toolrl.prompt_study run-all --root results/my-prompt-control --model-info models/model-location.json',
            'python -m toolrl.prompt_study analyze --root results/my-prompt-control',
            'python -m toolrl.prompt_report results/my-prompt-control','```','',
            '运行示意使用下载器生成的 models/model-location.json；本次使用 work 内的同一固定 snapshot。实验 runner 在单独进程中替换 baseline 的系统提示绑定，原 baseline/environment 文件没有修改，加载、生成、解析与工具执行仍由原引擎完成。','',
            '提示改善能证明表现对接口与示范敏感；持续失败只能表明这些提示尚未解决问题，不能证明模型绝对没有该能力。SFT 前应先固定有效接口，再用独立实体与独立模板评测学习收益。']
    (root/'metrics.zh-CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    examples={}
    for variant in aggregate:
        candidates=[]
        for seed in summary['plan']['seeds']:
            rows=[json.loads(x) for x in (root/f'{variant}-seed{seed}'/'episodes.jsonl').read_text(encoding='utf-8').splitlines()]
            for row in rows:
                if not row['success']:
                    candidates.append({k:row[k] for k in ['task_id','seed','kind','template_id','expected','answer','stop_reason','actions','observations']}
                                      | {'question':row['messages'][1]['content'],'raw_outputs':[r['text'] for r in row['requests']]})
        examples[variant]=candidates[:12]
    (root/'failure_examples.json').write_text(json.dumps(examples,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    colors=['#8b6f78','#3b719f','#218678']
    labels=['Original','Neutral','Balanced examples']
    x=np.arange(3)
    for i,((variant,m),label,color) in enumerate(zip(aggregate.items(),labels,colors)):
        vals=[100*m['by_kind'][k]['successes']/m['by_kind'][k]['attempts'] for k in kinds]
        bars=axes[0].bar(x+(i-1)*.25,vals,width=.23,color=color,label=label)
        for bar,value in zip(bars,vals):
            axes[0].text(bar.get_x()+bar.get_width()/2,value+1,f'{value:.0f}',ha='center',fontsize=9)
    axes[0].set_xticks(x,['Given in prompt','One lookup','Two lookups'])
    axes[0].set_ylabel('Task success (%)')
    axes[0].set_ylim(0,114)
    axes[0].legend(loc='upper center',ncol=3,fontsize=8)
    axes[0].set_title('Same tasks and sampling seeds')
    for i,((variant,m),label,color) in enumerate(zip(aggregate.items(),labels,colors)):
        vals=[100*m['two_lookup_started']/m['two_lookup_attempts'],100*m['strict_successes']/m['episodes']]
        bars=axes[1].bar(np.arange(2)+(i-1)*.25,vals,width=.23,color=color,label=label)
        for bar,value in zip(bars,vals):
            axes[1].text(bar.get_x()+bar.get_width()/2,value+1,f'{value:.0f}',ha='center',fontsize=9)
    axes[1].set_xticks(np.arange(2),['Two-hop starts lookup','Strict JSON + correct'])
    axes[1].set_ylabel('Attempt fraction (%)')
    axes[1].set_ylim(0,114)
    axes[1].set_title('Behavior and protocol')
    fig.suptitle('Qwen3-0.6B prompt controls | RTX 5060 Laptop | no training',fontsize=14,fontweight='bold')
    fig.text(.01,-.04,'60 shared synthetic development tasks x 3 seeds per prompt. Fenced parser; strict scores reported separately. Repeats are correlated.',fontsize=9,color='#52605c')
    fig.savefig(root/'comparison.png',dpi=180,bbox_inches='tight')
    fig.savefig(root/'comparison.svg',bbox_inches='tight')
    print('Wrote metrics.zh-CN.md, failure_examples.json, comparison.png and comparison.svg')


if __name__=='__main__':
    main()

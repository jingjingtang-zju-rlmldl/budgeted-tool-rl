"""Static scientific result figure; matplotlib is needed only for this command."""
import argparse
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('run',type=Path)
    args=parser.parse_args()
    summary=json.loads((args.run/'summary.json').read_text(encoding='utf-8'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    families=['direct','one_lookup','two_lookup']
    values=[summary['by_kind'][k] for k in families]
    bars=axes[0].bar(['Given in prompt','One lookup','Two lookups'],[v['success_rate']*100 for v in values],
                     color=['#218678','#3b719f','#8172a8'],width=.6)
    axes[0].set_ylim(0,110)
    axes[0].set_ylabel('Task success (%)')
    axes[0].set_title('Real model inference: correctness')
    for bar,value in zip(bars,values):
        axes[0].text(bar.get_x()+bar.get_width()/2,bar.get_height()+2,
                     f"{value['successes']}/{value['tasks']}",ha='center',fontweight='bold')
    failures=summary['overall']['failure_classes']
    labels={'answer_without_required_lookup':'Answered without lookup','premature_answer':'Stopped before enough lookups',
            'invalid_lookup_argument':'Invalid tool arguments','wrong_lookup_or_answer':'Wrong lookup / final value',
            'invalid_action':'Invalid JSON / schema','generation_truncated':'Generation limit',
            'max_turns':'Turn limit','tool_budget_exhausted':'Call budget limit','context_limit':'Context limit'}
    ordered=sorted(failures.items(),key=lambda x:x[1])
    if ordered:
        y=[labels.get(k,k) for k,n in ordered]
        bars=axes[1].barh(y,[n for k,n in ordered],color='#bd654b',height=.6)
        for bar,(_,n) in zip(bars,ordered):
            axes[1].text(n+.3,bar.get_y()+bar.get_height()/2,str(n),va='center')
        axes[1].set_xlim(0,max(n for k,n in ordered)*1.15+1)
    axes[1].set_xlabel('Failed tasks')
    axes[1].set_title('Failure classes (mutually exclusive)')
    m=summary['manifest']
    fig.suptitle('Qwen3-0.6B · RTX 5060 Laptop · BF16 · 100 development tasks',fontweight='bold',fontsize=14)
    fig.text(.01,-.04,
             f"Synthetic English environment; real inference; one sampled attempt/task; no training. "
             f"Peak CUDA allocated {m.get('cuda_peak_allocated_mib',0):.0f} MiB. "
             f"Parser: {m['config'].get('parser','strict')}.",fontsize=9,color='#52605c')
    fig.savefig(args.run/'baseline.png',dpi=180,bbox_inches='tight')
    fig.savefig(args.run/'baseline.svg',bbox_inches='tight')
    print('Wrote baseline.png and baseline.svg')


if __name__=='__main__':
    main()

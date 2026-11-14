"""Plot observed E12 budget points with the existing Python workflow."""
from pathlib import Path
import hashlib
import json
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(r'E:\语言模型组合路由\research\e12_shared_feedback_learning')
STYLES={
    'M_raw':dict(color='#3775BA',linestyle='--',marker='o',markerfacecolor='white'),
    'M_group_all':dict(color='#0F4D92',linestyle='-',marker='o'),
    'P_raw':dict(color='#B47EAA',linestyle='--',marker='s',markerfacecolor='white'),
    'P_group_all':dict(color='#843E79',linestyle='-',marker='s'),
    'P_group_revealed':dict(color='#42949E',linestyle='-.',marker='D',markerfacecolor='white'),
    'Global_B':dict(color='#767676',linestyle=':',marker='x'),
    'RBF_E7':dict(color='#C27630',linestyle=':',marker='^')}


def main():
    cpu,wall=time.process_time(),time.perf_counter()
    source=ROOT/'public/E12_RESULTS.json'
    data=json.loads(source.read_text(encoding='utf-8'))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'axes.linewidth':.8,'xtick.labelsize':9,'ytick.labelsize':9,
        'savefig.facecolor':'white'})
    budgets=data['budgets']

    def draw(axis,task=None):
        handles=[];values=[]
        for name,style in STYLES.items():
            if name=='RBF_E7':
                records=data['historical_references'][name]
                xs=[240,300,360,480]
            else:
                records=data['methods'][name];xs=budgets
            ys=[100*(records[str(b)]['accuracy'] if task is None else records[str(b)]['by_task'][task]['accuracy']) for b in xs]
            values.extend(ys)
            line,=axis.plot(xs,ys,label=name,linewidth=1.6,markersize=4.4,**style)
            handles.append(line)
        axis.set_xlabel('Total revealed questions')
        axis.set_ylabel('Final aggregate accuracy (%)')
        axis.set_xticks(budgets)
        axis.set_xlim(48,492)
        axis.set_ylim(np.floor((min(values)-.3)*2)/2,np.ceil((max(values)+.3)*2)/2)
        axis.grid(axis='y',color='#E8E8E8',linewidth=.6)
        axis.set_axisbelow(True)
        return handles

    figures=ROOT/'public/figures';figures.mkdir(exist_ok=True)
    figure,axis=plt.subplots(figsize=(8.5,5.2))
    handles=draw(axis)
    axis.set_title('Overall',loc='left',pad=10)
    figure.legend(handles,[h.get_label() for h in handles],loc='upper center',bbox_to_anchor=(.52,1.005),
        ncol=3,frameon=False,fontsize=9,columnspacing=1.8,handlelength=2.6)
    figure.subplots_adjust(top=.77,left=.095,right=.98,bottom=.13)
    figure.savefig(figures/'overall_budget_curve.png',dpi=240)
    plt.close(figure)
    figure,axes=plt.subplots(1,2,figsize=(11.0,5.1))
    for axis,task,title in zip(axes,('gsm8k','bbh'),('GSM8K','BBH')):
        handles=draw(axis,task)
        axis.set_title(title,loc='left',pad=10)
    figure.legend(handles,[h.get_label() for h in handles],loc='upper center',bbox_to_anchor=(.52,1.005),
        ncol=4,frameon=False,fontsize=9,columnspacing=1.5,handlelength=2.6)
    figure.subplots_adjust(top=.80,left=.075,right=.985,bottom=.135,wspace=.25)
    figure.savefig(figures/'task_budget_curves.png',dpi=240)
    plt.close(figure)
    receipt=dict(status='COMPLETE',backend='Python matplotlib',source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        plot_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        cpu_seconds=time.process_time()-cpu,wall_seconds=time.perf_counter()-wall,
        displayed_budget_points=budgets,RBF_displayed_budget_points=[240,300,360,480],
        interpolation='straight segments through measured values',new_fits=0,new_prefill=0,
        image_sha256={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in figures.glob('*.png')})
    (ROOT/'private/logs/PLOT_RUN.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt))


if __name__=='__main__':
    main()

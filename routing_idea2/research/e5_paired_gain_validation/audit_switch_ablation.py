"""Exploratory gate ablations using the already-paired E5-C bootstrap draws."""
from pathlib import Path
import json
import numpy as np
from e5_common import PUBLIC,PRIVATE,put,sha
from summarize_switching import _intervals,_holm


def _append_block(text,begin,end,addition):
    if begin in text:
        start=text.index(begin)
        finish=text.index(end,start)+len(end)
        return text[:start]+addition+text[finish:]
    return text.rstrip()+'\n\n'+addition+'\n'


def _metadata_appendix(path,text):
    if not path.exists():return text
    data=json.loads(path.read_text(encoding='utf-8'))
    if data['status']!='COMPLETE':raise ValueError('Metadata supplement is incomplete')
    begin='<!-- E5_C_METADATA_BEGIN -->'
    end='<!-- E5_C_METADATA_END -->'
    lines=[begin,'## 同权限元数据补充','',
        '该补充单列task-known和真实BBH子任务oracle。每种模式下默认规则与P2推荐器同时获得同样元数据，采用相同F/G/E角色、alpha100和四种门控结构。它评价了不同部署权限，未参与Text-only主方法选择、主family认证或REAL_ALLOCATION_GATE。','',
        '|权限|主评分|同权限默认准确率|策略准确率|净收益与95%CI pp|task匹配超额与95%CI pp|subtask匹配超额与95%CI pp|',
        '|---|---|---:|---:|---:|---:|---:|']
    for mode,block in data['table'].items():
        base=block['default_accuracy']['balanced_v2']
        for score,row in block['table'].items():
            def cell(name):
                interval=row['intervals'][name]
                low,high=interval['ci95']
                return f"{100*interval['point']:+.4f} [{100*low:+.4f}, {100*high:+.4f}]"
            accuracy=row['accuracy']['balanced_v2']
            lines.append(f"|{mode}|{score}|{100*base:.4f}%|{100*accuracy:.4f}%|{cell('gain')}|{cell('excess_task')}|{cell('excess_subtask')}|")
    lines.extend(['','这些八项为探索性权限敏感性，表中95%区间未按八项调整。真实BBH子任务ID属于特权输入，主Text-only不具备该输入。task-known因子化门控的净收益与任务内匹配超额区间覆盖零。subtask-oracle因子化门控的这两项未经八项校正的95%区间为正，但其子任务内匹配超额区间覆盖零，需将子任务构成与同子任务逐题排序区分。该特权信号单列保留，主关口依据Text-only预设比较。','',
        f"METADATA_PERMISSION_CONTROLS.json SHA256为 `{sha(path)}`。",end])
    return _append_block(text,begin,end,'\n'.join(lines))


def main():
    source=PUBLIC/'SWITCH_RANKING_RESULTS.json'
    result=json.loads(source.read_text(encoding='utf-8'))
    paired_path=PRIVATE/'switching'/'SWITCH_PAIRED.npz'
    draws_path=PRIVATE/'switching'/'SWITCH_BOOTSTRAP.npz'
    with np.load(paired_path) as a:
        labels=a['rule_names'].tolist()
        gates=a['gates'].astype(float)
        D=a['D'].mean(1)
        task=a['task']
    with np.load(draws_path) as a:
        gains=a['gain']
    weights=np.where(task=='gsm8k',.5/np.sum(task=='gsm8k'),.5/np.sum(task=='bbh'))
    focal=labels.index('main/factorized')
    table={}
    for alternative in ('gap','disagreement','direction'):
        other=labels.index('main/'+alternative)
        paired=(gates[focal]-gates[other])*D
        point=float(weights@paired)
        assert abs(point-result['table']['main/factorized']['gain']+result['table']['main/'+alternative]['gain'])<1e-12
        summary=_intervals(point,gains[:,focal]-gains[:,other])
        summary.pop('ci97_5_bonferroni_two_comparisons')
        table['factorized_minus_'+alternative]={key:(value.tolist() if isinstance(value,np.ndarray) else value)
            for key,value in summary.items()}
    adjusted=_holm({key:float(row['centered_bootstrap_two_sided_p']) for key,row in table.items()})
    for key,row in table.items():row['holm3_adjusted_centered_bootstrap_p']=adjusted[key]
    output=dict(status='COMPLETE',comparisons=table,independent_questions=len(D),
        frozen_primary_method='factorized',role='exploratory mechanism ablations, separate from the two declared primary contrasts',
        bootstrap='same existing outerfold x broad-task original-question draws for every method',
        source={source.name:sha(source),paired_path.name:sha(paired_path),draws_path.name:sha(draws_path),
                'audit_switch_ablation.py':sha(__file__)},new_llm_calls=0)
    put(PUBLIC/'SWITCH_ABLATION_COMPARISONS.json',output)
    lines=['<!-- E5_C_ABLATION_BEGIN -->','## 因子化消融的直接配对比较','',
        '下表用相同原题、相同候选与同一批bootstrap draws直接比较各类G冻结主门控。它补充前表的单策略结果，三个对比作为探索性家族另行Holm处理。原先两项主比较与REAL_ALLOCATION_GATE提供继续判断。','',
        '|比较|准确率差 pp|95%描述区间 pp|Holm3 p（开发描述）|',
        '|---|---:|---:|---:|']
    for key,row in table.items():
        low,high=row['ci95']
        lines.append(f"|{key}|{100*row['point']:+.4f}|[{100*low:+.4f}, {100*high:+.4f}]|{row['holm3_adjusted_centered_bootstrap_p']:.6f}|")
    lines.extend(['','纠错与破坏的直接差异已经包含在题级配对准确率差中。当前因子化评分尚未建立相对这些简单评分的稳定优势，有限G区方向支持与OOF/full-G头的差异在前文单列。','',
        '完整数值和来源绑定保存在SWITCH_ABLATION_COMPARISONS.json，原逐题draws保持private。','<!-- E5_C_ABLATION_END -->'])
    report=PUBLIC/'03_SWITCH_RANKING_AND_MATCHED_CONTROLS.md'
    text=report.read_text(encoding='utf-8')
    begin='<!-- E5_C_ABLATION_BEGIN -->'
    end='<!-- E5_C_ABLATION_END -->'
    addition='\n'.join(lines)
    text=_append_block(text,begin,end,addition)
    text=_metadata_appendix(PUBLIC/'METADATA_PERMISSION_CONTROLS.json',text)
    report.write_text(text,encoding='utf-8')
    print(json.dumps(output,ensure_ascii=False),flush=True)


if __name__=='__main__':main()

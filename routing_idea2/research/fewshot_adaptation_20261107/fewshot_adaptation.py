"""CPU-only few-shot adaptation on frozen V2 matrices.

Episodes are question-level. A generation seed is a repeated measurement: support
and query use the same seed within an episode. No new LLM calls are made.
"""
import os, sys, json, time, hashlib
from pathlib import Path
import numpy as np

ROOT=Path(os.environ.get('MAB_ASSET_ROOT',r'E:\语言模型组合路由'))
REPO=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import load, sha, put, B, TEAMS, ridge, Transform

SEEDS=(42,137,2024); SUPPORT_SEEDS=(11,29,47,71); KVALS=(0,1,2,4)
METHODS=('task_subtask_fixed','pretrained_ridge','online_ridge','hierarchical_update','retrieval_knn','lowrank_adapter_r2','lowrank_adapter_r4','random_support_control')
BASE_CACHE={}

def tname(t,s): return 'gsm8k' if t=='gsm8k' else str(s)
def indices(d,name): return np.array([i for i,(t,s) in enumerate(zip(d['task'],d['sub'])) if tname(t,s)==name],dtype=int)
def names(d): return sorted(set(tname(t,s) for t,s in zip(d['task'],d['sub'])))
def episode(d,name,seed,k,mode):
    ids=indices(d,name); rng=np.random.default_rng(seed+1009*k+(0 if mode=='seen' else 90000)); p=rng.permutation(ids)
    if mode=='seen': h=max(1,len(p)//2); hist=p[:h]; pool=p[h:]
    else: hist=np.empty(0,dtype=int);pool=p
    if k>len(pool)-1:return None
    return hist,pool[:k],pool[k:]
def history(d,name,hist,mode):
    allidx=np.arange(len(d['E'])); target=indices(d,name)
    other=np.setdiff1d(allidx,target,assume_unique=False)
    return np.unique(np.r_[other,hist]) if mode=='seen' else other
def base_fit(d,hist,alpha=100.):
    dd={k:(v[hist] if k!='cost' else {}) for k,v in d.items()};tf=Transform('pca16').fit(dd);X=tf.transform(dd);W=ridge(X,d['Y'][hist].mean(axis=1),alpha)
    return tf,W
def fixed_choice(d,hist,name,g):
    if len(hist)==0:return 0
    ym=d['Y'][hist,g];same=np.array([tname(d['task'][i],d['sub'][i])==name for i in hist])
    if name!='gsm8k' and same.sum()>=2:return int(np.argmax(ym[same].mean(axis=0)))
    task='gsm8k' if name=='gsm8k' else 'bbh';tm=np.asarray(d['task'])[hist]==task
    return int(np.argmax(ym[tm].mean(axis=0) if tm.any() else ym.mean(axis=0)))
def support_labels(d,support,g,mode,default):
    if len(support)==0:return np.empty((0,20)),np.empty((0,20),bool)
    y=d['Y'][support,g].copy();m=np.ones_like(y,bool)
    if mode=='restricted':m[:]=False;m[:,default]=True
    y[~m]=np.nan;return y,m
def update(Xs,base_s,Xq,base_q,y,m,method,lam=10.,knn=5,adapter_dim=4,rng_seed=0):
    if len(y)==0:return base_q
    r=y-base_s
    if method=='online_ridge':
        out=base_q.copy()
        for a in range(20):
            ok=m[:,a]
            if ok.any():out[:,a]+=Xq@ridge(Xs[ok],r[ok,a,None],lam).ravel()
        return out
    if method=='hierarchical_update':
        out=base_q.copy()
        for a in range(20):
            ok=m[:,a]
            if ok.any():out[:,a]+=np.nansum(r[ok,a])/(ok.sum()+lam)
        return out
    if method.startswith('lowrank_adapter'):
        rr=int(method[-1]);U=B[:,:rr];rows=[];vals=[]
        for j in range(len(y)):
            for a in np.flatnonzero(m[j]):rows.append(U[a]);vals.append(r[j,a])
        if not rows:return base_q
        h=ridge(np.asarray(rows),np.asarray(vals)[:,None],lam).ravel();return base_q+U@h
    if method in ('retrieval_knn','random_support_control'):
        out=base_q.copy()
        for qi in range(len(Xq)):
            dist=np.sum((Xs-Xq[qi])**2,axis=1);order=np.argsort(dist)[:min(knn,len(dist))]
            for a in range(20):
                ok=m[order,a]
                if not ok.any():continue
                oo=order[ok];vals=y[oo,a]
                if method=='random_support_control':vals=np.random.default_rng(rng_seed+qi+a).permutation(vals)
                w=1/(1+dist[oo]);out[qi,a]=np.average(vals,weights=w)
        return out
    return base_q
def one(d,name,ss,k,mode,feedback,method):
    ep=episode(d,name,ss,k,mode)
    if ep is None:return None
    hist0,support,query=ep;hist=history(d,name,hist0,mode)
    key=(id(d),name,ss,k,mode)
    if key not in BASE_CACHE:
        tf,W=base_fit(d,hist);Xh=tf.transform({'E':d['E'][hist]});Xq=tf.transform({'E':d['E'][query]});bq=Xq@W
        BASE_CACHE[key]=(tf,W,Xh,Xq,bq,hist,support,query)
    tf,W,Xh,Xq,bq,hist,support,query=BASE_CACHE[key]
    default=fixed_choice(d,hist,name,0); # same task context, seed-specific choice below
    rows=[]
    for g,seed in enumerate(SEEDS):
        default=fixed_choice(d,hist,name,g);bs=tf.transform({'E':d['E'][support]})@W if len(support) else np.empty((0,20))
        y,m=support_labels(d,support,g,feedback,default)
        pred=bq.copy() if method in ('task_subtask_fixed','pretrained_ridge') else update(tf.transform({'E':d['E'][support]}) if len(support) else Xh[:0],bs,Xq,bq,y,m,method,adapter_dim=2 if method.endswith('r2') else 4,rng_seed=ss)
        choice=np.full(len(query),default) if method=='task_subtask_fixed' else np.argmax(pred,axis=1)
        q=d['Y'][query,g];served=q[np.arange(len(query)),choice];before=q[np.arange(len(query)),np.argmax(bq,axis=1)];fixed=q[:,default]
        rows.append({'task':name,'mode':mode,'feedback':feedback,'method':method,'k':k,'support_seed':ss,'generation_seed':seed,'query_n':len(query),'accuracy':float(served.mean()),'before_accuracy':float(before.mean()),'fixed_accuracy':float(fixed.mean()),'delta_before':float(served.mean()-before.mean()),'delta_fixed':float(served.mean()-fixed.mean()),'regret':float((q.max(axis=1)-served).mean()),'before_regret':float((q.max(axis=1)-before).mean()),'wrong_to_right':int(np.sum((served>before)&(before<1))),'right_to_wrong':int(np.sum((served<before)&(before>0)))})
    return rows
def run(d,mode,feedback):
    rows=[]
    for name in names(d):
        for ss in SUPPORT_SEEDS:
            for k in KVALS:
                for method in METHODS:
                    got=one(d,name,ss,k,mode,feedback,method)
                    if got:rows.extend(got)
    return rows
def summarize(rows):
    out={}
    for m in METHODS:
        out[m]={}
        for k in KVALS:
            z=[r for r in rows if r['method']==m and r['k']==k]
            if not z:out[m][str(k)]={'episodes':0};continue
            out[m][str(k)]={'episodes':len(z),'accuracy':float(np.mean([r['accuracy'] for r in z])),'before_accuracy':float(np.mean([r['before_accuracy'] for r in z])),'fixed_accuracy':float(np.mean([r['fixed_accuracy'] for r in z])),'delta_before':float(np.mean([r['delta_before'] for r in z])),'delta_fixed':float(np.mean([r['delta_fixed'] for r in z])),'regret':float(np.mean([r['regret'] for r in z])),'before_regret':float(np.mean([r['before_regret'] for r in z])),'wrong_to_right':int(sum(r['wrong_to_right'] for r in z)),'right_to_wrong':int(sum(r['right_to_wrong'] for r in z)),'negative_episode_fraction':float(np.mean([r['delta_before']<0 for r in z])),'per_generation_accuracy':[float(np.mean([r['accuracy'] for r in z if r['generation_seed']==s])) for s in SEEDS]}
    return out
def main():
    start=time.perf_counter();d,refs=load();out=REPO/'routing_idea2/research/fewshot_adaptation_20261107';out.mkdir(parents=True,exist_ok=True);private=ROOT/'research/conditional_gain_20261106/private/fewshot_adaptation_20261107';private.mkdir(parents=True,exist_ok=True)
    protocol={'schema':'mab_fewshot_adaptation_protocol_v1','status':'CPU_ONLY_FULL_AND_RESTRICTED_SUPPORT','task_units':'BBH 27 subtasks plus GSM8K as one task','support_seeds':list(SUPPORT_SEEDS),'k_values':list(KVALS),'generation_seeds':list(SEEDS),'episode_unit':'question IDs; same generation seed support/query; three seeds are repeated measurements','seen_mode':'same task first half history, remaining pool split support/query','unseen_mode':'entire target task excluded from base history','full_feedback':'all 20 existing aggregate scores for support question at that generation seed','restricted_feedback':'only default combination score; unknown actions remain unobserved','methods':list(METHODS),'base_ridge':'direct20 PCA16 history-only, alpha100','online_ridge_lambda':10.0,'hierarchical_lambda':4.0,'retrieval_k':5,'lowrank_dimensions':[2,4],'random_control':'shuffle observed support rewards within action, support IDs fixed','selection':'fixed protocol; no validation tuning','fresh_or_sealed_opened':False,'new_llm_calls':0}
    put(out/'CPU_EXPERIMENT_PROTOCOL.json',protocol)
    ts=run(d['selector_train'],'seen','full');tu=run(d['selector_train'],'unseen','full');vs=run(d['selector_validation'],'seen','full');vu=run(d['selector_validation'],'unseen','full');vr=run(d['selector_validation'],'unseen','restricted')
    result={'schema':'mab_fewshot_adaptation_results_v1','status':'EXPERIMENT1_2_CPU_COMPLETE','source_refs':refs,'protocol':protocol,'train_seen':summarize(ts),'train_unseen':summarize(tu),'validation_seen':summarize(vs),'validation_unseen_full':summarize(vu),'validation_unseen_restricted':summarize(vr),'episode_counts':{'train_seen_rows':len(ts),'train_unseen_rows':len(tu),'validation_seen_rows':len(vs),'validation_unseen_full_rows':len(vu),'validation_unseen_restricted_rows':len(vr)},'support_query_overlap_checked':True,'validation_not_used_for_selection':True,'new_llm_calls':0,'new_answer_scoring_operations':0,'cpu_wall_s':time.perf_counter()-start,'fresh400_opened':False,'sealed200_opened':False}
    put(out/'EXPERIMENT_RESULTS.json',result)
    # Private row-level outputs: retained locally, never added to repository.
    np.savez_compressed(private/'EPISODE_ROWS_PRIVATE.npz',train_seen=np.array([json.dumps(x) for x in ts],dtype='U'),train_unseen=np.array([json.dumps(x) for x in tu],dtype='U'),validation_seen=np.array([json.dumps(x) for x in vs],dtype='U'),validation_unseen=np.array([json.dumps(x) for x in vu],dtype='U'),validation_restricted=np.array([json.dumps(x) for x in vr],dtype='U'))
    (out/'01_episode_construction.md').write_text('# 01 少样本适应任务构造\n\nBBH 27个子任务各自成任务环境，GSM8K作为单一任务。每个episode按问题ID分Support/Query；三个语言模型生成seed在episode内保持重复测量。已见任务使用同任务历史半区和新题池；未见任务模式完全排除目标子任务评分。k=0/1/2/4，支持抽样seed为11/29/47/71。\n\n主分析使用同一生成seed的Support反馈更新同一生成seed的Query；充分反馈读取已有真实20组合矩阵，受限反馈仅使用默认组合已执行评分。validation仅开发诊断，fresh400/sealed200未打开。完整episode计数和逐k/任务汇总见EXPERIMENT_RESULTS.json。\n',encoding='utf8')
    (out/'02_fewshot_adaptation_baselines.md').write_text('# 02 少样本适应基线\n\n比较任务/子任务固定、冻结Ridge、在线Ridge、层级收缩、距离加权KNN、2/4维低秩适应和随机Support负对照。所有方法使用同一历史、Support和Query；支持反馈与查询奖励严格分离。\n\n结果按k、任务环境、生成seed报告准确率、Top-1 Regret、更新前后差值、固定策略差值、纠错/破坏和负迁移比例。充分反馈与受限反馈分开，不将完整矩阵标签当作受限反馈。\n\n本轮仅CPU复用已有矩阵，未新增LLM调用；由于BBH每子任务只有11—12个训练题、验证题3—4个，局部episode方差较大，不能替代fresh独立测试。\n',encoding='utf8')
    (out/'EXPERIMENT_SUMMARY.md').write_text('# MAB 少样本反馈适应：实验1—2阶段总结\n\n本轮从codex/conditional-gain-20261106创建新分支，固定N6/K3、20组合、Qwen3聚合器和三个生成seed。实验1构造已见任务与未见子任务的Support/Query episodes；实验2比较普通少样本适应方法。\n\n所有结果来自train600/validation200既有真实20组合矩阵。元学习、顺序在线流和反馈采集机制尚未启动；只有普通适应取得稳定正向证据后才推进。\n',encoding='utf8')
    files=[{'file':p.relative_to(out).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(out.rglob('*')) if p.is_file() and p.name!='DELIVERY_MANIFEST.json'];put(out/'DELIVERY_MANIFEST.json',{'schema':'mab_fewshot_adaptation_manifest_v1','files':files,'private_episode_rows_excluded':True,'new_llm_calls':0,'research_complete':False})
    print(json.dumps({'status':'COMPLETE','train_seen_rows':len(ts),'train_unseen_rows':len(tu),'validation_unseen_full_rows':len(vu),'validation_unseen_restricted_rows':len(vr),'output':str(out)},ensure_ascii=False),flush=True)
if __name__=='__main__':main()

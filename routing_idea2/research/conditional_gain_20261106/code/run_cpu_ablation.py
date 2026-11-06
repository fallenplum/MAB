from common import *
import warnings,pickle,sys
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold,train_test_split

def main():
    d,refs=load();tr=d['selector_train'];va=d['selector_validation']
    foldfile=np.load(PRIVATE/'FOLDS_PRIVATE.npz');folds=[(foldfile[f'fold{j}_train'],foldfile[f'fold{j}_holdout']) for j in range(5)]
    candidates=[(f,'pca16',a) for f in ['structured','direct20','lowrank5','ranknet'] for a in [.1,1,10,100]]
    candidates += [('direct20',feat,a) for feat in ['pca32','pca64','full','full_meta'] for a in [.1,1,10,100]]
    cache={};baseline_oof={k:np.zeros((600,20)) for k in ['global_fixed','task_fixed','subtask_fixed','shrink_default']}
    for j,(it,iv) in enumerate(folds):
        a,b=subset(tr,it),subset(tr,iv)
        for k,v in defaults(a,b).items():baseline_oof[k][iv]=v
        baseline_oof['shrink_default'][iv]=defaults(a,b,10,4)['subtask_fixed']
        for feat in ['pca16','pca32','pca64','full','full_meta']:
            tf=Transform(feat).fit(a);assert not set(tf.train_ids)&set(b['ids'])
            cache[j,feat]=(tf.transform(a),tf.transform(b))
    np.savez_compressed(PRIVATE/'BASELINE_OOF.npz',**baseline_oof)
    records=[];preds={};total_cpu=time.process_time();total_wall=time.perf_counter()
    def run_one(family,feat,alpha):
        name=f'{family}__{feat}__a{alpha:g}';file=PRIVATE/'oof'/f'{name}.npz';meta_path=PRIVATE/'oof'/f'{name}.json'
        if file.exists() and meta_path.exists():
            z=np.load(file);p=z['pred'];info=read(meta_path);print('RESUME '+name,flush=True)
        else:
            p=np.zeros((600,20));fits=[];cw=time.perf_counter();cc=time.process_time()
            for j,(it,iv) in enumerate(folds):
                a,b=subset(tr,it),subset(tr,iv);x,xt=cache[j,feat]
                bt=np.argmax(defaults(a,a,10,4)['subtask_fixed'],axis=1);bv=np.argmax(baseline_oof['shrink_default'][iv],axis=1)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always',ConvergenceWarning)
                    pr,trainp,extra,state=fit_predict(family,x,a['Y'],xt,alpha,bt,bv)
                p[iv]=pr;fits.append({'fold':j,'train_questions':len(it),'holdout_questions':len(iv),'train_metrics':metrics(trainp,a,bt if family.startswith('delta_') else None),'fit_metadata':extra,'warnings':[str(w.message) for w in caught]})
            info={'name':name,'family':family,'feature':feat,'alpha':alpha,'oof':metrics(p,tr,baseline_oof['shrink_default'].argmax(1) if family.startswith('delta_') else None),'fold_fits':fits,'cpu_s':time.process_time()-cc,'wall_s':time.perf_counter()-cw}
            file.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(file,pred=p);put(meta_path,info)
            print(json.dumps({'candidate':name,'oof_acc':info['oof']['accuracy'],'wall_s':info['wall_s']}),flush=True)
        preds[name]=p;records.append(info);put(PUBLIC/'CPU_SEARCH_PROGRESS.json',{'completed_candidates':len(records),'last':name,'last_oof_accuracy':info['oof']['accuracy'],'validation_used_for_selection':False,'llm_calls':0})
        return info
    for c in candidates:run_one(*c)
    def best(rows):return max(rows,key=lambda r:(r['oof']['accuracy'],-r['oof']['top1_regret'],r['alpha'],r['name']))
    structure={f:best([r for r in records if r['family']==f and r['feature']=='pca16'])['name'] for f in ['structured','direct20','lowrank5','ranknet']}
    feature={f:best([r for r in records if r['family']=='direct20' and r['feature']==f])['name'] for f in ['pca16','pca32','pca64','full','full_meta']}
    winner=best([r for r in records if r['family']=='direct20']);selected_feature=winner['feature']
    # This durable selection precedes any new-candidate validation scoring.
    put(PUBLIC/'TRAIN_ONLY_STRUCTURE_FEATURE_SELECTION.json',{'structure_winners':structure,'feature_winners':feature,'conditional_feature':selected_feature,'decision_basis':'training600 OOF only','selection_timestamp_unix':time.time(),'validation_not_used_for_this_selection':True})
    for family in ['delta_linear','delta_default_interaction','delta_mlp']:
        for alpha in [.1,1,10,100]:run_one(family,selected_feature,alpha)
    # Conditional and absolute policies are compared at the same selected question representation.
    oof_shrink=np.argmax(baseline_oof['shrink_default'],axis=1)
    threshold_rows=[]
    for r in records:
        if not r['family'].startswith('delta_'):continue
        p=preds[r['name']]
        for tau in [0,.01,.025,.05,.1,.2,1e9]:
            choices=np.where(p.max(1)>tau,p.argmax(1),oof_shrink)
            served=selected(tr['Y'],choices)
            threshold_rows.append({'candidate':r['name'],'threshold':tau,'oof_accuracy':macro(served.mean(1),tr['task']),'switch_rate':float(np.mean(choices!=oof_shrink))})
    chosen=max(threshold_rows,key=lambda r:(r['oof_accuracy'],-r['switch_rate'],r['threshold'],r['candidate']))
    put(PUBLIC/'TRAIN_ONLY_MAIN_SELECTION_FROZEN.json',{'selected_conditional':chosen,'threshold_curve_all_configs':threshold_rows,'selected_feature':selected_feature,'selection_uses':'training OOF only','fresh_or_sealed_scores_opened':False,'validation_used_for_model_selection':False,'created_unix':time.time()})
    vpred={};tmetrics={};final_meta={};states={}
    bval=defaults(tr,va);btrain=defaults(tr,tr)
    for k,v in bval.items():vpred[k]=v
    bval['shrink_default']=defaults(tr,va,10,4)['subtask_fixed'];btrain['shrink_default']=defaults(tr,tr,10,4)['subtask_fixed'];vpred['shrink_default']=bval['shrink_default']
    bt=btrain['shrink_default'].argmax(1);bv=bval['shrink_default'].argmax(1)
    # All registered candidates receive development diagnostics, retaining losses and convergence failures.
    finaltf={feat:Transform(feat).fit(tr) for feat in ['pca16','pca32','pca64','full','full_meta']}
    for r in records:
        name=r['name'];tf=finaltf[r['feature']];x=tf.transform(tr);xv=tf.transform(va);c=time.process_time();w=time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always',ConvergenceWarning)
            p,tp,extra,state=fit_predict(r['family'],x,tr['Y'],xv,r['alpha'],bt,bv)
        vpred[name]=p;tmetrics[name]=metrics(tp,tr,bt if r['family'].startswith('delta_') else None);final_meta[name]={'fit_and_val_predict_cpu_s':time.process_time()-c,'fit_and_val_predict_wall_s':time.perf_counter()-w,'metadata':extra,'warnings':[str(z.message) for z in caught]};states[name]={'transform':tf,'state':state,'config':r}
    p=vpred[chosen['candidate']];vc=np.where(p.max(1)>chosen['threshold'],p.argmax(1),bv)
    chosen_scores=np.full_like(p,-1e6);chosen_scores[np.arange(200),vc]=0
    vpred['conditional_main_thresholded']=chosen_scores
    # Original frozen strategies and independentTop3 are read-only references, never reselected on validation.
    old=ROOT/'artifacts/private/routing_redesign_20261104/ff_v1'
    with np.load(old/'model/model.npz',allow_pickle=False) as z:oldmodel={k:z[k] for k in z.files}
    sys.path.insert(0,str(ROOT/'src'));from budget_router_v2 import full_feedback as ff
    inp={'embeddings':va['E'],'task_labels':va['task'],'subtask_labels':va['sub']}
    oldp=ff.predictions(oldmodel,inp);vpred['original_frozen_selector']=oldp['selector']
    with np.load(old/'it3_v1/validation/predictions.npz',allow_pickle=False) as z:
        keys=z.files
        if 'selected_team_index' in keys:ic=z['selected_team_index']
        else:raise ValueError('IndependentTop3 original selected choice field absent')
    ip=np.full((200,20),-1e6);ip[np.arange(200),ic]=0;vpred['independent_top3']=ip
    # Separate empirical uncertainty calibration. Never refit calibration questions into this variant.
    st=np.array([str(t)+'/'+str(s) for t,s in zip(tr['task'],tr['sub'])]);proper,cal=train_test_split(np.arange(600),test_size=120,stratify=st,random_state=20261106)
    a,b=subset(tr,proper),subset(tr,cal);rc=next(r for r in records if r['name']==chosen['candidate']);tf=Transform(rc['feature']).fit(a)
    ba=defaults(a,a,10,4)['subtask_fixed'].argmax(1);bc=defaults(a,b,10,4)['subtask_fixed'].argmax(1);bz=defaults(a,va,10,4)['subtask_fixed'].argmax(1)
    both={'E':np.concatenate((b['E'],va['E'])),'task':np.concatenate((b['task'],va['task'])),'sub':np.concatenate((b['sub'],va['sub']))}
    pp,_,_,state=fit_predict(rc['family'],tf.transform(a),a['Y'],tf.transform(both),rc['alpha'],ba,np.r_[bc,bz])
    actual=b['Y'].mean(1);actual-=actual[np.arange(120),bc,None]
    u=float(np.quantile(np.max(pp[:120]-actual,axis=1),min(1,np.ceil(121*.9)/120),method='higher'))
    cc=np.where(pp[120:].max(1)>max(u,0),pp[120:].argmax(1),bz);cs=np.full((200,20),-1e6);cs[np.arange(200),cc]=0;vpred['empirically_calibrated_switch']=cs
    put(PUBLIC/'UNCERTAINTY_CALIBRATION.json',{'proper_train':480,'calibration_questions':120,'quantile_level':.9,'simultaneous_over_actions_empirical_one_sided_error_threshold':u,'heldout_calibration_not_refit':True,'coverage_target':'observed3seed mean paired labels, not population conditional Delta','not_a_strict_population_error_bound':True,'validation_switch_rate':float(np.mean(cc!=bz))})
    np.savez_compressed(PRIVATE/'ALL_VALIDATION_PREDICTIONS.npz',**vpred)
    with (PRIVATE/'FITTED_CANDIDATE_STATES.pkl').open('wb') as f:pickle.dump({'states':states,'selection':chosen,'defaults_train_ids':tr['ids']},f)
    baseline_results={k:metrics(v,va) for k,v in vpred.items() if k in ['global_fixed','task_fixed','subtask_fixed','shrink_default','original_frozen_selector','independent_top3']}
    validation={r['name']:metrics(vpred[r['name']],va,bv if r['family'].startswith('delta_') else None) for r in records}
    # Fake sentinel scores encode only decisions: report service metrics, not surrogate value regression errors.
    switch={}
    for name in ['conditional_main_thresholded','empirically_calibrated_switch']:
        choice=vpred[name].argmax(1);own=selected(va['Y'],choice);default_choice=bv if name=='conditional_main_thresholded' else bz;default=selected(va['Y'],default_choice)
        delta=own-default;mask=choice!=default_choice
        switch[name]={'accuracy':macro(own.mean(1),va['task']),'switch_questions':int(mask.sum()),'switch_rate':float(mask.mean()),'wrong_to_right_cells':int(np.sum(delta>0)),'right_to_wrong_cells':int(np.sum(delta<0)),'net_corrected_cells':int(delta.sum()),'changed_question_mean_positive':int(np.sum(delta.mean(1)>0)),'changed_question_mean_negative':int(np.sum(delta.mean(1)<0)),
            'per_generation_accuracy':[macro(own[:,g],va['task']) for g in range(3)],'vs_original_baselines':{k:ci(own-selected(va['Y'],vpred[k].argmax(1)),va['task'],va['sub']) for k in ['global_fixed','task_fixed','subtask_fixed','independent_top3']},
            'within_bbh_subtask':{s:{'questions':int(np.sum(va['sub']==s)),'switch_questions':int(np.sum(mask&(va['sub']==s))),'net_corrected_cells':int(delta[va['sub']==s].sum()),'ci':ci(delta[va['sub']==s],va['task'][va['sub']==s],va['sub'][va['sub']==s])} for s in sorted(set(va['sub'][va['task']=='bbh']))},
            'per_task':{t:{'questions':int(np.sum(va['task']==t)),'switch_questions':int(np.sum(mask&(va['task']==t))),'wrong_to_right':int(np.sum(delta[va['task']==t]>0)),'right_to_wrong':int(np.sum(delta[va['task']==t]<0))} for t in ['gsm8k','bbh']}}
    variance={}
    for name in ['conditional_main_thresholded','original_frozen_selector','independent_top3']:
        dy=selected(va['Y'],vpred[name].argmax(1))-selected(va['Y'],vpred['subtask_fixed'].argmax(1))
        w=float(dy.var(1,ddof=1).mean());vraw=float(dy.mean(1).var(ddof=1)-w/3)
        variance[name]={'w_estimate':w,'v_unclipped':vraw,'v_clipped_for_planning_only':max(vraw,0),'iid_problem_variance_estimate_mean':vraw/200+w/600,'not_a_valid_bound_after_selection_on_same_data':True,'L_star_by_a_over_b':{str(r):(float(np.sqrt(r*w/max(vraw,1e-15))) if vraw>0 else None) for r in [.1,1,10,100]},'platform_and_stratification_caveat':True}
    results={'status':'CPU_DEVELOPMENT_COMPLETE_FRESH_NOT_EVALUATED','structure_selection':structure,'feature_selection':feature,'selected_conditional':chosen,'oof_baselines':{k:metrics(v,tr) for k,v in baseline_oof.items()},'all_candidate_oof':records,'final_train_metrics':tmetrics,'all_candidate_validation':validation,'original_validation_baselines':baseline_results,'conditional_switches':switch,'frozen_policy_variance':variance,'actual_cpu_s':time.process_time()-total_cpu,'actual_wall_s':time.perf_counter()-total_wall,'final_fit_costs':final_meta,'new_llm_calls':0,'fresh_gold_used':False,'sealed_gold_used':False,'selection_data':'train600 internal OOF; validation200 development diagnostics only'}
    put(PUBLIC/'CPU_COMPLETE_RESULTS.json',results)
    reports=[('02_second_order_vs_direct20.md',structure,'结构表示'),('03_feature_ablation.md',feature,'问题特征')]
    for filename,group,title in reports:
        text=f'# {title}消融\n\n## 研究问题\n相同600题分组五折与相同正则搜索，比较{title}对实际决策质量的影响。\n\n## 实验设置与对照\n四正则0.1/1/10/100，PCA及标准化仅训练折拟合，三个生成种子永不跨折。排序采用真实种子配对logistic；低秩固定rank5。其余详见预先锁定协议。\n\n## 数值结果\n|配置|OOF准确率|最终train准确率|val准确率|val Regret|val差值MAE|val非平局排序|\n|---|---:|---:|---:|---:|---:|---:|\n'
        for label,name in group.items():
            r=next(r for r in records if r['name']==name);v=validation[name];text+=f"|{label} ({name})|{r['oof']['accuracy']:.6f}|{tmetrics[name]['accuracy']:.6f}|{v['accuracy']:.6f}|{v['top1_regret']:.6f}|{v['pair_value_difference_mae']:.6f}|{v['non_tie_pair_ranking_accuracy']:.6f}|\n"
        text+='\n## 统计结论\n所有搜索配置的完整MSE、支持数、逐seed、逐任务/子任务结果和收敛警告均在CPU_COMPLETE_RESULTS.json，不挑选验证赢家。OOF用于调参的最佳值有选择乐观偏差，不是独立测试。\n\n## 失败原因\n训练误差、方差保留和参数自由度均不足以单独因果归因；需要比较OOF/validation服务质量。36k评分只有600个不同题。\n\n## 下一步\n仅使用训练OOF选择的表示进入条件增益比较；后续fresh用于一次独立开发评价。\n'
        (PUBLIC/filename).write_text(text,encoding='utf8')
    text='# 条件增益路由\n\n## 研究问题\n预测同题相对训练默认组合的配对增益，是否能找出值得切换的新题。\n\n## 实验设置与对照\n默认组按训练折选，收缩伪计数10、子任务至少4题，不足回退任务/全局。同表征比较线性、默认条件交互线性和32单隐层MLP；各4个正则。阈值仅按训练OOF选择。普通线性delta重定锚与直接20回归具有严格排序等价性，作为负对照。\n\n## 数值结果\n选定配置：'+json.dumps(chosen,ensure_ascii=False)+'\n\n'+json.dumps(switch,ensure_ascii=False,indent=2)+'\n\n## 统计结论\n完整配对CI按题聚类，三个seed一起重采样。所有结果为开发诊断；主候选由train OOF锁定，没有按validation选赢家。\n\n## 失败原因与边界\n简单改成相对标签未必改变决策；阈值可能选择永不切换，不能将其称逐题优势。校准变体仅proper480训练、120校准，目标是已观测3seed均值误差，不承诺真实条件Delta同时界。\n\n## 下一步\n冻结由训练选择的主候选，先构建fresh必要调用并集，再按原三个seed开展一次评价；sealed200继续封存。\n'
    (PUBLIC/'04_conditional_gain_router.md').write_text(text,encoding='utf8')
    put(PRIVATE/'CPU_STAGE_DONE.json',{'status':'COMPLETE','selected':chosen,'completed_unix':time.time()})
    print(json.dumps({'status':'CPU_COMPLETE','selected':chosen,'val_main_accuracy':switch['conditional_main_thresholded']['accuracy'],'paired_vs_subtask':switch['conditional_main_thresholded']['vs_original_baselines']['subtask_fixed'],'cpu_s':results['actual_cpu_s'],'wall_s':results['actual_wall_s']}),flush=True)
if __name__=='__main__':main()

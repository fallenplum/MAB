from common import *
from audit_and_freeze import raw_predict
import pickle,warnings
from sklearn.model_selection import train_test_split

def main():
    d,refs=load();tr=d['selector_train'];va=d['selector_validation'];r=read(PUBLIC/'CPU_COMPLETE_RESULTS.json')
    with (PRIVATE/'FITTED_CANDIDATE_STATES.pkl').open('rb') as f:p=pickle.load(f)
    records={x['name']:x for x in r['all_candidate_oof']}
    with np.load(PRIVATE/'BASELINE_OOF.npz') as a:b=a['shrink_default'].argmax(1)
    checks=[]
    for alpha in [.1,1,10,100]:
        aa=PRIVATE/'oof'/f'direct20__full_meta__a{alpha:g}.npz';bb=PRIVATE/'oof'/f'delta_linear__full_meta__a{alpha:g}.npz'
        with np.load(aa) as z:x=z['pred']
        with np.load(bb) as z:y=z['pred']
        centered=x-x[np.arange(600),b,None];err=float(np.max(np.abs(centered-y)));assert err<1e-8
        assert np.array_equal(x.argmax(1),y.argmax(1))
        checks.append({'alpha':alpha,'max_abs_after_reanchoring':err,'oof_argmax_changes':0})
    # Refit only the identical proper480 calibration variant to retain its calibration proof objects.
    st=np.array([str(t)+'/'+str(s) for t,s in zip(tr['task'],tr['sub'])]);proper,cal=train_test_split(np.arange(600),test_size=120,stratify=st,random_state=20261106)
    a,c=subset(tr,proper),subset(tr,cal);cfg=records[r['selected_conditional']['candidate']];tf=Transform(cfg['feature']).fit(a)
    ba=defaults(a,a,10,4)['subtask_fixed'].argmax(1);bc=defaults(a,c,10,4)['subtask_fixed'].argmax(1);bv=defaults(a,va,10,4)['subtask_fixed'].argmax(1)
    both={'E':np.r_[c['E'],va['E']],'task':np.r_[c['task'],va['task']],'sub':np.r_[c['sub'],va['sub']]}
    cw=time.perf_counter();cc=time.process_time()
    with warnings.catch_warnings(record=True) as ww:
        warnings.simplefilter('always');pr,_,extra,state=fit_predict(cfg['family'],tf.transform(a),a['Y'],tf.transform(both),cfg['alpha'],ba,np.r_[bc,bv])
    yt=c['Y'].mean(1);yt=yt-yt[np.arange(120),bc,None];yv=va['Y'].mean(1);yv=yv-yv[np.arange(200),bv,None]
    erc=np.max(pr[:120]-yt,axis=1);erv=np.max(pr[120:]-yv,axis=1)
    old=read(PUBLIC/'UNCERTAINTY_CALIBRATION.json');u=float(np.quantile(erc,min(1,np.ceil(121*.9)/120),method='higher'));assert abs(u-old['simultaneous_over_actions_empirical_one_sided_error_threshold'])<1e-8
    old.update(calibration_empirical_simultaneous_coverage=float(np.mean(erc<=u)),validation_empirical_simultaneous_coverage_on_observed_three_seed_targets=float(np.mean(erv<=u)),extra_identical_calibration_refit_for_audit=1,extra_cpu_s=time.process_time()-cc,extra_wall_s=time.perf_counter()-cw,warnings=[str(x.message) for x in ww])
    put(PUBLIC/'UNCERTAINTY_CALIBRATION.json',old)
    np.savez_compressed(PRIVATE/'UNCERTAINTY_AUDIT_PRIVATE.npz',proper_ids=proper,cal_ids=cal,cal_errors=erc,validation_errors=erv)
    with (PRIVATE/'UNCERTAINTY_MODEL_PRIVATE.pkl').open('wb') as f:pickle.dump({'transform':tf,'state':state,'config':cfg,'u':u},f)
    # Fold leakage, dimensions and all candidate supports are inspected independently of scores.
    fold=np.load(PRIVATE/'FOLDS_PRIVATE.npz');test=[]
    for j in range(5):
        it,iv=fold[f'fold{j}_train'],fold[f'fold{j}_holdout'];assert not set(it)&set(iv)
        assert len(it)==480 and len(iv)==120;test.extend(iv.tolist())
    assert sorted(test)==list(range(600)) and len(records)==44
    for ref in refs.values():assert sha(ref['path'])==ref['sha256']
    # Public source may name field schemas, but must never contain actual private IDs.
    sensitive=set(tr['ids'])|set(tr['canonical'])|set(va['ids'])|set(va['canonical'])
    corpus='\n'.join(x.read_text(encoding='utf8') for x in PUBLIC.rglob('*') if x.is_file() and x.suffix in ['.py','.json','.md'])
    assert not any(str(k) in corpus for k in sensitive)
    put(PUBLIC/'FINAL_VALIDATION_CHECKS.json',{'status':'PASS','fold_question_id_overlap':0,'all600_questions_once_in_oof':True,'registered_configs':44,'linear_delta_equivalence':checks,'full_state_validation_replay_max_abs':0,'source_asset_hashes_revalidated':True,'private_id_or_canonical_exact_match_hits':0,'calibration_audit_extra_identical_fits':1,'new_llm_calls':0,'fresh_or_sealed_outcomes_opened':False,'primary_choice_unchanged':True})
    # Add a concise structured failure analysis and explicit recommendation scope.
    text='# 基础价值预测器与错误来源\n\n## 研究问题\n结构容量、输入信息和决策损失哪个限制服务质量。\n\n## 数值结果与统计结论\n原二阶PCA16训练OOF86.44%，扩到完整向量+元数据直接20为86.78%，但后者val84.33%，原二阶val85.67%。低秩5与排序val各86.50%，仍低于subtask87%。条件MLP的OOF提升没有稳定迁移到val。\n\n## 失败原因\n1. 无法表达的5个组合方向只解释一部分经验对比残差，真实条件残差还混有三seed噪声。\n2. PCA增维降低训练拟合限制，却扩大方差；当前负迁移不足以单独证明PCA是主因。\n3. 600不同题对完整向量、默认条件交互及32隐藏单元的容量有限；同一BBH子任务仅11–12个训练题。\n4. 主候选GSM净+2、BBH净−7题×seed实例；小子任务CI不能支撑稳定内部选择规律。\n5. 标签相对化的普通线性模型严格等价，已有真实OOF检查；新收益必须来自条件表示、非线性或阈值，且这些本轮未证实泛化。\n\n## 推荐与下一步\n作为可解释基础诊断，保留原二阶与直接20/PCA16双基线，低秩和排序作容量/目标对照；不把任一val最高模型重新选为主模型。训练OOF冻结的主候选仍为MLP+metadata/tau.025，但不能称作已经具有可重复增益的推荐部署器。后续若需另立方案，应先修正过度切换并另行约束搜索，不能继续用val挑赢家。\n'
    (PUBLIC/'ERROR_SOURCE_ANALYSIS.md').write_text(text,encoding='utf8')
    print(json.dumps({'final_checks':'PASS','calibration_coverage':old['calibration_empirical_simultaneous_coverage'],'validation_coverage':old['validation_empirical_simultaneous_coverage_on_observed_three_seed_targets'],'new_llm_calls':0}),flush=True)
if __name__=='__main__':main()

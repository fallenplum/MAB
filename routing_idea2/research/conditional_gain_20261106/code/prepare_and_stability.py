from common import *
import ast,sys,platform

def main():
    PRIVATE.mkdir(parents=True,exist_ok=True);PUBLIC.mkdir(parents=True,exist_ok=True)
    paths=[]
    # Byte/AST coverage is recorded separately from scientific reproducibility.
    for base in [REPO/'routing_idea2',ROOT/'protocol',ROOT/'src',ROOT/'scripts']:
        for p in sorted(base.rglob('*')):
            if p.suffix not in ['.py','.md','.txt','.json'] or any(v in p.parts for v in ['__pycache__','.git']):continue
            raw=p.read_bytes();entry={'file':str(p.relative_to(base)),'root_role':base.name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
            if p.suffix=='.py':
                try:
                    tree=ast.parse(raw.decode('utf-8-sig'));entry['functions']=[n.name for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
                except (SyntaxError,UnicodeError):entry['ast_parse']='not_parsed'
            paths.append(entry)
    put(PUBLIC/'ASSET_AND_CODE_READING_INDEX.json',{'files':paths,'boundary':'Complete byte/AST inventory, not a claim of replaying every historical experiment.'})
    d,refs=load();put(PRIVATE/'INPUT_LINEAGE.json',refs)
    tr,va=d.values()
    sys.path.insert(0,str(ROOT/'src'))
    from budget_router_v2 import full_feedback as original
    from sklearn.model_selection import StratifiedKFold
    strata=np.array([str(t)+'/'+str(s) for t,s in zip(tr['task'],tr['sub'])])
    folds=list(StratifiedKFold(5,shuffle=True,random_state=20261106).split(tr['E'],strata))
    np.savez_compressed(PRIVATE/'FOLDS_PRIVATE.npz',**{f'fold{j}_train':a for j,(a,b) in enumerate(folds)},**{f'fold{j}_holdout':b for j,(a,b) in enumerate(folds)})
    protocol={'status':'LOCKED_BEFORE_NEW_CPU_SEARCH','seed':20261106,'cv':'5fold question-grouped task/BBH-subtask stratified; all3 generations together','ridge_grid':[.1,1,10,100],
      'structure_models':['structured','direct20','lowrank5','ranknet'],'structure_feature':'pca16','feature_ablation_model':'direct20','feature_variants':['pca16','pca32','pca64','full','full_meta'],
      'conditional_models':['delta_linear','delta_default_interaction','delta_mlp'],'default_shrinkage_pseudocount':10,'default_subtask_min_questions':4,
      'threshold_grid':[0,.01,.025,.05,.1,.2,1e9],'selection':'OOF service task-macro accuracy, then lower switching rate, then stronger regularization; validation cannot select configuration',
      'search_budget':'4regularization configurations per family/representation; lowrank fixed5, MLP fixed32 hidden/max300/early-stop train-only; ranknet max150iter',
      'ranking_objective':'pairwise logistic all190pairs across3 actual seeds, tied true pairs excluded; same ridge grid but objectives differ',
      'target':'mean3 generation rewards is sufficient for squared-loss fits; original sum-observation ridge penalty preserved via alpha/3',
      'meta_access':'Provided task/subtask identifiers same as original fixed policies; train-fold-only vocabulary with task/global fallback',
      'uncertainty_variant':'proper-train480/calibration120 split; separate calibration max error across20 observed paired reward means; exploratory empirical calibration, not strict population Delta bound; do not refit on cal',
      'independent_evaluation':'fresh400 all3seeds only after all policies frozen and exact necessary call union; sealed200 excluded',
      'feedback_gate':'fresh paired95CI lower>0 vs subtask and >=2/3seeds positive, adequate support','original_main':os.environ.get('MAB_BASE_HEAD'),'no_llm_retraining':True,'prior_cpu_diagnostic_validation_seen':True}
    put(PUBLIC/'CPU_EXPERIMENT_PROTOCOL.json',protocol)
    # Real-data exact-operator qualification, independent from CV conclusions.
    olddir=ROOT/'artifacts/private/routing_redesign_20261104/ff_v1/model'
    with np.load(olddir/'model.npz',allow_pickle=False) as a:old={k:a[k] for k in a.files}
    def phi(x):return np.column_stack((np.ones(len(x)),((x-old['pca_mean'])@old['pca_components'].T-old['scale_mean'])/old['scale_scale']))
    X=phi(tr['E']);V=phi(va['E']);W=structured(X,tr['Y'].mean(axis=1),1)
    expected=V@old['theta'].reshape(17,22)@B.T;actual=V@W@B.T
    error=float(np.max(np.abs(actual-expected)));assert error<2e-6 and np.array_equal(actual.argmax(1),expected.argmax(1))
    # Basis projection and exact affine-target invariance are meaningful implementation checks.
    assert np.linalg.matrix_rank(B)==15
    base=tr['Y'].mean(axis=1)[:,0];Wa=ridge(X,tr['Y'].mean(axis=1),1);Wd=ridge(X,tr['Y'].mean(axis=1)-base[:,None],1)
    gap=float(np.max(np.abs((V@Wa-(V@Wa)[:,0,None])-(V@Wd-(V@Wd)[:,0,None]))));assert gap<1e-8
    put(PUBLIC/'NUMERICAL_QUALIFICATION.json',{'source_frozen_model_sha256':sha(olddir/'model.npz'),'max_abs_vs_original_structured_prediction':error,'validation_argmax_changes':0,'basis_rank':15,'linear_delta_reanchoring_invariance_max_abs':gap,'not_bitwise_claim':True,'new_llm_calls':0})
    out={};private={}
    for name,dd in d.items():
        Y=dd['Y'];mu=Y.mean(1);within=Y.var(1,ddof=1)
        U,sv,_=np.linalg.svd(B,full_matrices=False);proj=U[:,:15]@U[:,:15].T
        residual=mu-mu@proj;centered=mu-mu.mean(1,keepdims=True)
        fixed=defaults(tr,dd);bs={k:np.argmax(v,axis=1) for k,v in fixed.items()}
        choices=np.zeros((len(Y),3),int);paired={k:[] for k in fixed}
        for g in range(3):
            choices[:,g]=np.argmax(np.delete(Y,g,axis=1).mean(1),axis=1)
            for k,b in bs.items():paired[k].append(Y[np.arange(len(Y)),g,choices[:,g]]-Y[np.arange(len(Y)),g,b])
        checks={}
        for k,v in paired.items():checks[k]=original._cluster_difference(np.array(v),dd['task'],dd['sub'],SEEDS,2000,20261104)
        centered_Y=Y-Y.mean(axis=2,keepdims=True)
        noise=centered_Y.var(axis=1,ddof=1).sum(axis=1)/(3*19)
        signal_raw=mu.var(axis=1,ddof=1)-noise
        pairs=Y[:,:,PAIRS[:,0]]-Y[:,:,PAIRS[:,1]]
        private[name+'_expected_rewards']=mu;private[name+'_within_variance']=within;private[name+'_mean_pair_differences']=pairs.mean(1);private[name+'_pair_variance']=pairs.var(1,ddof=1)
        out[name]={'unique_questions':len(Y),'observed_post_oracle':float(Y.max(2).mean()),'same_team_mean3_plugin_oracle':float(mu.max(1).mean()),'mean_within_generation_variance':float(within.mean()),'mean_pair_difference_within_variance':float(pairs.var(1,ddof=1).mean()),'stable_contrast_variance_unclipped':float(signal_raw.mean()),'mean_reward_projection_residual_mse':float((residual**2).mean()),'residual_fraction_of_contrast_mse':float((residual**2).mean()/(centered**2).mean()),'leave_one_generation_out_quality':float(np.mean([Y[np.arange(len(Y)),g,choices[:,g]].mean() for g in range(3)])),'comparisons':checks,'subtask_support':{s:int(np.sum(dd['sub']==s)) for s in set(dd['sub']) if s},'finite_seed_oracles_are_not_population_bounds':True}
    assert abs(out['selector_validation']['leave_one_generation_out_quality']-.88)<1e-12
    assert abs(out['selector_validation']['comparisons']['subtask_fixed']['task_macro_difference']-.01)<1e-12
    put(PUBLIC/'STABILITY_AND_VARIANCE.json',out);np.savez_compressed(PRIVATE/'PER_QUESTION_STABILITY.npz',**private)
    b='# 01 组合价值稳定性与Oracle空间\n\n## 研究问题\n复核同题跨生成价值是否可迁移，并区分有限种子事后机会与输入可学习价值。\n\n## 实验设置与对照\n冻结N6/K3、Qwen3聚合器及完整真实20组合矩阵。train600/validation200、三seed42/137/2024；两个seed选队、第三seed评价；全局/任务/子任务固定基线均只由train600选择。新增LLM调用和评分0。\n\n## 数值结果\n'
    for n,r in out.items():b+=f"- {n}: leave-one-seed quality={r['leave_one_generation_out_quality']:.6f}; observed post-oracle={r['observed_post_oracle']:.6f}; mean3同组合事后最大={r['same_team_mean3_plugin_oracle']:.6f}; 题内方差均值={r['mean_within_generation_variance']:.6f}; 二阶均值投影残差占组合对比MSE={r['residual_fraction_of_contrast_mse']:.6f}。\n"
    b+='\n验证复核88.00%对87.00%，原bootstrap规则与种子重算配对CI见JSON；不将600个题×seed实例当600道独立题。\n\n## 统计结论\n差值区间跨0，未建立未见题泛化。理论链需要策略仅使用Z且生成噪声独立，完整命题见05。经验mean3最大值使用三次观测共同选择，存在max选择偏差，不能当真实V_stable。V_phi不可由唯一题目上的三次重复非参数识别，需交叉拟合策略及独立测试。\n\n## 失败原因与限制\n三seed不足以可靠识别每题条件期望；42部分历史成员与137/2024混合平台；任务内异质性、投影残差和随机性不可仅由总体方差因果拆分。完整原排序相关性及严格同向率直接引用main已有完整统计，不重做生成。\n\n## 下一步\n同一折划分比较结构与特征，仅用训练OOF选择候选。\n'
    (PUBLIC/'01_stability_and_oracle_analysis.md').write_text(b,encoding='utf8')
    put(PRIVATE/'STAGE1_DONE.json',{'status':'COMPLETE','inputs':refs,'completed_unix':time.time(),'new_llm_calls':0})
    print(json.dumps({'stage1':'COMPLETE','validation_leaveout':.88,'private':str(PRIVATE),'public':str(PUBLIC)}),flush=True)
if __name__=='__main__':main()

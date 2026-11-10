from e5_common import *
from gate_models import gate_statistics,SCORE_NAMES
from summarize_switching import question_record_bootstrap
from run_gates import native


def main():
    data,_=load_train()
    result={}
    for mode in ('task-known','subtask-oracle'):
        gates=np.zeros((4,600),dtype=bool)
        D=np.zeros((600,3))
        Rb=np.zeros((600,3))
        fold=np.zeros(600,dtype=int)
        default_actions=np.zeros(600,dtype=int)
        candidate_actions=np.zeros(600,dtype=int)
        for f in range(5):
            with np.load(PRIVATE/'metadata_gates'/f'{mode}_f{f}.npz') as a:
                q=a['E']
                gates[:,q]=a['gate_masks_E'][np.arange(4),a['main_indices']]
                D[q],Rb[q],fold[q]=a['D_E'],a['Rb_E'],f
                default_actions[q],candidate_actions[q]=a['base_E'],a['candidate_E']
        assert np.all(default_actions!=candidate_actions)
        boot=question_record_bootstrap(gates,D,data['task'],data['sub'],fold,reps=5000,seed=20261110)
        table={}
        for j,name in enumerate(SCORE_NAMES):
            summary=gate_statistics(gates[j],D,data['task'],data['sub'])
            summary={k:v for k,v in summary.items() if not k.startswith('per_question_')}
            intervals={k:{field:(val[j] if isinstance(val,np.ndarray) else val) for field,val in entry.items()}
                       for k,entry in boot['summaries'].items()}
            table[name]=dict(statistics=summary,accuracy=scope_means(Rb.mean(1)+gates[j]*D.mean(1),data['task'],data['sub']),
                intervals=intervals)
        result[mode]=dict(default_accuracy=scope_means(Rb.mean(1),data['task'],data['sub']),table=table,
            permission='default and proposer both receive identical declared metadata; no cross-permission winner selected',
            bootstrap=boot['metadata'],independent_questions=600)
        np.savez_compressed(PRIVATE/'metadata_gates'/f'{mode}_PAIRED.npz',gates=gates,D=D,Rb=Rb,fold=fold,
                            default_actions=default_actions,candidate_actions=candidate_actions)
    put(PUBLIC/'METADATA_PERMISSION_CONTROLS.json',native(dict(status='COMPLETE',table=result,
        role='separate information-permission sensitivity; primary remains text-only',new_llm_calls=0)))
    print(json.dumps({mode:dict(default=v['default_accuracy']['balanced_v2'],factorized=v['table']['factorized']['statistics']['gain']) for mode,v in result.items()}))


if __name__=='__main__':main()

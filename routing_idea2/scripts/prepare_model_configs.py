"""Write candidate runtime specs without using any benchmark performance."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from budget_router.inference import ModelSpec,configuration_fingerprint
pool=json.loads((ROOT/'artifacts/audit/candidate_model_identities.json').read_text('utf-8-sig'))
items=[]
for entry in pool:
    alias=entry['alias']
    spec=ModelSpec(model_id=entry['repo_id'],revision=entry['revision'],model_path=str(ROOT/'models'/entry['repo_id'].replace('/','--')),
                   max_input_tokens=4096,max_new_tokens=2048,temperature=0.7,top_p=0.95,quantization='nf4',enable_thinking=False)
    spec.validate()
    from dataclasses import asdict
    destination=ROOT/'configs'/f'member_{alias.lower()}.json'
    destination.write_text(json.dumps(asdict(spec),ensure_ascii=False,indent=2),encoding='utf-8')
    items.append({'index':len(items),'alias':alias,'config':destination.relative_to(ROOT).as_posix(),
                  'model_id':spec.model_id,'revision':spec.revision,'fingerprint':configuration_fingerprint(spec),
                  'historical_cache_alias':{'Granite-3.3-8b-instruct':'granite-3.3-8b-instruct','Cogito-v1-preview-llama-8B':'cogito-v1-preview-llama-8B'}.get(alias,alias)})
(ROOT/'configs/pool_v1.json').write_text(json.dumps({'status':'CANDIDATE_PENDING_E0_QUALIFICATION','models':items,
 'formal_answer_source':'BOTH_TASKS_REGENERATED_USING_THESE_PINNED_MODELS_NF4',
 'historical_cache':'Audit only. Unknown original checkpoint revisions are not assigned current model revisions.'},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'status':'candidate_runtime_configs_written','models':len(items),'scientific_completion':False}))

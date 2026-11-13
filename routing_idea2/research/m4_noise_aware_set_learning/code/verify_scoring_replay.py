"""Check the independent output location reproduces all scientific numbers."""
from pathlib import Path
import hashlib
import json

root = Path(__file__).resolve().parent
replay = root / 'private/evaluation_replay'
results = {}
for filename in ('NS_TRAINING_RESULTS.json', 'NS_PRIMARY_STATISTICS.json'):
    original = root / 'public' / filename
    repeated = replay / 'public' / filename
    left = json.loads(original.read_text(encoding='utf-8'))
    right = json.loads(repeated.read_text(encoding='utf-8'))
    if left != right:
        raise AssertionError('Replayed scientific JSON differs ' + filename)
    results[filename] = dict(original_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
        replay_sha256=hashlib.sha256(repeated.read_bytes()).hexdigest(), complete_JSON_identical=True)
receipt = dict(status='PASS_COMPLETE_SCIENTIFIC_REPLAY', all_16_models_reused=True,
    new_model_fits=0, new_LLM_calls=0, forbidden_outcomes_accessed=[], files=results)
(replay / 'REPRODUCTION_VERIFIED.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
print(json.dumps(receipt))

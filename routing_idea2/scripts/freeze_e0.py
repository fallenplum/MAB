from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from budget_router.protocol_lock import freeze_e0
lock=freeze_e0(ROOT)
print(json.dumps({'status':lock['status'],'stage':'E0','e1_questions':lock['e1_questions'],'scientific_completion':False}))

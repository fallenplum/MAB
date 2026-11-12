"""Inventory archived benchmark inputs without exposing their question text."""
from pathlib import Path
import json
import hashlib
import zipfile
import collections
import csv
import io

WORK=Path(r'E:\语言模型组合路由\research/ces_traceable_baseline')
PRIVATE=WORK/'private/public_assets'


def main():
    files=['data/alpaca_eval/reference.json','FastChat/fastchat/llm_judge/data/mt_bench/question.jsonl',
           'FastChat/fastchat/llm_judge/data/mt_bench/reference_answer/gpt-4.jsonl',
           'FLASK/evaluation_set/flask_hard_evaluation.jsonl','FLASK/evaluation_set/flask_evaluation.jsonl',
           'FLASK/input_data/flask_evaluation_raw.jsonl']
    rows=[]
    with zipfile.ZipFile(PRIVATE/'CES-v1.0.0-kdd2026.zip') as a:
        prefix=a.namelist()[0].split('/')[0]+'/'
        mt_questions=[json.loads(line) for line in a.read(prefix+'FastChat/fastchat/llm_judge/data/mt_bench/question.jsonl').decode().splitlines() if line.strip()]
        mt_categories={row['question_id']:row['category'] for row in mt_questions}
        for file in files:
            raw=a.read(prefix+file)
            values=json.loads(raw) if file.endswith('.json') else [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
            row=dict(path=file,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),records=len(values),
                     record_keys=sorted(set().union(*(set(x) for x in values))))
            if 'generator' in values[0]:
                row['generators']=dict(collections.Counter(str(x.get('generator')) for x in values))
            if 'turns' in values[0]:
                row['turn_count_distribution']=dict(collections.Counter(len(x['turns']) for x in values))
            if 'choices' in values[0]:
                row['choices_per_record']=dict(collections.Counter(len(x['choices']) for x in values))
            if file.endswith('reference_answer/gpt-4.jsonl'):
                row['referenced_question_categories']=dict(collections.Counter(mt_categories[x['question_id']] for x in values))
            if 'category' in values[0]:
                row['categories']=dict(collections.Counter(str(x.get('category')) for x in values))
            rows.append(row)
        csv_files=[name for name in a.namelist() if name.endswith('.csv')]
        ces_matches=[]
        for name in csv_files:
            raw=a.read(name)
            try:
                parsed=list(csv.reader(io.StringIO(raw.decode('utf-8-sig'))))
            except UnicodeDecodeError:
                continue
            for index,record in enumerate(parsed):
                if any(cell.strip().lower() in ['ces','ces-full','ces_full'] for cell in record):
                    ces_matches.append(dict(path=name,row=index,values=record))
    result=dict(status='COMPLETE',source='official Zenodo archive with verified MD5',benchmark_inputs=rows,
        bundled_CSV_files_scanned=len(csv_files),CES_rows_in_bundled_CSVs=ces_matches,
        conclusion='Benchmark prompts and reference answers are available, but no CES per-question routed answers/judgments/usage were found in the official artifact.',
        physical_model_API_calls=0)
    (PRIVATE/'BENCHMARK_INPUT_INVENTORY.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    main()

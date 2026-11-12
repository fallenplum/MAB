"""Read-only CES public archive inventory and benchmark input counts."""
from pathlib import Path
import hashlib
import json
import requests
import zipfile
import re

ROOT=Path(r'E:\语言模型组合路由\research\ces_traceable_baseline')
PRIVATE=ROOT/'private/public_assets'
PUBLIC=ROOT/'public'


def put(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')


def download(url,path,max_bytes=60000000):
    if not path.exists():
        temporary=path.with_suffix(path.suffix+'.downloading')
        with requests.get(url,stream=True,timeout=120,headers={'User-Agent':'MAB-CES-reproducibility-audit/1.0'}) as r:
            r.raise_for_status()
            if int(r.headers.get('content-length',0))>max_bytes:
                raise ValueError('Public asset exceeds fixed60MB audit cap')
            size=0
            path.parent.mkdir(parents=True,exist_ok=True)
            with temporary.open('wb') as f:
                for chunk in r.iter_content(1048576):
                    size+=len(chunk)
                    if size>max_bytes:
                        raise ValueError('Audit cap exceeded')
                    f.write(chunk)
        temporary.replace(path)
    raw=path.read_bytes()
    return dict(url=url,file=path.name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),md5=hashlib.md5(raw).hexdigest())


def main():
    url='https://zenodo.org/records/20364184/files/JinkunXu/CES-v1.0.0-kdd2026.zip?download=1'
    zip_path=PRIVATE/'CES-v1.0.0-kdd2026.zip'
    receipt=download(url,zip_path)
    assert receipt['md5']=='ba087d464f99dbefcebeb618b6b0b413'
    with zipfile.ZipFile(zip_path) as archive:
        items=[dict(path=i.filename,bytes=i.file_size,compressed_bytes=i.compress_size,crc32=f'{i.CRC:08x}')
               for i in archive.infolist() if not i.is_dir()]
        core=[]
        for item in items:
            parts=item['path'].split('/')
            relative='/'.join(parts[1:])
            if not re.match(r'^(FLASK|FastChat|MoA_reproduce|KABB_reproduce)/',relative):
                core.append(dict(**item,relative=relative))
        candidates=[i for i in core if re.search(r'(^|/)(runs|outputs|results|cache|caches|model_answer|model_judgment)(/|\.)',i['relative'],re.I)]
        data_files=[i for i in core if re.search(r'\.(jsonl?|csv|parquet|npz|npy|pkl|pt|pth|log)$',i['relative'])]
        details=[]
        for item in data_files:
            raw=archive.read(item['path'])
            row={**item,'sha256':hashlib.sha256(raw).hexdigest()}
            try:
                value=json.loads(raw)
                row['json_type']=type(value).__name__
                row['top_level_count']=len(value)
                if isinstance(value,list) and value:
                    row['first_record_keys']=list(value[0]) if isinstance(value[0],dict) else None
                if isinstance(value,dict):
                    row['top_level_keys']=list(value)
            except Exception:
                pass
            details.append(row)
        licenses=[i for i in core if i['relative']=='LICENSE']
        license_text=archive.read(licenses[0]['path']).decode() if licenses else None
        put(PRIVATE/'ZENODO_ARCHIVE_INVENTORY.json',dict(receipt=receipt,files=items,core_files=core))
        if license_text is not None:
            (PRIVATE/'UPSTREAM_LICENSE.txt').write_text(license_text,encoding='utf-8')
        for item in core:
            if item['relative'].startswith('scripts/') and item['relative'].endswith('.py') or item['relative'] in ['README.md','.gitignore','eval_alpaca.py','eval_mt_bench.py','show_mt_bench_result.py','show_flask.py','artifacts/mmlu_meta_10d.json']:
                target=PRIVATE/'archive_core'/item['relative']
                target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes(archive.read(item['path']))
    result=dict(status='COMPLETE',archive=receipt,archive_file_count=len(items),core_file_count=len(core),
        source_result_directories=candidates,core_data_files=details,
        archive_root_license_present=bool(licenses),
        license='Current master9e071c51 root code is MIT. The archived tag may omit root LICENSE. Bundled benchmark/baseline subtrees retain separate licenses.',
        no_original_CES_outputs_found=not candidates,
        physical_model_API_calls=0,
        scope='Official Zenodo software ZIP inventory, original benchmark inputs and profiles. No live generation/judging.')
    put(PRIVATE/'ZENODO_ASSET_SUMMARY.json',result)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    main()

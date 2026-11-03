"""Resume-safe local E0/E1 supervisor. Never interrupts another GPU job.

This process runs the full 300-question/20-team E1 only after real E0 assets.
An E1 PASS leaves the overall project PARTIAL until E2-E5; no automatic upload.
"""
from pathlib import Path
import argparse,json,os,subprocess,sys,time,traceback,uuid,hashlib,csv,re
ROOT=Path(__file__).resolve().parents[1]
PYTHON=ROOT/'.venv/Scripts/python.exe'
GPU_MIN_FREE_MIB=6200
sys.path.insert(0,str(ROOT/'src'))


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    pending=path.with_suffix(path.suffix+'.pending')
    pending.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    os.replace(pending,path)


def state(step,**fields):
    path=ROOT/'RUN_STATUS.json'
    old=json.loads(path.read_text('utf-8-sig')) if path.exists() else {}
    old.update(dict(updated_at_unix=time.time(),supervisor_pid=os.getpid(),step=step,
      overall='IN_PROGRESS',supervisor_running=True,scientific_completion=False,
      gpu_min_free_mib=GPU_MIN_FREE_MIB,
      github_upload='NOT_UPLOADED_WAITING_FOR_REQUIRED_EXPERIMENTS'))
    old.update(fields)
    old['notes']=['Local first, full original scale authorized by user',
                  'Real output/source receipts are required. Unit tests are not experiment completion.']
    atomic_json(path,old)
    print(json.dumps({'event':'state','step':step,'gpu_min_free_mib':GPU_MIN_FREE_MIB,**fields},ensure_ascii=False),flush=True)


def run(script,*args,allowed_returncodes=(0,)):
    command=[str(PYTHON),'-X','utf8','-u',str(ROOT/'scripts'/script),*[str(x) for x in args]]
    print(json.dumps({'event':'start','script':script,'args':[str(x) for x in args]},ensure_ascii=False),flush=True)
    process=subprocess.Popen(command,cwd=ROOT,env=os.environ.copy())
    while True:
        try:
            process.wait(timeout=20)
            break
        except subprocess.TimeoutExpired:
            status=json.loads((ROOT/'RUN_STATUS.json').read_text('utf-8-sig'))
            status.update(updated_at_unix=time.time(),active_child_pid=process.pid,active_script=script)
            atomic_json(ROOT/'RUN_STATUS.json',status)
    status_path=ROOT/'RUN_STATUS.json'
    if status_path.is_file():
        status=json.loads(status_path.read_text('utf-8-sig'))
        status.update(active_child_pid=None,active_script=None,last_child_script=script,last_child_exit=process.returncode,updated_at_unix=time.time())
        atomic_json(status_path,status)
    if process.returncode not in allowed_returncodes:
        raise RuntimeError(f'Child script failed: {script}; exit={process.returncode}; inspect private log')
    return process.returncode


class NetworkAssetUnavailable(RuntimeError):
    pass

def wait_download(repo,revision,*,role):
    if role not in {'aggregator','members'}:raise ValueError('Explicit download role required')
    while True:
        matches=[]
        for path in (ROOT/'artifacts/audit').glob('*_model_download.json'):
            try:receipt=json.loads(path.read_text('utf-8-sig'))
            except (json.JSONDecodeError,OSError):continue
            if receipt.get('repo_id')==repo and receipt.get('revision')==revision and receipt.get('completed_at_unix') and receipt.get('files'):
                matches.append(path)
        if len(matches)>1:raise ValueError('Duplicate completed model receipts')
        if matches:return matches[0]
        status_path=ROOT/'artifacts/audit'/f'download_{role}_status.json'
        if status_path.exists():
            try:download_status=json.loads(status_path.read_text('utf-8-sig'))
            except (json.JSONDecodeError,OSError):download_status={}
            if download_status.get('status') in {'FAILED','COMPLETED'}:
                raise NetworkAssetUnavailable(f'{role} download {download_status["status"]}; exact receipt missing for {repo}@{revision}')
        state('WAITING_PINNED_MODEL_DOWNLOAD',waiting_model=repo,E0='IN_PROGRESS',E1='NOT_STARTED')
        time.sleep(30)


def host_memory_snapshot():
    try:
        import psutil
        memory=psutil.virtual_memory()
        return {'host_available_bytes':int(memory.available),'host_total_bytes':int(memory.total),'host_memory_probe':'psutil'}
    except ImportError:
        if os.name!='nt':raise RuntimeError('A real host RAM probe is required')
        import ctypes
        from ctypes import wintypes
        class MemoryStatus(ctypes.Structure):
            _fields_=[('dwLength',wintypes.DWORD),('dwMemoryLoad',wintypes.DWORD),
              ('ullTotalPhys',ctypes.c_ulonglong),('ullAvailPhys',ctypes.c_ulonglong),
              ('ullTotalPageFile',ctypes.c_ulonglong),('ullAvailPageFile',ctypes.c_ulonglong),
              ('ullTotalVirtual',ctypes.c_ulonglong),('ullAvailVirtual',ctypes.c_ulonglong),
              ('ullAvailExtendedVirtual',ctypes.c_ulonglong)]
        memory=MemoryStatus();memory.dwLength=ctypes.sizeof(memory)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
            raise RuntimeError('Host RAM measurement failed')
        return {'host_available_bytes':int(memory.ullAvailPhys),'host_total_bytes':int(memory.ullTotalPhys),'host_memory_probe':'GlobalMemoryStatusEx'}

def gpu_snapshot():
    completed=subprocess.run(['nvidia-smi','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],
                             capture_output=True,text=True,check=True)
    rows=[x.strip() for x in completed.stdout.split('\n') if x.strip()]
    if len(rows)!=1:raise ValueError('Expected the one authorized local GPU')
    free,util=map(int,rows[0].split(','))
    # An external training process retains its GPU context during CPU/fold
    # preparation. Low utilization/free VRAM alone does not grant admission.
    completed=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader,nounits'],
                             capture_output=True,text=True,check=True)
    processes=[];other_python=set()
    for row in csv.reader(completed.stdout.split('\n'),skipinitialspace=True):
        if not row or not any(value.strip() for value in row):continue
        if len(row)!=2:raise ValueError('GPU process query returned an unparseable row')
        pid=int(row[0].strip());process_name=row[1].strip()
        executable=process_name.replace('\\','/').rsplit('/',1)[-1].lower()
        python_process=re.fullmatch(r'(?:python(?:w|[0-9]+(?:\.[0-9]+)*)?(?:\.exe)?|torchrun(?:\.exe)?)',executable) is not None
        processes.append({'pid':pid,'process_name':process_name,'recognized_python':python_process})
        if python_process and pid!=os.getpid():other_python.add(pid)
    return {'free_mib':free,'utilization_percent':util,'compute_processes':processes,
            'other_python_gpu_pids':sorted(other_python),**host_memory_snapshot()}


def wait_gpu():
    # Two independent consecutive observations. This is resource admission, not
    # an inference-throughput estimate or permission to stop another process.
    streak=0
    while streak<2:
        sample=gpu_snapshot()
        okay=(sample['free_mib']>=GPU_MIN_FREE_MIB and sample['utilization_percent']<=20
              and sample['host_available_bytes']>=5*1024**3
              and not sample.get('other_python_gpu_pids',[]))
        streak=streak+1 if okay else 0
        state('WAITING_LOCAL_GPU_ADMISSION' if not okay else 'CHECKING_LOCAL_GPU_ADMISSION',gpu=sample)
        if streak<2:time.sleep(30)



def qualification_cache_complete(jobs_path,output_path,config_path,*,member=False):
    """Skip resource admission only after verifying every immutable qualification row.

    The ordinary CLI and independent runtime review still run afterward. Missing
    output is ordinary pending work; malformed or mismatched evidence raises.
    """
    jobs_path,output_path,config_path=map(Path,(jobs_path,output_path,config_path))
    if not output_path.is_file():return False
    if output_path.with_name(output_path.name+'.lock').exists():
        raise RuntimeError('Qualification execution is locked; reconcile before cache reuse')
    from budget_router.inference import ModelSpec
    from budget_router.execution import ExecutionStore,validate_job
    from budget_router.member_execution import MemberStore,validate_member_job,validate_resample_groups
    spec=ModelSpec(**json.loads(config_path.read_text('utf-8-sig')));spec.validate()
    jobs=[json.loads(line) for line in jobs_path.read_text('utf-8-sig').split('\n') if line.strip()]
    if not jobs or any(job.get('split')!='qualification' for job in jobs):
        raise ValueError('Cache admission is limited to nonempty qualification batches')
    ids=[]
    for job in jobs:
        ids.append(validate_member_job(job,spec) if member else validate_job(job,spec,require_token_policy=True))
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate qualification job IDs')
    if member:validate_resample_groups(jobs)
    store=MemberStore(output_path) if member else ExecutionStore(output_path)
    if store.unreconciled:raise RuntimeError('Unknown failed qualification cost requires reconciliation')
    if set(store.completed)-set(ids):raise ValueError('Unrelated successful output in qualification ledger')
    cached=[store._cached(job,spec) for job in jobs]
    return all(row is not None for row in cached)


def qualification_gpu_admission(jobs,output,config,*,member=False):
    if qualification_cache_complete(jobs,output,config,member=member):
        print(json.dumps({'event':'reuse_verified_qualification_cache','output':str(output),
                          'gpu_min_free_mib':GPU_MIN_FREE_MIB},ensure_ascii=False),flush=True)
        return
    wait_gpu()


def gpu_min_free_mib_argument(value):
    if not re.fullmatch(r'[1-9][0-9]*',str(value)) or int(value)>1024*1024:
        raise argparse.ArgumentTypeError('GPU free-memory threshold must be a positive integer MiB, at most 1048576')
    return int(value)


def runtime_review(jobs,output,config,receipt):
    run('review_runtime.py','--jobs',jobs,'--outputs',output,'--model-config',config,'--receipt',receipt)


def verify_e0_after_child():
    from budget_router.inference import ModelSpec,configuration_fingerprint
    def sha(path):
        h=hashlib.sha256()
        with Path(path).open('rb') as stream:
            for chunk in iter(lambda:stream.read(1048576),b''):h.update(chunk)
        return h.hexdigest()
    path=ROOT/'protocol/E0_LOCK.json'
    lock=json.loads(path.read_text('utf-8-sig'))
    if (lock.get('status'),lock.get('stage'),lock.get('N'),lock.get('K'),lock.get('e1_questions'))!=('PASS','E0',6,3,300):
        raise ValueError('Freeze child did not leave a full-scale E0 PASS lock')
    for field,relative in [('pool_config_sha256','configs/pool_v1.json'),
                           ('experiment_config_sha256','configs/experiment_v1.json'),
                           ('split_manifest_sha256','data/derived/split_manifest.json')]:
        if lock.get(field)!=sha(ROOT/relative):raise ValueError('E0 lock current configuration/source binding mismatch')
    agg=ModelSpec(**json.loads((ROOT/'configs/aggregator_v1.json').read_text('utf-8-sig')))
    if lock.get('aggregator_fingerprint')!=configuration_fingerprint(agg):raise ValueError('E0 aggregator configuration differs')
    if len(lock.get('qualification_receipts',[]))!=7 or len(lock.get('member_outputs',[]))!=6 or not lock.get('source_code'):
        raise ValueError('E0 lock has incomplete real source/qualification inventories')
    for field in ('qualification_receipts','member_outputs','source_code'):
        for item in lock[field]:
            target=(ROOT/item['path']).resolve()
            if not target.is_relative_to(ROOT.resolve()) or sha(target)!=item['sha256']:
                raise ValueError('E0 locked asset missing or changed')
    return lock

def evaluate_e1_receipt():
    destination=ROOT/'results/e1/e1_diagnostics.json'
    destination.parent.mkdir(parents=True,exist_ok=True)
    candidate=destination.with_name('e1_diagnostics.'+uuid.uuid4().hex+'.pending.json')
    preserve_for_review=False
    try:
        run('evaluate_real_e1.py','--root',ROOT,'--output',candidate,allowed_returncodes=(0,2))
        if not candidate.is_file():raise ValueError('Evaluation produced no scientific receipt')
        result=json.loads(candidate.read_text('utf-8-sig'))
        if result.get('status') not in {'PASS','STOP_E1','PARTIAL'}:
            raise ValueError('Unrecognized E1 scientific status')
        if destination.exists():
            previous=json.loads(destination.read_text('utf-8-sig'))
            if previous!=result:
                preserve_for_review=True
                raise ValueError('Existing immutable E1 receipt differs from current evidence; retain pending review receipt')
        else:os.replace(candidate,destination)
        return result
    finally:
        if not preserve_for_review:candidate.unlink(missing_ok=True)

def main():
    global GPU_MIN_FREE_MIB
    parser=argparse.ArgumentParser();parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--continue-online-after-e1-pass',action='store_true',
                        help='After exact E1 PASS, prepare real online matrices; does not execute E2 validation')
    parser.add_argument('--gpu-min-free-mib',type=gpu_min_free_mib_argument,default=6200,
                        help='Scheduling-only minimum free GPU MiB; does not change model execution')
    args=parser.parse_args()
    GPU_MIN_FREE_MIB=args.gpu_min_free_mib
    if args.dry_run:
        print(json.dumps({'mode':'dry_run_no_model_calls','gpu_min_free_mib':GPU_MIN_FREE_MIB,'E1_questions':300,'models':6,'K':3,'teams':20,
          'main_member_calls':1800,'team_aggregation_calls':6000,'control_new_member_calls':600,
          'control_aggregation_calls':400,'total_E1_model_calls_excluding_E0':8800,
          'scope':'E0 then E1, E2-E5 remain explicitly gated; no upload'},ensure_ascii=False));return
    lock=ROOT/'logs/local_e1_supervisor.lock';lock.parent.mkdir(parents=True,exist_ok=True)
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd,json.dumps({'pid':os.getpid(),'started_at_unix':time.time()}).encode());os.close(fd)
    try:
        cache_paths={'HF_HOME':ROOT/'.cache/huggingface','TORCH_HOME':ROOT/'.cache/torch',
                     'CUDA_CACHE_PATH':ROOT/'.cache/cuda','XDG_CACHE_HOME':ROOT/'.cache',
                     'TMP':ROOT/'.cache/tmp','TEMP':ROOT/'.cache/tmp'}
        for path in set(cache_paths.values()):path.mkdir(parents=True,exist_ok=True)
        os.environ.update({key:str(path) for key,path in cache_paths.items()})
        os.environ.update(HF_HUB_DISABLE_XET='1',TOKENIZERS_PARALLELISM='false')
        pool=json.loads((ROOT/'configs/pool_v1.json').read_text('utf-8-sig'))['models']
        aggregator=json.loads((ROOT/'configs/aggregator_v1.json').read_text('utf-8-sig'))
        state('PREPARING_DISJOINT_QUALIFICATION_JOBS',E0='IN_PROGRESS',E1='NOT_STARTED')
        run('prepare_jobs.py','--kind','members','--stage','qualification')
        run('prepare_jobs.py','--kind','members','--stage','e1')
        wait_download(aggregator['model_id'],aggregator['revision'],role='aggregator')
        run('prepare_jobs.py','--kind','aggregations','--stage','qualification','--historical-qualification')
        jobs=ROOT/'data/jobs/qualification/historical_runtime_qualification.jsonl'
        output=ROOT/'results/raw/qualification/aggregator.jsonl'
        config=ROOT/'configs/aggregator_v1.json';receipt=ROOT/'artifacts/qualification/aggregator.json'
        qualification_gpu_admission(jobs,output,config);state('E0_AGGREGATOR_REAL_QUALIFICATION')
        run('run_aggregation.py','--jobs',jobs,'--model-config',config,'--output',output,'--qualification','--max-jobs','4')
        runtime_review(jobs,output,config,receipt)
        # Qualify EVERY pool member before any formal generation, so a protocol
        # failure cannot be patched using E1 scores or only for one model.
        for entry in pool:
            wait_download(entry['model_id'],entry['revision'],role='members')
            alias=entry['alias'].lower();config=ROOT/entry['config']
            jobs=ROOT/'data/jobs/qualification'/f'members_{alias}.jsonl'
            output=ROOT/'results/raw/qualification'/f'members_{alias}.jsonl'
            receipt=ROOT/'artifacts/qualification'/f'member_{alias}.json'
            qualification_gpu_admission(jobs,output,config,member=True);state('E0_MEMBER_REAL_QUALIFICATION',active_model=entry['alias'])
            run('run_members.py','--jobs',jobs,'--model-config',config,'--output',output,'--qualification','--max-jobs','8')
            runtime_review(jobs,output,config,receipt)
        for entry in pool:
            alias=entry['alias'].lower();wait_gpu();state('E0_COLLECTING_FULL_REAL_MEMBER_ANSWERS',active_model=entry['alias'])
            run('run_members.py','--jobs',ROOT/'data/jobs/e1'/f'members_{alias}.jsonl',
                '--model-config',ROOT/entry['config'],'--output',ROOT/'results/raw/e1'/f'members_{alias}.jsonl',
                '--qualification-receipt',ROOT/'artifacts/qualification'/f'member_{alias}.json')
        run('freeze_e0.py');verify_e0_after_child();state('E0_REAL_ASSETS_LOCKED',E0='PASS',E1='PREPARING')
        run('prepare_jobs.py','--kind','aggregations','--stage','e1')
        run('prepare_jobs.py','--kind','alone','--stage','e1')
        for jobname,outputname in [('aggregations','aggregations'),('aggregator_alone','aggregator_alone')]:
            wait_gpu();state('E1_FULL_REAL_AGGREGATION',E1='IN_PROGRESS',active_batch=jobname)
            run('run_aggregation.py','--jobs',ROOT/'data/jobs/e1'/f'{jobname}.jsonl',
                '--model-config',ROOT/'configs/aggregator_v1.json','--output',ROOT/'results/raw/e1'/f'{outputname}.jsonl',
                '--qualification-receipt',ROOT/'artifacts/qualification/aggregator.json')
        run('prepare_e1_controls.py','--kind','samples')
        selection=json.loads((ROOT/'artifacts/audit/e1_strong_model_selection.json').read_text('utf-8-sig'))
        strong=selection['model_alias'].lower()
        wait_gpu();state('E1_STRONG_MODEL_THREE_INDEPENDENT_SAMPLES',active_model=selection['model_alias'])
        run('run_members.py','--jobs',ROOT/'data/jobs/e1/strong_samples.jsonl',
            '--model-config',ROOT/'configs/e1_strong_sampling.json','--output',ROOT/'results/raw/e1/strong_samples.jsonl',
            '--qualification-receipt',ROOT/'artifacts/qualification'/f'member_{strong}.json')
        run('prepare_e1_controls.py','--kind','aggregate')
        wait_gpu();state('E1_REAL_STRONG_SAMPLE_AGGREGATION')
        run('run_aggregation.py','--jobs',ROOT/'data/jobs/e1/strong_aggregations.jsonl',
            '--model-config',ROOT/'configs/aggregator_v1.json','--output',ROOT/'results/raw/e1/strong_aggregations.jsonl',
            '--qualification-receipt',ROOT/'artifacts/qualification/aggregator.json')
        state('E1_SCORING_COMPLETE_REAL_ARTIFACTS')
        result=evaluate_e1_receipt()
        outcome=result['status']
        state('E1_DIAGNOSTICS_RECORDED',E0='PASS',E1=outcome,
          E2='READY_FOR_E2_REAL_ASSETS' if outcome=='PASS' else 'STOPPED_OR_GATED',
          E3='GATED_BY_E2',E4='GATED_BY_E3',E5='GATED_BY_E4')
        if args.continue_online_after_e1_pass and outcome=='PASS':
            from budget_router.online_continuation import continue_online_assets
            continue_online_assets(ROOT,run_script=run,gpu_admission=wait_gpu,update_state=state)
        final=json.loads((ROOT/'RUN_STATUS.json').read_text('utf-8-sig'))
        final['overall']='STOP_E1' if outcome=='STOP_E1' else 'PARTIAL'
        final.update(supervisor_finished_at_unix=time.time(),supervisor_running=False,active_child_pid=None,active_script=None)
        atomic_json(ROOT/'RUN_STATUS.json',final)
    except Exception as error:
        previous=json.loads((ROOT/'RUN_STATUS.json').read_text('utf-8-sig')) if (ROOT/'RUN_STATUS.json').exists() else {}
        stopped='STOPPED_NETWORK_ASSET' if isinstance(error,NetworkAssetUnavailable) else 'STOPPED_FOR_REVIEW'
        prior_e1=previous.get('E1','NOT_STARTED')
        state(stopped,E0=previous.get('E0','NOT_STARTED'),failure_type=type(error).__name__,
              failure_summary=str(error),E1=prior_e1 if prior_e1 in {'PASS','PARTIAL','STOP_E1'} else 'NOT_COMPLETE',
              overall=stopped,supervisor_running=False,active_child_pid=None,active_script=None)
        traceback.print_exc();raise
    finally:
        lock.unlink(missing_ok=True)

if __name__=='__main__':main()

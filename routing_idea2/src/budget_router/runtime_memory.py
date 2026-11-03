"""Scheduler-only CUDA allocator housekeeping at two completed safe boundaries.

No torch import, model change, RNG operation, explicit synchronize call,
retry, score or reward update occurs here. empty_cache may wait for underlying
CUDA events or frees; measured host CPU/wall preserves that overhead.
Cached/failed rows return before touching a backend.
"""
from pathlib import Path
import hashlib
import json
import os
import time

HELPER_SOURCE_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class RuntimeMemoryHousekeepingError(RuntimeError):
    """Stop and audit at a safe boundary; never repeat a durable successful call."""


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def _append_audit(path,record):
    path.parent.mkdir(parents=True,exist_ok=True)
    payload=(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n').encode('utf8')
    descriptor=os.open(path,os.O_CREAT|os.O_WRONLY|os.O_APPEND)
    try:
        offset=0
        while offset<len(payload):
            written=os.write(descriptor,payload[offset:])
            if written<=0:raise OSError('Audit append made no progress')
            offset+=written
        os.fsync(descriptor)
    finally:os.close(descriptor)


def _allocator(cuda):
    allocated=int(cuda.memory_allocated());reserved=int(cuda.memory_reserved())
    if allocated<0 or reserved<0:raise ValueError('Invalid allocator measurements')
    return {'allocated_bytes':allocated,'reserved_bytes':reserved}


def cleanup_after_committed_success(backend,row,*,cached,project_root,output_path,runner_name,runner_source_sha256=None,phase="post_persisted_success"):
    """Default phase is after Store.run returns its fsynced fresh-success row.

    Initialization uses the dedicated wrapper after backend construction and
    before any generation; it has no job or successful-generation record.

    Returns an overhead event (or None for cached/non-success). The separate
    JSONL audit records allocator-only measurements and cleanup CPU/wall; batch
    overhead also includes audit serialization/fsync. Cleanup failures are
    recorded without exception text and stop before another generation.
    """
    if cached:return None
    if phase not in {'post_persisted_success','post_backend_initialization'}:raise RuntimeMemoryHousekeepingError('Unknown safe housekeeping boundary')
    initializing=phase=='post_backend_initialization'
    if initializing:
        if backend is None or row is not None:raise RuntimeMemoryHousekeepingError('Initialization cleanup requires a returned backend and no generation row')
        row={}
    elif not isinstance(row,dict) or row.get('status')!='success':return None
    total_cpu=time.process_time();total_wall=time.perf_counter()
    if not initializing:
        if row.get('cost_status')!='recorded_actual_usage':
            raise RuntimeMemoryHousekeepingError('Success cost is not settled; do not retry the recorded model call')
        original={key:value for key,value in row.items() if key!='record_sha256'}
        if row.get('record_sha256')!=_digest(original):
            raise RuntimeMemoryHousekeepingError('A durable unchanged Store success record is required before housekeeping')
    root=Path(project_root).resolve()
    output=Path(output_path).resolve();source=output.relative_to(root).as_posix() if output.is_relative_to(root) else None
    record={'schema':'runtime_memory_housekeeping_v1','observed_at_unix':time.time(),'pid':os.getpid(),
        'phase':phase,'boundary':('Backend construction returned successfully; no generate has started' if initializing else 'backend.generate returned; Store success append+fsync returned and its paid-call context exited'),
        'source_ledger':source,'runner':Path(runner_name).name,'runner_source_sha256':runner_source_sha256,
        'housekeeping_source_sha256':HELPER_SOURCE_SHA256,'generation_record_sha256':row.get('record_sha256'),
        'job_anonymous':None if initializing else 'sha256:'+hashlib.sha256(('runtime-memory-job-v1:'+str(row.get('job_id'))).encode()).hexdigest(),
        'attempt_id':row.get('attempt_id'),'cleanup_performed':False,'allocator_before':None,'allocator_after':None,
        'status':'NOT_STARTED','cleanup_process_time_s':None,'cleanup_wall_time_s':None,
        'memory_scope':'PyTorch allocated/reserved allocator counters only. Other WDDM/process residency is not classified as cache; no total physical GPU-memory reduction is inferred.',
        'timing_scope':'Measured host process CPU/wall for allocator queries and empty_cache; audit append/fsync overhead is separate in the returned batch event. Not GPU-kernel timing.',
        'generation_result_or_reward_modified':False,'model_rng_seed_or_token_caps_modified':False,'retry_permitted':False}
    if initializing:record['job_id']=None
    result=row.get('result',{})
    if initializing:
        spec=getattr(backend,'spec',None)
        result={'model_id':getattr(spec,'model_id',None),'model_revision':getattr(spec,'revision',None)}
    if isinstance(result,dict):
        for key in ('model_id','model_revision'):
            if isinstance(result.get(key),str):record[key]=result[key][:240]
    error_type=None;cleanup_cpu=time.process_time();cleanup_wall=time.perf_counter()
    try:
        torch=getattr(backend,'torch',None)
        if torch is None:record['status']='BACKEND_WITHOUT_TORCH_NOT_APPLICABLE'
        else:
            cuda=getattr(torch,'cuda',None)
            if cuda is None or not cuda.is_available():record['status']='CUDA_UNAVAILABLE_NOT_PERFORMED'
            else:
                record['allocator_before']=_allocator(cuda)
                cuda.empty_cache()
                record['cleanup_performed']=True
                record['allocator_after']=_allocator(cuda)
                record['status']='CUDA_ALLOCATOR_CACHE_RELEASE_COMPLETED'
    except Exception as error:
        error_type=type(error).__name__;record.update(status='HOUSEKEEPING_FAILED_BEFORE_FIRST_GENERATION' if initializing else 'HOUSEKEEPING_FAILED_AFTER_DURABLE_SUCCESS',error_type=error_type)
    record['cleanup_process_time_s']=time.process_time()-cleanup_cpu
    record['cleanup_wall_time_s']=time.perf_counter()-cleanup_wall
    record['record_sha256']=_digest(record)
    audit_path=root/'artifacts/audit/runtime_memory_housekeeping.jsonl';append_cpu=time.process_time();append_wall=time.perf_counter()
    try:_append_audit(audit_path,record)
    except Exception:
        raise RuntimeMemoryHousekeepingError('Initialization housekeeping audit could not be persisted; no generation has started' if initializing else 'Housekeeping audit could not be persisted; the successful model row must not be retried') from None
    event={key:record[key] for key in ('phase','status','cleanup_performed','allocator_before','allocator_after','cleanup_process_time_s','cleanup_wall_time_s')}
    event.update(audit_record_sha256=record['record_sha256'],audit_path=audit_path.relative_to(root).as_posix(),
        audit_append_process_time_s=time.process_time()-append_cpu,audit_append_wall_time_s=time.perf_counter()-append_wall,
        total_housekeeping_process_time_s=time.process_time()-total_cpu,total_housekeeping_wall_time_s=time.perf_counter()-total_wall)
    if error_type is not None:
        raise RuntimeMemoryHousekeepingError('CUDA housekeeping failed before the first generation; audit locally before any model job' if initializing else 'CUDA housekeeping failed after a durable model success; audit locally and do not retry that call') from None
    return event


def cleanup_after_backend_initialization(backend,*,project_root,output_path,runner_name,runner_source_sha256=None):
    """Only after a complete returned backend, before its first generate call."""
    return cleanup_after_committed_success(backend,None,cached=False,project_root=project_root,output_path=output_path,
        runner_name=runner_name,runner_source_sha256=runner_source_sha256,phase='post_backend_initialization')


class RuntimeMemorySummary:
    def __init__(self):
        self.records=0;self.success_records=0;self.initialization_records=0;self.cleanup_calls=0;self.not_applicable=0
        self.totals={key:0. for key in ('cleanup_process_time_s','cleanup_wall_time_s','audit_append_process_time_s',
            'audit_append_wall_time_s','total_housekeeping_process_time_s','total_housekeeping_wall_time_s')}
    def observe(self,event):
        if event is None:return
        self.records+=1;self.success_records+=int(event['phase']=='post_persisted_success');self.initialization_records+=int(event['phase']=='post_backend_initialization')
        self.cleanup_calls+=int(event['cleanup_performed']);self.not_applicable+=int(not event['cleanup_performed'])
        for key in self.totals:self.totals[key]+=event[key]
    def as_dict(self):
        return {'schema':'runtime_memory_housekeeping_batch_v1','fresh_success_records_observed':self.success_records,'post_backend_initialization_records':self.initialization_records,'housekeeping_records':self.records,
            'cuda_empty_cache_calls':self.cleanup_calls,'not_applicable_records':self.not_applicable,**self.totals,
            'audit':'artifacts/audit/runtime_memory_housekeeping.jsonl','housekeeping_source_sha256':HELPER_SOURCE_SHA256,
            'scope':'Only new post-initialization and per-fresh-success housekeeping overhead, including its audit append. Separate from inference wall time and pre-existing model teardown; never reward or model execution cost.'}

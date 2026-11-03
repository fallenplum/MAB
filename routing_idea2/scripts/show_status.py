"""Read-only factual status; never infers research success from GPU memory."""
from pathlib import Path
from collections import Counter
import json,subprocess,os
import psutil
ROOT=Path(__file__).resolve().parents[1]

def read(path):
 try:return json.loads(Path(path).read_text('utf-8-sig'))
 except FileNotFoundError:return None

def observe_process(pid,expected_script,*,recorded_start=None,recorded_update=None,required_args=(),process_factory=None):
 """Verify a process identity, including creation time to reject reused PIDs.

 Older records lack exact process creation time. A process created after the
 record's last update cannot own that record, even if its script name matches.
 """
 if isinstance(pid,bool) or not isinstance(pid,int) or pid<=0:
  return {'status':'PROCESS_NOT_RECORDED','alive':False,'identity_verified':False}
 factory=process_factory or psutil.Process
 try:
  process=factory(pid)
  created=process.create_time()
  command=process.cmdline()
  process_status=process.status()
  if process_status==psutil.STATUS_ZOMBIE:
   return {'status':'PROCESS_NOT_FOUND','pid':pid,'alive':False,'identity_verified':False}
  expected=os.path.normcase(os.path.abspath(expected_script))
  observed_scripts=[]
  for arg in command:
   if not isinstance(arg,str) or not arg.lower().endswith('.py'):continue
   location=arg if os.path.isabs(arg) else os.path.join(process.cwd(),arg)
   observed_scripts.append(os.path.normcase(os.path.abspath(location)))
  args_match=all(any(command[i:i+len(group)]==list(group) for i in range(len(command))) for group in required_args)
  if expected not in observed_scripts or not args_match:
   return {'status':'PROCESS_IDENTITY_MISMATCH','pid':pid,'alive':True,'identity_verified':False,'observed_create_time':created}
  # The legacy lock start timestamp was written just after process creation;
  # allow a two-second timestamp/granularity tolerance for that format.
  if recorded_start is not None and abs(created-float(recorded_start))>2:
   return {'status':'PROCESS_START_TIME_MISMATCH','pid':pid,'alive':True,'identity_verified':False,'observed_create_time':created}
  if recorded_update is not None and created>float(recorded_update)+1:
   return {'status':'PROCESS_NEWER_THAN_RECORD','pid':pid,'alive':True,'identity_verified':False,'observed_create_time':created}
  suspended=process_status==psutil.STATUS_STOPPED
  return {'status':'VERIFIED_SUSPENDED' if suspended else 'VERIFIED_RUNNING','pid':pid,
          'alive':True,'identity_verified':True,'execution_active':not suspended,
          'process_status':process_status,'observed_create_time':created}
 except psutil.NoSuchProcess:
  return {'status':'PROCESS_NOT_FOUND','pid':pid,'alive':False,'identity_verified':False}
 except (psutil.AccessDenied,PermissionError):
  return {'status':'PROCESS_UNVERIFIABLE','pid':pid,'alive':None,'identity_verified':False}


def effective_record(record,*,pid_field,status_field,script,active_statuses,required_args=(),recorded_start=None,process_factory=None):
 """Return a view; never rewrite the underlying status or an execution ledger."""
 if record is None:return None
 result=dict(record)
 recorded=record.get(status_field)
 observation=observe_process(record.get(pid_field),ROOT/'scripts'/script,
   recorded_start=recorded_start or record.get('process_create_time') or record.get('started_at_unix'),
   recorded_update=record.get('updated_at_unix'),required_args=required_args,process_factory=process_factory)
 result['recorded_status']=recorded
 result['process_observation']=observation
 active=recorded in active_statuses or (pid_field=='supervisor_pid' and record.get('supervisor_running') is True)
 if active and not observation['identity_verified']:
  effective='UNVERIFIED_PROCESS_STATE' if observation['alive'] is None else 'STOPPED_'+observation['status']
 elif active and observation.get('execution_active') is False:
  effective='PAUSED_PROCESS_SUSPENDED'
 else:effective=recorded
 result['effective_status']=effective
 result[status_field]=effective
 if pid_field=='supervisor_pid':
  result['recorded_supervisor_running']=record.get('supervisor_running')
  result['supervisor_running']=bool(active and observation['identity_verified'] and observation.get('execution_active',False)) if observation['alive'] is not None else None
 return result


def main():
 lock=read(ROOT/'logs/local_e1_supervisor.lock') or {}
 run=read(ROOT/'RUN_STATUS.json')
 start=lock.get('started_at_unix') if run and lock.get('pid')==run.get('supervisor_pid') else None
 report={'run':effective_record(run,pid_field='supervisor_pid',status_field='overall',
   script='run_local_e1.py',active_statuses={'IN_PROGRESS','IN_PROGRESS_TECHNICAL_RECOVERY'},recorded_start=start),
   'downloads':{},'actual_model_calls':{}}
 for role in ('aggregator','members'):
  recorded=read(ROOT/'artifacts/audit'/f'download_{role}_status.json')
  report['downloads'][role]=effective_record(recorded,pid_field='wrapper_pid',status_field='status',
    script='download_with_retry.py',active_statuses={'RUNNING','RETRYING'},required_args=(('--role',role),))
 for stage in ('qualification','e1','online_calibration','online_stream'):
  counts=Counter();partial=0
  for file in (ROOT/'results/raw'/stage).glob('*.jsonl'):
   with file.open(encoding='utf-8-sig') as f:
    for line in f:
     if not line.strip():continue
     try:row=json.loads(line)
     except json.JSONDecodeError:partial+=1;continue
     counts[row.get('status','unknown')]+=1
  report['actual_model_calls'][stage]={'status_counts':dict(counts),'incomplete_lines_observed':partial}
 try:
  sample=subprocess.run(['nvidia-smi','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=8)
  free,util=map(int,sample.stdout.strip().split(','));report['gpu_observation']={'free_mib':free,'utilization_percent':util,'does_not_prove_this_experiment_is_running':True}
 except Exception:report['gpu_observation']={'status':'UNAVAILABLE'}
 for stage in ('e1',):
  result=read(ROOT/f'results/{stage}/{stage}_diagnostics.json')
  report[stage+'_result']=None if result is None else {k:result.get(k) for k in ('status','n_total_e1','n_combinations','missing_requirements','research_complete')}
 print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()

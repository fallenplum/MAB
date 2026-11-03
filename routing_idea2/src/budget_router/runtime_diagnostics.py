"""Private local backend-init diagnostics. Never records remote provider bodies."""
from __future__ import annotations
from pathlib import Path
from importlib import metadata
import json,os,sys,time,traceback,uuid
from .inference import configuration_fingerprint

PACKAGES={'torch':'torch','transformers':'transformers','accelerate':'accelerate',
          'bitsandbytes':'bitsandbytes','huggingface-hub':'huggingface_hub',
          'tokenizers':'tokenizers','safetensors':'safetensors'}


class BackendInitializationError(RuntimeError):
    """Console-safe failure; the original local traceback stays private."""
    def __init__(self,diagnostic_path=None,write_error_type=None):
        super().__init__('local_backend_initialization_failed')
        self.diagnostic_path=diagnostic_path
        self.write_error_type=write_error_type

    def public_record(self):
        result={'status':'blocked','reason_code':'local_backend_initialization_failed',
                'diagnostic_path':self.diagnostic_path,'generation_calls_started':0,
                'scientific_completion':False}
        if self.write_error_type:result['diagnostic_write_error_type']=self.write_error_type
        return result


def _runtime_inventory():
    result={}
    for distribution,module_name in PACKAGES.items():
        try:
            dist=metadata.distribution(distribution)
            item={'version':dist.version,'distribution_path':str(Path(dist.locate_file('')).resolve())}
        except metadata.PackageNotFoundError:item={'version':None,'distribution_path':None}
        loaded=sys.modules.get(module_name)
        item['loaded_module_path']=str(getattr(loaded,'__file__',None)) if getattr(loaded,'__file__',None) else None
        result[distribution]=item
    return result


def make_backend_with_diagnostics(spec,*,project_root,factory):
    """Wrap construction only, before caller can issue any generate/store call.

    OpenAI-compatible and other nonlocal backends retain their original error
    path: no provider exception body or environment value is persisted here.
    """
    if spec.backend!='transformers':return factory(spec)
    started_at=time.time();started=time.perf_counter()
    try:return factory(spec)
    except Exception as error:
        wall=time.perf_counter()-started
        original_traceback=traceback.format_exc()
        root=Path(project_root).resolve()
        diagnostic_path=None;write_error_type=None
        try:
            folder=(root/'artifacts/private/runtime_failures').resolve()
            if not folder.is_relative_to(root):raise ValueError('Private diagnostics escaped project root')
            folder.mkdir(parents=True,exist_ok=True)
            token=uuid.uuid4().hex
            target=folder/f'backend_initialization_{int(started_at)}_{token}.json'
            pending=target.with_suffix('.json.pending')
            try:fingerprint=configuration_fingerprint(spec)
            except Exception:fingerprint=None
            payload={'schema_version':1,'scope':'PRIVATE_LOCAL_BACKEND_INITIALIZATION_ONLY',
                'status':'BACKEND_INITIALIZATION_FAILED','model_fingerprint':fingerprint,
                'model_id':spec.model_id,'model_revision':spec.revision,'backend':spec.backend,
                'model_path':spec.model_path,'generation_calls_started':0,'scientific_completion':False,
                'started_at_unix':started_at,'failed_at_unix':time.time(),
                'initialization_wall_seconds':wall,'exception_type':type(error).__name__,
                'traceback':original_traceback,'runtime':_runtime_inventory(),
                'python_executable':sys.executable,'python_prefix':sys.prefix,'python_base_prefix':sys.base_prefix}
            with pending.open('x',encoding='utf-8') as stream:
                stream.write(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False));stream.flush();os.fsync(stream.fileno())
            os.replace(pending,target)
            diagnostic_path=target.relative_to(root).as_posix()
        except Exception as write_error:
            write_error_type=type(write_error).__name__
        raise BackendInitializationError(diagnostic_path,write_error_type) from None

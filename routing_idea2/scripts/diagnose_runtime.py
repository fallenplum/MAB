"""Private technical diagnosis. No generation, scoring, E0 or E1 acceptance."""
from __future__ import annotations
from pathlib import Path
from importlib.metadata import version
import argparse,json,os,sys,time,traceback,threading
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--load-model',action='store_true',help='Explicitly load the local aggregator; never call generate')
    parser.add_argument('--model-config',type=Path,default=ROOT/'configs/aggregator_v1.json')
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/audit/runtime_diagnosis.json')
    args=parser.parse_args()
    for key,part in {'HF_HOME':'.cache/huggingface','TORCH_HOME':'.cache/torch','CUDA_CACHE_PATH':'.cache/cuda',
                     'XDG_CACHE_HOME':'.cache','TMP':'.cache/tmp','TEMP':'.cache/tmp'}.items():
        location=ROOT/part;location.mkdir(parents=True,exist_ok=True);os.environ[key]=str(location)
    os.environ.update(HF_HUB_DISABLE_XET='1',TOKENIZERS_PARALLELISM='false')
    os.chdir(ROOT)
    report={'scope':'TECHNICAL_ONLY_NO_GENERATION_OR_SCIENTIFIC_ACCEPTANCE','scientific_completion':False,
            'mode':'local_model_load_only' if args.load_model else 'imports_and_quantizer_environment_only',
            'started_at_unix':time.time(),'python_executable':sys.executable,'utf8_mode':sys.flags.utf8_mode,
            'packages':{name:version(name) for name in ['torch','transformers','accelerate','bitsandbytes']},
            'phases':[],'thread_errors':[],'model_generate_calls':0}
    previous=threading.excepthook
    def thread_failure(event):
        report['thread_errors'].append({'type':event.exc_type.__name__,
            'traceback':''.join(traceback.format_exception(event.exc_type,event.exc_value,event.exc_traceback))})
        previous(event)
    threading.excepthook=thread_failure
    namespace={}
    steps=[('torch','import torch'),
      ('transformers_classes','from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig'),
      ('quantization_config','quantization=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16)'),
      ('quantizer','from transformers.quantizers import AutoHfQuantizer; quantizer=AutoHfQuantizer.from_config(quantization)'),
      ('quantizer_environment','quantizer.validate_environment(device_map={"":"cuda"})')]
    okay=True
    try:
        for name,code in steps:
            started=time.perf_counter();print(json.dumps({'phase':name,'event':'begin'}),flush=True)
            try:
                exec(code,namespace)
                phase={'phase':name,'status':'PASS','wall_seconds':time.perf_counter()-started}
            except Exception as error:
                okay=False;phase={'phase':name,'status':'FAILED','wall_seconds':time.perf_counter()-started,
                    'error_type':type(error).__name__,'traceback':traceback.format_exc()}
                traceback.print_exc()
            report['phases'].append(phase);print(json.dumps({k:v for k,v in phase.items() if k!='traceback'}),flush=True)
            if not okay:break
        if okay and args.load_model:
            from budget_router.inference import ModelSpec,make_backend,configuration_fingerprint
            spec=ModelSpec(**json.loads(args.model_config.read_text('utf-8-sig')))
            if spec.backend!='transformers' or not spec.model_path:raise ValueError('Diagnosis only supports pinned local Transformers models')
            started=time.perf_counter();print(json.dumps({'phase':'actual_model_load','event':'begin'}),flush=True)
            try:
                backend=make_backend(spec)
                report['phases'].append({'phase':'actual_model_load','status':'PASS','wall_seconds':time.perf_counter()-started,
                    'model_fingerprint':configuration_fingerprint(spec),'actual_4bit':bool(getattr(backend.model,'is_loaded_in_4bit',False)),
                    'parameter_bytes':int(backend.model.get_memory_footprint()),
                    'hf_device_map':{k:str(v) for k,v in getattr(backend.model,'hf_device_map',{}).items()}})
            except Exception as error:
                okay=False;report['phases'].append({'phase':'actual_model_load','status':'FAILED','wall_seconds':time.perf_counter()-started,
                    'error_type':type(error).__name__,'traceback':traceback.format_exc()})
                traceback.print_exc()
        report['status']='TECHNICAL_PASS' if okay and not report['thread_errors'] else 'TECHNICAL_FAILURE'
    except Exception as error:
        report.update(status='TECHNICAL_FAILURE',error_type=type(error).__name__,traceback=traceback.format_exc())
        traceback.print_exc();okay=False
    finally:
        threading.excepthook=previous
        report['completed_at_unix']=time.time();args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({'status':report['status'],'output':str(args.output),'model_generate_calls':0,'scientific_completion':False}),flush=True)
    return 0 if okay and not report['thread_errors'] else 2
if __name__=='__main__':raise SystemExit(main())

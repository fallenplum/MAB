"""Execute synthetic temporary-source reference-adapter qualification only."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import platform
import time
import unittest


def main():
    code=Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=code/"E18_REFERENCE_ADAPTER_QUALIFICATION_v2.json")
    output = parser.parse_args().output
    if output.exists():raise FileExistsError("Existing v2 qualification is immutable")
    cpu,wall=time.process_time(),time.perf_counter()
    log=io.StringIO()
    suite=unittest.defaultTestLoader.discover(str(code/"tests"),pattern="test_reference_adapter.py")
    result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
    qualification=dict(
        schema="e18_reference_adapter_synthetic_qualification_v2",status="PASS" if result.wasSuccessful() else "FAIL",
        tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),python=platform.python_version(),
        source_sha256=sha(code/"e18_reference_adapter.py"),gate_source_sha256=sha(code/"e18_execution_adapter.py"),
        tests_sha256=sha(code/"tests/test_reference_adapter.py"),qualification_code_sha256=sha(Path(__file__)),
        scope="Toy temporary JSONL/official-style JSON only; actual reference source access remains unexercised",
        actual_fresh_reference_reads=0,actual_sealed_input_reads=0,actual_gold_values_decoded=0,
        new_llm_calls=0,new_encoder_forwards=0,confidence_intervals=0,bootstrap_runs=0,
        physical_boundary="A separately approved allow_structural_byte_scan=True scope is necessary for whole-source SHA and raw structural scans. Unrequested reference values are syntax-skipped, not JSON-decoded.",
        production_approval_required=True,real_source_runtime_status="NOT_RUNTIME_TESTED",
        cpu_s=time.process_time()-cpu,wall_s=time.perf_counter()-wall,test_log=log.getvalue())
    with output.open("x",encoding="utf-8") as stream:json.dump(qualification,stream,ensure_ascii=False,indent=2,allow_nan=False)
    print(log.getvalue())
    print(json.dumps({key:qualification[key] for key in ("status","tests","failures","errors","actual_fresh_reference_reads","actual_sealed_input_reads","source_sha256")},ensure_ascii=False))
    if not result.wasSuccessful():raise SystemExit(1)


if __name__=="__main__":main()

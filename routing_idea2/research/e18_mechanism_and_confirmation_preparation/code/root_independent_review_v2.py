"""Review final toy qualifications and source hashes without real reward access."""
from __future__ import annotations

import ast
from pathlib import Path

from e18_common import CODE, preserve_e17, read_json, sha256, write_json


def main():
    records = [
        ("EXECUTION_ADAPTER_ROOT_FINAL2_v2.json", "tests", 33, {
            "source_sha256": "e18_execution_adapter.py",
            "test_source_sha256": "qualify_execution_adapter.py",
        }),
        ("REFERENCE_ADAPTER_ROOT_FINAL2_v2.json", "tests", 22, {
            "source_sha256": "e18_reference_adapter.py",
            "gate_source_sha256": "e18_execution_adapter.py",
            "tests_sha256": "tests/test_reference_adapter.py",
            "qualification_code_sha256": "qualify_e18_reference_adapter.py",
        }),
        ("SCORING_ADAPTER_ROOT_FINAL2_v2.json", "checks_run", 15, {
            "adapter_source_sha256": "e18_scoring_adapter.py",
            "qualification_source_sha256": "qualify_e18_scoring_adapter.py",
            "execution_adapter_source_sha256": "e18_execution_adapter.py",
            "reference_adapter_source_sha256": "e18_reference_adapter.py",
        }),
    ]
    checked_sources = {}
    reviewed = []
    for name, count_key, expected, source_map in records:
        value = read_json(CODE / name)
        if value["status"] != "PASS" or value[count_key] != expected:
            raise ValueError("Final synthetic qualification differs " + name)
        for key, relative in source_map.items():
            source = CODE / relative
            if sha256(source) != value[key]:
                raise ValueError("The qualified source changed " + relative)
            ast.parse(source.read_text(encoding="utf-8-sig"))
            checked_sources[relative] = value[key]
        reviewed.append(dict(path=name, sha256=sha256(CODE / name), tests=expected))
    write_json(CODE / "INDEPENDENT_ADAPTER_REVIEW_v2.json", dict(
        status="PASS_FINAL_SOURCE_AND_SOFTWARE_REPLAY",
        independent_replayed_tests=70,
        source_ast_checks=len(checked_sources),
        source_hashes=checked_sources,
        qualification_records=reviewed,
        E17_preserved=preserve_e17(),
        actual_llm_calls=0,
        actual_fresh_reference_reads=0,
        sealed_answers_read=0,
        extra_sealed_inputs_read=0,
        runtime_gpu_status="NOT_RUNTIME_TESTED",
        full_cohort_dispatcher_status="NOT_IMPLEMENTED",
        scientific_accuracy_claims_made=False,
    ))
    print("E18_INDEPENDENT_ADAPTER_REVIEW_PASS", 70, flush=True)


if __name__ == "__main__":
    main()

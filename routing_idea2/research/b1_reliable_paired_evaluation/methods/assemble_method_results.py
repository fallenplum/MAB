"""Assemble public, aggregate-only method provenance and execution status."""
import json
import sys
from pathlib import Path

def assemble(public):
    public = Path(public)
    native = json.loads((public / "NATIVE_REFERENCE_RESULTS.json").read_text(encoding="utf-8"))
    qualification = json.loads((public / "PUBLIC_METHODS_QUALIFICATION.json").read_text(encoding="utf-8"))
    supplements = json.loads((public / "SUPPLEMENTARY_FINITE_POOL_RESULTS.json").read_text(encoding="utf-8"))
    manifest = json.loads((public / "UPSTREAM_SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    regression = json.loads((public / "ACTIVE_DENOMINATOR_REGRESSION.json").read_text(encoding="utf-8"))
    result = {
        "status": "TASK_ADAPTERS_AND_REFERENCES_COMPLETE_ROBUST_PEW_ORIGINAL_REPLICATION_PARTIAL",
        "physical_LLM_calls": 0,
        "sources": [{key: source[key] for key in ["repo", "commit", "license"]} for source in manifest["sources"]],
        "source_manifest": "UPSTREAM_SOURCE_MANIFEST.json",
        "source_byte_equivalence": "SOURCE_BYTE_EQUIVALENCE.json",
        "native_references": native,
        "software_qualification": qualification,
        "active_denominator_regression": regression,
        "real_finite_pool_supplements": supplements,
        "inference_boundaries": {
            "ASI_native": "author population asymptotic delta-method CI on author AlphaFold example",
            "Robust_ASI_original": "Pew original data absent; native MinMaxOptimizer_l2 operator check only",
            "matched_table": "same prediction-assisted estimator and fixed-pool design inference, sampler formulas adapted to paired residual labels and two expected budgets",
            "FixedSize_task": "author CPS sampler, actual pi and pi_ij, SYG unbiased design variance with asymptotic normal CI",
            "ActiveTesting_task": "LURE with frozen predicted bounded paired loss and separately proven conservative fixed-time martingale CI",
            "pool_target": "original V2 validation200 finite questions and three fixed generation repeats",
        },
        "implementation_corrections": "METHOD_IMPLEMENTATION_CORRECTIONS.md",
        "reproduction": "PUBLIC_METHODS_REPRODUCE.md",
    }
    output = public / "PUBLIC_METHOD_RESULTS.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return output

if __name__ == "__main__":
    print(assemble(sys.argv[1]))

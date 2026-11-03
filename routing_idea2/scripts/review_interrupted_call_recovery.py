"""Read-only operator-plan review. Contains no process-control or ledger writes.

An explicit --execute request is always refused by this review-only program:
there is currently no validated atomic-freeze/GPU-retirement execution adapter.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys


def review(path, *, execute=False, authorization=None):
    plan_path = Path(path)
    raw = plan_path.read_bytes()
    plan = json.loads(raw.decode("utf-8-sig"))
    if plan.get("schema") != "interrupted_call_operator_plan_review_v1":
        raise ValueError("Unknown review-only plan schema")
    if plan.get("execution_implemented") is not False or plan.get("new_user_authorization_obtained") is not False:
        raise ValueError("This review plan cannot assert execution or new user authorization")
    if execute:
        if authorization is None:
            raise PermissionError("New explicit user authorization bound to this exact plan is required")
        auth = json.loads(Path(authorization).read_text("utf-8-sig"))
        digest = hashlib.sha256(raw).hexdigest()
        if (auth.get("schema") != "explicit_interrupted_call_authorization_v1"
                or auth.get("plan_sha256") != digest
                or auth.get("explicit_user_approval") is not True
                or not isinstance(auth.get("user_message_evidence"), str)
                or not auth["user_message_evidence"].strip()):
            raise PermissionError("Authorization is not explicitly bound to this plan")
        # Even an authorization declaration cannot supply technical proof or turn
        # this reader into a process-control/reconciliation implementation.
        raise RuntimeError("REVIEW_ONLY: no validated freeze/retirement adapter; no actions performed")
    return {"schema": "interrupted_call_plan_dry_run_v1", "plan_sha256": hashlib.sha256(raw).hexdigest(),
            "execution_ready": False, "actions_performed": [],
            "target_identity_verified": False, "live_state_read": False,
            "assessment": plan["assessment"],
            "blocking_proofs": plan["blocking_proofs"],
            "committed_prefix_snapshot_is_final_usage": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--authorization")
    args = parser.parse_args(argv)
    try:
        output = review(args.plan, execute=args.execute, authorization=args.authorization)
    except (ValueError, PermissionError, RuntimeError) as error:
        print(json.dumps({"status": "REFUSED", "reason": str(error), "actions_performed": []}), file=sys.stderr)
        return 2
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

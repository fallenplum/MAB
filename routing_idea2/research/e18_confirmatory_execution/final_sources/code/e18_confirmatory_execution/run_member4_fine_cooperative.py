"""Production entry point for a frozen fine plan and cooperative resources."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cooperative_slot_admission_v2 as resources
import member4_tail_partition as tail


CONTROL_SOURCE_SHA = "45c88b376059a4b390d065ef61e38db24fa1862cf8b98f7658c110d89c423f9f"
COOPERATIVE_SOURCE_SHA = "3d20b299e26c03bafb189be795a21b54156b4bc0bf62001da178f13b16d25d10"


def run_cooperative(args):
    if tail.base.file_sha(tail.__file__) != CONTROL_SOURCE_SHA or tail.base.file_sha(resources.__file__) != COOPERATIVE_SOURCE_SHA:
        raise ValueError("Frozen control or operational resource source differs")
    if not args.gpu.isdigit() or args.gpu_slot not in (0, 1):
        raise ValueError("An explicit physical GPU and original slot are required")
    if tail.base.file_sha(args.plan) != args.plan_sha256:
        raise ValueError("Fine-plan bytes differ from the bound plan")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    dispatcher = tail.load_runtime(args.dispatcher)
    plan = tail.base.read_json(args.plan)
    manifests, _ = tail.admit_fine_plan(plan, args.plan_sha256, dispatcher)
    with resources.CooperativeSlotAdmission(
        dispatcher.worker, manifests[0][1], args.plan_sha256, args.gpu, args.gpu_slot,
        dict(path=str(args.resource_policy), sha256=args.resource_policy_sha256),
    ):
        return tail.execute_fine_plan(plan, args.plan_sha256, dispatcher)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--dispatcher", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--gpu-slot", type=int, choices=(0, 1), default=0)
    parser.add_argument("--resource-policy", type=Path, required=True)
    parser.add_argument("--resource-policy-sha256", required=True)
    result = run_cooperative(parser.parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

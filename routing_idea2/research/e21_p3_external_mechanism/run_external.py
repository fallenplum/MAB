"""P3 replay of the preregistered DARS single-model contrast."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
from importlib import metadata
import json
from pathlib import Path
import re
import sys
import time
import unicodedata

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from endpoints import fit_endpoint_pair, uniform_policy

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "e18_mechanism_and_confirmation_preparation"))
from e18_reference_adapter import _project_object

CELLS = (("tspo", 3), ("listnet", 3), ("tspo", 1), ("tspo", 5))
TEMPERATURES = (0.2, 0.7, 1.0)
ASSETS = Path(r"E:\语言模型组合路由")
SOURCE_SHA256 = {
    ("gpqa", "train"): "9a9f9c797647b8397a6b4eee2ce9915187dcf8dd5a3a271e8b33f4abec006025",
    ("gpqa", "test"): "5983fa6677fc1ce04616ab1307d3f4c9e3ac20ccc5016163097603dd69499353",
    ("math-500", "train"): "11c82f532eff977250cf049869b33c0e26705d2a3f85e885589ae1abb8ce7200",
    ("math-500", "test"): "1c2ddf9c702f4c78dc3d43842d403903faaf0376d3e1afd307d6d1647495543e",
}


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def norm(question):
    return " ".join(unicodedata.normalize("NFKC", question).split()).casefold()


def decode_number(value):
    if type(value) is int:
        return value
    found = re.fullmatch(r"(?:(?:sample|decode)[_-]?)?(\d+)", str(value))
    if not found:
        raise ValueError("Unrecognized DARS training decode index")
    return int(found[1])


def paths(data):
    return {split: data / f"{split}_scored_generations.jsonl" for split in ("train", "test")}


def release_identity(path, revision, expected):
    header = path.with_suffix(path.suffix + ".headers")
    observed = sha(path)
    if observed != expected:
        raise ValueError("DARS cache bytes do not match the released source pointer")
    if header.is_file():
        headers = header.read_text(encoding="utf-8", errors="replace")
        linked = re.search(r'(?mi)^x-linked-etag:\s*"([a-f0-9]{64})"\s*$', headers)
        if not linked or linked[1] != observed or not re.search(rf"(?mi)^x-repo-commit:\s*{revision}\s*$", headers):
            raise ValueError("Released DARS pointer identity differs")
    return observed


def project_inputs(path):
    rows = {}
    with path.open("rb") as stream:
        for raw in stream:
            row = _project_object(raw, {"query_id", "original_question"})
            key, text = str(row["query_id"]), row["original_question"]
            if not isinstance(text, str) or not norm(text):
                raise ValueError("Empty or non-text evaluation input")
            previous = rows.setdefault(key, text)
            if previous != text:
                raise ValueError("Original query text differs across the external test views")
    return rows


def finite_binary(value):
    if type(value) not in (int, float) or value not in (0, 1):
        raise ValueError("P3 selected only DARS binary tasks")
    return int(value)


def load_training(source, actions, task):
    observations, texts = defaultdict(dict), {}
    settings, generation_keys = set(), set()
    with source.open(encoding="utf-8") as stream:
        for raw in stream:
            row = json.loads(raw)
            if row["dataset"] != task or row["split"] != "train":
                raise ValueError("DARS training release split mismatched")
            if str(row["prompt_variant_id"]) != "0":
                continue
            key, model, decode = str(row["query_id"]), row["model"], decode_number(row["decode_id"])
            if model not in actions or decode in observations[(key, model)]:
                raise ValueError("DARS decoded row identity is unexpected or repeated")
            generation_key = row["generation_key"]
            if generation_key in generation_keys:
                raise ValueError("Repeated DARS generation key")
            generation_keys.add(generation_key)
            observations[(key, model)][decode] = (
                finite_binary(row["score"]), row["input_question"])
            text = row["original_question"]
            if not isinstance(text, str) or not norm(text) or texts.setdefault(key, text) != text:
                raise ValueError("Original training text changed across observations")
            request = row["request_config"]
            settings.add((float(request["temperature"]), request["top_p"], request["max_tokens"]))
    if len(settings) != 1 or not np.isclose(next(iter(settings))[0], 0.7):
        raise ValueError("Training repeats do not share the preregistered request configuration")
    common_decodes = {tuple(sorted(item)) for item in observations.values()}
    if len(common_decodes) != 1:
        raise ValueError("Inconsistent training-repeat identities")
    order = next(iter(common_decodes))
    if len(order) != 5 or len(observations) != len(texts) * len(actions):
        raise ValueError("The first rewrite must have complete five-repeat action grids")
    for key in texts:
        prompts = {observations[(key, model)][decode][1] for model in actions for decode in order}
        if len(prompts) != 1:
            raise ValueError("Multiple rewrites in the training-repeat cell")
    ids = sorted(texts)
    values = np.array([[[observations[(key, model)][index][0] for model in actions]
                        for index in order] for key in ids], dtype=np.float64)
    return texts, values, dict(request_configs=[list(item) for item in sorted(settings)],
                              decode_order=list(order), unique_chosen_rewrite_generations=len(generation_keys))


def encode_all(questions, assets):
    import torch

    sys.path.insert(0, str(assets / "src"))
    from budget_router.encoding import FrozenQuestionEncoder

    receipt = json.loads((assets / "artifacts/private/routing_redesign_20261104/full_feedback_inputs/v1"
                          / "INPUT_ASSETS_RECEIPT.json").read_text(encoding="utf-8-sig"))
    versions = {name: metadata.version(name)
                for name in ("numpy", "torch", "transformers", "tokenizers", "safetensors")}
    if any(receipt["runtime_versions"][name] != version for name, version in versions.items()):
        raise ValueError("Use the original qualified MiniLM runtime")
    revision = receipt["feature_encoder"]["revision"]
    torch.set_num_threads(2)
    encoder = FrozenQuestionEncoder(
        assets / "models/sentence-transformers--all-MiniLM-L6-v2", revision=revision,
        receipt=assets / "artifacts/audit/encoder_model_download.json", device="cpu", max_length=256)
    feature = encoder.encode(questions, batch_size=8)
    if feature.shape != (len(questions), 384) or not np.isfinite(feature).all():
        raise ValueError("Invalid MiniLM representation")
    return feature, dict(revision=revision, versions=versions,
                         encoder_source_sha256=sha(assets / "src/budget_router/encoding.py"))


def no_overlap(train, test):
    train_set, test_set = {norm(item) for item in train.values()}, {norm(item) for item in test.values()}
    if (train.keys() & test.keys() or train_set & test_set
            or len(train_set) != len(train) or len(test_set) != len(test)):
        raise ValueError("DARS has duplicate original question identity within or across splits")


def fit(args, protocol):
    timings = time.process_time(), time.perf_counter()
    external = protocol["external"]
    actions = tuple(external["action_order"])
    args.run.mkdir(parents=True, exist_ok=False)
    prepared, partitions = {}, {}
    for task, expected in external["tasks"].items():
        source = paths(args.data / task)
        hashes = {split: release_identity(path, external["revision"], SOURCE_SHA256[task, split])
                  for split, path in source.items()}
        train, rewards, detail = load_training(source["train"], actions, task)
        test = project_inputs(source["test"])
        if [len(train), len(test)] != expected:
            raise ValueError("DARS original-question counts differ from the release")
        no_overlap(train, test)
        prepared[task] = train, test, rewards
        partitions[task] = dict(source_sha256=hashes, train_questions=len(train),
                                test_questions=len(test), training_observation_audit=detail)
    seen = set()
    for train, test, _ in prepared.values():
        current = {norm(item) for item in [*train.values(), *test.values()]}
        if current & seen:
            raise ValueError("Identical original question across distinct DARS task files")
        seen.update(current)
    input_list = [text for train, test, _ in prepared.values()
                  for rows in (train, test) for _, text in sorted(rows.items())]
    encodings, encoder_audit = encode_all(input_list, args.asset_root)
    start, fit_count = 0, 0
    for task, (train, test, repeats) in prepared.items():
        stop_train, stop_test = start + len(train), start + len(train) + len(test)
        pca = PCA(n_components=16, svd_solver="full")
        scaler = StandardScaler()
        train_embedding = scaler.fit_transform(pca.fit_transform(encodings[start:stop_train]))
        test_embedding = scaler.transform(pca.transform(encodings[stop_train:stop_test]))
        start = stop_test
        phi_train = np.column_stack((np.ones(len(train)), train_embedding))
        phi_test = np.column_stack((np.ones(len(test)), test_embedding))
        packet = dict(question_ids=np.array(sorted(test)), pca_mean=pca.mean_, pca_components=pca.components_,
                      scaler_mean=scaler.mean_, scaler_scale=scaler.scale_)
        cell_audit = {}
        for family, repeats_used in CELLS:
            cell = f"{family}_L{repeats_used}"
            reverse = task == "gpqa" and repeats_used == 3
            coefficients, qualification = fit_endpoint_pair(
                family, phi_train, repeats[:, :repeats_used].mean(axis=1),
                regularization=external["lambda"], reverse=reverse, audit_features=phi_test)
            scores = np.stack([phi_test @ value for value in coefficients])
            packet[f"{cell}_scores"] = scores
            packet[f"{cell}_probabilities"] = uniform_policy(scores, protocol["primary_tau"])
            packet[f"{cell}_raw"] = uniform_policy(scores, protocol["raw_exact_tau"])
            packet[f"{cell}_coefficients"] = coefficients
            cell_audit[cell] = qualification
            fit_count += 2 + 2 * reverse
        baseline = repeats[:, :3].mean(axis=(0, 1), keepdims=True).reshape(1, -1)
        packet["global_fixed"] = uniform_policy(baseline, protocol["primary_tau"])[0]
        target = args.run / f"{task}_FROZEN.npz"
        with target.open("xb") as stream:
            np.savez_compressed(stream, **packet)
        partitions[task].update(frozen_sha256=sha(target), fit_qualification=cell_audit)
        print(json.dumps({"task": task, "status": "PREDICTIONS_FROZEN"}), flush=True)
    receipt = dict(status="ALL_EXTERNAL_PREDICTIONS_FROZEN", protocol_sha256=args.protocol_sha256,
                   runner_sha256=sha(Path(__file__)), encoder=encoder_audit, tasks=partitions,
                   test_score_values_decoded=0, main_fits=16, reverse_qualification_fits=fit_count - 16,
                   cpu_seconds=time.process_time()-timings[0], wall_seconds=time.perf_counter()-timings[1])
    write(args.run / "FIT_FREEZE.json", receipt)
    print(json.dumps({key: receipt[key] for key in ("status", "main_fits", "reverse_qualification_fits")}))


def evaluation_rewards(source, task, ids, actions):
    seen, entries, settings = set(), {}, defaultdict(set)
    task_positions, model_positions = {key: index for index, key in enumerate(ids)}, {key: index for index, key in enumerate(actions)}
    with source.open(encoding="utf-8") as stream:
        for raw in stream:
            row = json.loads(raw)
            if row["dataset"] != task or row["split"] != "test":
                raise ValueError("DARS test release split mismatched")
            if row["observation_type"] != "decoding_variation":
                continue
            if str(row["prompt_variant_id"]) != "-1":
                raise ValueError("A rewritten prompt entered original-query evaluation")
            config = row["request_config"]
            temperature = float(config["temperature"])
            if temperature not in TEMPERATURES:
                raise ValueError("Unexpected DARS original-query decoding condition")
            key = (temperature, task_positions[str(row["query_id"])], model_positions[row["model"]])
            generation_key = row["generation_key"]
            if key in entries or generation_key in seen:
                raise ValueError("Duplicate physical DARS test observation")
            settings[temperature].add((config["top_p"], config["max_tokens"]))
            entries[key] = finite_binary(row["score"])
            seen.add(generation_key)
    output = {}
    for temperature in TEMPERATURES:
        if len(settings[temperature]) != 1:
            raise ValueError("Different decoding configs pooled under one temperature")
        output[temperature] = np.array([[entries[(temperature, q, m)] for m in range(len(actions))]
                                        for q in range(len(ids))], dtype=np.float64)
    return output, {str(key): [list(item) for item in value] for key, value in settings.items()}


def quality(probabilities, rewards, strict_changed):
    utility = np.einsum("man,an->ma", probabilities, rewards)
    common = probabilities.min(axis=0)
    residual_b, residual_c = probabilities - common
    mass = residual_b.sum(axis=1)
    denom = np.where(mass > 1e-14, mass, np.inf)
    corrected = (residual_b * (1-rewards)).sum(axis=1) * (residual_c * rewards).sum(axis=1) / denom
    damaged = (residual_b * rewards).sum(axis=1) * (residual_c * (1-rewards)).sum(axis=1) / denom
    if not np.allclose(corrected-damaged, utility[1]-utility[0], rtol=0, atol=1e-12):
        raise ValueError("P3 correction accounting differs")
    return dict(B_accuracy_pct=float(100*utility[0].mean()), C_accuracy_pct=float(100*utility[1].mean()),
                C_minus_B_pp=float(100*(utility[1]-utility[0]).mean()),
                expected_corrections=float(corrected.sum()), expected_damages=float(damaged.sum()),
                strict_corrections=int(np.sum((utility[1] == 1) & (utility[0] == 0) & strict_changed)),
                strict_damages=int(np.sum((utility[1] == 0) & (utility[0] == 1) & strict_changed)))


def evaluate(args, protocol):
    timings = time.process_time(), time.perf_counter()
    frozen = json.loads((args.run / "FIT_FREEZE.json").read_text(encoding="utf-8"))
    if (frozen["protocol_sha256"] != args.protocol_sha256 or frozen["runner_sha256"] != sha(Path(__file__))
            or frozen["status"] != "ALL_EXTERNAL_PREDICTIONS_FROZEN"
            or frozen["test_score_values_decoded"] != 0):
        raise ValueError("Exact frozen two-task P3 packet required before test scores")
    external = protocol["external"]
    actions = tuple(external["action_order"])
    for task, evidence in frozen["tasks"].items():
        if (sha(args.run / f"{task}_FROZEN.npz") != evidence["frozen_sha256"]
                or sha(paths(args.data / task)["test"]) != evidence["source_sha256"]["test"]):
            raise ValueError("External evaluation input bytes differ from the freeze")
    by_task = {}
    score_count = 0
    for task, evidence in frozen["tasks"].items():
        with np.load(args.run / f"{task}_FROZEN.npz", allow_pickle=False) as packet:
            ids = packet["question_ids"].tolist()
            rewards, settings = evaluation_rewards(paths(args.data/task)["test"], task, ids, actions)
            score_count += sum(values.size for values in rewards.values())
            row = dict(train_questions=evidence["train_questions"], test_questions=len(ids),
                       source_sha256=evidence["source_sha256"], frozen_prediction_sha256=evidence["frozen_sha256"],
                       training_observation_audit=evidence["training_observation_audit"],
                       test_request_settings=settings,
                       train_only_global_fixed_accuracy_pct={
                           str(key): float(100*(value @ packet["global_fixed"]).mean())
                           for key, value in rewards.items()}, cells={})
            for family, repeats_used in CELLS:
                cell = f"{family}_L{repeats_used}"
                scores, probabilities, raw = (packet[f"{cell}_{kind}"] for kind in ("scores", "probabilities", "raw"))
                sorted_scores = np.sort(scores, axis=-1)
                strict = np.all(sorted_scores[:, :, -1]-sorted_scores[:, :, -2] > protocol["primary_tau"], axis=0)
                changed = strict & (scores[1].argmax(axis=1) != scores[0].argmax(axis=1))
                raw_strict = np.all(sorted_scores[:, :, -1]-sorted_scores[:, :, -2] > protocol["raw_exact_tau"], axis=0)
                raw_changed = raw_strict & (scores[1].argmax(axis=1) != scores[0].argmax(axis=1))
                centered = scores - scores.mean(axis=-1, keepdims=True)
                row["cells"][cell] = dict(fit_qualification=evidence["fit_qualification"][cell],
                    center_changed_questions=int(np.sum(np.max(np.abs(centered[1]-centered[0]), axis=1) > 1e-7)),
                    strict_unique_both=int(strict.sum()), strict_switches=int(changed.sum()),
                    probability_changed_questions=int(np.sum(np.any(probabilities[0] != probabilities[1], axis=1))),
                    margins_by_policy=np.median(sorted_scores[:, :, -1]-sorted_scores[:, :, -2], axis=1).tolist(),
                    support_size_counts_by_policy=[dict(Counter(map(int, (policy > 0).sum(axis=1))))
                                                   for policy in probabilities],
                    mean_total_variation=float(np.abs(probabilities[1]-probabilities[0]).sum(axis=1).mean()/2),
                    B_action_probabilities=probabilities[0].mean(axis=0).tolist(),
                    C_action_probabilities=probabilities[1].mean(axis=0).tolist(),
                    by_temperature={str(key): quality(probabilities, value, changed)
                                    for key, value in rewards.items()},
                    raw_exact_at_0_7=quality(raw, rewards[0.7], raw_changed))
        by_task[task] = row
    result = dict(schema="mab_p3_external_dars_results_v1", status="COMPLETE_EXTERNAL_MECHANISM_REPLAY",
                  protocol_sha256=args.protocol_sha256, dars_revision=external["revision"], action_order=list(actions),
                  encoder=frozen["encoder"], unique_external_test_questions=sum(row["test_questions"] for row in by_task.values()),
                  consumed_cached_binary_observations=score_count, primary_temperature=0.7, by_task=by_task,
                  macro_at_0_7={cell: {field: float(np.mean([
                                    row["cells"][cell]["by_temperature"]["0.7"][field] for row in by_task.values()]))
                                    for field in ("B_accuracy_pct", "C_accuracy_pct", "C_minus_B_pp")}
                                for cell in (f"{family}_L{length}" for family, length in CELLS)},
                  main_fits=frozen["main_fits"], reverse_qualification_fits=frozen["reverse_qualification_fits"],
                  fit_wall_seconds=frozen["wall_seconds"], fit_cpu_seconds=frozen["cpu_seconds"],
                  evaluation_wall_seconds=time.perf_counter()-timings[1], evaluation_cpu_seconds=time.process_time()-timings[0],
                  new_llm_calls=0, new_llm_tokens=0, fresh400_or_sealed200_reads=0, bootstrap_repetitions=0)
    write(args.output, result)
    print(json.dumps({key: result[key] for key in ("status", "unique_external_test_questions", "macro_at_0_7")}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=("fit", "evaluate"))
    for name in ("protocol", "data", "run", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, default=ASSETS)
    parser.add_argument("--protocol-sha256", required=True)
    args = parser.parse_args()
    if sha(args.protocol) != args.protocol_sha256:
        raise ValueError("P3 protocol changed after freeze")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8-sig"))
    if sha(HERE / "endpoints.py") != protocol["endpoint_source_sha256"]:
        raise ValueError("External endpoint source differs from the locked protocol")
    if sha(HERE.parent / "e19_symmetric_tie_constraint" / "weighted_objectives.py") != protocol["source_e19_weighted_sha256"]:
        raise ValueError("Original E19 objective source changed")
    (fit if args.phase == "fit" else evaluate)(args, protocol)


if __name__ == "__main__":
    main()

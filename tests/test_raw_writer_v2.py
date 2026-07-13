import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file

from wisent.scripts.activations.raw import extract_and_upload as raw


REVISION = {
    "activation_revision": "a" * 40,
    "model_revision": "b" * 40,
    "tokenizer_revision": "c" * 40,
}
STRATEGY = "chat_first"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def ref(uri, payload=b"x", generation="immutable-7"):
    return {
        "uri": uri,
        "generation": generation,
        "size": str(len(payload)),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def manifest():
    routes = []
    for layer in (1, 2):
        routes.append({
            "strategy": STRATEGY,
            "layer": layer,
            "completion_ref": ref(f"bundle:///source/{STRATEGY}/layer-{layer}.completion.json", f"completion-{layer}".encode()),
            "proof_ref": ref(f"bundle:///source/{STRATEGY}/layer-{layer}.proof.json", f"proof-{layer}".encode()),
        })
    value = {
        "schema_version": 2,
        "protocol": {"id": "steering_effectiveness_v1", "revision": 1},
        "target": {
            "target_id": "target-explicit-id",
            "result_id": "result-explicit-id",
            "model_name": "acme/model",
            "model_slug": "acme__model",
            "benchmark": "okapi/arc_multilingual/日本語",
            "expected_pairs": 2,
            "result_prefix": "results/acme__model/okapi/arc_multilingual/日本語",
        },
        "revisions": {"inventory_sha256": "d" * 64, **REVISION},
        "activation": {
            "status": "complete", "eligible": True, "layer_count": 2, "n_pairs": 2,
            "grouped": False, "strategies": {STRATEGY: 2}, "routes": routes,
            "proof": {"cache_sha256": "e" * 64, "record_sha256": "f" * 64},
        },
        "support": {
            "state": "prepared", "proof_sha256": "1" * 64, "pair_count": 2,
            "split_counts": {"train": 1, "validation": 0, "test": 1},
            "splits": {
                "train": [{"pair_id": 10, "stable_id": "stable-train"}],
                "validation": [],
                "test": [{"pair_id": 20, "stable_id": "stable-test"}],
            },
        },
        "evaluation": {"required_outputs": ["raw_trajectory"], "split": "test"},
        "calibration": {"methods": ["caa"], "strategies": [STRATEGY], "layer_count": 2, "expected_pairs": 2},
        "execution": {"state": "unprepared", "blocked": False, "rerun_locked": False, "publication": None},
    }
    value["manifest_sha256"] = hashlib.sha256(canonical(value)).hexdigest()
    return value


def pair(pair_id, stable_id, offset, pos_len, neg_len):
    pos_ids = torch.arange(offset, offset + pos_len, dtype=torch.int64)
    neg_ids = torch.arange(offset + 100, offset + 100 + neg_len, dtype=torch.int64)
    return {
        "pair_id": pair_id,
        "stable_id": stable_id,
        "pos_hidden_states": {
            "1": torch.stack((pos_ids.float(), pos_ids.float() + 0.5), dim=1),
            "2": torch.stack((pos_ids.float() + 1000, pos_ids.float() + 1000.5), dim=1),
        },
        "neg_hidden_states": {
            "1": torch.stack((neg_ids.float(), neg_ids.float() + 0.5), dim=1),
            "2": torch.stack((neg_ids.float() + 1000, neg_ids.float() + 1000.5), dim=1),
        },
        "pos_input_ids": pos_ids,
        "neg_input_ids": neg_ids,
        "pos_attention_mask": torch.ones(pos_len, dtype=torch.int64),
        "neg_attention_mask": torch.ones(neg_len, dtype=torch.int64),
        "pos_effective_length": pos_len,
        "neg_effective_length": neg_len,
        "pos_answer_onset": 2,
        "neg_answer_onset": 1,
    }


def pairs_shuffled():
    return [pair(20, "stable-test", 20, 4, 3), pair(10, "stable-train", 10, 3, 2)]


def write_bundle(root):
    target = manifest()
    target_bytes = canonical(target)
    target_ref = ref("targets/target-explicit-id.json", target_bytes, "manifest-generation")
    paths = raw.write_raw_activations_v2(
        pairs_shuffled(), target, target_ref, root,
        strategy=STRATEGY, layer_count=2,
        activation_revision=REVISION["activation_revision"],
        model_revision=REVISION["model_revision"],
        tokenizer_revision=REVISION["tokenizer_revision"],
        artifact_base_uri="evidence",
    )
    return target, target_ref, paths


def test_writer_packs_exact_order_metadata_refs_and_safe_nested_benchmark(tmp_path):
    target, target_ref, completion_paths = write_bundle(tmp_path)
    assert [p.parent.name for p in completion_paths] == ["layer_1", "layer_2"]
    assert all("raw_activations_v2" in p.parts for p in completion_paths)
    assert not any("okapi" in part or "日本語" in part for p in completion_paths for part in p.parts)

    completion_payload = json.loads(completion_paths[0].read_bytes())
    assert set(completion_payload) == {"schema_version", "complete", "kind", "target", "revisions", "support", "target_manifest_ref", "source_route_ref", "artifact", "manifest_sha256"}
    assert completion_payload["target"] == {
        "target_id": "target-explicit-id", "model": "acme/model", "model_slug": "acme__model",
        "benchmark": "okapi/arc_multilingual/日本語", "strategy": STRATEGY, "layer": 1, "layer_count": 2,
    }
    assert completion_payload["revisions"] == {"activation": "a" * 40, "model": "b" * 40, "tokenizer": "c" * 40}
    assert completion_payload["support"] == [
        {"pair_id": 10, "stable_id": "stable-train", "split": "train"},
        {"pair_id": 20, "stable_id": "stable-test", "split": "test"},
    ]
    assert completion_payload["target_manifest_ref"] == target_ref
    assert completion_payload["source_route_ref"] == target["activation"]["routes"][0]["completion_ref"]
    expected_self_hash = hashlib.sha256(canonical({
        key: value for key, value in completion_payload.items() if key != "manifest_sha256"
    })).hexdigest()
    assert completion_payload.get("manifest_sha256") == expected_self_hash

    artifact = completion_paths[0].parent / "activations.safetensors"
    blob = artifact.read_bytes()
    assert completion_payload["artifact"] == {
        "uri": "evidence/" + artifact.relative_to(tmp_path).as_posix(),
        "sha256": hashlib.sha256(blob).hexdigest(),
        "size": str(len(blob)),
    }
    assert completion_payload["artifact"]["size"].isdigit() and not completion_payload["artifact"]["size"].startswith("0")

    tensors = load_file(str(artifact))
    assert set(tensors) == {
        "positive_activations", "negative_activations", "positive_token_ids", "negative_token_ids",
        "positive_attention_mask", "negative_attention_mask",
    }
    assert tensors["positive_activations"].shape == (7, 2)
    assert tensors["negative_activations"].shape == (5, 2)
    assert tensors["positive_token_ids"].tolist() == [10, 11, 12, 20, 21, 22, 23]
    assert tensors["negative_token_ids"].tolist() == [110, 111, 120, 121, 122]
    assert tensors["positive_attention_mask"].tolist() == [1] * 7
    assert tensors["negative_attention_mask"].tolist() == [1] * 5
    assert tensors["positive_activations"][:, 0].tolist() == [10, 11, 12, 20, 21, 22, 23]
    with safe_open(str(artifact), framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
    assert set(metadata) == set(raw.METADATA_KEYS)
    assert {key: json.loads(value) for key, value in metadata.items()} == {
        "pair_ids": [10, 20], "stable_ids": ["stable-train", "stable-test"],
        "positive_lengths": [3, 4], "negative_lengths": [2, 3],
        "positive_prompt_lengths": [2, 2], "negative_prompt_lengths": [1, 1],
        "positive_answer_onsets": [2, 2], "negative_answer_onsets": [1, 1],
    }


def test_writer_is_byte_deterministic(tmp_path):
    _, _, first = write_bundle(tmp_path / "one")
    _, _, second = write_bundle(tmp_path / "two")
    for a, b in zip(first, second):
        assert (a.parent / "activations.safetensors").read_bytes() == (b.parent / "activations.safetensors").read_bytes()
        assert (a.parent / "manifest.json").read_bytes() == (b.parent / "manifest.json").read_bytes()
        assert a.read_bytes() == b.read_bytes()


def test_writer_retry_is_idempotent_and_existing_conflict_fails_closed(tmp_path):
    _, _, first = write_bundle(tmp_path)
    first_bytes = [(path.parent / "activations.safetensors").read_bytes() for path in first]
    _, _, retried = write_bundle(tmp_path)
    assert retried == first
    assert [(path.parent / "activations.safetensors").read_bytes() for path in retried] == first_bytes

    artifact = first[0].parent / "activations.safetensors"
    artifact.write_bytes(b"conflicting existing bytes")
    with pytest.raises((FileExistsError, RuntimeError, ValueError), match="conflict|immutable|existing"):
        write_bundle(tmp_path)


@pytest.mark.parametrize("mutation, match", [
    (lambda ps, m, r: ps[0].pop("pair_id"), "missing pair_id"),
    (lambda ps, m, r: ps[0].pop("stable_id"), "missing stable_id"),
    (lambda ps, m, r: ps[0].pop("pos_answer_onset"), "missing pos_answer_onset"),
    (lambda ps, m, r: ps[0].__setitem__("stable_id", "wrong"), "support is missing explicit identity"),
    (lambda ps, m, r: ps[0].__setitem__("pos_input_ids", torch.tensor([1])), "exceeds final encoding|different lengths"),
    (lambda ps, m, r: ps[0]["pos_hidden_states"].__setitem__("1", torch.ones(2, 3, 4)), "must be \\[tokens, hidden\\]"),
    (lambda ps, m, r: ps[0].__setitem__("pos_answer_onset", 999), "inside its final encoded sequence"),
    (lambda ps, m, r: m["revisions"].pop("tokenizer_revision"), "exact 40-character revision"),
    (lambda ps, m, r: r.__setitem__("size", "01"), "canonical positive decimal string"),
    (lambda ps, m, r: r.__setitem__("sha256", "0" * 63), "lowercase SHA-256"),
    (lambda ps, m, r: m["activation"]["routes"][0].pop("completion_ref"), "source_route_ref"),
])
def test_writer_rejects_partial_or_invalid_contract(tmp_path, mutation, match):
    ps, m = pairs_shuffled(), manifest()
    target_bytes = canonical(m)
    r = ref("targets/manifest.json", target_bytes)
    mutation(ps, m, r)
    if "manifest_sha256" in m:
        m["manifest_sha256"] = hashlib.sha256(canonical({k: v for k, v in m.items() if k != "manifest_sha256"})).hexdigest()
    if match not in {"canonical positive decimal string", "lowercase SHA-256"}:
        rebound = canonical(m)
        r["size"] = str(len(rebound))
        r["sha256"] = hashlib.sha256(rebound).hexdigest()
    with pytest.raises((ValueError, TypeError), match=match):
        raw.write_raw_activations_v2(
            ps, m, r, tmp_path, strategy=STRATEGY, layer_count=2,
            activation_revision="a" * 40, model_revision="b" * 40, tokenizer_revision="c" * 40,
            artifact_base_uri="evidence",
        )


def test_writer_rejects_old_prefix_and_path_traversal_uri(tmp_path):
    m = manifest()
    r = ref("targets/manifest.json", canonical(m))
    kwargs = dict(strategy=STRATEGY, layer_count=2, activation_revision="a" * 40, model_revision="b" * 40, tokenizer_revision="c" * 40)
    with pytest.raises(ValueError, match="clean 'raw_activations_v2' prefix"):
        raw.write_raw_activations_v2(pairs_shuffled(), m, r, tmp_path, prefix="raw_activations", **kwargs)
    with pytest.raises(ValueError, match="traversal|relative|URI|artifact_base_uri"):
        raw.write_raw_activations_v2(pairs_shuffled(), m, r, tmp_path, artifact_base_uri="../escape", **kwargs)


@pytest.mark.parametrize("slug", ["../escape", "/absolute", "nested/model", ".", ".."])
def test_writer_rejects_unsafe_model_slug_before_filesystem_write(tmp_path, slug):
    m = manifest()
    m["target"]["model_slug"] = slug
    m["manifest_sha256"] = hashlib.sha256(canonical({k: v for k, v in m.items() if k != "manifest_sha256"})).hexdigest()
    r = ref("targets/manifest.json", canonical(m))
    with pytest.raises(ValueError, match="model_slug|unsafe|path|traversal|absolute"):
        raw.write_raw_activations_v2(
            pairs_shuffled(), m, r, tmp_path, strategy=STRATEGY, layer_count=2,
            activation_revision="a" * 40, model_revision="b" * 40, tokenizer_revision="c" * 40,
            artifact_base_uri="evidence",
        )
    assert not (tmp_path / "raw_activations_v2").exists()


@pytest.mark.parametrize("field,value", [("sha256", "0" * 64), ("size", "1")])
def test_writer_binds_target_manifest_ref_to_exact_canonical_payload_bytes(tmp_path, field, value):
    m = manifest()
    r = ref("targets/manifest.json", canonical(m))
    r[field] = value
    with pytest.raises(ValueError, match="target_manifest_ref.*(hash|size|payload|canonical|bind|mismatch)"):
        raw.write_raw_activations_v2(
            pairs_shuffled(), m, r, tmp_path, strategy=STRATEGY, layer_count=2,
            activation_revision="a" * 40, model_revision="b" * 40, tokenizer_revision="c" * 40,
            artifact_base_uri="evidence",
        )
    assert not (tmp_path / "raw_activations_v2").exists()

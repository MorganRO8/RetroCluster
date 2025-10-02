import ast
import json
import logging
from collections import Counter

from pipelines.phase2_mechanism import (
    MechanismSignature,
    MechanismCluster,
    _cluster_signatures,
    _coarse_mechanism_key,
    _compute_signature_payload,
    _summarize_mechanism_outputs,
)


def _row(**overrides: object) -> dict[str, object]:
    base = {
        "rxn_vid": overrides.get("rxn_vid", 1),
        "cond_hash": overrides.get("cond_hash", "abcd"),
        "reactants_canonical": json.dumps(["CCO"]),
        "products_canonical": json.dumps(["CC=O"]),
        "role_main_substrate": "CCO",
    }
    base.update(overrides)
    return base


def test_compute_signature_prefers_mapped_branch():
    signature = _compute_signature_payload(_row())
    assert signature.signature_type == "mapped"
    assert signature.event_tokens
    assert signature.scaffold_key == "CCO"
    assert signature.coarse_key is not None


def test_compute_signature_falls_back_when_delta_missing():
    signature = _compute_signature_payload(
        _row(rxn_vid=5, reactants_canonical=json.dumps(["CCO"]), products_canonical=json.dumps(["CCO"]))
    )
    assert signature.signature_type == "unmapped"
    assert any(token.startswith("R:") for token in signature.event_tokens)


def test_cluster_assignment_groups_by_cond_and_signature():
    sig_a = _compute_signature_payload(_row(rxn_vid=1, cond_hash="hash1"))
    sig_b = _compute_signature_payload(_row(rxn_vid=2, cond_hash="hash1"))
    sig_c = _compute_signature_payload(
        _row(
            rxn_vid=3,
            cond_hash="hash2",
            reactants_canonical=json.dumps(["N#N"]),
            products_canonical=json.dumps(["N#N"]),
        )
    )
    clusters = _cluster_signatures([sig_a, sig_b, sig_c])
    assert len(clusters) == 2
    first = next(cluster for cluster in clusters if cluster.cond_hash == "hash1")
    assert json.loads(first.to_dict()["rxn_vids"]) == [1, 2]
    base_counts = json.loads(first.to_dict()["mech_sig_base_counts"])
    assert sum(base_counts.values()) == 2
    assert sig_a.cluster_id == first.cluster_id
    assert sig_b.cluster_id == first.cluster_id
    second = next(cluster for cluster in clusters if cluster.cond_hash == "hash2")
    assert json.loads(second.to_dict()["rxn_vids"]) == [3]
    assert sig_c.cluster_id == second.cluster_id


def test_cluster_coalesces_by_coarse_key():
    sig_a = MechanismSignature(
        rxn_vid=10,
        cond_hash="cond",
        mech_sig_base="base_a",
        mech_sig_r1=None,
        mech_sig_r2=None,
        signature_type="mapped",
        event_tokens=["alpha:x"],
        redox_events=0,
        stereo_events=0,
        ring_events=0,
        scaffold_key="Scaf",
    )
    sig_a.coarse_key = _coarse_mechanism_key(sig_a)

    sig_b = MechanismSignature(
        rxn_vid=11,
        cond_hash="cond",
        mech_sig_base="base_b",
        mech_sig_r1=None,
        mech_sig_r2=None,
        signature_type="mapped",
        event_tokens=["alpha:y"],
        redox_events=0,
        stereo_events=0,
        ring_events=0,
        scaffold_key="Scaf",
    )
    sig_b.coarse_key = _coarse_mechanism_key(sig_b)

    clusters = _cluster_signatures([sig_a, sig_b])
    assert len(clusters) == 1
    cluster = clusters[0]
    assert json.loads(cluster.to_dict()["rxn_vids"]) == [10, 11]
    assert sig_a.cluster_id == cluster.cluster_id == sig_b.cluster_id
    base_counts = json.loads(cluster.to_dict()["mech_sig_base_counts"])
    assert base_counts == {"base_a": 1, "base_b": 1}


def test_coarse_key_uses_scaffold_family():
    sig_a = MechanismSignature(
        rxn_vid=20,
        cond_hash="cond",
        mech_sig_base="base",
        mech_sig_r1=None,
        mech_sig_r2=None,
        signature_type="mapped",
        event_tokens=["+C:1", "-1:1"],
        redox_events=0,
        stereo_events=0,
        ring_events=0,
        scaffold_key="c1ccccc1",
    )
    sig_b = MechanismSignature(
        rxn_vid=21,
        cond_hash="cond",
        mech_sig_base="base",
        mech_sig_r1=None,
        mech_sig_r2=None,
        signature_type="mapped",
        event_tokens=["+N:1", "-2:1"],
        redox_events=0,
        stereo_events=0,
        ring_events=0,
        scaffold_key="c1ccc(cc1)C",
    )

    key_a = _coarse_mechanism_key(sig_a)
    key_b = _coarse_mechanism_key(sig_b)

    assert key_a == key_b
    assert key_a[0] == "aromatic"

    clusters = _cluster_signatures([sig_a, sig_b])
    assert len(clusters) == 1
    cluster = clusters[0]
    assert json.loads(cluster.to_dict()["rxn_vids"]) == [20, 21]


def test_coarse_key_buckets_similar_event_tokens():
    sig_a = MechanismSignature(
        rxn_vid=30,
        cond_hash="cond",
        mech_sig_base="base",
        mech_sig_r1=None,
        mech_sig_r2=None,
        signature_type="mapped",
        event_tokens=["+C:1", "-1:1"],
        redox_events=0,
        stereo_events=0,
        ring_events=0,
        scaffold_key="CCO",
    )
    sig_b = MechanismSignature(
        rxn_vid=31,
        cond_hash="cond",
        mech_sig_base="base",
        mech_sig_r1=None,
        mech_sig_r2=None,
        signature_type="mapped",
        event_tokens=["+N:2", "-2:1"],
        redox_events=0,
        stereo_events=0,
        ring_events=0,
        scaffold_key="CCO",
    )

    key_a = _coarse_mechanism_key(sig_a)
    key_b = _coarse_mechanism_key(sig_b)

    assert key_a == key_b
    assert key_a[1] == (("gain:atom_upper", "one"), ("loss:ring_index", "one"))

    clusters = _cluster_signatures([sig_a, sig_b])
    assert len(clusters) == 1
    cluster = clusters[0]
    assert json.loads(cluster.to_dict()["rxn_vids"]) == [30, 31]


def test_summarize_mechanism_outputs_logs_percentiles(caplog):
    caplog.set_level(logging.INFO)

    main_coarse_key = ("scaf_family", (("family", "one"),))
    single_coarse_key = ("other_family", (("other", "one"),))

    signatures = [
        MechanismSignature(
            rxn_vid=idx,
            cond_hash="cond_a" if idx in {1, 2} else "cond_b",
            mech_sig_base=f"base_{idx % 2}",
            mech_sig_r1=None,
            mech_sig_r2=None,
            signature_type="mapped",
            event_tokens=["token"],
            redox_events=0,
            stereo_events=0,
            ring_events=0,
            scaffold_key="scaf",
        )
        for idx in range(1, 5)
    ]

    extra_signature = MechanismSignature(
        rxn_vid=5,
        cond_hash="cond_c",
        mech_sig_base="base_extra",
        mech_sig_r1=None,
        mech_sig_r2=None,
        signature_type="mapped",
        event_tokens=["extra"],
        redox_events=0,
        stereo_events=0,
        ring_events=0,
        scaffold_key="other_scaf",
    )

    signatures.append(extra_signature)

    for signature in signatures[:4]:
        signature.coarse_key = main_coarse_key
    signatures[4].coarse_key = single_coarse_key

    clusters = [
        MechanismCluster(
            cond_hash="cond_a",
            cluster_id="c1",
            coarse_key=main_coarse_key,
            mech_sig_base_counts=Counter({"base_1": 1, "base_0": 1}),
            rxn_vids=[1, 2],
        ),
        MechanismCluster(
            cond_hash="cond_b",
            cluster_id="c2",
            coarse_key=main_coarse_key,
            mech_sig_base_counts=Counter({"base_1": 1, "base_0": 1}),
            rxn_vids=[3, 4],
        ),
        MechanismCluster(
            cond_hash="cond_c",
            cluster_id="c3",
            coarse_key=single_coarse_key,
            mech_sig_base_counts=Counter({"base_extra": 1}),
            rxn_vids=[5],
        ),
    ]

    _summarize_mechanism_outputs(signatures, clusters)

    messages = [record.getMessage() for record in caplog.records]
    size_summary = next(msg for msg in messages if msg.startswith("Level 2 cluster sizes:"))
    base_summary = next(
        msg for msg in messages if msg.startswith("Distinct mech_sig_base per cluster:")
    )
    top_coarse_summary = next(msg for msg in messages if msg.startswith("Top coarse signatures:"))
    single_condition_summary = next(
        msg
        for msg in messages
        if msg.startswith("Coarse signatures limited to one condition bucket:")
    )
    family_distribution_summary = next(
        msg for msg in messages if msg.startswith("Scaffold family distribution:")
    )
    family_cluster_size_summary = next(
        msg for msg in messages if msg.startswith("Scaffold family average cluster sizes:")
    )

    _, top_payload = top_coarse_summary.split(": ", 1)
    top_entries = ast.literal_eval(top_payload)
    _, single_payload = single_condition_summary.split(": ", 1)
    single_entries = ast.literal_eval(single_payload)
    _, family_payload = family_distribution_summary.split("top=", 1)
    family_entries = ast.literal_eval(family_payload)
    _, avg_payload = family_cluster_size_summary.split("top=", 1)
    avg_entries = ast.literal_eval(avg_payload)

    assert "pct10=" in size_summary and "pct90=" in size_summary
    assert "pct25=" in base_summary and "pct75=" in base_summary
    assert any(entry["condition_buckets"] == 2 for entry in top_entries)
    assert any(entry["reactions"] == 1 for entry in single_entries)
    assert any(entry[0] == "scaf_family" and entry[1] == 4 for entry in family_entries)
    assert any(item["family"] == "scaf_family" for item in avg_entries)

import json

from pipelines.phase2_mechanism import (
    MechanismSignature,
    _cluster_signatures,
    _coarse_mechanism_key,
    _compute_signature_payload,
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

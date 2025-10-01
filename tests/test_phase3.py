import json

from pipelines.phase3_structures import (
    StructureFeature,
    _assign_structure_clusters,
    _build_structure_feature,
    _tanimoto,
)


def _row(**overrides: object) -> dict[str, object]:
    base = {
        "rxn_vid": overrides.get("rxn_vid", 1),
        "CID": overrides.get("CID", "cid-1"),
        "MID": overrides.get("MID", "mid-1"),
        "scaffold_key": overrides.get("scaffold_key", "Scaffold"),
        "mech_sig_base": overrides.get("mech_sig_base", "base"),
        "event_tokens": json.dumps(overrides.get("event_tokens", ["tokA"])),
    }
    base.update(overrides)
    return base


def test_structure_feature_uses_scaffold_and_events():
    feature_a = _build_structure_feature(_row(rxn_vid=1), fp_bits=32)
    feature_b = _build_structure_feature(_row(rxn_vid=2), fp_bits=32)
    assert feature_a.scaffold_key == "Scaffold"
    assert feature_a.bits == feature_b.bits
    assert feature_a.vector == [float(bit) for bit in sorted(feature_a.bits)]


def test_tanimoto_similarity_respects_shared_bits():
    feature = StructureFeature(rxn_vid=1, cid="c", mid="m", scaffold_key="a", bits={1, 2, 3})
    other = StructureFeature(rxn_vid=2, cid="c", mid="m", scaffold_key="b", bits={1, 2, 4})
    disjoint = StructureFeature(rxn_vid=3, cid="c", mid="m", scaffold_key="c", bits={7})
    assert _tanimoto(feature.bits, other.bits) >= 0.5
    assert _tanimoto(feature.bits, disjoint.bits) == 0.0


def test_assign_structure_clusters_groups_similar_canopies():
    rows = [
        _row(rxn_vid=1, event_tokens=["tokA", "tokB"]),
        _row(rxn_vid=2, event_tokens=["tokA", "tokB"]),
        _row(rxn_vid=3, event_tokens=["tokZ"], MID="mid-2"),
        _row(rxn_vid=4, event_tokens=["tokB", "tokC"]),
    ]
    features = [_build_structure_feature(row, fp_bits=64) for row in rows]
    clusters = _assign_structure_clusters(features, tanimoto_threshold=0.5)
    assert len(clusters) == 2

    # first two rows should share the same SID because of identical fingerprints
    sid_first = next(feature.sid for feature in features if feature.rxn_vid == 1)
    sid_second = next(feature.sid for feature in features if feature.rxn_vid == 2)
    assert sid_first == sid_second

    # row 3 belongs to a different MID, hence its own cluster
    sid_third = next(feature.sid for feature in features if feature.rxn_vid == 3)
    assert sid_third.startswith("mid-2-")

    # verify cluster dictionary payload is sorted and exemplar limited to 3 entries
    payload = next(cluster.to_dict() for cluster in clusters if cluster.sid == sid_first)
    assert json.loads(payload["rxn_vids"]) == [1, 2, 4]
    assert json.loads(payload["exemplar_vids"]) == [1, 2, 4]


def test_structure_feature_reads_suffixed_columns():
    base = {
        "rxn_vid": 10,
        "CID_lvl2": "cid-sfx",
        "cluster_id_lvl2": "mid-sfx",
        "scaffold_key_sig": "ScafX",
    }
    row_a = {**base, "event_tokens_sig": json.dumps(["tokA"]) }
    row_b = {**base, "event_tokens_sig": json.dumps(["tokB"]) }

    feature_a = _build_structure_feature(row_a, fp_bits=128)
    feature_b = _build_structure_feature(row_b, fp_bits=128)

    assert feature_a.cid == "cid-sfx"
    assert feature_a.mid == "mid-sfx"
    assert feature_a.scaffold_key == "ScafX"
    assert feature_a.bits != feature_b.bits

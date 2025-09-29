import json

import pytest

from pipelines.phase4_fusion import (
    ClusterRecord,
    _build_cluster_cards,
    _build_cross_view_edges,
    _merge_cluster_assignments,
    _top_values,
)


class Frame(list):
    """Minimal stand-in for a pandas DataFrame used in tests."""

    def to_dict(self, orient: str = "records"):
        assert orient == "records"
        return list(self)


def test_top_values_prefers_frequency_then_lexicographic():
    values = ["water", "ethanol", "water", "methanol", "ethanol", "water"]
    assert _top_values(values, limit=2) == [("water", 3), ("ethanol", 2)]


def test_merge_cluster_assignments_resolves_all_levels():
    level1 = Frame([
        {"CID": "C1", "rxn_vids": json.dumps([1, 2])},
        {"CID": "C2", "rxn_vids": json.dumps([3])},
    ])
    level2 = Frame(
        [
            {"CID": "C1", "MID": "M1", "rxn_vids": json.dumps([1])},
            {"CID": "C1", "MID": "M2", "rxn_vids": json.dumps([2])},
            {"CID": "C2", "MID": "M3", "rxn_vids": json.dumps([3])},
        ]
    )
    level3 = Frame(
        [
            {"CID": "C1", "MID": "M1", "SID": "S1", "rxn_vids": json.dumps([1])},
            {"CID": "C1", "MID": "M2", "SID": "S2", "rxn_vids": json.dumps([2])},
            {"CID": "C2", "MID": "M3", "SID": "S3", "rxn_vids": json.dumps([3])},
        ]
    )

    assignments = _merge_cluster_assignments(level1, level2, level3)
    assert assignments == [
        ClusterRecord(rxn_vid=1, cid="C1", mid="M1", sid="S1"),
        ClusterRecord(rxn_vid=2, cid="C1", mid="M2", sid="S2"),
        ClusterRecord(rxn_vid=3, cid="C2", mid="M3", sid="S3"),
    ]


def test_build_cluster_cards_aggregates_multisource_features():
    assignments = [
        ClusterRecord(rxn_vid=1, cid="C1", mid="M1", sid="S1"),
        ClusterRecord(rxn_vid=2, cid="C1", mid="M1", sid="S1"),
        ClusterRecord(rxn_vid=3, cid="C2", mid="M2", sid="S2"),
    ]
    structures = Frame(
        [
            {"rxn_vid": 1, "scaffold_key": "ScafA"},
            {"rxn_vid": 2, "scaffold_key": "ScafA"},
            {"rxn_vid": 3, "scaffold_key": "ScafB"},
        ]
    )
    mechanisms = Frame(
        [
            {"rxn_vid": 1, "event_tokens": json.dumps(["tokA", "tokB"]), "mech_sig_base": "base1"},
            {"rxn_vid": 2, "event_tokens": json.dumps(["tokA"]), "mech_sig_base": "base1"},
            {"rxn_vid": 3, "event_tokens": json.dumps(["tokC"]), "mech_sig_base": "base2"},
        ]
    )
    reactions = Frame(
        [
            {
                "rxn_vid": 1,
                "solvent_normalized": json.dumps(["water"]),
                "temperature_K": 298.2,
                "yield": 0.8,
            },
            {
                "rxn_vid": 2,
                "solvent_normalized": json.dumps(["water"]),
                "temperature_K": 300.5,
                "yield": 0.6,
            },
            {
                "rxn_vid": 3,
                "solvent_normalized": json.dumps(["ethanol"]),
                "temperature_K": 310.0,
                "yield": 0.4,
            },
        ]
    )

    cards = _build_cluster_cards(assignments, structures, mechanisms, reactions)
    key = ("C1", "M1", "S1")
    assert key in cards
    card = cards[key]
    assert card.size == 2
    assert card.dominant_scaffold == "ScafA"
    assert card.top_solvents[0] == ("water", 2)
    assert card.mechanism_summary[0][0] == "base1"
    assert card.yield_mean == pytest.approx(0.7)
    assert card.yield_median == pytest.approx(0.7)
    assert card.exemplars == ["rxn_1.svg", "rxn_2.svg"]


def test_build_cross_view_edges_links_shared_clusters():
    assignments = [
        ClusterRecord(rxn_vid=1, cid="C1", mid="M1", sid="S1"),
        ClusterRecord(rxn_vid=2, cid="C2", mid="M1", sid="S2"),
        ClusterRecord(rxn_vid=3, cid="C2", mid="M2", sid="S2"),
        ClusterRecord(rxn_vid=4, cid="C3", mid="M1", sid="S2"),
    ]

    edges = _build_cross_view_edges(assignments)
    edge_kinds = {(edge.kind, edge.shared_id) for edge in edges}
    assert ("MID", "M1") in edge_kinds
    assert ("SID", "S2") in edge_kinds
    mid_edge = next(edge for edge in edges if edge.kind == "MID" and edge.shared_id == "M1")
    assert mid_edge.weight == 1  # min count across shared CIDs
    sid_edge = next(edge for edge in edges if edge.kind == "SID" and edge.shared_id == "S2")
    assert sid_edge.weight == 1

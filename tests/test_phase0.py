from pipelines.phase0_data_readiness import _deduplicate, _normalize_row


def test_normalize_row_generates_hash_and_roles():
    row = {
        "rxn_id": "rxn-001",
        "reactants": "CCO.OCC",
        "products": "CCOC",
        "agents": "NaOH",
        "solvent": "water",
        "temperature": "25 C",
        "time": "30 min",
        "pressure": "1 atm",
        "pH": "7",
        "phase": "liquid",
        "light": "dark",
        "atmosphere": "air",
        "yield": 0.8,
        "source": "unit-test",
        "timestamp": "2024-01-01",
    }
    result = _normalize_row(row)
    assert not result.failures
    payload = result.payload
    assert payload["reaction_hash_v1"]
    assert payload["reactants_canonical"]
    assert payload["role_main_substrate"] in payload["reactants_canonical"]


def test_deduplicate_merges_provenance():
    row = {
        "rxn_id": "rxn-001",
        "reactants": "CCO",
        "products": "CCOC",
        "agents": "",
        "solvent": "ethanol",
        "temperature": 298,
    }
    result = _normalize_row(row)
    deduped = _deduplicate([result.payload, result.payload.copy()])
    assert len(deduped) == 1
    assert deduped[0]["rxn_vid"] == 1

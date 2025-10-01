import pytest

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


def test_normalize_row_handles_rxn_smiles_and_synonyms():
    row = {
        "rxn_id": "rxn-002",
        "rxn_smiles": "CCO.OCC>>CCOC",
        "solvents": ["water"],
        "agents": ["NaOH"],
        "catalysts": ["Pd/C"],
        "temperature_c": 25,
        "time_min": 45,
        "pressure_atm": 1,
        "ph": 8,
        "phase": "liquid",
        "atmosphere": "argon",
    }
    result = _normalize_row(row)
    assert not result.failures
    payload = result.payload
    assert payload["reactants_canonical"]
    assert payload["products_canonical"]
    assert payload["agents_canonical"]
    assert payload["temperature_K"] == pytest.approx(298.15, rel=1e-3)
    assert payload["time_s"] == pytest.approx(2700.0, rel=1e-3)
    assert payload["pressure_Pa"] == pytest.approx(101325.0, rel=1e-3)
    assert payload["solvent_normalized"]
    # mask bits: temperature=1<<0, time=1<<1, pressure=1<<2, atmosphere=1<<3, pH=1<<5, phase=1<<6, solvent=1<<7
    assert payload["present_mask"] & 0b1
    assert payload["present_mask"] & 0b10
    assert payload["present_mask"] & 0b100
    assert payload["present_mask"] & 0b1000
    assert payload["present_mask"] & 0b100000
    assert payload["present_mask"] & 0b1000000
    assert payload["present_mask"] & 0b10000000

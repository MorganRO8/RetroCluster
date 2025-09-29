import json

from pipelines.phase1_conditions import (
    _bin_ph,
    _bin_temperature,
    _build_condition_record,
)


def _sample_row() -> dict[str, object]:
    return {
        "rxn_vid": 42,
        "present_mask": 5,
        "temperature_K": 298.15,
        "time_s": 1800,
        "pressure_Pa": 101325,
        "pH": 7.2,
        "atmosphere": "Air",
        "light": "Dark",
        "phase": "Liquid",
        "solvent_normalized": json.dumps(["water", "ethanol"]),
        "agents_canonical": json.dumps(["NaOH"]),
        "spectators_moved": json.dumps(["NaCl"]),
        "resolution_confidence": 0.92,
    }


def test_temperature_binning_changes_width_after_400k():
    assert _bin_temperature(298.15) == "290-300"
    assert _bin_temperature(450.0) == "450-475"


def test_ph_bin_clamps_to_valid_range():
    assert _bin_ph(-1) == "0-1"
    assert _bin_ph(14.7) == "14-14"


def test_condition_record_generates_hash_and_keys():
    row = _sample_row()
    record = _build_condition_record(row)
    assert record.cond_hash
    solvent_key = json.loads(record.solvent_key)
    assert solvent_key == [["ethanol", None], ["water", None]]
    catalyst_key = json.loads(record.catalyst_key)
    assert catalyst_key == ["NaOH"]
    agent_key = json.loads(record.agent_key)
    assert agent_key == ["NaCl"]
    assert record.temp_K_bin == "290-300"
    assert record.time_s_bin == "1000-10000"
    assert record.pressure_Pa_bin == "100000-1000000"
    assert record.pH_bin == "7-8"


def test_present_mask_influences_hash():
    row = _sample_row()
    record_a = _build_condition_record(row)
    row_b = dict(row)
    row_b["present_mask"] = row["present_mask"] + 1
    record_b = _build_condition_record(row_b)
    assert record_a.cond_hash != record_b.cond_hash

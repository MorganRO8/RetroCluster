"""Phase 0 – data readiness pipeline implementation."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import click
try:  # pragma: no cover - optional dependency in minimal environments
    import pandas as pd
except Exception:  # pragma: no cover - allow CLI helpers to signal requirement
    pd = None  # type: ignore[assignment]

from normalizers.chem_standardize import (
    canonical_lists,
    split_entities,
    standardize_reaction_entities,
)
from normalizers.conditions_normalize import normalize_conditions
from normalizers.role_assign import assign_roles
from utils.hashing import HashableReaction

LOGGER = logging.getLogger(__name__)


@dataclass
class NormalizationResult:
    payload: dict[str, Any]
    failures: list[str]


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s - %(message)s")


def _require_pandas() -> Any:
    if pd is None:
        raise click.ClickException(
            "pandas is required for file IO operations in phase0; install pandas to continue"
        )
    return pd


def _load_frame(path: str) -> Any:
    pandas = _require_pandas()
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return pandas.read_csv(path)
    if suffix in {".json", ".jsonl"}:
        return pandas.read_json(path, lines=suffix == ".jsonl")
    if suffix in {".parquet", ".pq"}:
        return pandas.read_parquet(path)
    raise click.ClickException(f"Unsupported input format: {suffix}")


def _present_mask(row: dict[str, Any]) -> int:
    fields = [
        "temperature",
        "time",
        "pressure",
        "atmosphere",
        "light",
        "pH",
        "phase",
        "solvent",
    ]
    mask = 0
    for idx, field in enumerate(fields):
        if row.get(field) not in (None, "", []):
            mask |= 1 << idx
    return mask


def _normalize_row(row: dict[str, Any]) -> NormalizationResult:
    failures: list[str] = []
    reactants_raw = split_entities(row.get("reactants"))
    products_raw = split_entities(row.get("products"))
    agents_raw = split_entities(row.get("agents"))

    reactants_norm, products_norm, agents_norm, spectators = standardize_reaction_entities(
        reactants_raw, products_raw, agents_raw
    )

    if any(record.failed for record in (*reactants_norm, *products_norm, *agents_norm)):
        failures.append("molecule_standardization")

    role_info = assign_roles(canonical_lists(reactants_norm))

    conditions = normalize_conditions(row)

    normalized_reaction_string = (
        ".".join(canonical_lists(reactants_norm))
        + ">>"
        + ".".join(canonical_lists(products_norm))
        + "|"
        + ".".join(canonical_lists(agents_norm))
    )

    conditions_payload = {
        "temperature_K": conditions.temperature_k,
        "time_s": conditions.time_s,
        "pressure_Pa": conditions.pressure_pa,
        "solvent": conditions.solvent,
        "atmosphere": conditions.atmosphere,
        "light": conditions.light,
        "phase": conditions.phase,
        "pH": conditions.ph,
    }
    condition_string = json.dumps(conditions_payload, sort_keys=True, default=str)

    reaction_hash = HashableReaction(
        reaction_string=normalized_reaction_string,
        conditions_string=condition_string,
    ).hexdigest()

    payload = {
        "rxn_id": row.get("rxn_id"),
        "reactants_canonical": canonical_lists(reactants_norm),
        "products_canonical": canonical_lists(products_norm),
        "agents_canonical": canonical_lists(agents_norm),
        "spectators_moved": spectators,
        "role_main_substrate": role_info.main_substrate,
        "role_scores": role_info.role_scores,
        "temperature_K": conditions.temperature_k,
        "time_s": conditions.time_s,
        "pressure_Pa": conditions.pressure_pa,
        "solvent_normalized": conditions.solvent,
        "solvent_confidence": conditions.solvent_confidence,
        "atmosphere": conditions.atmosphere,
        "light": conditions.light,
        "phase": conditions.phase,
        "pH": conditions.ph,
        "yield": row.get("yield"),
        "source": row.get("source"),
        "timestamp": row.get("timestamp"),
        "reaction_hash_v1": reaction_hash,
        "present_mask": _present_mask(row),
        "resolution_confidence": conditions.solvent_confidence,
        "provenance": [row.get("rxn_id")],
    }

    return NormalizationResult(payload=payload, failures=failures)


def _deduplicate(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}
    for record in records:
        key = record["reaction_hash_v1"]
        existing = deduped.get(key)
        if existing is None:
            deduped[key] = record
        else:
            provenance = existing.setdefault("provenance", [])
            provenance.extend(record.get("provenance", []))
    for idx, record in enumerate(sorted(deduped.values(), key=lambda item: item["reaction_hash_v1"])):
        record["rxn_vid"] = idx + 1
    return list(deduped.values())


def _write_output(records: list[dict[str, Any]], output: Path) -> None:
    pandas = _require_pandas()
    frame = pandas.DataFrame(records)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not frame.empty:
        json_columns = [
            "reactants_canonical",
            "products_canonical",
            "agents_canonical",
            "spectators_moved",
            "role_scores",
            "provenance",
            "solvent_normalized",
        ]
        for column in json_columns:
            if column in frame.columns:
                frame[column] = frame[column].apply(
                    lambda value: json.dumps(value, sort_keys=True)
                    if isinstance(value, (list, dict))
                    else value
                )
    frame.to_parquet(output, index=False)


def _write_failures(failures: list[dict[str, Any]], path: Path) -> None:
    if not failures:
        return
    pandas = _require_pandas()
    serialised = []
    for entry in failures:
        record = dict(entry)
        if isinstance(record.get("failures"), list):
            record["failures"] = json.dumps(record["failures"], sort_keys=True)
        serialised.append(record)
    frame = pandas.DataFrame(serialised)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


@click.command()
@click.option("--input", "input_path", required=True, type=click.Path(exists=True))
@click.option("--output", "output_path", required=True, type=click.Path())
@click.option(
    "--failed-output",
    "failed_output_path",
    default=None,
    type=click.Path(),
    help="Optional path for rows that fail normalization.",
)
@click.option("--sample", is_flag=True, help="Process only the first 1,000 rows for smoke tests.")
@click.option("--verbose", is_flag=True, help="Enable debug logging.")
def main(
    input_path: str,
    output_path: str,
    failed_output_path: str | None,
    sample: bool,
    verbose: bool,
) -> None:
    """Run Phase 0: standardize, normalize, and deduplicate reactions."""

    _configure_logging(verbose)
    LOGGER.info("Loading raw reactions from %s", input_path)
    frame = _load_frame(input_path)
    if sample:
        frame = frame.head(1000)
    total_rows = len(frame)
    LOGGER.info("Processing %s rows", total_rows)

    normalized_records: list[dict[str, Any]] = []
    failed_records: list[dict[str, Any]] = []
    failure_counts: dict[str, int] = {}

    for row in frame.to_dict(orient="records"):
        rxn_id = row.get("rxn_id")
        if rxn_id is None:
            rxn_id = len(normalized_records) + len(failed_records)
            row["rxn_id"] = rxn_id
        result = _normalize_row(row)
        if result.failures:
            failed_records.append(
                {
                    "rxn_id": row.get("rxn_id"),
                    "failures": result.failures,
                    "source": row.get("source"),
                }
            )
            for failure in result.failures:
                failure_counts[failure] = failure_counts.get(failure, 0) + 1
            continue
        normalized_records.append(result.payload)

    success_ratio = 1.0
    if total_rows:
        success_ratio = len(normalized_records) / total_rows
    if success_ratio < 0.99:
        LOGGER.warning("Canonical molecule success ratio %.2f < 0.99", success_ratio)
    else:
        LOGGER.info("Canonical molecule success ratio: %.2f", success_ratio)
    if failure_counts:
        LOGGER.info("Normalization failures: %s", failure_counts)

    deduped = _deduplicate(normalized_records)
    LOGGER.info("Deduped %s reactions into %s unique hashes", len(normalized_records), len(deduped))

    _write_output(deduped, Path(output_path))

    if failed_output_path:
        _write_failures(failed_records, Path(failed_output_path))

    success_flag = Path(output_path).parent / "_SUCCESS"
    success_flag.write_text("phase0 completed\n")
    LOGGER.info("Wrote %s and success sentinel %s", output_path, success_flag)


if __name__ == "__main__":  # pragma: no cover
    main()

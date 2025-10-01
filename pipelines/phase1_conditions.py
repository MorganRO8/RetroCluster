"""Phase 1 – condition bucketing implementation."""
from __future__ import annotations

import json
import logging
import math
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import click

try:  # pragma: no cover - optional dependency in minimal environments
    import pandas as pd
except Exception:  # pragma: no cover - allow CLI helpers to signal requirement
    pd = None  # type: ignore[assignment]

from utils.hashing import blake3_hexdigest

LOGGER = logging.getLogger(__name__)


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s - %(message)s")


def _require_pandas() -> Any:
    if pd is None:
        raise click.ClickException(
            "pandas is required for file IO operations in phase1; install pandas to continue"
        )
    return pd


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def _ensure_list(value: Any) -> list[str]:
    if _is_missing(value):
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
        # Fallback: split by comma/semicolon
        tokens = [token.strip() for token in text.replace(";", ",").split(",")]
        return [token for token in tokens if token]
    return [str(value).strip()]


def _normalize_scalar(value: Any) -> str | None:
    if _is_missing(value):
        return None
    text = str(value).strip()
    return text.lower() if text else None


def _bin_temperature(value: Any) -> str | None:
    if _is_missing(value):
        return None
    numeric = float(value)
    if numeric <= 400:
        width = 10
        lower = math.floor(numeric / width) * width
    else:
        width = 25
        lower = 400 + math.floor((numeric - 400) / width) * width
    upper = lower + width
    return f"{int(lower)}-{int(upper)}"


def _bin_log_scaled(value: Any) -> str | None:
    if _is_missing(value):
        return None
    numeric = float(value)
    if numeric <= 0:
        return "0-1"
    power = math.floor(math.log10(numeric))
    lower = 10 ** power
    upper = 10 ** (power + 1)
    if lower == upper:
        upper *= 10
    return f"{int(lower)}-{int(upper)}"


def _bin_pressure(value: Any) -> str | None:
    return _bin_log_scaled(value)


def _bin_time(value: Any) -> str | None:
    return _bin_log_scaled(value)


def _bin_ph(value: Any) -> str | None:
    if _is_missing(value):
        return None
    numeric = float(value)
    numeric = max(0.0, min(14.0, numeric))
    lower = math.floor(numeric)
    upper = min(lower + 1, 14)
    return f"{int(lower)}-{int(upper)}"


def _serialize_key(values: Iterable[str], include_none: bool = False) -> tuple[list[Any], str]:
    structured: list[Any] = []
    for value in sorted({item for item in values if item}):
        if include_none:
            structured.append([value, None])
        else:
            structured.append(value)
    serialized = json.dumps(structured, separators=(",", ":"), sort_keys=True)
    return structured, serialized


def _cond_hash_components(
    solvent_struct: list[Any],
    temp_bin: str | None,
    time_bin: str | None,
    pressure_bin: str | None,
    ph_bin: str | None,
    atmosphere: str | None,
    light: str | None,
    phase: str | None,
    catalyst_struct: list[Any],
    present_mask: int,
) -> str:
    payload = [
        solvent_struct,
        temp_bin,
        time_bin,
        pressure_bin,
        ph_bin,
        atmosphere,
        light,
        phase,
        catalyst_struct,
        present_mask,
    ]
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return blake3_hexdigest(serialized.encode("utf-8"))


def _log_counter(name: str, counter: Counter[Any], limit: int = 5) -> None:
    if not counter:
        LOGGER.info("%s: no observations", name)
        return
    top = counter.most_common(limit)
    LOGGER.info("%s (top %s): %s", name, min(limit, len(counter)), top)


def _summarize_condition_records(records: list[ConditionRecord]) -> None:
    if not records:
        LOGGER.warning("No condition records produced; skipping statistics")
        return

    cond_counter = Counter(record.cond_hash for record in records)
    LOGGER.info("Unique condition hashes: %d", len(cond_counter))
    _log_counter("Most common condition hashes", cond_counter)

    prefix_counter = Counter(record.cond_hash_prefix for record in records)
    _log_counter("cond_hash_prefix distribution", prefix_counter)

    present_counter = Counter(record.present_mask for record in records)
    _log_counter("Present mask distribution", present_counter)

    temp_bins = Counter(record.temp_K_bin for record in records if record.temp_K_bin)
    _log_counter("Temperature bin usage", temp_bins)

    time_bins = Counter(record.time_s_bin for record in records if record.time_s_bin)
    _log_counter("Time bin usage", time_bins)

    pressure_bins = Counter(
        record.pressure_Pa_bin for record in records if record.pressure_Pa_bin
    )
    _log_counter("Pressure bin usage", pressure_bins)

    ph_bins = Counter(record.pH_bin for record in records if record.pH_bin)
    _log_counter("pH bin usage", ph_bins)

    solvent_keys = Counter(record.solvent_key for record in records if record.solvent_key)
    _log_counter("Solvent key distribution", solvent_keys)

    catalyst_keys = Counter(record.catalyst_key for record in records if record.catalyst_key)
    _log_counter("Catalyst key distribution", catalyst_keys)

    agent_keys = Counter(record.agent_key for record in records if record.agent_key)
    _log_counter("Spectator/agent key distribution", agent_keys)

    confidences = [
        record.resolution_confidence
        for record in records
        if record.resolution_confidence is not None
    ]
    if confidences:
        LOGGER.info(
            "Solvent resolution confidence: mean=%.2f median=%.2f",
            statistics.fmean(confidences),
            statistics.median(confidences),
        )


@dataclass
class ConditionRecord:
    rxn_vid: int
    cond_hash: str
    cond_hash_prefix: str
    present_mask: int
    temp_K_bin: str | None
    time_s_bin: str | None
    pressure_Pa_bin: str | None
    pH_bin: str | None
    atmosphere: str | None
    light: str | None
    phase: str | None
    solvent_key: str
    catalyst_key: str
    agent_key: str
    resolution_confidence: float | None

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def _build_condition_record(row: dict[str, Any]) -> ConditionRecord:
    rxn_vid = int(row.get("rxn_vid") or 0)
    present_mask = int(row.get("present_mask") or 0)

    temp_bin = _bin_temperature(row.get("temperature_K"))
    time_bin = _bin_time(row.get("time_s"))
    pressure_bin = _bin_pressure(row.get("pressure_Pa"))
    ph_bin = _bin_ph(row.get("pH"))

    atmosphere = _normalize_scalar(row.get("atmosphere"))
    light = _normalize_scalar(row.get("light"))
    phase = _normalize_scalar(row.get("phase"))

    solvents = _ensure_list(row.get("solvent_normalized"))
    catalysts = _ensure_list(row.get("agents_canonical"))
    agents = _ensure_list(row.get("spectators_moved"))

    solvent_struct, solvent_key = _serialize_key(solvents, include_none=True)
    catalyst_struct, catalyst_key = _serialize_key(catalysts)
    agent_struct, agent_key = _serialize_key(agents)

    cond_hash = _cond_hash_components(
        solvent_struct,
        temp_bin,
        time_bin,
        pressure_bin,
        ph_bin,
        atmosphere,
        light,
        phase,
        catalyst_struct,
        present_mask,
    )

    cond_hash_prefix = cond_hash[:8]

    resolution = row.get("resolution_confidence")
    if _is_missing(resolution):
        resolution_confidence = None
    else:
        resolution_confidence = float(resolution)

    return ConditionRecord(
        rxn_vid=rxn_vid,
        cond_hash=cond_hash,
        cond_hash_prefix=cond_hash_prefix,
        present_mask=present_mask,
        temp_K_bin=temp_bin,
        time_s_bin=time_bin,
        pressure_Pa_bin=pressure_bin,
        pH_bin=ph_bin,
        atmosphere=atmosphere,
        light=light,
        phase=phase,
        solvent_key=solvent_key,
        catalyst_key=catalyst_key,
        agent_key=agent_key,
        resolution_confidence=resolution_confidence,
    )


def _write_output(records: list[ConditionRecord], output: Path) -> None:
    pandas = _require_pandas()
    frame = pandas.DataFrame([record.to_dict() for record in records])
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)


SAMPLE_ROW_LIMIT = 1000


def _load_input(path: str, *, sample: bool = False) -> tuple[list[dict[str, Any]], int]:
    pandas = _require_pandas()
    frame = pandas.read_parquet(path)
    total = len(frame)
    if sample:
        frame = frame.head(SAMPLE_ROW_LIMIT)
    return frame.to_dict(orient="records"), total


@click.command()
@click.option("--input", "input_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--output", "output_path", required=True, type=click.Path(dir_okay=False))
@click.option("--sample", is_flag=True, help="Process only the first 1,000 rows for smoke tests.")
@click.option("--verbose/--quiet", default=False, show_default=True)
def main(input_path: str, output_path: str, sample: bool, verbose: bool) -> None:
    """CLI entry point for phase 1 condition bucketing."""

    _configure_logging(verbose)
    LOGGER.info("Loading normalized reactions from %s", input_path)
    rows, total_rows = _load_input(input_path, sample=sample)

    if sample and total_rows > len(rows):
        LOGGER.info("Sampling %d of %d rows", len(rows), total_rows)
    else:
        LOGGER.info("Processing %d records", len(rows))

    records = [_build_condition_record(row) for row in rows]

    _summarize_condition_records(records)

    output_path_obj = Path(output_path)
    LOGGER.info("Writing %d condition keys to %s", len(records), output_path_obj)
    _write_output(records, output_path_obj)

    success_flag = output_path_obj.parent / "_SUCCESS"
    success_flag.write_text("phase1 completed\n")
    LOGGER.info("Wrote success sentinel to %s", success_flag)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

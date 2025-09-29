"""Condition normalization utilities for Phase 0."""
from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from difflib import get_close_matches
from typing import Iterable

LOGGER = logging.getLogger(__name__)

_TEMP_UNITS = {
    "k": lambda value: value,
    "c": lambda value: value + 273.15,
    "f": lambda value: (value - 32) * 5.0 / 9.0 + 273.15,
}

_TIME_UNITS = {
    "s": 1.0,
    "sec": 1.0,
    "secs": 1.0,
    "second": 1.0,
    "seconds": 1.0,
    "m": 60.0,
    "min": 60.0,
    "mins": 60.0,
    "minute": 60.0,
    "minutes": 60.0,
    "h": 3600.0,
    "hr": 3600.0,
    "hrs": 3600.0,
    "hour": 3600.0,
    "hours": 3600.0,
    "day": 86400.0,
    "days": 86400.0,
}

_PRESSURE_UNITS = {
    "pa": 1.0,
    "kpa": 1e3,
    "mpa": 1e6,
    "bar": 1e5,
    "mbar": 1e2,
    "atm": 101325.0,
    "torr": 133.322,
    "psi": 6894.757,
}

_SOLVENT_DICTIONARY = {
    "water": {"aliases": {"h2o", "aqueous"}},
    "ethanol": {"aliases": {"etoh", "ethyl alcohol"}},
    "methanol": {"aliases": {"meoh", "methyl alcohol"}},
    "acetonitrile": {"aliases": {"mecn", "acn"}},
}


@dataclass
class NormalizedConditions:
    temperature_k: float | None
    time_s: float | None
    pressure_pa: float | None
    solvent: list[str]
    solvent_confidence: float
    atmosphere: str | None
    light: str | None
    ph: float | None
    phase: str | None
    additional_agents: list[str]


def _parse_number(raw: str) -> float | None:
    try:
        return float(raw)
    except ValueError:
        cleaned = raw.replace(",", "").strip()
        return float(cleaned)


def _extract_value_and_unit(value: object) -> tuple[float | None, str | None]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None, None
    if isinstance(value, (int, float)):
        return float(value), None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None, None
        match = re.match(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*([A-Za-z°%]+)?", stripped)
        if match:
            number = match.group(1)
            unit = match.group(2)
            try:
                return float(number), unit.lower() if unit else None
            except ValueError:
                return _parse_number(number), unit.lower() if unit else None
        try:
            return float(stripped), None
        except ValueError:
            return None, stripped.lower()
    return None, None


def _normalize_temperature(value: object) -> float | None:
    numeric, unit = _extract_value_and_unit(value)
    if numeric is None:
        return None
    unit_key = (unit or "c").lower()
    converter = _TEMP_UNITS.get(unit_key)
    if converter is None:
        LOGGER.debug("Unknown temperature unit %s; assuming Celsius", unit)
        converter = _TEMP_UNITS["c"]
    return converter(numeric)


def _normalize_time(value: object) -> float | None:
    numeric, unit = _extract_value_and_unit(value)
    if numeric is None:
        return None
    multiplier = _TIME_UNITS.get((unit or "s").lower())
    if multiplier is None:
        LOGGER.debug("Unknown time unit %s; defaulting to seconds", unit)
        multiplier = 1.0
    return numeric * multiplier


def _normalize_pressure(value: object) -> float | None:
    numeric, unit = _extract_value_and_unit(value)
    if numeric is None:
        return None
    multiplier = _PRESSURE_UNITS.get((unit or "pa").lower())
    if multiplier is None:
        LOGGER.debug("Unknown pressure unit %s; defaulting to Pa", unit)
        multiplier = 1.0
    return numeric * multiplier


def _coerce_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except ValueError:
        return None


def _tokenize(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value)
    if not text.strip():
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except Exception:
        pass
    delimiters = [";", "|", ","]
    for delimiter in delimiters:
        if delimiter in text:
            return [part.strip() for part in text.split(delimiter) if part.strip()]
    return [text.strip()]


def _resolve_from_dictionary(tokens: Iterable[str]) -> tuple[list[str], float]:
    resolved: list[str] = []
    confidences: list[float] = []
    for token in tokens:
        lowered = token.lower()
        if lowered in _SOLVENT_DICTIONARY:
            resolved.append(lowered)
            confidences.append(1.0)
            continue
        match = None
        for canonical, payload in _SOLVENT_DICTIONARY.items():
            aliases = payload.get("aliases", set())
            if lowered in aliases:
                match = canonical
                confidences.append(0.8)
                break
        if match is None:
            candidates = get_close_matches(lowered, list(_SOLVENT_DICTIONARY.keys()), n=1)
            if candidates:
                match = candidates[0]
                confidences.append(0.5)
        if match is None:
            resolved.append(lowered)
            confidences.append(0.1)
        else:
            resolved.append(match)
    if not resolved:
        return [], 0.0
    return resolved, sum(confidences) / len(confidences)


def normalize_conditions(row: dict) -> NormalizedConditions:
    solvent_tokens = _tokenize(row.get("solvent"))
    agents_tokens = _tokenize(row.get("agents"))
    solvents, solvent_confidence = _resolve_from_dictionary(solvent_tokens)
    atmosphere = row.get("atmosphere")
    light = row.get("light")
    phase = row.get("phase")
    ph = _coerce_float(row.get("pH"))

    return NormalizedConditions(
        temperature_k=_normalize_temperature(row.get("temperature")),
        time_s=_normalize_time(row.get("time")),
        pressure_pa=_normalize_pressure(row.get("pressure")),
        solvent=solvents,
        solvent_confidence=solvent_confidence,
        atmosphere=str(atmosphere).strip() if atmosphere else None,
        light=str(light).strip() if light else None,
        phase=str(phase).strip() if phase else None,
        ph=ph,
        additional_agents=agents_tokens,
    )

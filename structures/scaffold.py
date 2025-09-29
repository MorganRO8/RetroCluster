"""Simplified scaffold extraction helpers."""
from __future__ import annotations

from typing import Iterable


def _select_candidate(reactants: Iterable[str]) -> str | None:
    for item in reactants:
        text = item.strip()
        if text:
            return text
    return None


def main_scaffold(reactants: list[str], products: list[str], role_info: dict | None) -> str:
    """Return a deterministic scaffold identifier for the "main" substrate."""

    if role_info:
        primary = role_info.get("main_substrate")
        if isinstance(primary, str) and primary.strip():
            return primary.strip()
    candidate = _select_candidate(reactants) or _select_candidate(products)
    return candidate or "unknown"

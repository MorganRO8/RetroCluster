"""Simplified scaffold extraction helpers."""
from __future__ import annotations

import re
from typing import Iterable

_METAL_ELEMENTS = {
    "Li",
    "Na",
    "K",
    "Mg",
    "Al",
    "Ti",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "Sn",
    "Sb",
    "Te",
    "Pt",
    "Au",
    "Hg",
    "Pb",
}


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


def coarse_scaffold_family(scaffold_key: str) -> str:
    """Map a raw scaffold key to a coarse structural family."""

    text = (scaffold_key or "").strip()
    if not text or text.lower() == "unknown":
        return "unknown"

    # Split multi-fragment entries into a dedicated bucket – these tend to be
    # mixture-like scaffolds rather than a single motif.
    if "." in text:
        return "multi_fragment"

    # Organometallic complexes typically encode the metal in brackets.
    bracketed = re.findall(r"\[([A-Z][a-z]?)", text)
    if any(element in _METAL_ELEMENTS for element in bracketed):
        return "organometallic"

    lowered = text.lower()

    # Aromatic scaffolds commonly include lowercase aromatic carbons followed by
    # ring indices. Require both to avoid false positives on arbitrary strings.
    aromatic = any(f"c{digit}" in lowered for digit in "123456789")
    hetero_aromatic = aromatic and any(char in lowered for char in "nosp")

    if aromatic:
        return "aromatic_hetero" if hetero_aromatic else "aromatic"

    hetero_upper = any(symbol in text for symbol in ("N", "O", "S", "P", "F", "Cl", "Br", "I"))
    hetero_lower = any(char in lowered for char in ("n", "o", "s", "p"))

    has_carbon = "C" in text

    if has_carbon:
        return "aliphatic_hetero" if (hetero_upper or hetero_lower) else "aliphatic"

    if hetero_upper or hetero_lower:
        return "hetero_only"

    return "inorganic"

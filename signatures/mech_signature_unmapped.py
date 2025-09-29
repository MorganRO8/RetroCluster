"""Fallback signature heuristics when no reliable mapping is available."""
from __future__ import annotations

from typing import Iterable

from utils.hashing import blake3_hexdigest


def _tokenise(items: Iterable[str], prefix: str) -> list[str]:
    tokens: list[str] = []
    for item in items:
        text = item.strip()
        if not text:
            continue
        tokens.append(f"{prefix}:{text}")
    return sorted(tokens)


def mech_sig_unmapped(reactants: list[str], products: list[str]) -> dict[str, object]:
    """Create a heuristic signature using string level fingerprints."""

    reactant_tokens = _tokenise(reactants, "R")
    product_tokens = _tokenise(products, "P")
    combined = tuple(reactant_tokens + product_tokens)
    if combined:
        mech_sig_base = blake3_hexdigest("|".join(combined).encode("utf-8"))
    else:
        mech_sig_base = blake3_hexdigest(b"noop")
    return {
        "mech_sig_base": mech_sig_base,
        "mech_sig_r1": None,
        "mech_sig_r2": None,
        "event_tokens": list(combined),
        "redox_events": 0,
        "stereo_events": 0,
        "ring_events": 0,
        "radius": 0,
    }

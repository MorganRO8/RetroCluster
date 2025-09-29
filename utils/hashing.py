"""Hashing utilities with graceful fallbacks."""
from __future__ import annotations

from dataclasses import dataclass

try:  # pragma: no cover - optional dependency
    from blake3 import blake3 as _blake3  # type: ignore
except Exception:  # pragma: no cover - fallback when blake3 is unavailable
    _blake3 = None  # type: ignore

import hashlib


def blake3_hexdigest(data: bytes) -> str:
    """Return a BLAKE3 hash hex digest.

    The repository's data model specifies BLAKE3 for reaction hashes.  The
    dependency is optional to keep the development environment lightweight; when
    unavailable, we fall back to ``hashlib.blake2b`` with a matching 256-bit
    digest size so downstream components still receive a stable identifier.
    """

    if _blake3 is not None:  # pragma: no branch - fast path when available
        return _blake3(data).hexdigest()
    return hashlib.blake2b(data, digest_size=32).hexdigest()


@dataclass(frozen=True)
class HashableReaction:
    """Container for the pieces used to compute reaction hashes."""

    reaction_string: str
    conditions_string: str

    def hexdigest(self) -> str:
        payload = f"{self.reaction_string}||{self.conditions_string}".encode(
            "utf-8"
        )
        return blake3_hexdigest(payload)

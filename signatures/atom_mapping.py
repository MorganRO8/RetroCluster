"""Toy atom-mapping helpers for the unit-test harness.

The production system described in :mod:`AGENTS.md` expects a fairly capable
reaction mapper.  For the unit tests in this kata we only need a deterministic
and dependency-free approximation that provides enough structure for the
mechanism signature routines to exercise branching logic.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class MappedRxn:
    """Simple container holding the parsed reaction and pseudo events."""

    reactants: Sequence[str]
    products: Sequence[str]
    event_tokens: tuple[str, ...]


def _tokenise(molecules: Iterable[str]) -> Counter[str]:
    """Break molecules into crude character level tokens.

    The goal is to return a stable representation that changes whenever the
    molecules differ.  We purposely keep the implementation tiny to avoid
    bringing heavyweight cheminformatics libraries into the exercises.
    """

    counter: Counter[str] = Counter()
    for molecule in molecules:
        if not molecule:
            continue
        for char in molecule:
            if char.isspace():
                continue
            counter[char] += 1
    return counter


def map_reaction(rxnsmi: str) -> tuple[MappedRxn, float]:
    """Return a pseudo atom mapping and a confidence score in ``[0, 1]``.

    The mapper recognises the generic SMILES-like ``reactants>>products`` shape
    and compares the multiset of characters on the two sides.  Whenever the
    composition changes we emit simple ``"+X"``/``"-X"`` tokens describing the
    delta.  A non-empty delta yields a relatively high confidence score, which
    nudges the mechanism pipeline to take the "mapped" branch.  When the delta
    is empty the confidence is kept low, signalling that a fallback signature
    should be used instead.
    """

    if ">>" not in rxnsmi:
        raise ValueError("Expected reaction SMILES with '>>' separator")

    lhs, rhs = rxnsmi.split(">>", 1)
    reactants = tuple(token.strip() for token in lhs.split(".") if token.strip())
    products = tuple(token.strip() for token in rhs.split(".") if token.strip())

    reactant_tokens = _tokenise(reactants)
    product_tokens = _tokenise(products)

    delta = product_tokens - reactant_tokens
    reverse_delta = reactant_tokens - product_tokens

    events = []
    for atom, count in sorted(delta.items()):
        events.append(f"+{atom}:{count}")
    for atom, count in sorted(reverse_delta.items()):
        events.append(f"-{atom}:{count}")

    confidence = 0.9 if events else 0.2
    mapped = MappedRxn(reactants=reactants, products=products, event_tokens=tuple(events))
    return mapped, confidence

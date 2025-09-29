"""Simple heuristics for reaction role assignment."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass
class RoleAssignment:
    main_substrate: str | None
    role_scores: dict[str, float]


def score_molecule(smiles: str) -> float:
    """Score a molecule by an approximate size/complexity heuristic."""

    # Use a naive heuristic based on string length and the presence of aromatic
    # markers.  The real system would use heavy atom counts and ring perception
    # from RDKit; the heuristic keeps behaviour deterministic for tests.
    length_score = len(smiles)
    aromatic_bonus = 2.0 if "ar" in smiles.lower() or "c1" in smiles else 0.0
    hetero_penalty = smiles.lower().count("n") * 0.1
    return length_score + aromatic_bonus - hetero_penalty


def assign_roles(reactants: Iterable[str]) -> RoleAssignment:
    scores: dict[str, float] = {}
    for smiles in reactants:
        scores[smiles] = score_molecule(smiles)
    if not scores:
        return RoleAssignment(main_substrate=None, role_scores={})
    main = max(scores.items(), key=lambda item: item[1])[0]
    return RoleAssignment(main_substrate=main, role_scores=scores)

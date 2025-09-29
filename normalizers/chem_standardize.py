"""Utilities for reaction molecule normalization.

The real production system would rely on RDKit for thorough standardisation.  We
provide a pragmatic implementation that keeps the interface and failure
semantics while falling back to lightweight heuristics when RDKit is not
available.  The goal is to keep the rest of the pipeline deterministic and easy
to test in constrained environments.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from typing import Iterable, Sequence

try:  # pragma: no cover - optional heavy dependency
    from rdkit import Chem  # type: ignore
    from rdkit.Chem import rdMolStandardize  # type: ignore
    from rdkit.Chem import inchi  # type: ignore
except Exception:  # pragma: no cover - allow running without RDKit
    Chem = None  # type: ignore
    rdMolStandardize = None  # type: ignore
    inchi = None  # type: ignore

LOGGER = logging.getLogger(__name__)


@dataclass
class MoleculeRecord:
    """Normalized representation of a molecular entity."""

    original: str
    canonical_smiles: str
    inchi_key: str | None
    failed: bool = False
    failure_reason: str | None = None


def _load_json_list(value: str) -> list[str]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:  # pragma: no cover - control flows to fallback
        return []
    if isinstance(parsed, list):
        return [str(item) for item in parsed if str(item).strip()]
    return []


def split_entities(value: object) -> list[str]:
    """Normalize raw entity columns to a list of strings."""

    if value is None:
        return []
    if isinstance(value, list | tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("[") and stripped.endswith("]"):
            parsed = _load_json_list(stripped)
            if parsed:
                return [item.strip() for item in parsed]
        # use common delimiters; SMILES uses '.' for separate fragments so we
        # keep it as the last resort.
        for delimiter in (";", "|", ","):
            if delimiter in stripped:
                return [part.strip() for part in stripped.split(delimiter) if part.strip()]
        return [part.strip() for part in stripped.split(".") if part.strip()]
    return [str(value).strip()]


def _canonicalize_with_rdkit(smiles: str) -> tuple[str, str | None]:
    mol = Chem.MolFromSmiles(smiles)  # type: ignore[attr-defined]
    if mol is None:
        raise ValueError("RDKit could not parse SMILES")
    if rdMolStandardize is not None:  # pragma: no branch
        clean = rdMolStandardize.Cleanup(mol)  # type: ignore[attr-defined]
        uncharger = rdMolStandardize.Uncharger()  # type: ignore[attr-defined]
        mol = uncharger.uncharge(clean)
        mol = rdMolStandardize.TautomerEnumerator().Canonicalize(mol)
    canonical = Chem.MolToSmiles(mol, canonical=True)  # type: ignore[attr-defined]
    inchikey = None
    if inchi is not None:  # pragma: no branch
        try:
            inchikey = inchi.MolToInchiKey(mol)  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - RDKit edge cases
            inchikey = None
    return canonical, inchikey


def _heuristic_canonicalize(smiles: str) -> tuple[str, None]:
    canonical = "".join(smiles.split())
    return canonical, None


def canonicalize_smiles(smiles: str) -> MoleculeRecord:
    """Canonicalize a single SMILES string."""

    try:
        if Chem is not None:
            canonical, inchikey = _canonicalize_with_rdkit(smiles)
        else:  # pragma: no cover - deterministic fallback
            canonical, inchikey = _heuristic_canonicalize(smiles)
        return MoleculeRecord(
            original=smiles,
            canonical_smiles=canonical,
            inchi_key=inchikey,
        )
    except Exception as exc:  # pragma: no cover - error path is logged for audit
        LOGGER.debug("Failed to canonicalize %s: %s", smiles, exc)
        canonical, inchikey = _heuristic_canonicalize(smiles)
        return MoleculeRecord(
            original=smiles,
            canonical_smiles=canonical,
            inchi_key=inchikey,
            failed=True,
            failure_reason=str(exc),
        )


def canonicalize_entities(values: Sequence[str]) -> list[MoleculeRecord]:
    return [canonicalize_smiles(value) for value in values]


def standardize_reaction_entities(
    reactants: Sequence[str],
    products: Sequence[str],
    agents: Sequence[str],
) -> tuple[list[MoleculeRecord], list[MoleculeRecord], list[MoleculeRecord], list[str]]:
    """Standardize the reaction participants.

    Returns normalized reactants, products, agents, and a list of spectator
    molecules that were moved from the main sides to the agents list.
    """

    norm_reactants = canonicalize_entities(reactants)
    norm_products = canonicalize_entities(products)
    norm_agents = canonicalize_entities(agents)

    # Move spectators (molecules appearing identically on both sides) into agents.
    reactant_keys = {mol.canonical_smiles for mol in norm_reactants}
    product_keys = {mol.canonical_smiles for mol in norm_products}
    spectators = sorted(reactant_keys & product_keys)
    if spectators:
        norm_agents.extend(
            MoleculeRecord(original=spec, canonical_smiles=spec, inchi_key=None)
            for spec in spectators
        )
        norm_reactants = [mol for mol in norm_reactants if mol.canonical_smiles not in spectators]
        norm_products = [mol for mol in norm_products if mol.canonical_smiles not in spectators]

    norm_reactants.sort(key=lambda rec: rec.canonical_smiles)
    norm_products.sort(key=lambda rec: rec.canonical_smiles)
    norm_agents.sort(key=lambda rec: rec.canonical_smiles)

    return norm_reactants, norm_products, norm_agents, spectators


def canonical_lists(records: Iterable[MoleculeRecord]) -> list[str]:
    return [record.canonical_smiles for record in records]

"""Phase 2 – mechanism / transformation clustering."""
from __future__ import annotations

import json
import logging
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import click

try:  # pragma: no cover - optional dependency
    import pandas as pd
except Exception:  # pragma: no cover - allow CLI helpers to signal requirement
    pd = None  # type: ignore[assignment]

from signatures import map_reaction, mech_sig_from_mapping, mech_sig_unmapped
from structures import main_scaffold

LOGGER = logging.getLogger(__name__)


@dataclass
class MechanismSignature:
    rxn_vid: int
    cond_hash: str
    mech_sig_base: str
    mech_sig_r1: str | None
    mech_sig_r2: str | None
    signature_type: str
    event_tokens: list[str]
    redox_events: int
    stereo_events: int
    ring_events: int
    scaffold_key: str
    cluster_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = self.__dict__.copy()
        payload["event_tokens"] = json.dumps(self.event_tokens, sort_keys=True)
        return payload


@dataclass
class MechanismCluster:
    cond_hash: str
    cluster_id: str
    mech_sig_base: str
    rxn_vids: list[int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "cond_hash": self.cond_hash,
            "cluster_id": self.cluster_id,
            "mech_sig_base": self.mech_sig_base,
            "rxn_count": len(self.rxn_vids),
            "rxn_vids": json.dumps(sorted(self.rxn_vids)),
        }


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s - %(message)s")


def _require_pandas() -> Any:
    if pd is None:
        raise click.ClickException(
            "pandas is required for file IO operations in phase2; install pandas to continue"
        )
    return pd


def _loads(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return [text]
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
        return [str(parsed)]
    return [str(value)]


def _reaction_smiles(reactants: list[str], products: list[str]) -> str:
    lhs = ".".join(reactants)
    rhs = ".".join(products)
    return f"{lhs}>>{rhs}"


def _compute_signature_payload(row: dict[str, Any]) -> MechanismSignature:
    rxn_vid = int(row["rxn_vid"])
    cond_hash = str(row["cond_hash"])
    reactants = _loads(row.get("reactants_canonical"))
    products = _loads(row.get("products_canonical"))
    role_info = {"main_substrate": row.get("role_main_substrate")}

    signature_type = "unmapped"
    signature_payload: dict[str, Any]

    try:
        mapped, confidence = map_reaction(_reaction_smiles(reactants, products))
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.debug("Atom mapping failed for rxn_vid=%s: %s", rxn_vid, exc)
        mapped = None
        confidence = 0.0

    if mapped is not None and confidence >= 0.5:
        signature_type = "mapped"
        signature_payload = mech_sig_from_mapping(mapped)
    else:
        signature_payload = mech_sig_unmapped(reactants, products)

    scaffold_key = main_scaffold(reactants, products, role_info)

    return MechanismSignature(
        rxn_vid=rxn_vid,
        cond_hash=cond_hash,
        mech_sig_base=str(signature_payload.get("mech_sig_base")),
        mech_sig_r1=signature_payload.get("mech_sig_r1"),
        mech_sig_r2=signature_payload.get("mech_sig_r2"),
        signature_type=signature_type,
        event_tokens=list(signature_payload.get("event_tokens", [])),
        redox_events=int(signature_payload.get("redox_events", 0)),
        stereo_events=int(signature_payload.get("stereo_events", 0)),
        ring_events=int(signature_payload.get("ring_events", 0)),
        scaffold_key=scaffold_key,
    )


def _cluster_signatures(signatures: Iterable[MechanismSignature]) -> list[MechanismCluster]:
    clusters: list[MechanismCluster] = []
    by_cond: dict[str, dict[str, list[MechanismSignature]]] = {}
    for signature in signatures:
        cond_group = by_cond.setdefault(signature.cond_hash, {})
        cond_group.setdefault(signature.mech_sig_base, []).append(signature)

    for cond_hash, base_groups in sorted(by_cond.items()):
        for idx, (mech_sig_base, members) in enumerate(sorted(base_groups.items()), start=1):
            cluster_id = f"{cond_hash}-{idx}"
            for member in members:
                member.cluster_id = cluster_id
            clusters.append(
                MechanismCluster(
                    cond_hash=cond_hash,
                    cluster_id=cluster_id,
                    mech_sig_base=mech_sig_base,
                    rxn_vids=[member.rxn_vid for member in members],
                )
            )
    return clusters


def _write_parquet(records: list[dict[str, Any]], path: Path) -> None:
    pandas = _require_pandas()
    frame = pandas.DataFrame(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def _log_counter(name: str, counter: Counter[Any], limit: int = 5) -> None:
    if not counter:
        LOGGER.info("%s: no observations", name)
        return
    top = counter.most_common(limit)
    LOGGER.info("%s (top %s): %s", name, min(limit, len(counter)), top)


def _summarize_mechanism_outputs(
    signatures: list[MechanismSignature], clusters: list[MechanismCluster]
) -> None:
    if not signatures:
        LOGGER.warning("No mechanism signatures produced; skipping statistics")
        return

    type_counter = Counter(signature.signature_type for signature in signatures)
    _log_counter("Signature type distribution", type_counter)

    cond_counter = Counter(signature.cond_hash for signature in signatures)
    LOGGER.info("Mechanism condition buckets: %d", len(cond_counter))

    base_counter = Counter(
        signature.mech_sig_base for signature in signatures if signature.mech_sig_base
    )
    _log_counter("Mechanism base signature usage", base_counter)

    event_lengths = [len(signature.event_tokens) for signature in signatures]
    if event_lengths:
        LOGGER.info(
            "Event token counts: mean=%.2f median=%.2f min=%d max=%d",
            statistics.fmean(event_lengths),
            statistics.median(event_lengths),
            min(event_lengths),
            max(event_lengths),
        )

    event_counter = Counter(
        token for signature in signatures for token in signature.event_tokens
    )
    if event_counter:
        _log_counter("Most common event tokens", event_counter)
    else:
        LOGGER.warning("No event tokens present in signatures")

    scaffold_counter = Counter(
        signature.scaffold_key for signature in signatures if signature.scaffold_key
    )
    _log_counter("Scaffold usage", scaffold_counter)

    redox_with_events = sum(1 for signature in signatures if signature.redox_events)
    stereo_with_events = sum(1 for signature in signatures if signature.stereo_events)
    ring_with_events = sum(1 for signature in signatures if signature.ring_events)
    LOGGER.info(
        "Reactions with redox/stereo/ring events: %d/%d/%d",
        redox_with_events,
        stereo_with_events,
        ring_with_events,
    )

    if clusters:
        cluster_sizes = [len(cluster.rxn_vids) for cluster in clusters]
        LOGGER.info(
            "Level 2 cluster sizes: mean=%.2f median=%.2f min=%d max=%d",
            statistics.fmean(cluster_sizes),
            statistics.median(cluster_sizes),
            min(cluster_sizes),
            max(cluster_sizes),
        )
        cluster_counter = Counter(cluster.cond_hash for cluster in clusters)
        _log_counter("Clusters per condition bucket", cluster_counter)
    else:
        LOGGER.warning("No level 2 clusters were formed")


@click.command()
@click.option("--input", "input_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--cond-table", "cond_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--output-dir", "output_dir", required=True, type=click.Path(file_okay=False))
@click.option("--sample", is_flag=True, help="Process only a sample of rows for smoke tests.")
@click.option("--verbose/--quiet", default=False, show_default=True)
def main(
    input_path: str,
    cond_path: str,
    output_dir: str,
    sample: bool,
    verbose: bool,
) -> None:
    """CLI entry point for mechanism clustering."""

    _configure_logging(verbose)
    pandas = _require_pandas()

    LOGGER.info("Loading normalized reactions from %s", input_path)
    reactions = pandas.read_parquet(input_path)
    LOGGER.info("Loading condition keys from %s", cond_path)
    conditions = pandas.read_parquet(cond_path)

    if sample:
        LOGGER.info("Sampling first 1000 reactions for quick iteration")
        reactions = reactions.head(1000)
        conditions = conditions.head(1000)

    LOGGER.info("Joining tables on rxn_vid")
    merged = conditions.merge(reactions, on="rxn_vid", how="inner", suffixes=("_cond", "_rxn"))
    LOGGER.info("Computing mechanism signatures for %d reactions", len(merged))

    signatures = [_compute_signature_payload(row) for row in merged.to_dict(orient="records")]
    clusters = _cluster_signatures(signatures)

    _summarize_mechanism_outputs(signatures, clusters)

    output_dir_path = Path(output_dir)
    mechanisms_path = output_dir_path / "mechanism_sigs.parquet"
    clusters_path = output_dir_path / "clusters_level2.parquet"

    LOGGER.info("Writing %d mechanism signatures to %s", len(signatures), mechanisms_path)
    _write_parquet([signature.to_dict() for signature in signatures], mechanisms_path)

    LOGGER.info("Writing %d clusters to %s", len(clusters), clusters_path)
    _write_parquet([cluster.to_dict() for cluster in clusters], clusters_path)

    success_flag = output_dir_path / "_SUCCESS"
    success_flag.parent.mkdir(parents=True, exist_ok=True)
    success_flag.write_text("phase2 completed\n")
    LOGGER.info("Wrote success sentinel to %s", success_flag)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

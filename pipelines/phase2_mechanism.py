"""Phase 2 – mechanism / transformation clustering."""
from __future__ import annotations

import hashlib
import json
import logging
import math
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
from structures import coarse_scaffold_family, main_scaffold

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
    coarse_key: tuple[str, tuple[tuple[str, str], ...]] | None = None
    cluster_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = self.__dict__.copy()
        payload["event_tokens"] = json.dumps(self.event_tokens, sort_keys=True)
        if self.coarse_key is None:
            payload["coarse_key"] = None
            payload["coarse_key_hash"] = None
        else:
            payload["coarse_key"] = json.dumps(_serialize_coarse_key(self.coarse_key), sort_keys=True)
            payload["coarse_key_hash"] = _hash_coarse_key(self.coarse_key)
        return payload


@dataclass
class MechanismCluster:
    cond_hash: str
    cluster_id: str
    coarse_key: tuple[str, tuple[tuple[str, str], ...]]
    mech_sig_base_counts: Counter[str]
    rxn_vids: list[int]

    def to_dict(self) -> dict[str, Any]:
        representative_base = ""
        if self.mech_sig_base_counts:
            representative_base = self.mech_sig_base_counts.most_common(1)[0][0]
        return {
            "cond_hash": self.cond_hash,
            "cluster_id": self.cluster_id,
            "mech_sig_base": representative_base,
            "mech_sig_base_counts": json.dumps(dict(self.mech_sig_base_counts), sort_keys=True),
            "coarse_key": json.dumps(_serialize_coarse_key(self.coarse_key), sort_keys=True),
            "coarse_key_hash": _hash_coarse_key(self.coarse_key),
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


def _serialize_coarse_key(coarse_key: tuple[str, tuple[tuple[str, str], ...]]) -> dict[str, Any]:
    scaffold, families = coarse_key
    return {"scaffold": scaffold, "families": list(families)}


def _hash_coarse_key(coarse_key: tuple[str, tuple[tuple[str, str], ...]]) -> str:
    serialized = json.dumps(_serialize_coarse_key(coarse_key), sort_keys=True)
    return hashlib.blake2s(serialized.encode("utf-8"), digest_size=16).hexdigest()


def _bucket_event_count(count: int) -> str:
    if count <= 1:
        return "one"
    if count <= 4:
        return "few"
    return "many"


def _coarse_event_family(token: str) -> str:
    """Bucket raw event tokens into broad categories for coarse grouping."""

    if not token:
        return "noop"

    base, _, _ = token.partition(":")
    base = base.strip()
    if not base:
        return "noop"

    polarity = "neutral"
    remainder = base

    if base[0] in {"+", "-"}:
        polarity = "gain" if base[0] == "+" else "loss"
        remainder = base[1:]
    elif base[0] in {"=", "#", "~"}:
        polarity = "bond"
        remainder = base[1:]

    remainder = remainder.strip()

    if not remainder:
        kind = "other"
    elif ":" in remainder or "," in remainder:
        kind = "composite"
    elif remainder.isdigit() or any(char.isdigit() for char in remainder):
        kind = "ring_index"
    elif remainder.isalpha():
        if remainder.isupper():
            kind = "atom_upper"
        elif remainder.islower():
            kind = "atom_lower"
        else:
            kind = "atom_mixed"
    elif set(remainder) <= set("-=#"):
        kind = "bond_symbol"
    elif set(remainder) <= set("()[]{}"):  # topology / parentheses markers
        kind = "topology"
    else:
        kind = "other"

    return f"{polarity}:{kind}"


def _coarse_mechanism_key(signature: MechanismSignature) -> tuple[str, tuple[tuple[str, str], ...]]:
    families: list[str] = []
    for token in signature.event_tokens:
        family = _coarse_event_family(token)
        if family:
            families.append(family)
    if not families:
        families = ["noop"]
    counts = Counter(families)
    bucketed = tuple(sorted((family, _bucket_event_count(count)) for family, count in counts.items()))
    scaffold_family = coarse_scaffold_family(signature.scaffold_key)
    return (scaffold_family, bucketed)


def _format_coarse_key(coarse_key: tuple[str, tuple[tuple[str, str], ...]]) -> str:
    scaffold, families = coarse_key
    family_part = ",".join(f"{family}:{bucket}" for family, bucket in families) if families else "noop"
    scaffold_part = scaffold or "no_scaffold"
    return f"{scaffold_part}|{family_part}"


def _compute_percentiles(values: list[float], percentiles: Iterable[int]) -> dict[int, float]:
    if not values:
        return {}
    sorted_values = sorted(values)
    count = len(sorted_values)
    results: dict[int, float] = {}
    for percentile in percentiles:
        if percentile < 0 or percentile > 100:
            continue
        if count == 1:
            results[percentile] = float(sorted_values[0])
            continue
        rank = (percentile / 100) * (count - 1)
        lower = math.floor(rank)
        upper = math.ceil(rank)
        lower_value = sorted_values[lower]
        upper_value = sorted_values[upper]
        if lower == upper:
            value = float(lower_value)
        else:
            weight = rank - lower
            value = float(lower_value) + (float(upper_value) - float(lower_value)) * weight
        results[percentile] = value
    return results


def _format_percentiles(percentiles: dict[int, float]) -> str:
    if not percentiles:
        return ""
    ordered = sorted(percentiles.items())
    formatted = " ".join(f"pct{percentile}={value:.2f}" for percentile, value in ordered)
    return formatted


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

    signature = MechanismSignature(
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
    signature.coarse_key = _coarse_mechanism_key(signature)
    return signature


def _cluster_signatures(signatures: Iterable[MechanismSignature]) -> list[MechanismCluster]:
    clusters: list[MechanismCluster] = []
    by_cond: dict[str, dict[tuple[str, tuple[tuple[str, str], ...]], list[MechanismSignature]]] = {}
    for signature in signatures:
        cond_group = by_cond.setdefault(signature.cond_hash, {})
        coarse_key = signature.coarse_key or _coarse_mechanism_key(signature)
        signature.coarse_key = coarse_key
        cond_group.setdefault(coarse_key, []).append(signature)

    for cond_hash, coarse_groups in sorted(by_cond.items()):
        sorted_groups = sorted(coarse_groups.items(), key=lambda item: _format_coarse_key(item[0]))
        for idx, (coarse_key, members) in enumerate(sorted_groups, start=1):
            cluster_id = f"{cond_hash}-{idx}"
            for member in members:
                member.cluster_id = cluster_id
            base_counts = Counter(
                member.mech_sig_base for member in members if member.mech_sig_base
            )
            clusters.append(
                MechanismCluster(
                    cond_hash=cond_hash,
                    cluster_id=cluster_id,
                    coarse_key=coarse_key,
                    mech_sig_base_counts=base_counts,
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
        size_percentiles = _compute_percentiles([float(size) for size in cluster_sizes], [10, 25, 50, 75, 90])
        LOGGER.info(
            "Level 2 cluster sizes: mean=%.2f median=%.2f min=%d max=%d %s",
            statistics.fmean(cluster_sizes),
            statistics.median(cluster_sizes),
            min(cluster_sizes),
            max(cluster_sizes),
            _format_percentiles(size_percentiles),
        )

        base_diversity = [len(cluster.mech_sig_base_counts) for cluster in clusters]
        base_percentiles = _compute_percentiles([float(count) for count in base_diversity], [10, 25, 50, 75, 90])
        LOGGER.info(
            "Distinct mech_sig_base per cluster: mean=%.2f median=%.2f min=%d max=%d %s",
            statistics.fmean(base_diversity),
            statistics.median(base_diversity),
            min(base_diversity),
            max(base_diversity),
            _format_percentiles(base_percentiles),
        )
        cluster_counter = Counter(cluster.cond_hash for cluster in clusters)
        _log_counter("Clusters per condition bucket", cluster_counter)

        coarse_usage = Counter(
            signature.coarse_key for signature in signatures if signature.coarse_key
        )
        LOGGER.info("Unique coarse mechanism signatures: %d", len(coarse_usage))
        if coarse_usage:
            top_coarse = [
                (count, _format_coarse_key(coarse_key))
                for coarse_key, count in coarse_usage.most_common(5)
            ]
            LOGGER.info("Top coarse signatures (count, key): %s", top_coarse)

        top_clusters = [
            (len(cluster.rxn_vids), _format_coarse_key(cluster.coarse_key))
            for cluster in sorted(clusters, key=lambda c: len(c.rxn_vids), reverse=True)[:5]
        ]
        LOGGER.info("Top coarse clusters by size: %s", top_clusters)
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

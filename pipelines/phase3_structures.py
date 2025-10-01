"""Phase 3 – structural sub-clustering."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence, Any

import click

try:  # pragma: no cover - optional dependency
    import pandas as pd
except Exception:  # pragma: no cover - allow CLI helpers to signal requirement
    pd = None  # type: ignore[assignment]

from indices.build_ann import build_hnsw
from utils.hashing import blake3_hexdigest

LOGGER = logging.getLogger(__name__)


@dataclass
class StructureFeature:
    """Light-weight container for structure level fingerprints."""

    rxn_vid: int
    cid: str
    mid: str
    scaffold_key: str
    bits: set[int]
    canopy_id: str | None = None
    sid: str | None = None

    def to_record(self) -> dict[str, object]:
        """Serialise the feature for tabular storage."""

        return {
            "rxn_vid": self.rxn_vid,
            "CID": self.cid,
            "MID": self.mid,
            "SID": self.sid,
            "canopy_id": self.canopy_id,
            "scaffold_key": self.scaffold_key,
            "fingerprint_bits": json.dumps(sorted(self.bits)),
        }

    @property
    def vector(self) -> list[float]:
        """Return a deterministic dense representation for ANN building."""

        return [float(bit) for bit in sorted(self.bits)]


@dataclass
class StructureCluster:
    """Simple structural cluster description."""

    cid: str
    mid: str
    sid: str
    rxn_vids: list[int]
    exemplars: list[int]

    def to_dict(self) -> dict[str, object]:
        return {
            "CID": self.cid,
            "MID": self.mid,
            "SID": self.sid,
            "rxn_vids": json.dumps(self.rxn_vids),
            "exemplar_vids": json.dumps(self.exemplars),
            "cluster_size": len(self.rxn_vids),
        }


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s - %(message)s")


def _require_pandas() -> object:
    if pd is None:
        raise click.ClickException(
            "pandas is required for file IO operations in phase3; install pandas to continue"
        )
    return pd


def _loads_list(value: object | None) -> list[str]:
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


def _parse_rxn_vids(value: object) -> list[int]:
    rxn_vids: list[int] = []
    for token in _loads_list(value):
        text = token.strip()
        if not text:
            continue
        try:
            rxn_vids.append(int(text))
        except ValueError as exc:  # pragma: no cover - defensive conversion guard
            raise click.ClickException(
                f"Unable to parse rxn_vid value '{token}' from clusters table"
            ) from exc
    return rxn_vids


def _expand_clusters_by_rxn(clusters_frame: Any) -> Any:
    """Ensure the clusters table has one row per reaction."""

    if "rxn_vid" in clusters_frame.columns:
        return clusters_frame
    if "rxn_vids" not in clusters_frame.columns:
        raise click.ClickException(
            "clusters_level2 table must include either 'rxn_vid' or 'rxn_vids' column"
        )

    records: list[dict[str, object]] = []
    for row in clusters_frame.to_dict(orient="records"):
        rxn_vids = _parse_rxn_vids(row.get("rxn_vids"))
        if not rxn_vids:
            LOGGER.debug("Cluster %s has no member reactions; skipping", row.get("cluster_id"))
            continue
        base = {key: value for key, value in row.items() if key != "rxn_vids"}
        for rxn_vid in rxn_vids:
            member = base.copy()
            member["rxn_vid"] = int(rxn_vid)
            records.append(member)

    pandas = _require_pandas()
    if not records:
        columns = [col for col in clusters_frame.columns if col != "rxn_vids"] + ["rxn_vid"]
        return pandas.DataFrame(columns=columns)

    return pandas.DataFrame(records)


_SUFFIX_CANDIDATES: tuple[str, ...] = ("", "_lvl2", "_sig")


def _resolve_with_suffixes(row: dict[str, object], *names: str) -> object | None:
    """Return the first non-empty value among candidate column names.

    Phase 3 joins tables that frequently introduce suffixes (``_lvl2``/``_sig``).
    The helper mirrors pandas' suffix behaviour so the downstream feature builder
    can remain agnostic to the exact origin of the column.
    """

    seen: set[str] = set()
    for name in names:
        for variant in {name, name.lower()}:
            if not variant:
                continue
            for suffix in _SUFFIX_CANDIDATES:
                if suffix and variant.endswith(suffix):
                    key = variant
                elif suffix:
                    key = f"{variant}{suffix}"
                else:
                    key = variant
                if key in seen:
                    continue
                seen.add(key)
                value = row.get(key)
                if value is None:
                    continue
                if isinstance(value, str) and not value.strip():
                    continue
                return value
    return None


def _normalise_identifier(row: dict[str, object], *candidates: str, default: str) -> str:
    value = _resolve_with_suffixes(row, *candidates)
    if value is None:
        return default
    text = str(value).strip()
    return text or default


def _hash_token_to_bit(token: str, fp_bits: int) -> int:
    digest = blake3_hexdigest(token.encode("utf-8"))
    return int(digest[:8], 16) % max(fp_bits, 1)


def _build_structure_feature(row: dict[str, object], fp_bits: int = 128) -> StructureFeature:
    rxn_vid = int(row["rxn_vid"])
    cid = _normalise_identifier(row, "CID", "cid", "cond_hash", default="unknown")
    mid = _normalise_identifier(row, "MID", "mid", "cluster_id", default="unknown")
    scaffold_key = _normalise_identifier(row, "scaffold_key", default="unknown")

    tokens: set[str] = {f"scaffold:{scaffold_key}"}
    mech_sig_base = row.get("mech_sig_base")
    if mech_sig_base:
        tokens.add(f"mech:{mech_sig_base}")
    event_source = _resolve_with_suffixes(row, "event_tokens")
    for token in _loads_list(event_source):
        tokens.add(f"event:{token}")

    bits = {_hash_token_to_bit(token, fp_bits) for token in tokens}

    return StructureFeature(rxn_vid=rxn_vid, cid=cid, mid=mid, scaffold_key=scaffold_key, bits=bits)


def _tanimoto(a: Sequence[int] | set[int], b: Sequence[int] | set[int]) -> float:
    set_a = set(a)
    set_b = set(b)
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    if union == 0:
        return 0.0
    return intersection / union


def _build_canopies(features: list[StructureFeature], tanimoto_threshold: float) -> list[list[StructureFeature]]:
    remaining = list(features)
    canopies: list[list[StructureFeature]] = []
    while remaining:
        seed = remaining.pop(0)
        canopy_id = f"{seed.mid}-C{len(canopies) + 1}"
        seed.canopy_id = canopy_id
        members = [seed]
        still_pending: list[StructureFeature] = []
        for feature in remaining:
            if _tanimoto(seed.bits, feature.bits) >= tanimoto_threshold:
                feature.canopy_id = canopy_id
                members.append(feature)
            else:
                still_pending.append(feature)
        remaining = still_pending
        canopies.append(members)
    return canopies


def _assign_structure_clusters(
    features: Iterable[StructureFeature],
    tanimoto_threshold: float,
) -> list[StructureCluster]:
    by_group: dict[tuple[str, str], list[StructureFeature]] = {}
    for feature in features:
        by_group.setdefault((feature.cid, feature.mid), []).append(feature)

    clusters: list[StructureCluster] = []
    for (cid, mid), members in sorted(by_group.items()):
        canopies = _build_canopies(sorted(members, key=lambda f: f.rxn_vid), tanimoto_threshold)
        for idx, canopy in enumerate(canopies, start=1):
            sid = f"{mid}-S{idx}"
            rxn_vids = [feature.rxn_vid for feature in canopy]
            for feature in canopy:
                feature.sid = sid
            exemplars = rxn_vids[: min(3, len(rxn_vids))]
            clusters.append(
                StructureCluster(
                    cid=cid,
                    mid=mid,
                    sid=sid,
                    rxn_vids=rxn_vids,
                    exemplars=exemplars,
                )
            )
    return clusters


def _write_parquet(records: list[dict[str, object]], path: Path) -> None:
    pandas = _require_pandas()
    frame = pandas.DataFrame(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def _build_ann_indices(
    features: Iterable[StructureFeature],
    output_dir: Path,
    m: int = 32,
    efc: int = 400,
) -> None:
    index_dir = output_dir / "indices"
    index_dir.mkdir(parents=True, exist_ok=True)
    by_group: dict[tuple[str, str], list[StructureFeature]] = {}
    for feature in features:
        by_group.setdefault((feature.cid, feature.mid), []).append(feature)

    for (cid, mid), members in sorted(by_group.items()):
        vectors = [feature.vector for feature in members]
        index = build_hnsw(vectors, m=m, efc=efc)
        payload = {"cid": cid, "mid": mid, "size": len(vectors), "m": index.m, "efc": index.efc}
        target = index_dir / f"index_{cid}_{mid}.json"
        target.write_text(json.dumps(payload, sort_keys=True, indent=2))


@click.command()
@click.option("--mechanism-sigs", "mechanism_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--clusters-level2", "clusters_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--output-dir", "output_dir", required=True, type=click.Path(file_okay=False))
@click.option("--sample", is_flag=True, help="Process only a sample of rows for smoke tests.")
@click.option("--fp-bits", default=128, show_default=True, type=int)
@click.option("--canopy-threshold", default=0.6, show_default=True, type=float)
@click.option("--verbose/--quiet", default=False, show_default=True)
def main(
    mechanism_path: str,
    clusters_path: str,
    output_dir: str,
    sample: bool,
    fp_bits: int,
    canopy_threshold: float,
    verbose: bool,
) -> None:
    """CLI entry point for the structural clustering phase."""

    _configure_logging(verbose)
    pandas = _require_pandas()

    LOGGER.info("Loading mechanism signatures from %s", mechanism_path)
    mechanisms = pandas.read_parquet(mechanism_path)
    LOGGER.info("Loading level 2 clusters from %s", clusters_path)
    clusters = pandas.read_parquet(clusters_path)
    clusters = _expand_clusters_by_rxn(clusters)

    if sample:
        LOGGER.info("Sampling first 1000 rows for quick iteration")
        mechanisms = mechanisms.head(1000)
        clusters = clusters.head(1000)

    LOGGER.info("Joining tables on rxn_vid")
    merged = clusters.merge(mechanisms, on="rxn_vid", how="inner", suffixes=("_lvl2", "_sig"))
    LOGGER.info("Computing structure features for %d reactions", len(merged))

    features = [
        _build_structure_feature(row, fp_bits=fp_bits)
        for row in merged.to_dict(orient="records")
    ]

    LOGGER.info("Assigning structural clusters using Tanimoto canopy threshold %.2f", canopy_threshold)
    clusters_lvl3 = _assign_structure_clusters(features, tanimoto_threshold=canopy_threshold)

    output_dir_path = Path(output_dir)
    features_path = output_dir_path / "structure_feats.parquet"
    clusters_path_lvl3 = output_dir_path / "clusters_level3.parquet"

    LOGGER.info("Writing %d structure features to %s", len(features), features_path)
    _write_parquet([feature.to_record() for feature in features], features_path)

    LOGGER.info("Writing %d structural clusters to %s", len(clusters_lvl3), clusters_path_lvl3)
    _write_parquet([cluster.to_dict() for cluster in clusters_lvl3], clusters_path_lvl3)

    LOGGER.info("Building ANN indices for %d groups", len({(f.cid, f.mid) for f in features}))
    _build_ann_indices(features, output_dir_path)

    success_flag = output_dir_path / "_SUCCESS"
    success_flag.parent.mkdir(parents=True, exist_ok=True)
    success_flag.write_text("phase3 completed\n")
    LOGGER.info("Wrote success sentinel to %s", success_flag)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

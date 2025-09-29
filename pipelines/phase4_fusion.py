"""Phase 4 – fusion of multi-level clusters into final IDs."""
from __future__ import annotations

import json
import logging
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping

import click

try:  # pragma: no cover - optional dependency in minimal environments
    import pandas as pd
except Exception:  # pragma: no cover - allow CLI helpers to signal requirement
    pd = None  # type: ignore[assignment]

LOGGER = logging.getLogger(__name__)


@dataclass
class ClusterRecord:
    """Mapping from a reaction to its hierarchical cluster IDs."""

    rxn_vid: int
    cid: str
    mid: str
    sid: str

    def to_dict(self) -> dict[str, Any]:
        return {"rxn_vid": self.rxn_vid, "CID": self.cid, "MID": self.mid, "SID": self.sid}


@dataclass
class ClusterCard:
    """Rich summary of a `{CID, MID, SID}` cluster."""

    cid: str
    mid: str
    sid: str
    size: int
    dominant_scaffold: str | None
    top_solvents: list[tuple[str, int]]
    top_temperatures: list[tuple[str, int]]
    mechanism_summary: list[tuple[str, int]]
    yield_mean: float | None
    yield_median: float | None
    yield_count: int
    exemplars: list[str]

    def to_dict(self) -> dict[str, Any]:
        def _pairs(pairs: Iterable[tuple[str, int]]) -> list[dict[str, Any]]:
            return [{"value": value, "count": count} for value, count in pairs]

        return {
            "CID": self.cid,
            "MID": self.mid,
            "SID": self.sid,
            "size": self.size,
            "dominant_scaffold": self.dominant_scaffold,
            "top_solvents": _pairs(self.top_solvents),
            "top_temperatures": _pairs(self.top_temperatures),
            "mechanism_summary": _pairs(self.mechanism_summary),
            "yield_mean": self.yield_mean,
            "yield_median": self.yield_median,
            "yield_count": self.yield_count,
            "exemplars": list(self.exemplars),
        }


@dataclass
class CrossViewEdge:
    """Edge linking clusters that share a mechanism/structure across CIDs."""

    kind: str
    source_cid: str
    target_cid: str
    shared_id: str
    weight: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "source_cid": self.source_cid,
            "target_cid": self.target_cid,
            "shared_id": self.shared_id,
            "weight": self.weight,
        }


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s - %(message)s")


def _require_pandas() -> Any:
    if pd is None:
        raise click.ClickException(
            "pandas is required for file IO operations in phase4; install pandas to continue"
        )
    return pd


def _loads_rxn_vids(value: Any) -> list[int]:
    """Best-effort parsing of reaction member lists from JSON strings."""

    def _normalise(item: Any) -> int | None:
        if item is None:
            return None
        if isinstance(item, (int, float)) and not math.isnan(float(item)):
            return int(item)
        text = str(item).strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            try:
                return int(float(text))
            except ValueError:
                return None

    if value is None:
        return []
    if isinstance(value, list):
        return [vid for vid in (_normalise(item) for item in value) if vid is not None]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            tokens = [token.strip() for token in text.replace(";", ",").split(",") if token.strip()]
            return [vid for vid in (_normalise(token) for token in tokens) if vid is not None]
        if isinstance(parsed, list):
            return [vid for vid in (_normalise(item) for item in parsed) if vid is not None]
        return [vid for vid in (_normalise(parsed),) if vid is not None]
    return [vid for vid in (_normalise(value),) if vid is not None]


def _loads_json_list(value: Any) -> list[str]:
    """Parse a JSON-like payload into a list of strings."""

    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            tokens = [token.strip() for token in text.replace(";", ",").split(",") if token.strip()]
            return tokens
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
        return [str(parsed).strip()]
    return [str(value).strip()]


def _top_values(values: Iterable[str], limit: int = 3) -> list[tuple[str, int]]:
    counter = Counter(value for value in values if value)
    return sorted(counter.items(), key=lambda item: (-item[1], item[0]))[:limit]


def _yield_summary(values: Iterable[float]) -> dict[str, Any]:
    numeric: list[float] = []
    for value in values:
        try:
            numeric.append(float(value))
        except (TypeError, ValueError):
            continue
        else:
            if math.isnan(numeric[-1]):
                numeric.pop()
    if not numeric:
        return {"mean": None, "median": None, "count": 0}
    return {
        "mean": statistics.mean(numeric),
        "median": statistics.median(numeric),
        "count": len(numeric),
    }


def _format_temperature(value: Any) -> str | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        text = str(value).strip()
        return text or None
    if math.isnan(numeric):
        return None
    return f"{int(round(numeric))}K"


def _select_exemplars(rxn_vids: Iterable[int], limit: int = 5) -> list[str]:
    unique_vids = list(dict.fromkeys(sorted(rxn_vids)))
    return [f"rxn_{vid}.svg" for vid in unique_vids[:limit]]


def _merge_cluster_assignments(
    level1: Any, level2: Any, level3: Any
) -> list[ClusterRecord]:
    """Join level 1–3 cluster tables into a per-reaction assignment list."""

    cid_lookup: dict[int, str] = {}
    if level1 is not None:
        for row in level1.to_dict(orient="records"):
            cid = str(row.get("CID") or row.get("cid") or "").strip()
            for rxn_vid in _loads_rxn_vids(row.get("rxn_vids") or row.get("rxn_vid")):
                cid_lookup[rxn_vid] = cid

    mid_lookup: dict[int, tuple[str, str]] = {}
    if level2 is not None:
        for row in level2.to_dict(orient="records"):
            cid = str(row.get("CID") or row.get("cid") or "").strip()
            mid = str(row.get("MID") or row.get("mid") or row.get("cluster_id") or "").strip()
            for rxn_vid in _loads_rxn_vids(row.get("rxn_vids") or row.get("rxn_vid")):
                mid_lookup[rxn_vid] = (cid or cid_lookup.get(rxn_vid, ""), mid)

    assignments: dict[int, ClusterRecord] = {}
    if level3 is not None:
        for row in level3.to_dict(orient="records"):
            cid = str(row.get("CID") or row.get("cid") or "").strip()
            mid = str(row.get("MID") or row.get("mid") or "").strip()
            sid = str(row.get("SID") or row.get("sid") or row.get("cluster_id") or "").strip()
            for rxn_vid in _loads_rxn_vids(row.get("rxn_vids") or row.get("rxn_vid")):
                resolved_cid = cid or mid_lookup.get(rxn_vid, (cid_lookup.get(rxn_vid, ""), ""))[0]
                resolved_mid = mid or mid_lookup.get(rxn_vid, ("", ""))[1]
                assignments[rxn_vid] = ClusterRecord(
                    rxn_vid=rxn_vid,
                    cid=resolved_cid,
                    mid=resolved_mid,
                    sid=sid,
                )

    # Ensure we cover reactions that may have been present in lower levels only.
    for rxn_vid, (cid, mid) in mid_lookup.items():
        assignments.setdefault(
            rxn_vid,
            ClusterRecord(rxn_vid=rxn_vid, cid=cid or cid_lookup.get(rxn_vid, ""), mid=mid, sid=""),
        )
    for rxn_vid, cid in cid_lookup.items():
        assignments.setdefault(
            rxn_vid, ClusterRecord(rxn_vid=rxn_vid, cid=cid, mid="", sid="")
        )

    missing = [record.rxn_vid for record in assignments.values() if not (record.cid and record.mid and record.sid)]
    if missing:
        raise click.ClickException(
            f"Missing cluster IDs for rxn_vids: {sorted(missing)[:10]} (total={len(missing)})"
        )

    return sorted(assignments.values(), key=lambda record: record.rxn_vid)


def _build_cluster_cards(
    assignments: Iterable[ClusterRecord],
    structure_frame: Any,
    mechanism_frame: Any,
    reaction_frame: Any,
    *,
    top_k: int = 3,
    exemplar_limit: int = 5,
) -> dict[tuple[str, str, str], ClusterCard]:
    """Generate cluster cards by aggregating multi-source features."""

    scaffold_map: dict[int, list[str]] = defaultdict(list)
    if structure_frame is not None:
        for row in structure_frame.to_dict(orient="records"):
            try:
                rxn_vid = int(row["rxn_vid"])
            except (KeyError, TypeError, ValueError):
                continue
            scaffold = str(row.get("scaffold_key") or "").strip()
            if scaffold:
                scaffold_map[rxn_vid].append(scaffold)

    token_map: dict[int, list[str]] = defaultdict(list)
    if mechanism_frame is not None:
        for row in mechanism_frame.to_dict(orient="records"):
            try:
                rxn_vid = int(row["rxn_vid"])
            except (KeyError, TypeError, ValueError):
                continue
            for token in _loads_json_list(row.get("event_tokens")):
                token_map[rxn_vid].append(token)
            base = row.get("mech_sig_base")
            if base:
                token_map[rxn_vid].append(str(base))

    reaction_map: dict[int, dict[str, Any]] = {}
    if reaction_frame is not None:
        for row in reaction_frame.to_dict(orient="records"):
            try:
                rxn_vid = int(row["rxn_vid"])
            except (KeyError, TypeError, ValueError):
                continue
            solvents = _loads_json_list(row.get("solvent_normalized"))
            solvent = solvents[0] if solvents else None
            reaction_map[rxn_vid] = {
                "solvent": solvent,
                "temperature": row.get("temperature_K"),
                "yield": row.get("yield"),
            }

    grouped: dict[tuple[str, str, str], dict[str, list[Any]]] = defaultdict(
        lambda: {
            "rxn_vids": [],
            "scaffolds": [],
            "solvents": [],
            "temperatures": [],
            "yields": [],
            "tokens": [],
        }
    )

    for record in assignments:
        key = (record.cid, record.mid, record.sid)
        bucket = grouped[key]
        bucket["rxn_vids"].append(record.rxn_vid)
        bucket["scaffolds"].extend(scaffold_map.get(record.rxn_vid, []))
        info = reaction_map.get(record.rxn_vid, {})
        if solvent := info.get("solvent"):
            bucket["solvents"].append(solvent)
        if temp := _format_temperature(info.get("temperature")):
            bucket["temperatures"].append(temp)
        if (yield_value := info.get("yield")) is not None:
            try:
                bucket["yields"].append(float(yield_value))
            except (TypeError, ValueError):
                LOGGER.debug("Skipping non-numeric yield for rxn_vid=%s", record.rxn_vid)
        bucket["tokens"].extend(token_map.get(record.rxn_vid, []))

    cards: dict[tuple[str, str, str], ClusterCard] = {}
    for key, payload in grouped.items():
        cid, mid, sid = key
        rxn_vids = payload["rxn_vids"]
        scaffolds = _top_values(payload["scaffolds"], limit=1)
        solvents = _top_values(payload["solvents"], limit=top_k)
        temperatures = _top_values(payload["temperatures"], limit=top_k)
        tokens = _top_values(payload["tokens"], limit=max(top_k, 5))
        yield_stats = _yield_summary(payload["yields"])
        cards[key] = ClusterCard(
            cid=cid,
            mid=mid,
            sid=sid,
            size=len(rxn_vids),
            dominant_scaffold=scaffolds[0][0] if scaffolds else None,
            top_solvents=solvents,
            top_temperatures=temperatures,
            mechanism_summary=tokens,
            yield_mean=yield_stats["mean"],
            yield_median=yield_stats["median"],
            yield_count=yield_stats["count"],
            exemplars=_select_exemplars(rxn_vids, limit=exemplar_limit),
        )

    return cards


def _build_cross_view_edges(assignments: Iterable[ClusterRecord]) -> list[CrossViewEdge]:
    by_mid: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    by_sid: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for record in assignments:
        by_mid[record.mid][record.cid] += 1
        by_sid[record.sid][record.cid] += 1

    edges: list[CrossViewEdge] = []
    for kind, lookup in (("MID", by_mid), ("SID", by_sid)):
        for shared_id, cid_counts in lookup.items():
            if len(cid_counts) <= 1:
                continue
            for source, target in combinations(sorted(cid_counts), 2):
                weight = min(cid_counts[source], cid_counts[target])
                edges.append(
                    CrossViewEdge(
                        kind=kind,
                        source_cid=source,
                        target_cid=target,
                        shared_id=shared_id,
                        weight=weight,
                    )
                )
    return edges


def _write_clusters_final(assignments: Iterable[ClusterRecord], cards: Mapping[tuple[str, str, str], ClusterCard], path: Path) -> None:
    pandas = _require_pandas()
    records: list[dict[str, Any]] = []
    for record in assignments:
        card = cards.get((record.cid, record.mid, record.sid))
        records.append(
            {
                "rxn_vid": record.rxn_vid,
                "CID": record.cid,
                "MID": record.mid,
                "SID": record.sid,
                "cluster_card": json.dumps(card.to_dict(), sort_keys=True) if card else None,
            }
        )
    frame = pandas.DataFrame(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def _write_jsonl(records: Iterable[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True))
            handle.write("\n")


@click.command()
@click.option("--clusters-level1", "level1_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--clusters-level2", "level2_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--clusters-level3", "level3_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--mechanism-sigs", "mechanism_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--structure-feats", "structure_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--reactions", "reactions_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--output-dir", "output_dir", required=True, type=click.Path(file_okay=False))
@click.option("--sample", is_flag=True, help="Process only a sample of rows for smoke tests.")
@click.option("--verbose/--quiet", default=False, show_default=True)
def main(
    level1_path: str,
    level2_path: str,
    level3_path: str,
    mechanism_path: str,
    structure_path: str,
    reactions_path: str,
    output_dir: str,
    sample: bool,
    verbose: bool,
) -> None:
    """CLI entry point for the fusion phase."""

    _configure_logging(verbose)
    pandas = _require_pandas()

    LOGGER.info("Loading cluster tables")
    level1 = pandas.read_parquet(level1_path)
    level2 = pandas.read_parquet(level2_path)
    level3 = pandas.read_parquet(level3_path)

    LOGGER.info("Loading mechanism signatures from %s", mechanism_path)
    mechanisms = pandas.read_parquet(mechanism_path)
    LOGGER.info("Loading structure features from %s", structure_path)
    structures = pandas.read_parquet(structure_path)
    LOGGER.info("Loading normalized reactions from %s", reactions_path)
    reactions = pandas.read_parquet(reactions_path)

    if sample:
        LOGGER.info("Sampling first 1000 rows per table for quick iteration")
        level1 = level1.head(1000)
        level2 = level2.head(1000)
        level3 = level3.head(1000)
        mechanisms = mechanisms.head(1000)
        structures = structures.head(1000)
        reactions = reactions.head(1000)

    assignments = _merge_cluster_assignments(level1, level2, level3)
    LOGGER.info("Merged %d assignments", len(assignments))

    cards = _build_cluster_cards(assignments, structures, mechanisms, reactions)
    LOGGER.info("Built %d cluster cards", len(cards))

    edges = _build_cross_view_edges(assignments)
    LOGGER.info("Constructed %d cross-view edges", len(edges))

    output_dir_path = Path(output_dir)
    final_path = output_dir_path / "clusters_final.parquet"
    stats_path = output_dir_path / "cluster_stats.jsonl"

    LOGGER.info("Writing final cluster assignments to %s", final_path)
    _write_clusters_final(assignments, cards, final_path)

    LOGGER.info("Writing cluster stats and cross-view edges to %s", stats_path)
    stats_records = [
        {"type": "cluster_card", **card.to_dict()} for card in sorted(cards.values(), key=lambda c: (c.cid, c.mid, c.sid))
    ]
    stats_records.extend(
        {"type": "cross_view_edge", **edge.to_dict()} for edge in sorted(edges, key=lambda e: (e.kind, e.shared_id, e.source_cid, e.target_cid))
    )
    _write_jsonl(stats_records, stats_path)

    success_flag = output_dir_path / "_SUCCESS"
    success_flag.parent.mkdir(parents=True, exist_ok=True)
    success_flag.write_text("phase4 completed\n")
    LOGGER.info("Wrote success sentinel to %s", success_flag)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

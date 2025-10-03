#!/usr/bin/env python3
"""Visual diagnostics for mechanism clusters."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional, Sequence

import importlib.util

if importlib.util.find_spec("matplotlib") is None:
    raise SystemExit("matplotlib is required for dashboards.cluster_coherence; install matplotlib first")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.switch_backend("Agg")


# --------- Parsing helpers -------------------------------------------------

def _load_json_mapping(value: object) -> dict[str, int]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return {str(k): int(v) for k, v in value.items()}
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        try:
            loaded = json.loads(text)
        except json.JSONDecodeError:
            return {text: 1}
        if isinstance(loaded, dict):
            return {str(k): int(v) for k, v in loaded.items()}
        if isinstance(loaded, list):
            counter: Counter[str] = Counter()
            for item in loaded:
                counter[str(item)] += 1
            return dict(counter)
        return {str(loaded): 1}
    return {str(value): 1}


def _load_json_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            loaded = json.loads(text)
        except json.JSONDecodeError:
            return [text]
        if isinstance(loaded, list):
            return [str(item) for item in loaded]
        return [str(loaded)]
    return [str(value)]


def _parse_coarse_key(value: object) -> tuple[str, tuple[tuple[str, str], ...]] | tuple[None, tuple[()]]:
    if value is None:
        return (None, tuple())
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return (None, tuple())
        try:
            loaded = json.loads(text)
        except json.JSONDecodeError:
            return (value, tuple())
    elif isinstance(value, dict):
        loaded = value
    else:
        return (str(value), tuple())

    scaffold = loaded.get("scaffold") if isinstance(loaded, dict) else None
    families_raw: Iterable[Sequence[str]] = loaded.get("families", []) if isinstance(loaded, dict) else []
    families: list[tuple[str, str]] = []
    for item in families_raw:
        if not isinstance(item, Sequence) or len(item) < 2:
            continue
        family = str(item[0])
        bucket = str(item[1])
        families.append((family, bucket))
    families.sort()
    return (str(scaffold) if scaffold is not None else None, tuple(families))


# --------- Metric helpers --------------------------------------------------

def _shannon_entropy(counts: Iterable[int]) -> float:
    total = float(sum(max(int(c), 0) for c in counts))
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counts:
        if count <= 0:
            continue
        p = float(count) / total
        entropy -= p * math.log(p, 2)
    return entropy


def _top_fraction(counts: Iterable[int]) -> float:
    prepared = [int(c) for c in counts if int(c) > 0]
    if not prepared:
        return 0.0
    total = float(sum(prepared))
    return float(max(prepared)) / total if total > 0 else 0.0


def _coarse_event_family(token: str) -> str:
    if not token:
        return "noop"
    base = token.split(":", 1)[0].strip()
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
    elif remainder.isalpha():
        if remainder.isupper():
            kind = "atom_upper"
        elif remainder.islower():
            kind = "atom_lower"
        else:
            kind = "atom_mixed"
    elif any(ch.isdigit() for ch in remainder):
        kind = "ring_index"
    elif set(remainder) <= set("-=#"):
        kind = "bond_symbol"
    elif set(remainder) <= set("()[]{}"):
        kind = "topology"
    else:
        kind = "other"
    return f"{polarity}:{kind}"


# --------- Data wrangling --------------------------------------------------

def load_clusters(path: Path) -> pd.DataFrame:
    clusters = pd.read_parquet(path)
    if "cluster_id" not in clusters.columns:
        raise ValueError("clusters table must include 'cluster_id'")
    if "rxn_count" not in clusters.columns and "rxn_vids" in clusters.columns:
        clusters["rxn_count"] = clusters["rxn_vids"].apply(lambda text: len(_load_json_list(text)))
    clusters["rxn_count"] = clusters["rxn_count"].astype(int)
    counts = clusters["mech_sig_base_counts"].apply(_load_json_mapping)
    clusters["base_diversity"] = counts.apply(lambda mapping: len(mapping))
    clusters["base_entropy"] = counts.apply(lambda mapping: _shannon_entropy(mapping.values()))
    clusters["top1_base_share"] = counts.apply(lambda mapping: _top_fraction(mapping.values()))
    coarse = clusters.get("coarse_key")
    if coarse is not None:
        parsed = coarse.apply(_parse_coarse_key)
        clusters["coarse_scaffold"] = [item[0] for item in parsed]
        clusters["coarse_families"] = [item[1] for item in parsed]
    else:
        clusters["coarse_scaffold"] = None
        clusters["coarse_families"] = None
    return clusters


def load_signatures(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if path is None:
        return None
    signatures = pd.read_parquet(path)
    if "cluster_id" not in signatures.columns:
        return None
    signatures["event_tokens_list"] = signatures.get("event_tokens", pd.Series([], dtype=object)).apply(
        _load_json_list
    )
    if "signature_type" in signatures.columns:
        signatures["is_mapped"] = signatures["signature_type"].astype(str).str.lower().eq("mapped")
    else:
        signatures["is_mapped"] = False
    return signatures


def aggregate_signature_metrics(signatures: pd.DataFrame) -> pd.DataFrame:
    if signatures.empty:
        return pd.DataFrame(columns=["cluster_id", "event_entropy", "event_token_mean", "mapped_share"])

    def summarise(group: pd.DataFrame) -> dict[str, float]:
        tokens: list[str] = []
        for row_tokens in group["event_tokens_list"]:
            tokens.extend(row_tokens)
        families = [_coarse_event_family(token) for token in tokens]
        event_entropy = _shannon_entropy(Counter(families).values())
        token_counts = [len(row_tokens) for row_tokens in group["event_tokens_list"]]
        mean_tokens = float(np.mean(token_counts)) if token_counts else 0.0
        mapped_share = float(group["is_mapped"].mean()) if "is_mapped" in group else 0.0
        return {
            "event_entropy": event_entropy,
            "event_token_mean": mean_tokens,
            "mapped_share": mapped_share,
        }

    records = [
        {"cluster_id": cluster_id, **summarise(group)}
        for cluster_id, group in signatures.groupby("cluster_id", sort=False)
    ]
    return pd.DataFrame.from_records(records)


def enrich_with_signatures(clusters: pd.DataFrame, signatures: Optional[pd.DataFrame]) -> pd.DataFrame:
    if signatures is None:
        clusters["event_entropy"] = np.nan
        clusters["event_token_mean"] = np.nan
        clusters["mapped_share"] = np.nan
        return clusters
    metrics = aggregate_signature_metrics(signatures)
    merged = clusters.merge(metrics, on="cluster_id", how="left")
    return merged


# --------- Plot helpers ----------------------------------------------------

def ensure_outdir(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)


def savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def fig_cluster_size_hist(df: pd.DataFrame, outdir: Path) -> Path:
    plt.figure()
    sizes = df["rxn_count"].astype(int)
    try:
        q1, q3 = np.percentile(sizes, [25, 75])
        iqr = q3 - q1
        bin_width = 2 * iqr / (len(sizes) ** (1 / 3)) if len(sizes) > 0 else 1
        span = sizes.max() - sizes.min()
        bins = max(5, min(80, int(span / bin_width))) if bin_width > 0 else 30
    except Exception:
        bins = 30
    plt.hist(sizes, bins=bins)
    plt.xlabel("Cluster size (rxn_count)")
    plt.ylabel("Count of clusters")
    plt.title("Distribution of mechanism cluster sizes")
    path = outdir / "01_cluster_size_hist.png"
    savefig(path)
    return path


def fig_cumulative_coverage(df: pd.DataFrame, outdir: Path) -> Path:
    plt.figure()
    sizes = np.sort(df["rxn_count"].astype(int).values)[::-1]
    total = sizes.sum() or 1
    cum = np.cumsum(sizes) / total
    x = np.arange(1, len(sizes) + 1) / max(len(sizes), 1)
    plt.plot(x, cum, marker="o", linewidth=1)
    plt.xlabel("Fraction of clusters (largest to smallest)")
    plt.ylabel("Fraction of reactions covered")
    plt.title("Cumulative coverage by cluster")
    path = outdir / "02_cumulative_coverage.png"
    savefig(path)
    return path


def fig_size_vs_entropy(df: pd.DataFrame, outdir: Path) -> Path:
    plt.figure()
    plt.scatter(df["rxn_count"], df["base_entropy"], s=30, alpha=0.6)
    plt.xlabel("Cluster size (rxn_count)")
    plt.ylabel("Base signature entropy (bits)")
    plt.title("Cluster size vs base signature entropy")
    if len(df) > 2:
        corr = np.corrcoef(df["rxn_count"], df["base_entropy"])[0, 1]
        plt.annotate(f"Pearson r = {corr:.3f}", xy=(0.05, 0.95), xycoords="axes fraction", va="top", ha="left")
    path = outdir / "03_size_vs_base_entropy.png"
    savefig(path)
    return path


def fig_size_vs_top1(df: pd.DataFrame, outdir: Path) -> Path:
    plt.figure()
    plt.scatter(df["rxn_count"], df["top1_base_share"], s=30, alpha=0.6)
    plt.xlabel("Cluster size (rxn_count)")
    plt.ylabel("Top1 base signature share")
    plt.title("Cluster size vs dominant signature share")
    path = outdir / "04_size_vs_top1_share.png"
    savefig(path)
    return path


def fig_entropy_boxplot_by_scaffold(df: pd.DataFrame, outdir: Path) -> Optional[Path]:
    if "coarse_scaffold" not in df.columns:
        return None
    scaffolds = [scaffold for scaffold in df["coarse_scaffold"].dropna().unique() if scaffold]
    if len(scaffolds) < 2:
        return None
    plt.figure()
    data = [df.loc[df["coarse_scaffold"] == scaffold, "base_entropy"].dropna().values for scaffold in scaffolds]
    plt.boxplot(data, labels=scaffolds, vert=True)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Base signature entropy (bits)")
    plt.title("Entropy by coarse scaffold family")
    path = outdir / "05_entropy_by_scaffold.png"
    savefig(path)
    return path


def fig_entropy_vs_event_entropy(df: pd.DataFrame, outdir: Path) -> Optional[Path]:
    if "event_entropy" not in df.columns or df["event_entropy"].isna().all():
        return None
    plt.figure()
    subset = df.dropna(subset=["event_entropy", "base_entropy"])
    if subset.empty:
        plt.close()
        return None
    plt.scatter(subset["base_entropy"], subset["event_entropy"], s=30, alpha=0.6)
    plt.xlabel("Base signature entropy (bits)")
    plt.ylabel("Event family entropy (bits)")
    plt.title("Signature vs event diversity")
    if len(subset) > 2:
        corr = np.corrcoef(subset["base_entropy"], subset["event_entropy"])[0, 1]
        plt.annotate(f"Pearson r = {corr:.3f}", xy=(0.05, 0.95), xycoords="axes fraction", va="top", ha="left")
    path = outdir / "06_entropy_vs_event_entropy.png"
    savefig(path)
    return path


def fig_mapped_share_hist(df: pd.DataFrame, outdir: Path) -> Optional[Path]:
    if "mapped_share" not in df.columns or df["mapped_share"].isna().all():
        return None
    plt.figure()
    subset = df["mapped_share"].dropna()
    if subset.empty:
        plt.close()
        return None
    plt.hist(subset, bins=20, range=(0.0, 1.0))
    plt.xlabel("Mapped signature share")
    plt.ylabel("Count of clusters")
    plt.title("Distribution of mapped signature coverage")
    path = outdir / "07_mapped_share_hist.png"
    savefig(path)
    return path


# --------- CLI -------------------------------------------------------------

def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize mechanism cluster coherence")
    parser.add_argument(
        "--clusters",
        type=Path,
        required=True,
        help="Path to clusters_level2.parquet produced by phase 2",
    )
    parser.add_argument(
        "--signatures",
        type=Path,
        default=None,
        help="Optional path to mechanism_sigs.parquet for per-cluster event stats",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("cluster_figs"),
        help="Output directory for generated figures",
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    clusters = load_clusters(args.clusters)
    signatures = load_signatures(args.signatures) if args.signatures else None
    clusters = enrich_with_signatures(clusters, signatures)

    ensure_outdir(args.outdir)

    print("\n=== Cluster summary metrics ===")
    print(f"Clusters: {len(clusters)}")
    print(f"Reactions covered: {int(clusters['rxn_count'].sum())}")
    print(
        "Base entropy stats (bits): min={:.2f} median={:.2f} p90={:.2f} max={:.2f}".format(
            clusters["base_entropy"].min(),
            clusters["base_entropy"].median(),
            np.percentile(clusters["base_entropy"], 90),
            clusters["base_entropy"].max(),
        )
    )
    if "event_entropy" in clusters.columns and clusters["event_entropy"].notna().any():
        subset = clusters["event_entropy"].dropna()
        print(
            "Event entropy stats (bits): min={:.2f} median={:.2f} p90={:.2f} max={:.2f}".format(
                subset.min(), subset.median(), np.percentile(subset, 90), subset.max()
            )
        )
    if len(clusters) > 2:
        corr = np.corrcoef(clusters["rxn_count"], clusters["base_entropy"])[0, 1]
        print(f"corr(rxn_count, base_entropy) = {corr:.4f}")
        corr2 = np.corrcoef(clusters["rxn_count"], clusters["top1_base_share"])[0, 1]
        print(f"corr(rxn_count, top1_base_share) = {corr2:.4f}")

    outputs: list[Path] = []
    outputs.append(fig_cluster_size_hist(clusters, args.outdir))
    outputs.append(fig_cumulative_coverage(clusters, args.outdir))
    outputs.append(fig_size_vs_entropy(clusters, args.outdir))
    outputs.append(fig_size_vs_top1(clusters, args.outdir))
    out_entropy_by_scaffold = fig_entropy_boxplot_by_scaffold(clusters, args.outdir)
    if out_entropy_by_scaffold:
        outputs.append(out_entropy_by_scaffold)
    out_entropy_vs_event = fig_entropy_vs_event_entropy(clusters, args.outdir)
    if out_entropy_vs_event:
        outputs.append(out_entropy_vs_event)
    out_mapped_hist = fig_mapped_share_hist(clusters, args.outdir)
    if out_mapped_hist:
        outputs.append(out_mapped_hist)

    print("\nWrote figures:")
    for path in outputs:
        print(f" - {path}")


if __name__ == "__main__":
    main()

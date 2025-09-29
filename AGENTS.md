This document specifies **end‑to‑end implementation details** for automation agents (or human contributors) to build the full pipeline. Each phase includes: **Inputs → Outputs**, **Specs**, **Acceptance tests**, and **Performance targets**.

> All file/glob paths and constants referenced below are configurable via `configs/local.yaml`, `configs/hparams.yaml`, and `configs/bins.yaml`.

---

## Global conventions

**IDs**

* `rxn_id`: immutable identifier from source ingest.
* `rxn_vid`: versioned internal ID after canonicalization (monotonic within repo).
* `reaction_hash_v1`: dedup hash over normalized reaction string + conditions.
* `CID`, `MID`, `SID`: integer cluster IDs for conditions, mechanism, structures.

**Partitioning**

* Shard key: `cond_hash_prefix` (first N bits of `cond_hash`, N∈[10,14]).
* Target shard size: 1–5M reactions; enforce via salt if necessary.

**Compression & formats**

* Columnar Parquet (Snappy/Zstd), 256MB row groups, dictionary encoding on low‑card fields.

**Presence masks**

* Bitfield `present_mask` indicating which condition fields were present; must match for bucket equality unless wildcard logic is explicitly enabled.

---

## Phase 0 — Data Readiness

**Input**: `reactions_raw` (Parquet/CSV/JSON), columns (minimum):

* `rxn_id`, `reactants`, `products`, `agents` (reagents/catalysts), `solvent`, `temperature`, `time`, `pressure`, `atmosphere`, `light`, `pH`, `potential`, `phase`, `yield`, `source`, `timestamp`, optional `mapped_rxn` if present.

**Output**: `reactions_normalized.parquet`, `reaction_hash_v1`, `rxn_vid`.

**Spec**

1. **Molecule standardization** (RDKit):

   * Neutralize, reionize (charge preference table), remove salts/counterions, tautomer canonicalization, normalize aromaticity/stereo; fail‑closed with reason codes.
   * Canonical SMILES/InChIKey for each entity.
2. **Reaction normalization**:

   * Sort reactants/products by canonical key; move spectators to `agents` if flagged.
   * `role_assign`: choose main substrate (largest organic OR center‑bearing molecule). Emit `role_scores` for audit.
3. **Condition normalization**:

   * Units → SI; parse temperature/time/pressure/concentration; `solvent` and `agents` resolved via dictionaries (Levenshtein + synonyms + CAS/name mapping). Emit `resolution_confidence` 0–1.
4. **Dedup**:

   * `reaction_hash_v1 = blake3( normalized_reaction_string || normalized_conditions_minimal )`
   * Drop exact duplicates; keep provenance list.

**Acceptance**

* ≥ 99% of rows have valid canonical molecules; log failures separately.
* Random 1k sample: median #unique `reaction_hash_v1` ≤ 0.999 × #rows.

**Performance**

* Throughput goal: ≥ 50k reactions/s per 16 vCPU shard (excluding I/O) on typical corpora.

---

## Phase 1 — Conditions Bucketing

**Input**: `reactions_normalized.parquet`

**Output**: `conditions_keys.parquet` with fields:

* `rxn_vid`, `cond_hash`, `present_mask`, normalized condition columns (`temp_K_bin`, `time_s_bin`, `pressure_Pa_bin`, `pH_bin`, `atmosphere`, `light`, `phase`),
* `solvent_key` (sorted multiset), `catalyst_key` (sorted multiset), `agent_key` (optional), `resolution_confidence`.

**Spec**

* Bin scalars using `configs/bins.yaml` (e.g., T: 10 K bins up to 400 K, wider above; time: log10 bins).
* `solvent_key`: tuple of (canonical_id, fraction_bin). Fractions rounded to nearest 0.1 if present; else omit.
* `catalyst_key`: tuple of canonical IDs (structures if resolved, else controlled vocab).
* Construct `cond_hash = blake3( serialize([solvent_key, temp_bin, time_bin, pressure_bin, pH_bin, atmosphere, light, phase, catalyst_key, present_mask]) )`.
* **Missingness policy**: keep `present_mask`; do **not** merge buckets with different masks.

**Partitioning**

* Write out partitioned by `cond_hash_prefix` and `present_mask`.

**Acceptance**

* Top 100 buckets by count: manual spot‑check shows coherent solvent/temperature distributions (variance small; no unit leaks).

**Performance**

* Map/GroupBy only; expect linear throughput close to I/O bound.

---

## Phase 2 — Mechanism / Transformation Clustering

**Input**: `conditions_keys.parquet` (for partitioning), `reactions_normalized.parquet`

**Output**: `mechanism_sigs.parquet` and `clusters_level2.parquet` (after clustering within each `(cond_hash)` bucket).

**Spec**

1. **Signatures**

   * **Mapped path (preferred when confidence high):**

     * Derive bond changes: tuples `(atomA_env, atomB_env, bond_before→after)` with environments defined by atom type, hybridization, charge, and 1–2 bond radius context hashed to short strings.
     * Produce `mech_sig_base` (sorted multiset hash) and optional `mech_sig_r1/r2` including context.
   * **Unmapped fallback:**

     * `ΔFP`: signed top‑k differences of product vs reactant ECFP features (k configurable; default 64).
     * `FG_transforms`: from `fg_library` on both sides → multiset of `FG_from→FG_to` codes.
     * Compose a composite signature `mech_sig_unmapped = hash([ΔFP_topk, FG_transforms])`.
   * **Ancillary deltas:** redox (sum of formal oxidation state deltas per element), `charge_delta`, `stereo_events`, `ring_events` (formation/cleavage count). Store as columns.
   * **Mapping confidence:** continuous 0–1 score; gate mapped pipeline when ≥ threshold.

2. **Pre‑bins & neighbors**

   * Prebin by exact `mech_sig_base` (or `mech_sig_unmapped`), then find near neighbors via:

     * For multiset signatures: MinHash LSH (128–256 perm) on tokenized events.
     * For vector features (ΔFP hashed to 8–16k dims): ANN (Faiss HNSW, M=32, efSearch=200).

3. **Clustering**

   * Build shard‑local k‑NN graph (k=15–50). Edge weights: Jaccard (events) or cosine/Tanimoto (vectors).
   * Run **HDBSCAN** (min_cluster_size tuned per shard) or **Leiden** on weighted graph.
   * Emit `MID` sequentially within `(cond_hash_prefix)` to keep IDs compact; maintain a global mapping table.

4. **Optional supervision**

   * Predict RXNO/name‑reaction with a weak classifier on a labeled subset; compute per‑cluster entropy as QA metric only.

**Acceptance**

* For 20 random clusters in 5 largest condition buckets: ≥ 80% human‑sensible mechanism homogeneity on spot‑checks.
* RXNO entropy median improving vs naive baseline (report).

**Performance**

* Target ≤ 2–4 hours per 5M‑reaction shard on 32 vCPU + 64 GB RAM (dominated by neighbor search). Persist intermediate neighbor lists to avoid rebuilds.

---

## Phase 3 — Structural Sub‑clustering

**Input**: `clusters_level2.parquet`, `structure_feats.parquet` (built here), `mechanism_sigs.parquet` (for centers)

**Output**: `clusters_level3.parquet` with `SID` per reaction; ANN indices per `(CID,MID)` shard in `artifacts/indices/`.

**Spec**

1. **Feature extraction**

   * `scaffold_key`: Bemis–Murcko scaffold of the main substrate.
   * `ecfp_hashed`: feature‑hashed ECFP (nBits 8–16k) of main substrate or concatenated `[ECFP(product) ‖ ECFP(reactant)]`.
   * `delta_fp_sparse`: ΔFP near the inferred reaction center (subset by radius).

2. **Index & canopy**

   * Build Faiss HNSW (CPU) with `M=32, efConstruction=400`. Optionally IVF‑PQ when shard > 2M.
   * Canopy by Tanimoto≥τ (τ≈0.6) using fast popcount kernels to reduce density clustering cost.

3. **Clustering**

   * Within canopy, run HDBSCAN/k‑medoids on Tanimoto/cosine distances.
   * Assign stable `SID` within `(CID,MID)`; export top‑k exemplars per cluster.

**Acceptance**

* Median intra‑cluster Tanimoto ≥ 0.55 and inter‑cluster median gap ≥ 0.15 for sampled shards.
* Qualitative: exemplars show consistent scaffolds/substituent patterns around the reaction center.

**Performance**

* ANN build ≤ 45 min per 2M vectors on CPU; search throughput ≥ 200k qps per node (batching).

---

## Phase 4 — Fusion & Final IDs

**Input**: `clusters_level1/2/3.parquet`, ancillary stats.

**Output**: `clusters_final.parquet` with `{CID, MID, SID}` and per‑cluster summaries; `cluster_stats.jsonl`.

**Spec**

* Join keys by `rxn_vid`.
* Generate **cluster cards** with: dominant scaffold, top solvents/temps, mechanism summary (most common `mech_sig` tokens), typical yields, and 5 exemplars (SVGs).
* Build cross‑view graphs: edges for `(MID across CIDs)` and `(SID across CIDs)` to analyze condition sensitivity.

**Acceptance**

* No missing IDs; coverage ≥ 98% of non‑failed rows.
* Spot‑checks: cluster cards read like coherent “families”.

---

## QA & Dashboards

* `qas.py` computes:

  * **Condition tightness**: variance of binned scalars per `{CID}`.
  * **Mechanism purity**: RXNO entropy per `{CID,MID}`.
  * **Structure coherence**: intra vs inter Tanimoto gaps per `{CID,MID,SID}`.
  * **Outcome coherence**: yield/selectivity variance where available.
* `dashboards/cluster_report.py` renders HTML with tables + exemplar SVGs.
* `dashboards/build_umap.py` makes quick 2D projections by shard for visual drift detection.

---

## Configuration

* `configs/example.local.yaml` (copy → `local.yaml`):

```yaml
paths:
  raw_reactions: "/data/reactions/*.parquet"
  workdir: "/data/retrosyn-clustering"
  artifacts: "artifacts"
compute:
  spark_master: "local[*]"   # or spark://...
  ray_address: "auto"
  num_shards: 1024
  shard_bits: 12              # cond_hash_prefix bits
  sample_rows: 5000000
flags:
  use_atom_mapping: true
  mapping_conf_threshold: 0.65
  use_gpu_faiss: false
versions:
  schema: 1
  params: 1
```

* `configs/hparams.yaml` (default hyper‑params):

```yaml
phase1:
  temp_bin_K: 10
  time_log_bins: [ -3, 7 ]   # 1 ms to ~4 months
phase2:
  delta_fp_topk: 64
  lsh_perm: 256
  hdbscan:
    min_cluster_size: 50
    min_samples: 10
phase3:
  fp_bits: 16384
  canopy_tanimoto: 0.6
  hdbscan:
    min_cluster_size: 30
    min_samples: 8
```

* `configs/bins.yaml`: explicit edges for temperature, time, pH, potential.

---

## Function contracts (key modules)

**normalizers/chem_standardize.py**

```python
def canonicalize_mol(smiles: str) -> dict:
    """Return {smiles, inchikey, molblock, flags} or raise CanonError."""

def canonicalize_reaction(reactants: list[str], products: list[str], agents: list[str]) -> dict:
    """Return normalized R/P/A lists, canonical reaction SMILES, role tags, errors."""
```

**normalizers/conditions_normalize.py**

```python
def normalize_conditions(row: dict, ont: Ontology) -> dict:
    """Parse units, map solvents/catalysts, bin scalars, build present_mask and cond_hash."""
```

**signatures/atom_mapping.py**

```python
def map_reaction(rxnsmi: str) -> tuple[MappedRxn, float]:
    """Return mapping object + confidence in [0,1]. Cache aggressively."""
```

**signatures/mech_signature_mapped.py**

```python
def mech_sig_from_mapping(mapped: MappedRxn, radius: int = 1) -> dict:
    """Return mech_sig_base, mech_sig_r1/r2, event tokens, redox/stereo/ring counters."""
```

**signatures/mech_signature_unmapped.py**

```python
def mech_sig_unmapped(reactants: list[str], products: list[str]) -> dict:
    """Return composite signature from ΔFP top‑k and FG transforms."""
```

**structures/scaffold.py**

```python
def main_scaffold(reactants: list[str], products: list[str], role_info: dict) -> str:
    """Return Bemis–Murcko scaffold key of main substrate."""
```

**indices/build_ann.py**

```python
def build_hnsw(vectors: np.ndarray, m: int = 32, efc: int = 400) -> FaissIndex:
    ...
```

---

## Testing strategy

* **Unit tests** for normalizers/signatures on curated toy sets in `tests/data/` with expected hashes.
* **Golden sample**: freeze outputs for a 100k public subset; diff on CI for regressions.
* **Property tests**: random fuzzing of units/strings to harden parsers.

---

## Performance playbook

* Cache everything (LMDB or Arrow IPC) between steps.
* Use memory‑mapped float32 arrays for vectors; prefer feature hashing to fixed 8–16k dims.
* Progressive clustering: tune on 1–5% then lock hyper‑params for full runs.
* Monitor ANN recall@k on held‑out neighbors; ensure ≥ 0.9 before HDBSCAN.

---

## Failure handling

* Any parsing/mapping failure must **not** crash the shard; send to quarantine lanes with rich error codes.
* Produce `failed_rows.parquet` with `rxn_id`, phase, reason, and raw payload.

---

## Makefile (excerpt)

```makefile
.PHONY: sample phase0 phase1 phase2 phase3 phase4 dashboards clean

sample:
	python -m pipelines.phase0_data_readiness --sample
	python -m pipelines.phase1_conditions --sample
	python -m pipelines.phase2_mechanism --sample
	python -m pipelines.phase3_structures --sample
	python -m pipelines.phase4_fusion --sample

phase0:
	python -m pipelines.phase0_data_readiness
phase1:
	python -m pipelines.phase1_conditions
phase2:
	python -m pipelines.phase2_mechanism
phase3:
	python -m pipelines.phase3_structures
phase4:
	python -m pipelines.phase4_fusion

dashboards:
	python -m dashboards.cluster_report

clean:
	rm -rf artifacts/* intermediates/* logs/*
```

---

## Minimal CLI pattern (all phases)

All phase scripts expose a consistent CLI via `click`:

```bash
python -m pipelines.phase2_mechanism \
  --input reactions_normalized.parquet \
  --cond-table conditions_keys.parquet \
  --out mechanisms/ --shard-bits 12 --ray-address auto
```

Each script writes JSONL logs and a `_SUCCESS` sentinel per shard directory for idempotent retries.

---

## Security & provenance

* Preserve original source IDs and timestamps.
* Keep a provenance array per deduped reaction (list of contributing records).

---

## Roadmap (nice‑to‑haves)

* GPU Faiss and RAPIDS‑cuDF lanes for large shards.
* Semi‑supervised mechanism labeling using contrastive learning on mech signatures.
* One‑pot step inference to split cascade reactions.
* Domain adaptation experiments across `CID` using cluster graphs.

---
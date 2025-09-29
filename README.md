> Retrosyn‑Clustering: factorized, scalable reaction clustering for 10^9+ records

This repository provides a **production‑grade scaffold** to cluster ~1.6B chemical reactions along three interpretable axes:

1. **Conditions** (solvent(s), temperature, time, atmosphere, light, catalysts, etc.)
2. **Mechanism / Transformation** (bond changes, FG transforms, redox/stereo/ring events)
3. **Structures** (scaffolds and local substituent patterns of the main substrate)

The pipeline is explicitly designed for **scale** (Spark/Dask, Faiss ANN, Ray) and **interpretability** (stable hierarchical IDs). Deliverables are tidy Parquet tables and shard‑local ANN indices, plus sanity‑check dashboards.

---

## Quickstart

```bash
# 0) Create environment (conda recommended)
conda env create -f env.yml
conda activate retrosyn-clust

# 1) Configure paths
cp configs/example.local.yaml configs/local.yaml
# edit configs/local.yaml to point to your raw reaction data (Parquet/JSON/CSV)

# 2) Run Phase 0–3 on a sample (1–5M rows) to validate
make sample

# 3) Scale out per-shard processing (Ray/Spark cluster)
make cluster-all   # end-to-end (can be resumed)

# 4) Explore clusters
make dashboards
```

---

## Repository layout

```
retrosyn-clustering/
├─ configs/
│  ├─ example.local.yaml         # paths & compute knobs (copy to local.yaml)
│  ├─ bins.yaml                  # scalar bin edges (T, time, pH, potential, etc.)
│  ├─ ontology/                  # solvent/catalyst/agent dictionaries + synonyms
│  │  ├─ solvents.csv
│  │  ├─ catalysts.csv
│  │  └─ reagents.csv
│  └─ hparams.yaml               # clustering hyper-parameters per phase
│
├─ schemas/
│  ├─ reactions_raw.schema.json
│  ├─ reactions_normalized.schema.json
│  ├─ conditions_keys.schema.json
│  ├─ mechanism_sigs.schema.json
│  ├─ structure_feats.schema.json
│  └─ clusters.schema.json       # {level1,2,3,final}
│
├─ normalizers/
│  ├─ __init__.py
│  ├─ chem_standardize.py        # RDKit molecule & reaction canonicalization
│  ├─ conditions_normalize.py    # units, synonyms, binning, condition hashing
│  └─ role_assign.py             # choose main substrate / role tagging
│
├─ signatures/
│  ├─ __init__.py
│  ├─ atom_mapping.py            # optional/tiers; caching & confidence scoring
│  ├─ mech_signature_mapped.py   # bond-change multisets w/ context
│  ├─ mech_signature_unmapped.py # ΔFP + FG transforms + heuristics
│  ├─ fg_library.py              # SMARTS-based FG tagging
│  └─ redox_stereo_ring.py       # redox/stereo/ring event extraction
│
├─ structures/
│  ├─ __init__.py
│  ├─ scaffold.py                # Bemis–Murcko & position maps
│  ├─ fingerprints.py            # ECFP/feature hashing utilities
│  └─ pairing.py                 # ΔFP, concatenated R/P FPs, proximity to center
│
├─ pipelines/
│  ├─ phase0_data_readiness.py   # Parquet IO, standardize, dedupe
│  ├─ phase1_conditions.py       # condition keys + hashing + partitioning
│  ├─ phase2_mechanism.py        # mech signatures + prebin + graph clustering
│  ├─ phase3_structures.py       # ANN indices + micro-clusters
│  ├─ phase4_fusion.py           # assign {CID,MID,SID} + final tables
│  ├─ qas.py                     # quality metrics & sampling galleries
│  └─ utils_spark.py             # Spark/Dask helpers
│
├─ indices/
│  └─ build_ann.py               # Faiss (HNSW/IVF-PQ) builders per shard
│
├─ dashboards/
│  ├─ build_umap.py              # quick visual sanity checks per shard
│  └─ cluster_report.py          # HTML report with exemplars
│
├─ scripts/
│  ├─ run_sample.sh
│  ├─ run_cluster_all.sh
│  └─ export_cluster_summaries.py
│
├─ tests/
│  ├─ data/                      # tiny toy sets with expected outputs
│  ├─ test_normalizers.py
│  ├─ test_signatures.py
│  └─ test_pipelines.py
│
├─ env.yml                       # conda environment
├─ pyproject.toml                # build & deps (PEP 621)
├─ Makefile                      # handy targets
└─ AGENTS.md                     # detailed build plan for automation agents
```

---

## Outputs (Parquet tables)

* `reactions_normalized.parquet`: canonical molecules/reactions, roles, dedup keys
* `conditions_keys.parquet`: `rxn_id`, `cond_hash`, normalized condition fields, presence mask
* `mechanism_sigs.parquet`: `rxn_id`, `mech_sig_*`, ancillary deltas (redox/stereo/ring), mapping_confidence
* `structure_feats.parquet`: `rxn_id`, `scaffold_key`, `ecfp_hashed`, ΔFP sparse, etc.
* `clusters_level1.parquet`: `rxn_id`, `CID`
* `clusters_level2.parquet`: `rxn_id`, `CID`, `MID`
* `clusters_level3.parquet`: `rxn_id`, `CID`, `MID`, `SID`
* `clusters_final.parquet`: one row per `rxn_id` with `{CID,MID,SID}` + per-cluster summary stats

Each table is **append‑safe** with schema versioning (field `schema_version`).

---

## Design choices

* **Factorized clustering** yields interpretable, reusable IDs.
* **Hash → ANN → density** pattern lets us scale without losing local fidelity.
* **Tiered atom mapping** avoids burning CPU on hard cases; ΔFP/FG fallback keeps coverage high.
* **Strict normalization** for conditions prevents bucket bleeding and preserves semantic meaning.

---

## Dependencies

* Python 3.11, RDKit, PySpark or Dask, Ray, Faiss (CPU; GPU optional), NumPy, Pandas, PyArrow, blake3, scikit‑learn, hdbscan, networkx, tqdm, click, pydantic.

Install via `env.yml` or `pip install -e .` after creating a base conda env with RDKit.

---

## Make targets

```make
make sample         # run phases on a 1–5M sample for validation
make phase0         # standardize & dedupe
make phase1         # condition keys & shards
make phase2         # mechanism signatures & clusters
make phase3         # structure features & sub-clusters
make phase4         # fusion & final IDs
make dashboards     # simple reports & UMAPs
make clean          # remove intermediates (keeps logs)
```

---

## Logging & reproducibility

* Structured logs (JSON) to `logs/` with per‑phase timers and sample exemplars.
* Every output includes `code_version` (git SHA), `schema_version`, `params_version`.

---
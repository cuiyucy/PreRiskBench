# Reconstruction order and executable scope

## 1. Upstream field-harmonization dependency

Obtain VitalDB, MOVER, INSPIRE and applicable terminology access independently.
Resource-specific field extraction, cleaning, aggregation and harmonization are
prerequisites. This module starts from their authorized exports; it does not include
the upstream database adapters or database configuration. At this module's initial
publication, the parent repository provides the PreRiskBench overview, not a pinned
release of those upstream adapters. End-to-end reconstruction therefore requires
independently preparing the following interface. Record the adapter version used.

For each resource (`vitaldb`, `mover`, `inspire`), this module expects these local-only paths
below `pipelines/benchmark/dataset/<resource>/`:

- `prerisk_bench_feat/feat_pat_demo.csv`
- `prerisk_bench_feat/feat_past_his.csv`
- `prerisk_bench_feat/feat_lab_res.csv`
- `prerisk_bench_feat/feat_med_rec.csv`
- `prerisk_bench_feat/feat_surg_vars.csv`
- `prerisk_bench_feat/feat_anes_vars.csv`
- `prerisk_bench_labels/` containing the ASA label table expected by the loader

Every feature table uses `surgery_id`; `feat_anes_vars.csv` must additionally provide
the complete `surgery_id -> patient_id` relation for grouped splitting. Source names
such as `mrn` and `log_id` are upstream linkage fields, not model features. The public
wrapper rejects missing or incomplete patient maps. No interface file is distributed.

## 2. Semantic mapping

Configure `pipelines/alignment/config/config.yaml` using the template. All `${...}`
variables must resolve locally. Model paths, RF2 content, indices and frozen
mapping artifacts are intentionally absent. Scripts retain their numbered
historical filenames to preserve the observable processing order.

For direct script execution, add the `semantic-alignment` module directory to
`PYTHONPATH` so `release_runtime` can be imported. Set `ALIGNMENT_WORK_DIR` to the
alignment work directory, `BENCHMARK_WORK_DIR` to the benchmark work directory and
`LOCAL_WORK_DIR` to a private runtime directory. Set the database, API, model and
index variables required by the YAML configuration in the local process environment.
Do not commit their values.

Data/model path literals that used historical container mounts now resolve through
`ALIGNMENT_WORK_DIR` and `BENCHMARK_WORK_DIR`. The K-ablation orchestrator still
contains historical `docker exec` commands for a container named `ml-experiments`
with benchmark code mounted at `/ml_training`; configure an isolated container or
adapt those commands locally before using that optional script. The public benchmark
wrapper sets work-directory variables automatically, but direct lower-level scripts
require the environment above. Full dependency and container reconstruction has not
been validated in this release.

The API client uses `llm.api_url` from the local configuration. Model names and
parameters in the historical configuration are examples, not a verified record of
the frozen hosted run. Set the intended endpoint/model explicitly. Hugging Face
downloads use the library's endpoint unless the caller sets `HF_ENDPOINT`.

String matching → vector retrieval/reranking → LLM/RAG-assisted difficult cases →
FSN correction → merge. Physician review is a subsequent sampled assessment,
not the source of the automatic similarity/LLM scores. Full row-level review
records are not part of the release. Raw ICD string matches are not a clinical
crosswalk and must not be interpreted as semantic equivalence.

## 3. Clustering and ASA benchmark

The included `clustering_pipeline.py` implements the paper's frequency-first
representative rule. The restricted dictionary with SHA-256 `744e95...` remains the
authoritative frozen paper artifact. Verify an authorized local copy with
`tools/verify_frozen_artifact.py`; never commit or redistribute the dictionary.

The original benchmark structure is retained so `data`, `features`, `models`,
`experiments` and `utils` imports resolve as they did in the research tree.
The default config and public wrapper select ASA only and six model families.
Historical lower-level modules may contain utilities for other tasks; their
presence is not a claim that those tasks were evaluated in this paper.

For a real build, populate `pipelines/benchmark/dataset` with authorized feature
exports and required maps, then use the public wrapper. It fails closed when the
patient key map is missing. The copied sampler also checks map completeness before
splitting, disabling the historical surgery-level fallback. This is an explicit
safety modification recorded in the transformation manifest; original source
files were not modified.

```sh
python -B release.py benchmark -- build --task asa --dry-run
# Only after local inputs and schema preflight have been approved:
python -B release.py benchmark --execute -- build --task asa
python -B release.py benchmark --execute -- run --task asa --exp baseline
```

The first command is a plan only. `--execute -- ... --dry-run` invokes the original
builder's dry run and can inspect local authorized input paths. No such real-data
dry run was performed during packaging.

## 4. Validation and outputs

Recheck patient intersections after construction; group keys must cover every
eligible surgery, and identical splits must be used across representations.
Save runtime data only to ignored local working directories. Package results as
paper tables/figures after author review; no automatic export of aggregate CSVs
or row-level artifacts is provided here.

# Semantic Alignment for PreRiskBench

Research code and manuscript for **Semantic Alignment of Heterogeneous Perioperative
Data Resources for Computational Intelligence**.

This module studies semantic alignment across VitalDB, MOVER and INSPIRE. It compares
Field-Aligned, Concept-Aligned and Concept-Clustered representations through semantic
overlap, cross-resource retrieval and patient-grouped ASA physical status prediction.
The parent [PreRiskBench](../README.md) repository covers the broader project.

## Contents

| Path | Contents |
| --- | --- |
| [`paper/`](paper/) | Final manuscript PDF, LaTeX sources, figures and tables |
| [`pipelines/alignment/`](pipelines/alignment/) | Text extraction, terminology matching, retrieval, reranking and LLM-assisted mapping |
| [`pipelines/benchmark/`](pipelines/benchmark/) | Representation construction, patient-grouped splitting, classifiers and analyses |
| [`figures/`](figures/) | Plotting sources and regeneration prerequisites |
| [`configs/`](configs/) | Local configuration examples |
| [`environment/`](environment/) | Smoke-test requirements and historical dependency declarations |
| [`tests/`](tests/) | Synthetic interface and patient-splitting tests |
| [`provenance/`](provenance/) | Source provenance and restricted-artifact hashes; no artifact payloads |

Read the [manuscript](paper/manuscript.pdf) for methods and results, and
[reconstruction guide](docs/RECONSTRUCTION.md) for the input interface and execution order.

## Quick start

From the repository root, use an isolated Python environment:

```sh
cd semantic-alignment
python -m pip install -r environment/requirements-smoke.txt
python -B release.py plan
python -B release.py check
python -B release.py benchmark -- --help
```

These checks use synthetic inputs. They do not require clinical datasets, model
weights, terminology downloads or hosted API access. The recorded checks and their
environment are in [`docs/VALIDATION.md`](docs/VALIDATION.md).

Real mapping and benchmark runs require independently authorized data, terminology,
model assets and the full dependencies. Upstream field-harmonization adapters are
not included in this module. The public wrapper supports ASA only; benchmark commands
are plans unless `--execute` is supplied. Missing or incomplete patient linkage is
rejected before benchmark construction. See the reconstruction guide before execution.

## Reproducibility and data access

The published code preserves the research module layout, with configuration
sanitization and explicit patient-map checks described in the provenance records.
The code has not been rerun end to end on real data during publication. Synthetic
tests do not establish reproduction of the reported experiment results.

No patient records, identifiers, split lists, row-level predictions, full terminology
mappings, RF2 content, model weights, embeddings, retrieval indices or API logs are
distributed. Paper figures and tables contain the reported evidence. See
[`DATA_ACCESS.md`](DATA_ACCESS.md) and [frozen-artifact metadata](docs/SNAPSHOTS.md).

## Manuscript and citation

The final manuscript and its sources are preserved in `paper/`. To compile the source,
use XeLaTeX with the required TeX packages and a locally licensed Times New Roman font;
font files are not redistributed. The provided PDF is the archived final manuscript,
not a new typesetting revision. Figure regeneration is documented in `figures/README.md`.

Author and title metadata are provided in [`CITATION.cff`](CITATION.cff). No DOI or
publication-status claim is assigned by this code release.

## License

This module follows the licensing arrangements of the parent PreRiskBench repository
and does not grant a separate license. At initial publication, the parent repository
has no LICENSE file. Existing third-party notices, including those in `paper/IEEEtran.cls`,
remain applicable. Data and terminology access are governed separately by their providers.

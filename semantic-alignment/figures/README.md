# Paper figures

The final figure PDFs used by the manuscript are in `../paper/figures/`.
Scripts in `scripts/` generate Figures 1, 3, 4 and 5. Figure 2 is supplied as the
final vector PDF. Supplementary figures from earlier drafts are not included.

Regeneration requires locally prepared aggregate inputs, which are not distributed:

- `data/analysis_source_schema_fields.csv`
- `data/analysis_source_schema_overlap.csv`
- `data/analysis_concept_space.csv`
- `data/analysis_knn_retrieval.csv`
- `data/experiment_master.csv`
- `data/analysis_prior_baseline.csv`
- `figures/source_data/cohort_summary.csv`
- `figures/source_data/mapping_review.csv`

Paths are relative to `semantic-alignment/`. The scripts validate historical row
counts and totals; they are intended for the paper's aggregate schema. Some validation
helpers require the complete input set even when rendering a single figure.

After preparing authorized inputs in an isolated environment:

```sh
python -m pip install -r figures/requirements.txt
python -B figures/scripts/fig01_field_alignment.py
python -B figures/scripts/fig03_concept_space_tradeoff.py
python -B figures/scripts/fig04_retrieval_coverage.py
python -B figures/scripts/fig05_asa_model_readiness.py
```

Outputs go to the ignored `figures/outputs/` directory. Font availability and graphics
library versions can affect rendering. Figure regeneration was not part of the
publication smoke tests; the supplied paper figure PDFs remain the final artifacts.

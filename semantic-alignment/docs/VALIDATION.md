# Publication validation

Validation date: 2026-10-22. Checks were run in an isolated Windows x86-64 Python 3.12.14
environment using `environment/requirements-smoke.txt`. The exact installed
package versions are recorded in `environment/smoke-tested.txt`; this is a smoke-test
environment record, not the historical experiment environment or a GPU training lock.

## Passed checks

- All 16 synthetic unit tests passed with `python -B release.py check`.
- The tests exercise environment expansion, missing configuration rejection,
  patient leakage detection, complete patient linkage, deterministic grouped
  splitting, exclusion of identifiers from model features, ASA-only command
  handling, benchmark help/imports, frequency-first cluster representatives,
  synthetic Macro-F1 and configured/missing LLM endpoints with a mocked API client.
- The package audit found no excluded data/model payloads, private endpoints or
  literal credential dictionaries, and parsed all 72 Python files.
- A separate publication scan checked credential assignments/arguments, private-key
  blocks, credential-bearing URLs, personal paths and malformed service placeholders.
- The documented plan and benchmark dry-run plan completed without reading data.
- `pip check` found no broken requirements in the isolated test environment.

The final manuscript PDF and 13 manuscript source/figure/table files were copied
unchanged from the archived final submission. Their hashes are recorded in
`provenance/publication.json`. The package manifest can be checked with:

```sh
python -B tools/manifest.py --verify
```

## Not rerun during publication

No clinical database extraction, real-data benchmark construction, hosted LLM call,
terminology processing, embedding generation, full model training, figure regeneration
or manuscript compilation was performed. Upstream field harmonization, historical
container orchestration and GPU dependencies are not covered by these smoke tests.
The tests establish interface behavior on synthetic inputs, not reproduction of
the paper's reported results or clinical validity.

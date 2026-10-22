# Frozen paper artifacts

Only metadata/hashes are included. Never redistribute the dictionaries or maps.

| Artifact | SHA-256 |
|---|---|
| VitalDB mapping | `9ce96b36641370272b1071645b266c2b586f5036e67bbaec70ecacd72fd60a4e` |
| MOVER mapping | `bd44b3e45fc1dc1964c491600d96712838402e5ff539a065e4460fe4be16d0ae` |
| INSPIRE mapping | `699d239939e56dfa47483411354e42cd833de2f51392b2291a5e5eef232d494a` |
| Frequency-representative cluster dictionary | `744e95a840ede013b0d9cf96838327c21d0b1ebc3ceadee44c85ac3ea582368c` |

`provenance/restricted_artifacts.json` records availability and independently
computed hashes from the archived research snapshot. The `available` field refers
to the private source snapshot, not to files distributed in this repository. A missing file is not treated as matching the table.
The cluster dictionary is the version used throughout the manuscript and experiments.

The public clustering code uses frequency-first representative selection. New
reconstructions resolve equal-frequency concept IDs lexically for deterministic
behavior. The frozen hash, rather than a newly generated file, is authoritative
for the historical paper snapshot. Verify an authorized local copy with:

```sh
python -B tools/verify_frozen_artifact.py /local/authorized/cid_to_centroid.json
```

Hosted LLM/RAG calls need model/version, dates, retrieval assets, prompts and
parameters. Current sanitized configuration is not evidence that every value
matches the original hosted run. The paper reports a frozen-result experiment;
independent reruns are not guaranteed bitwise identical.

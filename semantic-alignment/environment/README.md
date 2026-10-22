# Environment records

The paper's reported Python environment was 3.10.12. The copied research
requirements preserve dependency declarations from the source directories;
they are not a proven portable lock or a guarantee of GPU/CPU binary compatibility.

`requirements-smoke.txt` is deliberately lightweight. Its tests exercise copied
patient splitting, identifier exclusion and ASA normalization plus wrapper safety.
They do not load ClinicalBERT, FAISS, HDBSCAN, a database or an LLM service.
The validation report records the actually available environment used for checks.

Full mapping/training dependencies, model snapshots and CUDA variants must be
installed in a new isolated environment. Never modify an active experiment
environment simply to satisfy these dependency files.

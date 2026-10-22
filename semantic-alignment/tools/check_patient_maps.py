"""Refuse real benchmark builds if a source key map is missing or incomplete."""
from pathlib import Path
import sys
import pandas as pd
from release_runtime import require_patient_map

root=Path('dataset')
errors=[]
for resource in ('vitaldb','mover','inspire'):
    path=root/resource/'prerisk_bench_feat/feat_anes_vars.csv'
    if not path.is_file():
        errors.append(resource+': required key-map feature file is absent'); continue
    data=pd.read_csv(path,dtype=str)
    if not {'surgery_id','patient_id'}.issubset(data.columns):
        errors.append(resource+': key columns are missing'); continue
    if data.groupby('surgery_id')['patient_id'].nunique().gt(1).any():
        errors.append(resource+': conflicting patient mapping'); continue
    try:
        require_patient_map(data['surgery_id'],dict(zip(data['surgery_id'],data['patient_id'])))
    except ValueError:
        errors.append(resource+': incomplete patient mapping')
if errors:
    print('\n'.join(errors)); sys.exit(1)
print('Key-map preflight passed; post-build split audit remains required.')

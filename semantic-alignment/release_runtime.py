"""Small, explicit configuration helpers; never connects to external services."""
import os
from pathlib import Path
import re


def expand_path(value):
    expanded=os.path.expandvars(str(value))
    missing=re.findall(r'\$\{([A-Za-z_][A-Za-z_0-9]*)\}',expanded)
    if missing:
        raise ValueError('Set required local environment variables: '+', '.join(sorted(set(missing))))
    return os.path.expanduser(expanded)


def expand_config(value):
    if isinstance(value,dict): return {k:expand_config(v) for k,v in value.items()}
    if isinstance(value,list): return [expand_config(v) for v in value]
    if isinstance(value,str): return expand_path(value)
    return value


def require_patient_map(surgery_ids, patient_map):
    """Publication wrapper is stricter than historical surgery-level fallback."""
    if not patient_map: raise ValueError('A complete surgery-to-patient mapping is required')
    normalized={str(k):str(v) for k,v in patient_map.items()
                if v is not None and str(v).strip().lower() not in {'','nan','none','<na>'}}
    missing=set(map(str,surgery_ids))-set(normalized)
    if missing: raise ValueError('Patient mapping is incomplete; refusing surgery-level fallback')
    return normalized


def verify_patient_disjoint(splits,patient_map):
    require_patient_map([sid for ids in splits.values() for sid in ids],patient_map)
    groups={name:{str(patient_map[str(sid)]) for sid in ids} for name,ids in splits.items()}
    names=list(groups)
    for i,a in enumerate(names):
        for b in names[i+1:]:
            if groups[a]&groups[b]: raise ValueError('Patient leakage between '+a+' and '+b)
    return True

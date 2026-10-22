"""Compute ablation overlaps for k=2000, 5000, 10000."""
import os as _release_os
from release_runtime import expand_path as _release_expand, expand_config as _release_expand_config
import pickle, csv, json
import numpy as np
from pathlib import Path
from sklearn.cluster import MiniBatchKMeans
data_path = Path(_release_expand('${BENCHMARK_WORK_DIR}/mapping/final_clustered'))
with open(data_path / 'core_embeddings.pkl', 'rb') as f:
    bert_data = pickle.load(f)
embs = bert_data['embeddings']
core_list = bert_data['core_list']
print(f'Cores: {len(core_list)}, Embs: {embs.shape}')
results = {}
for target_k, label in [(2000, 'k2000'), (5000, 'k5000'), (10000, 'k10000')]:
    k = min(target_k, len(core_list) // 2)
    print(f'\n--- {label}: k={k} ---')
    km = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=4096, n_init=3, max_iter=300)
    labels = km.fit_predict(embs)
    core_to_cluster = {core_list[i]: int(labels[i]) for i in range(len(core_list))}
    cid_to_centroid = {}
    for ds in ['vitaldb', 'mover', 'inspire']:
        with open(_release_expand(f'${{ALIGNMENT_WORK_DIR}}/data/mapping/final_output/{ds}_final_mapping.csv')) as f:
            for row in csv.DictReader(f):
                cid = row['conceptId'].strip()
                fsn = row.get('fsn', '')
                bare = fsn.rsplit('(', 1)[0].strip().rstrip().lower() if fsn else ''
                if bare in core_to_cluster:
                    cid_to_centroid[cid] = str(core_to_cluster[bare])
                else:
                    cid_to_centroid[cid] = cid
    unique_clusters = len(set(cid_to_centroid.values()))
    print(f'Mapped: {len(cid_to_centroid)} concepts -> {unique_clusters} clusters')
    with open(data_path / f'cid_to_centroid_{label}.json', 'w') as f:
        json.dump(cid_to_centroid, f)
    ds_centroids = {}
    for ds in ['vitaldb', 'mover', 'inspire']:
        cents = set()
        with open(_release_expand(f'${{ALIGNMENT_WORK_DIR}}/data/mapping/final_output/{ds}_final_mapping.csv')) as f:
            for row in csv.DictReader(f):
                cid = row['conceptId'].strip()
                cents.add(cid_to_centroid.get(cid, cid))
        ds_centroids[ds] = cents
    k_res = {'actual_k': k, 'unique_clusters': unique_clusters}
    for a, b in [('vitaldb', 'mover'), ('vitaldb', 'inspire'), ('mover', 'inspire')]:
        inter = ds_centroids[a] & ds_centroids[b]
        union = ds_centroids[a] | ds_centroids[b]
        smaller = min(len(ds_centroids[a]), len(ds_centroids[b]))
        j = round(len(inter) / len(union) * 100, 2) if union else 0
        cv = round(len(inter) / smaller * 100, 1) if smaller else 0
        triple = len(ds_centroids['vitaldb'] & ds_centroids['mover'] & ds_centroids['inspire'])
        k_res[f'{a}_vs_{b}'] = {'jaccard': j, 'coverage': cv, 'intersection': len(inter)}
        print(f'  {a}↔{b}: J={j}% cov={cv}% int={len(inter)}')
    k_res['triple_shared'] = triple
    k_res['ds_clusters'] = {ds: len(cents) for ds, cents in ds_centroids.items()}
    results[label] = k_res
with open(data_path / 'ablation_overlaps.json', 'w') as f:
    json.dump(results, f, indent=2)
print('\nSaved to ablation_overlaps.json')
print(json.dumps(results, indent=2))

"""
分析脚本 C：S-Cluster vs S-Aligned 信息损失量化（互信息）
运行: docker exec ml-experiments python3 /ml_training/scripts/analysis_C_mutual_info.py
"""
import os as _release_os
from release_runtime import expand_path as _release_expand, expand_config as _release_expand_config
from pathlib import Path
from collections import defaultdict, Counter
import json
import math
import pandas as pd
CLUSTER_JSON = _release_expand('${BENCHMARK_WORK_DIR}/mapping/final_clustered/cid_to_centroid.json')
BENCH_DIR = Path(_release_expand('${BENCHMARK_WORK_DIR}/benchmark'))
TEXT_COLS = ['preop_diagnosis', 'scheduled_procedure_name', 'hx_circ_detail', 'hx_endo_detail']
cid2cen = json.load(open(CLUSTER_JSON))

def load_patient_concepts(mode, ds, col):
    """返回该数据集某文本列的患者级概念计数（概念ID）"""
    path = BENCH_DIR / 'asa' / mode / ds / 'train.csv'
    cnt = Counter()
    df = pd.read_csv(path, index_col=0, usecols=lambda c: c in (col,))
    if col not in df.columns:
        return cnt
    split_items = col.startswith('hx_')
    for v in df[col].dropna().astype(str):
        if not v.strip():
            continue
        if split_items:
            for item in v.split(';'):
                item = item.strip().lower()
                if item:
                    cnt[item] += 1
        else:
            cnt[v.strip().lower()] += 1
    return cnt

def entropy(dist):
    """dist: Counter 或 dict value->count, 返回熵"""
    total = sum(dist.values())
    if total == 0:
        return 0.0
    h = 0.0
    for c in dist.values():
        if c <= 0:
            continue
        p = c / total
        h -= p * math.log2(p)
    return h
print('=' * 70)
print('信息损失量化: S-Aligned(概念) vs S-Cluster(簇) 互信息')
print('=' * 70)
print('\n加载映射表建立 概念文本->概念ID->簇 链路...')
import re
fsn2cid = {}
raw2cid = {}
for ds in ['vitaldb', 'mover', 'inspire']:
    df = pd.read_csv(_release_expand(f'${{BENCHMARK_WORK_DIR}}/mapping/{ds}_final_mapping.csv'), dtype={'conceptId': str})
    for _, r in df.iterrows():
        fsn = r['fsn']
        if isinstance(fsn, str) and fsn.strip():
            key = fsn.strip().lower()
            fsn2cid.setdefault(key, r['conceptId'])
            bare = re.sub('\\s*\\([^)]*\\)\\s*$', '', key).strip()
            fsn2cid.setdefault(bare, r['conceptId'])
        fv = r['field_value']
        if isinstance(fv, str) and fv.strip():
            raw2cid.setdefault(fv.strip().lower(), r['conceptId'])
concept2cluster = {}
for cid, cen in cid2cen.items():
    concept2cluster[cid] = cen
print('\n逐库逐列计算互信息...')
overall = []
for ds in ['vitaldb', 'mover', 'inspire']:
    for col in TEXT_COLS:
        path = BENCH_DIR / 'asa' / 'aligned' / ds / 'train.csv'
        if not path.exists():
            continue
        df = pd.read_csv(path, usecols=lambda c: c in (col, 'surgery_id'))
        if col not in df.columns:
            continue
        split_items = col.startswith('hx_')
        concept_counts = Counter()
        for v in df[col].dropna().astype(str):
            if not v.strip():
                continue
            vals = v.split(';') if split_items else [v]
            for item in vals:
                item = item.strip().lower()
                if not item:
                    continue
                cid = fsn2cid.get(item)
                if cid is None:
                    cid = raw2cid.get(item)
                if cid is None:
                    continue
                concept_counts[cid] += 1
        if not concept_counts:
            continue
        H_concept = entropy(concept_counts)
        cluster_counts = Counter()
        cluster_of_concept = {}
        for cid, cnt in concept_counts.items():
            cl = concept2cluster.get(cid)
            if cl is None:
                continue
            cluster_of_concept[cid] = cl
            cluster_counts[cl] += cnt
        H_cluster = entropy(cluster_counts)
        H_cond = 0.0
        total_n = sum(concept_counts.values())
        cl_grp = defaultdict(list)
        for cid, cl in cluster_of_concept.items():
            cl_grp[cl].append(cid)
        for cl, cids in cl_grp.items():
            sub = Counter({c: concept_counts[c] for c in cids})
            H_cond += sum(sub.values()) / total_n * entropy(sub)
        MI = H_concept - H_cond
        loss_ratio = H_cond / H_concept if H_concept > 0 else 0
        NMI = MI / H_concept if H_concept > 0 else 0
        row = {'dataset': ds, 'column': col, 'n_concepts': len(concept_counts), 'n_clusters': len(cluster_counts), 'H_concept': round(H_concept, 4), 'H_cluster': round(H_cluster, 4), 'H_cond_loss': round(H_cond, 4), 'MI': round(MI, 4), 'NMI': round(NMI, 4), 'loss_ratio': round(loss_ratio, 4)}
        overall.append(row)
        print(f"  [{ds}] {col:25s} 概念{row['n_concepts']:4d} 簇{row['n_clusters']:4d} Hc={H_concept:.3f} Hcl={H_cluster:.3f} Hcond={H_cond:.3f} MI={MI:.3f} NMI={NMI:.3f} 损失比={100 * loss_ratio:.1f}%")
if overall:
    df_all = pd.DataFrame(overall)
    print('\n' + '=' * 70)
    print('汇总')
    print('=' * 70)
    print(df_all.to_string(index=False))
    mi_total = df_all['MI'].sum()
    loss_total = df_all['H_cond_loss'].sum()
    h_total = df_all['H_concept'].sum()
    print(f'\n总 MI={mi_total:.3f}, 总概念熵={h_total:.3f}, 总条件熵损失={loss_total:.3f}, 整体信息损失比例={100 * loss_total / h_total:.1f}%')
    df_all.to_csv(_release_expand('${BENCHMARK_WORK_DIR}/outputs/analysis/mutual_info_loss.csv'), index=False)
    print('已保存: /ml_training/outputs/analysis/mutual_info_loss.csv')

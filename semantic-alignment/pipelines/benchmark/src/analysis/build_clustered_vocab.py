"""
构建聚类后的 one-hot 词汇表 — aligned_clustered 模式

修复点：
1. 枚举字段（anesthesia_type, sex, race, department, admission_chars 等）保留 aligned 原值，不做聚类
2. 只有文本字段（hx_*_detail, preop_diagnosis, scheduled_procedure_name, med_rec_summary）做聚类
3. conceptId 统一用整数格式（避免科学计数法）
"""
import os as _release_os
from release_runtime import expand_path as _release_expand, expand_config as _release_expand_config
import sys, os, csv, json, re
from pathlib import Path
from collections import defaultdict, Counter
import pandas as pd
MAPPING_DIR = Path(_release_expand('${ALIGNMENT_WORK_DIR}/data/mapping/final_output'))
CLUSTER_DIR = Path(_release_expand('${BENCHMARK_WORK_DIR}/mapping/final_clustered'))
VOCAB_DIR = Path(_release_expand('${BENCHMARK_WORK_DIR}/onehot_feature/aligned_clustered'))
VOCAB_DIR.mkdir(parents=True, exist_ok=True)
BENCHMARK_DIR = Path(_release_expand('${BENCHMARK_WORK_DIR}/benchmark'))
BENCHMARK_TASKS = ['asa', 'icu_admit', 'hospital_los']
ENUM_COLUMNS = {'scheduled_anesthesia_type', 'scheduled_operative_type', 'scheduled_surgical_approach', 'scheduled_surgical_position', 'surg_admission_chars', 'surgical_department', 'race', 'sex'}
TEXT_COLUMNS = {'preop_diagnosis', 'scheduled_procedure_name', 'med_rec_summary', 'hx_circ_detail', 'hx_resp_detail', 'hx_endo_detail', 'hx_neuro_detail', 'hx_dige_detail', 'hx_spine_detail', 'hx_immu_detail', 'hx_urine_detail', 'hx_other_detail', 'hx_surgery_detail'}

def normalize_cid(cid: str) -> str:
    """将 conceptId 归一化为整数格式（消除科学计数法）"""
    try:
        if 'e' in cid.lower() or '.' in cid:
            return str(int(float(cid)))
    except (ValueError, OverflowError):
        pass
    return cid

def main():
    print('=' * 70)
    print('构建聚类后词汇表 (aligned_clustered)')
    print('=' * 70)
    print('\n[1] 加载聚类映射...')
    with open(CLUSTER_DIR / 'cid_to_centroid.json') as f:
        raw_map = json.load(f)
    cid_to_centroid = {normalize_cid(k): normalize_cid(v) for k, v in raw_map.items()}
    print(f'  {len(cid_to_centroid)} 条 conceptId→代表概念映射')
    print('\n[2] 加载映射表...')
    fsn_to_cid = {}
    fv_to_cid = {}
    for ds in ['vitaldb', 'mover', 'inspire']:
        with open(MAPPING_DIR / f'{ds}_final_mapping.csv') as f:
            for row in csv.DictReader(f):
                fsn = row.get('fsn', '')
                cid = normalize_cid(row['conceptId'].strip())
                bare = re.sub('\\s*\\([^)]*\\)\\s*$', '', fsn).strip().lower() if fsn else ''
                if bare:
                    fsn_to_cid[bare] = cid
                fv_to_cid[row['field_value'].strip().lower()] = cid
    print(f'  FSN→conceptId: {len(fsn_to_cid)} | field_value→conceptId: {len(fv_to_cid)}')
    print('\n[3] 构建 generalized_mapping...')
    gen_map = {}
    for term in list(fsn_to_cid.keys()) + list(fv_to_cid.keys()):
        cid = fsn_to_cid.get(term) or fv_to_cid.get(term)
        if cid and cid in cid_to_centroid:
            gen_map[term] = cid_to_centroid[cid]
        elif cid:
            gen_map[term] = cid
    print(f'  {len(gen_map)} 条 text→聚类代表概念映射')
    print('\n[4] 从 benchmark 收集全部 FSN 值并按 centroid 频次聚类...')
    aligned_vocabs = {}
    for ds in ['vitaldb', 'mover', 'inspire']:
        with open(_release_expand(f'${{BENCHMARK_WORK_DIR}}/onehot_feature/aligned/{ds}.json')) as f:
            aligned_vocabs[ds] = json.load(f)
    fsn_freq = defaultdict(lambda: defaultdict(Counter))
    for ds in ['vitaldb', 'mover', 'inspire']:
        for task in BENCHMARK_TASKS:
            train_path = BENCHMARK_DIR / task / 'aligned' / ds / 'train.csv'
            if not train_path.exists():
                continue
            df = pd.read_csv(train_path, index_col=0)
            for col in TEXT_COLUMNS:
                if col in df.columns:
                    vals = df[col].dropna().astype(str)
                    fsn_freq[ds][col].update(vals)
    all_clustered = {}
    TOP_N = 500
    for ds in ['vitaldb', 'mover', 'inspire']:
        ds_vocab = aligned_vocabs[ds]
        clustered_columns = {}
        for col, terms in ds_vocab.get('columns', {}).items():
            if not terms:
                continue
            if col in ENUM_COLUMNS:
                clustered_columns[col] = terms
            elif col in TEXT_COLUMNS:
                centroid_counts = Counter()
                col_freq = fsn_freq.get(ds, {}).get(col, Counter())
                for t, freq in col_freq.items():
                    cid = fsn_to_cid.get(t.lower()) or fv_to_cid.get(t.lower())
                    if cid and cid in cid_to_centroid:
                        centroid_counts[cid_to_centroid[cid]] += freq
                    elif cid:
                        centroid_counts[cid] += freq
                top_centroids = [cid for cid, _ in centroid_counts.most_common(TOP_N)]
                if top_centroids:
                    clustered_columns[col] = sorted(top_centroids, key=lambda x: int(x) if x.isdigit() else x)
            else:
                clustered_columns[col] = terms
        vocab = {'version': 'aligned_clustered', 'source': 'Clustered: enum fields from aligned, text fields via core+synonym+bert clustering', 'columns': clustered_columns, 'generalized_mapping': gen_map}
        with open(VOCAB_DIR / f'{ds}.json', 'w', encoding='utf-8') as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)
        all_clustered[ds] = clustered_columns
        total_terms = sum((len(v) for v in clustered_columns.values()))
        total_text = sum((len(v) for col, v in clustered_columns.items() if col in TEXT_COLUMNS))
        total_enum = sum((len(v) for col, v in clustered_columns.items() if col in ENUM_COLUMNS))
        print(f'    {ds}: {len(clustered_columns)} 列, {total_terms} 值 (文本:{total_text}, 枚举:{total_enum})')
    combined_columns = {}
    for ds in ['vitaldb', 'mover', 'inspire']:
        for col, terms in all_clustered[ds].items():
            if col not in combined_columns:
                combined_columns[col] = set()
            combined_columns[col].update(terms)
    combined_columns = {col: sorted(terms, key=lambda x: int(x) if x.isdigit() else x) for col, terms in combined_columns.items()}
    combined_vocab = {'version': 'aligned_clustered_combined', 'columns': combined_columns, 'generalized_mapping': gen_map}
    with open(VOCAB_DIR / 'combined.json', 'w', encoding='utf-8') as f:
        json.dump(combined_vocab, f, ensure_ascii=False, indent=2)
    total_all = sum((len(v) for v in combined_columns.values()))
    print(f'\n  combined: {len(combined_columns)} 列, {total_all} 值')
    print(f'\n✅ 词汇表已保存至: {VOCAB_DIR}')
if __name__ == '__main__':
    main()

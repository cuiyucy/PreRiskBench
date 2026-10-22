"""
实验④ — 基于 SNOMED 层级的概念分布可视化与数据画像

将三个数据集的所有映射概念按 SNOMED 顶级语义层级归类，
展示各数据集在这些类别下的概念数量分布和样本覆盖分布。
使用 FSN 末尾括号标签（disorder, procedure, finding, product, ...）作为层级分类。
"""
import os as _release_os
from release_runtime import expand_path as _release_expand, expand_config as _release_expand_config
import pandas as pd
import json
from pathlib import Path
from collections import defaultdict, Counter
import re
MAPPING_DIR = Path(_release_expand('${ALIGNMENT_WORK_DIR}/data/mapping/final_output'))
OUTPUT_DIR = Path(_release_expand('${BENCHMARK_WORK_DIR}/outputs/analysis/exp04'))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TOP_LEVEL_MAP = {'disorder': 'Clinical finding', 'finding': 'Clinical finding', 'situation': 'Clinical finding', 'morphologic abnormality': 'Clinical finding', 'procedure': 'Procedure', 'regime/therapy': 'Procedure', 'product': 'Pharmaceutical / biologic product', 'medicinal product': 'Pharmaceutical / biologic product', 'clinical drug': 'Pharmaceutical / biologic product', 'medicinal product form': 'Pharmaceutical / biologic product', 'dose form': 'Pharmaceutical / biologic product', 'body structure': 'Body structure', 'organism': 'Organism', 'substance': 'Substance', 'observable entity': 'Observable entity', 'qualifier value': 'Qualifier value', 'event': 'Event', 'person': 'Social context', 'physical object': 'Physical object', 'navigational concept': 'Navigational concept', 'assessment scale': 'Staging / scales', 'record artifact': 'Record artifact', 'context-dependent category': 'Context-dependent', 'specialty': 'Specialty', 'environment': 'Environment / geographic'}

def extract_semantic_label(fsn: str) -> str:
    m = re.search('\\(([^)]+)\\)$', fsn)
    if m:
        return m.group(1).lower()
    return 'unknown'

def load_mapping(dataset: str) -> pd.DataFrame:
    path = MAPPING_DIR / f'{dataset}_final_mapping.csv'
    df = pd.read_csv(path, dtype={'conceptId': str})
    df['semantic_label'] = df['fsn'].apply(extract_semantic_label)
    df['top_level'] = df['semantic_label'].map(lambda x: TOP_LEVEL_MAP.get(x, x))
    return df

def concept_distribution():
    datasets = ['vitaldb', 'mover', 'inspire']
    top_level_stats = []
    for ds in datasets:
        df = load_mapping(ds)
        unique_by_tl = df.groupby('top_level')['conceptId'].nunique()
        total_mappings = df.groupby('top_level').size()
        for tl in sorted(set(df['top_level'].unique()) | set(TOP_LEVEL_MAP.values())):
            if tl not in unique_by_tl.index:
                continue
            top_level_stats.append({'数据集': ds, '顶级层级': tl, '唯一概念数': int(unique_by_tl[tl]), '映射记录数': int(total_mappings[tl])})
    tl_df = pd.DataFrame(top_level_stats)
    pivot_concept = tl_df.pivot_table(index='顶级层级', columns='数据集', values='唯一概念数', fill_value=0, aggfunc='sum')
    pivot_record = tl_df.pivot_table(index='顶级层级', columns='数据集', values='映射记录数', fill_value=0, aggfunc='sum')
    print('=' * 90)
    print('实验④ 基于 SNOMED 层级的概念分布可视化')
    print('=' * 90)
    print('\n[表1] 各数据集按顶级层级的唯一概念数分布\n')
    print(pivot_concept.to_string())
    print('\n[表2] 各数据集按顶级层级的映射记录数分布\n')
    print(pivot_record.to_string())
    print('\n[表3] 按顶级层级的三库概念重叠\n')
    all_datasets_data = {}
    for ds in datasets:
        all_datasets_data[ds] = {}
        by_tl = load_mapping(ds).groupby('top_level')
        for tl, grp in by_tl:
            all_datasets_data[ds][tl] = set(grp['conceptId'].unique())
    overlap_rows = []
    for tl in sorted(set(TOP_LEVEL_MAP.values())):
        sets_for_tl = []
        for ds in datasets:
            s = all_datasets_data[ds].get(tl, set())
            sets_for_tl.append(s)
        union = set().union(*sets_for_tl) if sets_for_tl else set()
        intersection = set.intersection(*sets_for_tl) if all((s for s in sets_for_tl)) else set()
        if len(union) > 0:
            overlap_rows.append({'顶级层级': tl, '并集概念数': len(union), '交集概念数': len(intersection), '重叠率': f'{len(intersection) / len(union) * 100:.1f}%' if union else '0%'})
    overlap_df = pd.DataFrame(overlap_rows)
    print(overlap_df.to_string(index=False))
    pivot_concept.to_csv(OUTPUT_DIR / 'concept_by_top_level.csv', encoding='utf-8-sig')
    pivot_record.to_csv(OUTPUT_DIR / 'records_by_top_level.csv', encoding='utf-8-sig')
    overlap_df.to_csv(OUTPUT_DIR / 'overlap_by_top_level.csv', index=False, encoding='utf-8-sig')
    print(f'\n结果已保存至: {OUTPUT_DIR}')
    return (pivot_concept, pivot_record, overlap_df)
if __name__ == '__main__':
    concept_distribution()

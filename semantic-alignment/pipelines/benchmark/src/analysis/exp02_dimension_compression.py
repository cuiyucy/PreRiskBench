"""
实验② — 概念空间冗余消除与维度压缩统计

量化 SNOMED 对齐对多源数据特征空间的"净化"能力：
- N_raw: 各数据集原始表述（field_value）去重后的独立 token 数
- N_snomed: 映射后唯一 SNOMED conceptId 数
- 压缩比 = N_raw / N_snomed
- 按 SNOMED 语义标签分解各层级的压缩效果
"""
import os as _release_os
from release_runtime import expand_path as _release_expand, expand_config as _release_expand_config
import pandas as pd
import json
from pathlib import Path
from collections import defaultdict
import re
MAPPING_DIR = Path(_release_expand('${ALIGNMENT_WORK_DIR}/data/mapping/final_output'))
OUTPUT_DIR = Path(_release_expand('${BENCHMARK_WORK_DIR}/outputs/analysis/exp02'))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def extract_semantic_label(fsn: str) -> str:
    """从 FSN 末尾提取语义标签，如 '(disorder)', '(procedure)'"""
    m = re.search('\\(([^)]+)\\)$', fsn)
    if m:
        return m.group(1).lower()
    return 'unknown'

def load_mapping(dataset: str) -> pd.DataFrame:
    path = MAPPING_DIR / f'{dataset}_final_mapping.csv'
    df = pd.read_csv(path, dtype={'conceptId': str})
    df['semantic_label'] = df['fsn'].apply(extract_semantic_label)
    return df

def compute_stats():
    datasets = ['vitaldb', 'mover', 'inspire']
    rows = []
    detail_rows = []
    for ds in datasets:
        df = load_mapping(ds)
        total_mappings = len(df)
        unique_raw = df['field_value'].nunique()
        unique_concepts = df['conceptId'].nunique()
        compression_ratio = unique_raw / unique_concepts if unique_concepts > 0 else 0
        low_conf = df[df['score'] < 0.7].shape[0] if 'score' in df.columns else 0
        avg_score = df['score'].mean() if 'score' in df.columns else 0
        rows.append({'数据集': ds, '总映射记录数': total_mappings, '原始表述数(N_raw)': unique_raw, '唯一概念数(N_snomed)': unique_concepts, '压缩比': round(compression_ratio, 2), '低置信度映射': low_conf, '平均置信度': round(avg_score, 4)})
        label_stats = df.groupby('semantic_label').agg(raw_count=('field_value', 'nunique'), concept_count=('conceptId', 'nunique')).reset_index()
        label_stats['压缩比'] = label_stats['raw_count'] / label_stats['concept_count']
        label_stats['数据集'] = ds
        detail_rows.append(label_stats)
    summary = pd.DataFrame(rows)
    detail = pd.concat(detail_rows, ignore_index=True)
    detail = detail.sort_values(['数据集', '压缩比'], ascending=[True, False])
    print('=' * 90)
    print('实验② 概念空间冗余消除与维度压缩统计')
    print('=' * 90)
    print('\n[表1] 各数据集压缩总览\n')
    print(summary.to_string(index=False))
    print('\n[表2] 按语义标签分解\n')
    print(detail.to_string(index=False))
    print('\n[表3] 跨数据集概念重叠\n')
    concept_sets = {}
    for ds in datasets:
        df = load_mapping(ds)
        concept_sets[ds] = set(df['conceptId'].unique())
    all_combined = set()
    for s in concept_sets.values():
        all_combined |= s
    all_intersection = set.intersection(*concept_sets.values()) if len(concept_sets) > 1 else set()
    all_others_union = {}
    for ds in datasets:
        others = [concept_sets[d] for d in datasets if d != ds]
        all_others_union[ds] = set()
        for s in others:
            all_others_union[ds] |= s
    print(f"{'指标':30s} {'数值':>10s}")
    print('-' * 42)
    print(f"{'三库概念并集':30s} {len(all_combined):>10d}")
    print(f"{'三库概念交集 (共有)':30s} {len(all_intersection):>10d}")
    for ds in datasets:
        unique_to_ds = concept_sets[ds] - all_others_union[ds]
        print(f"{ds + ' 特有概念':30s} {len(unique_to_ds):>10d}")
    summary.to_csv(OUTPUT_DIR / 'compression_summary.csv', index=False, encoding='utf-8-sig')
    detail.to_csv(OUTPUT_DIR / 'compression_by_label.csv', index=False, encoding='utf-8-sig')
    report_md = f'# 实验② — 概念空间冗余消除与维度压缩\n\n## 表1: 压缩总览\n\n| 数据集 | 原始表述数 | 唯一概念数 | 压缩比 | 平均置信度 |\n| :--- | ---: | ---: | ---: | ---: |\n'
    for _, r in summary.iterrows():
        report_md += f"| {r['数据集']} | {r['原始表述数(N_raw)']} | {r['唯一概念数(N_snomed)']} | {r['压缩比']} | {r['平均置信度']} |\n"
    report_md += f'\n## 表2: 跨数据集概念重叠\n\n| 指标 | 数值 |\n| :--- | ---: |\n| 三库概念并集 | {len(all_combined)} |\n| 三库概念交集 | {len(all_intersection)} |\n'
    for ds in datasets:
        unique_to_ds = concept_sets[ds] - all_others_union[ds]
        report_md += f'| {ds} 特有概念 | {len(unique_to_ds)} |\n'
    (OUTPUT_DIR / 'report.md').write_text(report_md, encoding='utf-8')
    print(f'\n结果已保存至: {OUTPUT_DIR}')
    return (summary, detail)
if __name__ == '__main__':
    compute_stats()

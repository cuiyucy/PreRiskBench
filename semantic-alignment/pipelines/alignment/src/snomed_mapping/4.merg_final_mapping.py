"""
最终映射合并器：整合字符串匹配、向量检索、LLM 处理结果
输出每个数据集一个 CSV + 一份全局统计 MD

去重优先级:
  1. special_fields_mapping
  2. 字符串匹配 (source=exact, score=1.0)
  3. 向量检索 (source=retrieval, score>0.85)
  4. llm_mapping_corrected (score≥0.85)
  5. 2postpro
"""
import os as _release_os
from release_runtime import expand_path as _release_expand, expand_config as _release_expand_config
import os
import csv
import pandas as pd
from pathlib import Path
from collections import defaultdict
PROJECT_ROOT = _release_expand('${ALIGNMENT_WORK_DIR}')
STRING_DIR = os.path.join(PROJECT_ROOT, 'data', 'mapping', 'string_mapping')
RETRIEVAL_DIR = os.path.join(PROJECT_ROOT, 'data', 'mapping', 'retrieval_mapping')
LLM_DIR = os.path.join(PROJECT_ROOT, 'data', 'mapping', 'llm_process')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'mapping', 'final_output')
os.makedirs(OUTPUT_DIR, exist_ok=True)
DATASETS = ['inspire', 'mover', 'vitaldb']
FIELDNAMES = ['dataset', 'field_name', 'field_value', 'conceptId', 'fsn', 'source', 'score']

def load_source(name: str, path: str, dataset: str=None) -> pd.DataFrame:
    """加载一个数据源，统一添加 dataset 列和 score 列"""
    if not os.path.exists(path):
        print(f'  ⚠ 文件不存在: {path}')
        return pd.DataFrame(columns=FIELDNAMES)
    df = pd.read_csv(path, keep_default_na=False, na_values=[], dtype={'conceptId': str})
    df = df.fillna('')
    if 'dataset' not in df.columns and dataset:
        df['dataset'] = dataset
    for col in FIELDNAMES:
        if col not in df.columns:
            df[col] = ''
    if 'score' in df.columns:
        df['score'] = pd.to_numeric(df['score'], errors='coerce').fillna(0)
    else:
        df['score'] = 1.0
    return df

def build_priority_map(dataset: str) -> pd.DataFrame:
    """按优先级合并单个数据集的所有来源"""
    string_path = os.path.join(STRING_DIR, f'{dataset}_field_values_mapping.csv')
    retrieval_path = os.path.join(RETRIEVAL_DIR, f'{dataset}_field_values_mapping_keyword_rerank.csv')
    special_path = os.path.join(LLM_DIR, 'special_fields_mapping.csv')
    llm_corrected_path = os.path.join(LLM_DIR, 'llm_mapping_corrected.csv')
    postpro_path = os.path.join(LLM_DIR, '2postpro.csv')

    def load_by_dataset(path: str, label: str) -> pd.DataFrame:
        df = load_source(label, path)
        if df.empty:
            return df
        return df[df['dataset'] == dataset]
    special_df = load_by_dataset(special_path, 'special')
    llm_corrected_df = load_by_dataset(llm_corrected_path, 'llm_corrected')
    postpro_df = load_by_dataset(postpro_path, '2postpro')
    string_df = load_source('string', string_path, dataset)
    string_exact = string_df[(string_df['source'] == 'exact') & (string_df['score'] == 1.0)].copy()
    retrieval_df = load_source('retrieval', retrieval_path, dataset)
    retrieval_gt = retrieval_df[(retrieval_df['source'] == 'retrieval') & (retrieval_df['score'] > 0.85)].copy()
    layers = [(5, postpro_df, '2postpro'), (4, llm_corrected_df, 'llm_corrected'), (3, retrieval_gt, 'retrieval'), (2, string_exact, 'string_exact'), (1, special_df, 'special')]
    merged = {}
    stats_counter = defaultdict(int)
    for priority, df, label in layers:
        if df.empty:
            continue
        for _, row in df.iterrows():
            key = (row['dataset'], row['field_name'], row['field_value'])
            row_dict = {k: row.get(k, '') for k in FIELDNAMES}
            if label == 'special':
                row_dict['score'] = 0.95
            if row_dict['conceptId']:
                cid = str(row_dict['conceptId'])
                if cid.endswith('.0'):
                    cid = cid[:-2]
                row_dict['conceptId'] = cid
            row_dict['_priority'] = priority
            row_dict['_source_label'] = label
            merged[key] = row_dict
            stats_counter[label] += 1
    result = pd.DataFrame(merged.values())
    result = result.sort_values(['dataset', 'field_name', 'field_value']).reset_index(drop=True)
    actual_stats = {'special': len(result[result['_source_label'] == 'special']), 'string_exact': len(result[result['_source_label'] == 'string_exact']), 'retrieval_gt': len(result[result['_source_label'] == 'retrieval']), 'llm_corrected': len(result[result['_source_label'] == 'llm_corrected']), '2postpro': len(result[result['_source_label'] == '2postpro'])}
    result = result[FIELDNAMES]
    return (result, actual_stats, {'special': len(special_df), 'string_exact': len(string_exact), 'retrieval_gt': len(retrieval_gt), 'llm_corrected': len(llm_corrected_df), '2postpro': len(postpro_df)})

def generate_stats(all_stats: dict, all_counts: dict, results: dict):
    """生成全局统计 MD 文件"""
    md_path = os.path.join(OUTPUT_DIR, 'mapping_stats.md')
    lines = []
    lines.append('# SNOMED CT 映射统计报告')
    lines.append('')
    lines.append('## 各数据集概览')
    lines.append('')
    lines.append('| 数据集 | 总记录 | 字符串匹配(exact) | 向量检索(>0.85) | LLM(≥0.85) | LLM(后处理) | 平均分 |')
    lines.append('|--------|--------|-------------------|----------------|------------|-------------|--------|')
    total_all = 0
    total_string = 0
    total_retrieval = 0
    total_llm_high = 0
    total_llm_low = 0
    all_scores = []
    for ds in DATASETS:
        df = results[ds]
        n = len(df)
        total_all += n
        scores = pd.to_numeric(df['score'], errors='coerce').fillna(0)
        avg = scores.mean()
        s_exact = all_stats[ds].get('string_exact', 0)
        r_gt = all_stats[ds].get('retrieval_gt', 0)
        llm_high = len(scores[(df['source'] == 'llm') & (scores >= 0.85)])
        llm_low = len(scores[(df['source'] == 'llm') & (scores < 0.85)])
        total_string += s_exact
        total_retrieval += r_gt
        total_llm_high += llm_high
        total_llm_low += llm_low
        all_scores.extend(scores.tolist())
        lines.append(f'| {ds} | {n} | {s_exact} ({s_exact / n * 100:.1f}%) | {r_gt} ({r_gt / n * 100:.1f}%) | {llm_high} ({llm_high / n * 100:.1f}%) | {llm_low} ({llm_low / n * 100:.1f}%) | {avg:.3f} |')
    avg_all = pd.Series(all_scores).mean() if all_scores else 0
    lines.append(f'| **合计** | **{total_all}** | **{total_string}** ({total_string / total_all * 100:.1f}%) | **{total_retrieval}** ({total_retrieval / total_all * 100:.1f}%) | **{total_llm_high}** ({total_llm_high / total_all * 100:.1f}%) | **{total_llm_low}** ({total_llm_low / total_all * 100:.1f}%) | **{avg_all:.3f}** |')
    lines.append('')
    lines.append('## 来源数据量')
    lines.append('')
    lines.append('| 来源 | inspire | mover | vitaldb |')
    lines.append('|------|---------|-------|---------|')
    for label in ['special', 'string_exact', 'retrieval_gt', 'llm_corrected', '2postpro']:
        vals = [str(all_counts[ds].get(label, 0)) for ds in DATASETS]
        lines.append(f"| {label} | {' | '.join(vals)} |")
    lines.append('')
    lines.append('## 最终合并文件评分分布')
    lines.append('')
    lines.append('| 区间 | inspire | mover | vitaldb | 合计 |')
    lines.append('|------|---------|-------|---------|------|')
    bins = [(0, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.85), (0.85, 0.9), (0.9, 0.95), (0.95, 1.0)]
    for lo, hi in bins:
        label = f'{lo}-{hi}' if hi < 1.0 else f'{lo}-1.0'
        row_vals = []
        for ds in DATASETS:
            df = results[ds]
            cnt = len(df[(df['score'] >= lo) & (df['score'] < hi)])
            row_vals.append(str(cnt))
        total_bin = sum((int(v) for v in row_vals))
        lines.append(f"| {label} | {' | '.join(row_vals)} | {total_bin} |")
    lines.append('')
    lines.append('## 最终合并文件统计')
    lines.append('')
    for ds in DATASETS:
        df = results[ds]
        scores = pd.to_numeric(df['score'], errors='coerce').fillna(0)
        lines.append(f'### {ds}')
        lines.append(f'')
        lines.append(f'- **总记录**: {len(df)}')
        lines.append(f'- **平均分**: {scores.mean():.4f}')
        lines.append(f'- **中位数**: {scores.median():.4f}')
        lines.append(f'- **标准差**: {scores.std():.4f}')
        lines.append(f'- **≥0.85**: {len(scores[scores >= 0.85])} ({len(scores[scores >= 0.85]) / len(scores) * 100:.1f}%)')
        lines.append(f'- **<0.85**: {len(scores[scores < 0.85])} ({len(scores[scores < 0.85]) / len(scores) * 100:.1f}%)')
        lines.append(f'')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'✓ 统计报告: {md_path}')
    return md_path

def main():
    print('=' * 60)
    print('最终映射合并器')
    print('=' * 60)
    all_results = {}
    all_stats = {}
    all_counts = {}
    for ds in DATASETS:
        print(f'\n处理 {ds}...')
        merged_df, stats, counts = build_priority_map(ds)
        output_path = os.path.join(OUTPUT_DIR, f'{ds}_final_mapping.csv')
        merged_df.to_csv(output_path, index=False, encoding='utf-8')
        all_results[ds] = merged_df
        all_stats[ds] = stats
        all_counts[ds] = counts
        n = len(merged_df)
        s = stats.get('string_exact', 0)
        r = stats.get('retrieval_gt', 0)
        l = n - s - r
        print(f'  ✓ {n} 条: string_exact={s}, retrieval>{r}, llm_others={l}')
        print(f'    输出: {output_path}')
    generate_stats(all_stats, all_counts, all_results)
    print(f"\n{'=' * 60}")
    print('全部完成!')
    print(f'输出目录: {OUTPUT_DIR}')
    print(f"{'=' * 60}")
if __name__ == '__main__':
    main()

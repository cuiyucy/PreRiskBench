"""
分析实验: 计算 unaligned / aligned / aligned_clustered 三种模式下的
两两重叠率和 JS 散度

数据源 (三种模式基于完全相同的 surgery_id 队列):
- unaligned:         dataset/{ds}/prerisk_bench_feat/*.csv  (原始文本/编码)
- aligned:           dataset/{ds}/alignment_feat/*_aligned.csv (已对齐 FSN) → conceptId
- aligned_clustered: 同 aligned 源, conceptId → 聚类 centroid

只统计临床概念文本列 (preop_diagnosis / scheduled_procedure_name /
med_rec_summary / hx_*_detail)，不含枚举列(麻醉类型/科室等)与数值列。
统计维度: sample-level (每个概念在多少个样本中出现)。
"""
import os as _release_os
from release_runtime import expand_path as _release_expand, expand_config as _release_expand_config
import json
import re
import csv
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
MAPPING_DIR = Path(_release_expand('${ALIGNMENT_WORK_DIR}/data/mapping/final_output'))
CLUSTER_DIR = Path(_release_expand('${BENCHMARK_WORK_DIR}/mapping/final_clustered'))
DATASET_DIR = Path(_release_expand('${BENCHMARK_WORK_DIR}/dataset'))
OUTPUT_DIR = Path(_release_expand('${BENCHMARK_WORK_DIR}/outputs/analysis/overlap_js'))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATASETS = ['vitaldb', 'mover', 'inspire']
UNALIGNED_FILES = ['feat_surg_vars.csv', 'feat_past_his.csv', 'feat_med_rec.csv', 'feat_anes_vars.csv']
ALIGNED_FILES = ['feat_surg_vars_aligned.csv', 'feat_past_his_aligned.csv', 'feat_med_rec_aligned.csv', 'feat_anes_vars_aligned.csv']
EXACT_TEXT_COLS = {'preop_diagnosis', 'scheduled_procedure_name', 'med_rec_summary'}
ENUM_COLS = {'surg_admission_chars', 'surgical_department', 'scheduled_anesthesia_type', 'scheduled_operative_type', 'scheduled_surgical_approach', 'scheduled_surgical_position'}

def is_text_col(col: str) -> bool:
    """文本概念列 (聚类模式下 → centroid)"""
    c = col.strip().lower()
    if c in EXACT_TEXT_COLS:
        return True
    if c.startswith('hx_') and (c.endswith('_detail') or c.endswith('_details')) and (not c.endswith('_cn')):
        return True
    return False

def is_enum_col(col: str) -> bool:
    """枚举概念列 (聚类模式下保留 aligned cid, 不做 centroid)"""
    return col.strip().lower() in ENUM_COLS

def is_concept_col(col: str) -> bool:
    return is_text_col(col) or is_enum_col(col)

def normalize_cid(cid: str) -> str:
    """conceptId 归一化为整数字符串 (消除科学计数法)"""
    try:
        if 'e' in cid.lower() or '.' in cid:
            return str(int(float(cid)))
    except (ValueError, OverflowError):
        pass
    return cid

def load_cid_to_centroid() -> dict:
    with open(CLUSTER_DIR / 'cid_to_centroid.json') as f:
        raw = json.load(f)
    return {normalize_cid(k): normalize_cid(v) for k, v in raw.items()}

def load_text_to_cid() -> dict:
    """构建 text(bare FSN / field_value, 小写) → conceptId 映射"""
    fsn_to_cid = {}
    fv_to_cid = {}
    for ds in DATASETS:
        with open(MAPPING_DIR / f'{ds}_final_mapping.csv') as f:
            for row in csv.DictReader(f):
                cid = normalize_cid(row['conceptId'].strip())
                fsn = row.get('fsn', '') or ''
                bare = re.sub('\\s*\\([^)]*\\)\\s*$', '', fsn).strip().lower()
                if bare:
                    fsn_to_cid.setdefault(bare, cid)
                fv = (row.get('field_value', '') or '').strip().lower()
                if fv:
                    fv_to_cid.setdefault(fv, cid)
    merged = dict(fv_to_cid)
    merged.update(fsn_to_cid)
    return merged

def _load_feat_frame(ds: str, files: list) -> pd.DataFrame:
    """加载并横向拼接特征文件，只保留 surgery_id + 概念列(文本+枚举)"""
    subdir = 'prerisk_bench_feat' if files is UNALIGNED_FILES else 'alignment_feat'
    dfs = []
    for filename in files:
        fp = DATASET_DIR / ds / subdir / filename
        if not fp.exists():
            continue
        df = pd.read_csv(fp, low_memory=False)
        keep = [c for c in df.columns if c == 'surgery_id' or is_concept_col(c)]
        if keep:
            dfs.append(df[keep])
    if not dfs:
        return pd.DataFrame()
    combined = pd.concat(dfs, axis=1)
    combined = combined.loc[:, ~combined.columns.duplicated()]
    return combined

def get_sample_level_counts(ds: str, mode: str, text_to_cid: dict, cid_to_centroid: dict) -> Counter:
    """统计每个概念在多少个样本中出现 (sample-level)"""
    counts = Counter()
    files = UNALIGNED_FILES if mode == 'unaligned' else ALIGNED_FILES
    combined = _load_feat_frame(ds, files)
    if combined.empty:
        return counts
    concept_cols = [c for c in combined.columns if is_concept_col(c)]
    for _, row in combined.iterrows():
        sample_items = set()
        for col in concept_cols:
            val = row[col]
            if pd.isna(val) or not isinstance(val, str) or (not val.strip()):
                continue
            enum_col = is_enum_col(col)
            for p in (x.strip() for x in val.split(';')):
                if not p:
                    continue
                if mode == 'unaligned':
                    sample_items.add(p.lower())
                else:
                    cid = text_to_cid.get(p.lower())
                    if not cid:
                        continue
                    if mode == 'aligned' or enum_col:
                        sample_items.add(cid)
                    else:
                        sample_items.add(cid_to_centroid.get(cid, cid))
        for item in sample_items:
            counts[item] += 1
    return counts

def compute_overlap_rates(counts_by_ds: dict) -> list:
    datasets = list(counts_by_ds.keys())
    results = []
    for i in range(len(datasets)):
        for j in range(i + 1, len(datasets)):
            a, b = (datasets[i], datasets[j])
            set_a = set(counts_by_ds[a].keys())
            set_b = set(counts_by_ds[b].keys())
            inter = len(set_a & set_b)
            union = len(set_a | set_b)
            results.append({'pair': f'{a} ↔ {b}', f'{a}_unique': len(set_a), f'{b}_unique': len(set_b), 'intersection': inter, 'union': union, 'jaccard_pct': round(inter / union * 100, 2) if union else 0, f'{a}_coverage_pct': round(inter / len(set_a) * 100, 2) if set_a else 0, f'{b}_coverage_pct': round(inter / len(set_b) * 100, 2) if set_b else 0})
    all_sets = [set(counts_by_ds[d].keys()) for d in datasets]
    triple = all_sets[0] & all_sets[1] & all_sets[2]
    results.append({'pair': '三库共有', 'intersection': len(triple)})
    return results

def _js(p_counts: Counter, q_counts: Counter, use_log2: bool) -> float:
    keys = sorted(set(p_counts) | set(q_counts))
    p = np.array([p_counts.get(k, 0) for k in keys], dtype=np.float64)
    q = np.array([q_counts.get(k, 0) for k in keys], dtype=np.float64)
    if p.sum() > 0:
        p /= p.sum()
    if q.sum() > 0:
        q /= q.sum()
    m = 0.5 * (p + q)
    log = np.log2 if use_log2 else np.log

    def kl(x, y):
        mask = x > 0
        return np.sum(x[mask] * log(x[mask] / y[mask]))
    val = 0.5 * (kl(p, m) + kl(q, m))
    return round(float(val), 6) if not np.isnan(val) else 0.0

def compute_js_divergence(counts_by_ds: dict, use_log2: bool) -> list:
    datasets = list(counts_by_ds.keys())
    results = []
    for i in range(len(datasets)):
        for j in range(i + 1, len(datasets)):
            a, b = (datasets[i], datasets[j])
            results.append({'pair': f'{a} ↔ {b}', 'js_divergence': _js(counts_by_ds[a], counts_by_ds[b], use_log2)})
    return results

def _generate_markdown(all_results: dict):
    lines = ['# 概念重叠率 & JS 散度报告 (Sample-level)\n']
    lines.append(f"> 更新时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n")
    lines.append('> 数据源: unaligned=prerisk_bench_feat, aligned/clustered=alignment_feat\n')
    lines.append('> 统计维度: sample-level, 临床概念文本列 + 枚举概念列(麻醉/入院/科室/术式)\n')
    lines.append('## 概览: 三种模式的概念空间统计\n')
    lines.append('| 模式 | Vitaldb 概念数 | Mover 概念数 | Inspire 概念数 |')
    lines.append('| :--- | ---: | ---: | ---: |')
    for mode in ['unaligned', 'aligned', 'aligned_clustered']:
        uc = all_results[mode]['unique_counts']
        lines.append(f"| {mode} | {uc['vitaldb']} | {uc['mover']} | {uc['inspire']} |")
    lines.append('\n## 两两重叠率\n')
    for mode in ['unaligned', 'aligned', 'aligned_clustered']:
        lines.append(f'### {mode}\n')
        lines.append('| 数据集对 | Jaccard(%) | Vitaldb覆盖率(%) | Mover覆盖率(%) | Inspire覆盖率(%) | 交集 | 并集 |')
        lines.append('| :--- | ---: | ---: | ---: | ---: | ---: | ---: |')
        for r in all_results[mode]['overlap']:
            if r['pair'] == '三库共有':
                lines.append(f"\n**三库共有概念数**: {r['intersection']}\n")
                continue
            ds1, ds2 = r['pair'].split(' ↔ ')
            cov1 = r.get(f'{ds1}_coverage_pct', '-')
            cov2 = r.get(f'{ds2}_coverage_pct', '-')
            cov_v = cov1 if ds1 == 'vitaldb' else cov2 if ds2 == 'vitaldb' else '-'
            cov_m = cov1 if ds1 == 'mover' else cov2 if ds2 == 'mover' else '-'
            cov_i = cov1 if ds1 == 'inspire' else cov2 if ds2 == 'inspire' else '-'
            lines.append(f"| {r['pair']} | {r['jaccard_pct']} | {cov_v} | {cov_m} | {cov_i} | {r['intersection']} | {r['union']} |")
        lines.append('')
    for base_key, base_name in [('js_ln', '自然对数 ln')]:
        lines.append(f'\n## JS 散度 ({base_name})\n')
        lines.append('| 模式 | Vitaldb ↔ Mover | Vitaldb ↔ Inspire | Mover ↔ Inspire |')
        lines.append('| :--- | ---: | ---: | ---: |')
        for mode in ['unaligned', 'aligned', 'aligned_clustered']:
            js = {x['pair']: x['js_divergence'] for x in all_results[mode][base_key]}
            lines.append(f"| {mode} | {js.get('vitaldb ↔ mover', 0):.6f} | {js.get('vitaldb ↔ inspire', 0):.6f} | {js.get('mover ↔ inspire', 0):.6f} |")
    lines.append('')
    md_path = OUTPUT_DIR / 'overlap_js_report.md'
    with open(md_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f'\nMarkdown 报告: {md_path}')
    print('\n'.join(lines))

def main():
    print('=' * 80)
    print(' 概念重叠率 & JS 散度分析 (Sample-level)')
    print('=' * 80)
    cid_to_centroid = load_cid_to_centroid()
    text_to_cid = load_text_to_cid()
    print(f'cid_to_centroid: {len(cid_to_centroid)} | text_to_cid: {len(text_to_cid)}')
    all_results = {}
    for mode in ['unaligned', 'aligned', 'aligned_clustered']:
        print(f"\n{'=' * 60}\n 模式: {mode}\n{'=' * 60}")
        counts_by_ds = {}
        for ds in DATASETS:
            counts = get_sample_level_counts(ds, mode, text_to_cid, cid_to_centroid)
            counts_by_ds[ds] = counts
            print(f'  {ds}: {len(counts)} 唯一概念, {sum(counts.values())} 概念-样本对')
        overlap = compute_overlap_rates(counts_by_ds)
        print('  --- 重叠率 ---')
        for r in overlap:
            if r['pair'] == '三库共有':
                print(f"    三库共有: {r['intersection']}")
            else:
                print(f"    {r['pair']}: Jaccard={r['jaccard_pct']}%")
        js_log2 = compute_js_divergence(counts_by_ds, use_log2=True)
        js_ln = compute_js_divergence(counts_by_ds, use_log2=False)
        print('  --- JS 散度 (log2 | ln) ---')
        for r2, rn in zip(js_log2, js_ln):
            print(f"    {r2['pair']}: log2={r2['js_divergence']:.4f}, ln={rn['js_divergence']:.4f}")
        all_results[mode] = {'overlap': overlap, 'js_log2': js_log2, 'js_ln': js_ln, 'unique_counts': {ds: len(c) for ds, c in counts_by_ds.items()}}
    with open(OUTPUT_DIR / 'overlap_js_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    _generate_markdown(all_results)
    print(f'\n结果已保存至: {OUTPUT_DIR}')
if __name__ == '__main__':
    main()

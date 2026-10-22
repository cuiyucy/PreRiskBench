"""
ICD 编码消融实验：直接用 ICD 原始编码走管线映射 vs 用 ICD 英文释义（desc_en）走管线映射

目的:
    消融"用 ICD 英文释义（desc_en）先翻译再映射"这一步是否必要。
    同样的 SNOMED 映射管线（string exact match + vector retrieval），
    对比两种输入的映射质量。

设计:
    - desc (基线): 使用原始管线输入 `inspire_unique_values.json` 中的 desc_en 值
      （即之前实际用于映射的 ICD 英文释义），复现基线质量。
    - code (消融): 字段名不变（保证 field_type_mapping 相同），但取值换成原始 ICD 编码
      （从 feat 文件中提取的原始编码，如 K40 / 0YQ50 / I10）。

数据来源:
    /ml_training/dataset/inspire/prerisk_bench_feat
      - feat_surg_vars.csv : preop_diagnosis(ICD-10-CM), scheduled_procedure_name(ICD-10-PCS)
      - feat_past_his.csv  : hx_*_detail(ICD编码)

对比指标（按 field, 按 variant）:
    - exact 匹配率 (string mapper source=exact)
    - retrieval 命中率 / 分数均值 / 中位数 / >=0.85 比例 / >=0.8 比例
    - 无映射(空 conceptId) 数量
"""
import os as _release_os
from release_runtime import expand_path as _release_expand, expand_config as _release_expand_config
import os
import sys
import json
import re
import csv
import time
import importlib.util
from collections import defaultdict
import numpy as np
import pandas as pd
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FEAT_DIR = _release_expand('${BENCHMARK_WORK_DIR}/dataset/inspire/prerisk_bench_feat')
CODE2DESC_PATH = _release_expand('${BENCHMARK_WORK_DIR}/mapping/icd_code2desc_inspire.json')
if not os.path.exists(FEAT_DIR):
    FEAT_DIR = os.path.join(os.path.dirname(PROJECT_ROOT), '1.1ml_training', 'dataset', 'inspire', 'prerisk_bench_feat')
    CODE2DESC_PATH = os.path.join(os.path.dirname(PROJECT_ROOT), '1.1ml_training', 'mapping', 'icd_code2desc_inspire.json')
OUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'mapping', 'ablation_icd_code')
STRING_DIR = os.path.join(OUT_DIR, 'string_mapping')
RETRIEVAL_DIR = os.path.join(OUT_DIR, 'retrieval_mapping')
os.makedirs(STRING_DIR, exist_ok=True)
os.makedirs(RETRIEVAL_DIR, exist_ok=True)
INDEX_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'snomed', 'snomed_optimized_index.json')
CONFIG_PATH = os.path.join(PROJECT_ROOT, 'config', 'field_mapping.json')
BASELINE_UNIQUE_VALUES = os.path.join(PROJECT_ROOT, 'data', 'raw', 'unique_values', 'inspire_unique_values.json')
FIELD_CM = 'data_process.diagnosis_icd10_cm_desc.desc_en'
FIELD_PCS = 'data_process.diagnosis_icd10_pcs_desc.desc_en'
CM_PAT = re.compile('\\(ICD-10-CM\\)\\s*([A-Z][A-Z0-9]+(?:\\.[0-9]+)?)')
PCS_PAT = re.compile('\\(ICD-10-PCS\\)\\s*([A-Z0-9][A-Z0-9]+)')

def load_pipeline_module(rel_path: str, name: str):
    """加载以数字开头的管线模块（无法直接 import）"""
    path = os.path.join(PROJECT_ROOT, 'src', 'snomed_mapping', rel_path)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def clean_desc(t: str) -> str:
    t = re.sub('^\\d+ (?:days?|years?|months?) ago (?:diagnosed as|underwent)\\s*', '', str(t))
    return t.strip()

def load_matched_pairs():
    """
    构建 (code, clean_desc) 匹配对，要求:
      - code2desc 表中 code 的英文释义(清洗后) 命中基线 desc_en 集合
      - 保证 desc 是干净的单疾病英文释义，code 与 desc 一一对应（同一概念两种表示）
    返回 (cm_pairs: dict[code->desc], pcs_pairs: dict[code->desc])
    """
    baseline = json.load(open(BASELINE_UNIQUE_VALUES, encoding='utf-8'))['unique_values']
    base_cm = set(baseline.get(FIELD_CM, []))
    base_pcs = set(baseline.get(FIELD_PCS, []))
    raw = json.load(open(CODE2DESC_PATH, encoding='utf-8'))
    cm_pairs, pcs_pairs = ({}, {})
    for c, d in raw.items():
        cd = clean_desc(str(d))
        if re.match('^[A-Z][0-9]', c) and cd in base_cm:
            cm_pairs[c] = cd
        elif re.match('^[0-9]', c) and cd in base_pcs:
            pcs_pairs[c] = cd
    return (cm_pairs, pcs_pairs)

def extract_raw_codes():
    """返回 (cm_codes: set, pcs_codes: set) —— 数据中出现的原始 ICD 编码"""
    surg = pd.read_csv(os.path.join(FEAT_DIR, 'feat_surg_vars.csv'), keep_default_na=False, na_values=[])
    his = pd.read_csv(os.path.join(FEAT_DIR, 'feat_past_his.csv'), keep_default_na=False, na_values=[])
    cm = set()
    pcs = set()
    for code in surg['preop_diagnosis'].astype(str):
        code = code.strip()
        if code and re.match('^[A-Z][0-9]', code):
            cm.add(code)
    for code in surg['scheduled_procedure_name'].astype(str):
        code = code.strip()
        if code and re.match('^[0-9A-Z]', code):
            pcs.add(code)
    for col in [c for c in his.columns if c.endswith('_detail')]:
        for v in his[col]:
            cm.update(CM_PAT.findall(v))
            pcs.update(PCS_PAT.findall(v))
    cm.discard('')
    pcs.discard('')
    return (cm, pcs)

def build_ablation_input(cm_codes, pcs_codes, out_path):
    """字段名与基线一致，取值换成原始 ICD 编码"""
    uv = {FIELD_CM: sorted(cm_codes), FIELD_PCS: sorted(pcs_codes)}
    data = {'dataset': 'INSPIRE', 'table_name': None, 'unique_values': uv, 'statistics': {'total_fields': len(uv), 'total_values': sum((len(v) for v in uv.values()))}}
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f'  [build] code: CM={len(uv[FIELD_CM])}, PCS={len(uv[FIELD_PCS])} -> {out_path}')

def run_string_mapper(input_json, output_csv):
    string_mod = load_pipeline_module('1.string_mapper.py', 'string_mapper')
    mapper = string_mod.StringSNOMEDMapper(INDEX_PATH, CONFIG_PATH)
    field_type_mapping = mapper.field_type_mapping
    with open(input_json, encoding='utf-8') as f:
        data = json.load(f)
    uv = data.get('unique_values', {})
    rows = []
    for field_name, values in uv.items():
        field_type = field_type_mapping.get(field_name)
        for val in values:
            if not val:
                continue
            m = mapper.map_term(val, field_type)
            rows.append({'field_name': field_name, 'field_value': val, 'conceptId': m['conceptId'], 'fsn': m['fsn'], 'source': m['source'], 'score': m['score']})
    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['field_name', 'field_value', 'conceptId', 'fsn', 'source', 'score'])
        writer.writeheader()
        writer.writerows(rows)
    n_exact = sum((1 for r in rows if r['source'] == 'exact'))
    print(f'  [string] {os.path.basename(output_csv)}: {len(rows)} rows, exact={n_exact} ({n_exact / len(rows) * 100:.1f}%)')

def run_vector_retrieval(string_csv, output_csv):
    retr_mod = load_pipeline_module('2.1vector_retrieval_mapper.py', 'vector_retrieval')
    mapper = retr_mod.VectorRetrievalMapper()
    df = pd.read_csv(string_csv, keep_default_na=False, na_values=[])
    fuzzy_mask = df['source'] == 'fuzzy'
    fuzzy_df = df[fuzzy_mask].copy()
    if fuzzy_df.empty:
        df.to_csv(output_csv, index=False)
        print(f'  [retrieval] {os.path.basename(output_csv)}: no fuzzy items')
        return
    fuzzy_items = fuzzy_df[['field_name', 'field_value']].drop_duplicates().values.tolist()
    value_to_field_names = defaultdict(list)
    for fname, val in fuzzy_items:
        value_to_field_names[val].append(fname)
    unique_fuzzy_values = fuzzy_df['field_value'].unique()

    def build_field_type(val):
        for fn in value_to_field_names.get(val, []):
            ft = mapper.field_mapping.get(fn, '')
            if ft:
                return ft
        return ''
    all_queries = []
    for val in unique_fuzzy_values:
        if pd.isna(val):
            continue
        field_type = build_field_type(val)
        raw_val = mapper._clean_cms_suffix(str(val))
        clean_val = mapper.clean_icd10_suffixes(str(val))
        raw_query = f'{field_type}:{raw_val}' if field_type else raw_val
        clean_query = f'{field_type}:{clean_val}' if field_type and clean_val.lower() != raw_val.lower() else clean_val if clean_val.lower() != raw_val.lower() else None
        all_queries.append((val, raw_query, 'raw'))
        if clean_query and clean_query.lower() != raw_query.lower():
            all_queries.append((val, clean_query, 'clean'))
    uncached = []
    uncached_idx = []
    for i, (val, qtext, qtype) in enumerate(all_queries):
        qlower = qtext.lower().strip()
        if qlower not in mapper.query_cache:
            uncached.append(qtext)
            uncached_idx.append(i)
    if uncached:
        vecs = mapper.model.encode(uncached, batch_size=128, convert_to_numpy=True)
        if len(vecs.shape) == 1:
            vecs = vecs.reshape(1, -1)
        vecs = vecs.astype(np.float32)
        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        for i, v in zip(uncached_idx, vecs):
            mapper.query_cache[all_queries[i][1].lower().strip()] = v.reshape(1, -1)
    stacked = []
    query_keys = []
    for val, qtext, qtype in all_queries:
        qlower = qtext.lower().strip()
        if qlower in mapper.query_cache:
            stacked.append(mapper.query_cache[qlower])
            query_keys.append((val, qtext, qtype))
    value_to_best = {}
    if stacked:
        S = np.vstack(stacked)
        top_k = 50
        distances, indices = mapper.faiss_index.search(S, top_k)
        cosine_similarities = 1.0 - distances / 2.0
        for qi, (val, qtext, qtype) in enumerate(query_keys):
            candidates = []
            for vi, cos_sim in zip(indices[qi], cosine_similarities[qi]):
                if vi == -1:
                    continue
                concept_id = mapper.mapping_data['concept_ids'][vi]
                fsn = mapper.mapping_data['concept_to_fsn'].get(concept_id, 'Unknown FSN')
                original_text = mapper.mapping_data['texts'][vi]
                candidates.append({'conceptId': str(concept_id), 'fsn': fsn, 'original_text': original_text, 'score': float(cos_sim)})
            if candidates:
                best = mapper.rerank_by_keywords(qtext, candidates)
                score = best['combined_score']
                existing = value_to_best.get(val)
                if existing is None or score > existing['score']:
                    value_to_best[val] = {'conceptId': str(best['conceptId']), 'fsn': best['fsn'], 'score': round(score, 4), 'source': qtype}
    out_rows = []
    for _, row in df.iterrows():
        if row['source'] == 'exact':
            out_rows.append({'field_name': row['field_name'], 'field_value': row['field_value'], 'conceptId': str(row['conceptId']), 'fsn': row['fsn'], 'source': 'exact', 'score': row['score']})
        elif row['source'] == 'fuzzy':
            mapping = value_to_best.get(row['field_value'])
            if mapping:
                out_rows.append({'field_name': row['field_name'], 'field_value': row['field_value'], 'conceptId': mapping['conceptId'], 'fsn': mapping['fsn'], 'source': 'retrieval', 'score': mapping['score']})
            else:
                out_rows.append({'field_name': row['field_name'], 'field_value': row['field_value'], 'conceptId': '', 'fsn': '', 'source': 'fuzzy', 'score': ''})
    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['field_name', 'field_value', 'conceptId', 'fsn', 'source', 'score'])
        writer.writeheader()
        writer.writerows(out_rows)
    n_retr = sum((1 for r in out_rows if r['source'] == 'retrieval'))
    n_unmapped = sum((1 for r in out_rows if r['source'] == 'fuzzy'))
    scores = pd.Series(pd.to_numeric([r['score'] for r in out_rows if r['source'] == 'retrieval'], errors='coerce')).dropna()
    p85 = (scores >= 0.85).mean() * 100 if len(scores) else 0
    print(f'  [retrieval] {os.path.basename(output_csv)}: retrieval={n_retr}, unmapped={n_unmapped}, avg={scores.mean():.4f} if any, >=0.85={p85:.1f}%')

def summarize(csv_path, field):
    df = pd.read_csv(csv_path, keep_default_na=False, na_values=[])
    df = df[df['field_name'] == field].copy()
    n = len(df)
    if n == 0:
        return None
    exact = (df['source'] == 'exact').sum()
    retr = (df['source'] == 'retrieval').sum()
    unmapped = (df['source'] == 'fuzzy').sum()
    df['score'] = pd.to_numeric(df['score'], errors='coerce')
    mapped = df[df['source'].isin(['exact', 'retrieval'])]
    scores = mapped['score'].dropna()
    res = {'n': n, 'exact': int(exact), 'exact_rate': exact / n * 100, 'retrieval': int(retr), 'retrieval_rate': retr / n * 100, 'unmapped': int(unmapped), 'unmapped_rate': unmapped / n * 100, 'mapped': int(len(mapped)), 'mapped_rate': len(mapped) / n * 100, 'score_avg': scores.mean() if len(scores) else 0, 'score_median': scores.median() if len(scores) else 0, 'p85': (scores >= 0.85).mean() * 100 if len(scores) else 0, 'p80': (scores >= 0.8).mean() * 100 if len(scores) else 0}
    return res

def main():
    t0 = time.time()
    print('=' * 70)
    print('ICD 编码消融实验: 原始编码 vs 英文释义 的 SNOMED 映射质量对比')
    print('=' * 70)
    print('\n[1] 从 feat 文件提取原始 ICD 编码...')
    cm_codes, pcs_codes = extract_raw_codes()
    print(f'  CM codes: {len(cm_codes)}, PCS codes: {len(pcs_codes)}')
    print('\n[2] 构建两种输入...')
    code_json = os.path.join(OUT_DIR, 'inspire_unique_values_code.json')
    build_ablation_input(cm_codes, pcs_codes, code_json)
    desc_json = BASELINE_UNIQUE_VALUES
    print(f'  [build] desc: 直接使用基线输入 {desc_json}')
    print('\n[3] string mapper (exact match)...')
    desc_string = os.path.join(STRING_DIR, 'inspire_desc_field_values_mapping.csv')
    code_string = os.path.join(STRING_DIR, 'inspire_code_field_values_mapping.csv')
    run_string_mapper(desc_json, desc_string)
    run_string_mapper(code_json, code_string)
    print('\n[4] vector retrieval (FAISS, fuzzy -> retrieval)...')
    desc_retr = os.path.join(RETRIEVAL_DIR, 'inspire_desc_field_values_mapping_keyword_rerank.csv')
    code_retr = os.path.join(RETRIEVAL_DIR, 'inspire_code_field_values_mapping_keyword_rerank.csv')
    run_vector_retrieval(desc_string, desc_retr)
    run_vector_retrieval(code_string, code_retr)
    print('\n[5] 生成对比报告...')
    rows = []
    for field, label in [(FIELD_CM, 'ICD-10-CM'), (FIELD_PCS, 'ICD-10-PCS')]:
        s_desc = summarize(desc_retr, field)
        s_code = summarize(code_retr, field)
        rows.append((label, s_desc, s_code))
    print('\n[6] 匹配子集对比（code↔desc 一一对应）...')
    cm_pairs, pcs_pairs = load_matched_pairs()
    print(f'  matched CM pairs: {len(cm_pairs)}, PCS pairs: {len(pcs_pairs)}')
    matched_json_desc = os.path.join(OUT_DIR, 'inspire_matched_unique_values_desc.json')
    matched_json_code = os.path.join(OUT_DIR, 'inspire_matched_unique_values_code.json')
    matched_uv_desc = {FIELD_CM: sorted(set(cm_pairs.values())), FIELD_PCS: sorted(set(pcs_pairs.values()))}
    matched_uv_code = {FIELD_CM: sorted(cm_pairs.keys()), FIELD_PCS: sorted(pcs_pairs.keys())}
    with open(matched_json_desc, 'w', encoding='utf-8') as f:
        json.dump({'unique_values': matched_uv_desc}, f, ensure_ascii=False, indent=2)
    with open(matched_json_code, 'w', encoding='utf-8') as f:
        json.dump({'unique_values': matched_uv_code}, f, ensure_ascii=False, indent=2)
    matched_desc_string = os.path.join(STRING_DIR, 'inspire_matched_desc_field_values_mapping.csv')
    matched_code_string = os.path.join(STRING_DIR, 'inspire_matched_code_field_values_mapping.csv')
    run_string_mapper(matched_json_desc, matched_desc_string)
    run_string_mapper(matched_json_code, matched_code_string)
    matched_desc_retr = os.path.join(RETRIEVAL_DIR, 'inspire_matched_desc_field_values_mapping_keyword_rerank.csv')
    matched_code_retr = os.path.join(RETRIEVAL_DIR, 'inspire_matched_code_field_values_mapping_keyword_rerank.csv')
    run_vector_retrieval(matched_desc_string, matched_desc_retr)
    run_vector_retrieval(matched_code_string, matched_code_retr)
    matched_rows = []
    for field, label in [(FIELD_CM, 'ICD-10-CM'), (FIELD_PCS, 'ICD-10-PCS')]:
        s_desc = summarize(matched_desc_retr, field)
        s_code = summarize(matched_code_retr, field)
        matched_rows.append((label, s_desc, s_code))
    report_path = os.path.join(OUT_DIR, 'icd_code_ablation_report.md')
    lines = []
    lines.append('# ICD 编码消融实验报告: 原始编码 vs 英文释义')
    lines.append('')
    lines.append(f"实验时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append('')
    lines.append('## 实验设计')
    lines.append('')
    lines.append('同样的 SNOMED 映射管线（string exact match + vector retrieval），仅改变 ICD 输入表示:')
    lines.append('- **desc (基线)**: 使用原始管线的 ICD 英文释义（desc_en）输入')
    lines.append('- **code (消融)**: 直接使用原始 ICD 编码（字段名不变，field_type_mapping 相同）')
    lines.append('')
    lines.append('## 指标定义')
    lines.append('')
    lines.append('- **exact**: 字符串精确匹配命中')
    lines.append('- **retrieval**: 向量检索得到映射（score 为融合相似度）')
    lines.append('- **unmapped**: 未能映射（空 conceptId）')
    lines.append('- **mapped**: exact + retrieval')
    lines.append('- **score >= 0.85 比例**: 高质量映射比例（当前管线阈值）')
    lines.append('')
    lines.append('## 对比结果')
    lines.append('')
    lines.append('| 字段 | 输入 | 总数 | exact (率) | retrieval (率) | unmapped (率) | mapped (率) | score均值 | score中位 | >=0.85 | >=0.8 |')
    lines.append('|------|------|------|-----------|----------------|---------------|-------------|-----------|-----------|--------|-------|')
    for label, s_desc, s_code in rows:
        lines.append(f"| {label} | desc | {s_desc['n']} | {s_desc['exact']} ({s_desc['exact_rate']:.1f}%) | {s_desc['retrieval']} ({s_desc['retrieval_rate']:.1f}%) | {s_desc['unmapped']} ({s_desc['unmapped_rate']:.1f}%) | {s_desc['mapped']} ({s_desc['mapped_rate']:.1f}%) | {s_desc['score_avg']:.4f} | {s_desc['score_median']:.4f} | {s_desc['p85']:.1f}% | {s_desc['p80']:.1f}% |")
        lines.append(f"| {label} | code | {s_code['n']} | {s_code['exact']} ({s_code['exact_rate']:.1f}%) | {s_code['retrieval']} ({s_code['retrieval_rate']:.1f}%) | {s_code['unmapped']} ({s_code['unmapped_rate']:.1f}%) | {s_code['mapped']} ({s_code['mapped_rate']:.1f}%) | {s_code['score_avg']:.4f} | {s_code['score_median']:.4f} | {s_code['p85']:.1f}% | {s_code['p80']:.1f}% |")
    lines.append('')
    lines.append('## 关键结论')
    lines.append('')
    for label, s_desc, s_code in rows:
        delta_mapped = s_code['mapped_rate'] - s_desc['mapped_rate']
        delta_p85 = s_code['p85'] - s_desc['p85']
        lines.append(f"- **{label}** (全量): 用英文释义 mapped={s_desc['mapped_rate']:.1f}% (>=0.85: {s_desc['p85']:.1f}%)，直接用原始编码 mapped={s_code['mapped_rate']:.1f}% (>=0.85: {s_code['p85']:.1f}%)，mapped 率下降 {delta_mapped:.1f} 个百分点，>=0.85 高质量比例下降 {delta_p85:.1f} 个百分点。")
    lines.append('')
    lines.append('## 匹配子集对比（同一概念两种表示，控制概念集一致）')
    lines.append('')
    lines.append(f'匹配方式: 对 code2desc 表中每个 ICD 编码，取其英文释义（清洗后）命中基线 desc_en 集合的对。同一概念分别用 desc 与 code 两种表示走同一管线，控制概念集完全一致。')
    lines.append('')
    lines.append('| 字段 | 输入 | 总数 | exact (率) | retrieval (率) | unmapped (率) | mapped (率) | score均值 | score中位 | >=0.85 | >=0.8 |')
    lines.append('|------|------|------|-----------|----------------|---------------|-------------|-----------|-----------|--------|-------|')
    for label, s_desc, s_code in matched_rows:
        lines.append(f"| {label} | desc | {s_desc['n']} | {s_desc['exact']} ({s_desc['exact_rate']:.1f}%) | {s_desc['retrieval']} ({s_desc['retrieval_rate']:.1f}%) | {s_desc['unmapped']} ({s_desc['unmapped_rate']:.1f}%) | {s_desc['mapped']} ({s_desc['mapped_rate']:.1f}%) | {s_desc['score_avg']:.4f} | {s_desc['score_median']:.4f} | {s_desc['p85']:.1f}% | {s_desc['p80']:.1f}% |")
        lines.append(f"| {label} | code | {s_code['n']} | {s_code['exact']} ({s_code['exact_rate']:.1f}%) | {s_code['retrieval']} ({s_code['retrieval_rate']:.1f}%) | {s_code['unmapped']} ({s_code['unmapped_rate']:.1f}%) | {s_code['mapped']} ({s_code['mapped_rate']:.1f}%) | {s_code['score_avg']:.4f} | {s_code['score_median']:.4f} | {s_code['p85']:.1f}% | {s_code['p80']:.1f}% |")
    lines.append('')
    lines.append('### 匹配子集结论')
    lines.append('')
    for label, s_desc, s_code in matched_rows:
        delta_p85 = s_code['p85'] - s_desc['p85']
        lines.append(f"- **{label}** (匹配子集): 同一概念，用英文释义 >=0.85: {s_desc['p85']:.1f}% (score均值 {s_desc['score_avg']:.4f})，直接用原始编码 >=0.85: {s_code['p85']:.1f}% (score均值 {s_code['score_avg']:.4f})，高质量比例下降 {delta_p85:.1f} 个百分点。")
    lines.append('')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'\n✓ 报告已保存: {report_path}')
    csv_out = os.path.join(OUT_DIR, 'icd_code_ablation_summary.csv')
    with open(csv_out, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['subset', 'field', 'variant', 'n', 'exact', 'exact_rate', 'retrieval', 'retrieval_rate', 'unmapped', 'unmapped_rate', 'mapped', 'mapped_rate', 'score_avg', 'score_median', 'p85', 'p80'])
        for label, s_desc, s_code in rows:
            for v, s in [('desc', s_desc), ('code', s_code)]:
                w.writerow(['full', label, v, s['n'], s['exact'], round(s['exact_rate'], 2), s['retrieval'], round(s['retrieval_rate'], 2), s['unmapped'], round(s['unmapped_rate'], 2), s['mapped'], round(s['mapped_rate'], 2), round(s['score_avg'], 4), round(s['score_median'], 4), round(s['p85'], 2), round(s['p80'], 2)])
        for label, s_desc, s_code in matched_rows:
            for v, s in [('desc', s_desc), ('code', s_code)]:
                w.writerow(['matched', label, v, s['n'], s['exact'], round(s['exact_rate'], 2), s['retrieval'], round(s['retrieval_rate'], 2), s['unmapped'], round(s['unmapped_rate'], 2), s['mapped'], round(s['mapped_rate'], 2), round(s['score_avg'], 4), round(s['score_median'], 4), round(s['p85'], 2), round(s['p80'], 2)])
    print(f'✓ CSV 已保存: {csv_out}')
    print(f'\n总耗时: {time.time() - t0:.1f}s')
    print('=' * 70)
if __name__ == '__main__':
    main()

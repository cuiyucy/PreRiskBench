import os as _release_os
from release_runtime import expand_path as _release_expand, expand_config as _release_expand_config
import csv
import os
import pandas as pd
STRING_MAPPING_DIR = _release_expand('${ALIGNMENT_WORK_DIR}/data/mapping/string_mapping')
RETRIEVAL_MAPPING_DIR = _release_expand('${ALIGNMENT_WORK_DIR}/data/mapping/retrieval_mapping')
OUTPUT_DIR = _release_expand('${ALIGNMENT_WORK_DIR}/data/mapping/llm_process')
MERGED_DIR = _release_expand('${ALIGNMENT_WORK_DIR}/data/mapping/merged_mapping')
SPECIAL_FILE = os.path.join(OUTPUT_DIR, 'special_fields.csv')
POSTPRO_FILE = os.path.join(OUTPUT_DIR, 'llm_postpro_fields.csv')
os.makedirs(MERGED_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def merge_string_and_retrieval_mappings(dataset):
    """合并字符串匹配和向量匹配的结果"""
    string_file = os.path.join(STRING_MAPPING_DIR, f'{dataset}_field_values_mapping.csv')
    retrieval_file = os.path.join(RETRIEVAL_MAPPING_DIR, f'{dataset}_field_values_mapping_keyword_rerank.csv')
    merged_file = os.path.join(MERGED_DIR, f'{dataset}_merged_mapping.csv')
    print(f'合并 {dataset} 数据集...')
    string_df = pd.read_csv(string_file, keep_default_na=False, na_values=[], dtype={'conceptId': str})
    string_df = string_df.fillna('')
    string_df['source'] = 'string'
    retrieval_df = pd.read_csv(retrieval_file, keep_default_na=False, na_values=[], dtype={'conceptId': str})
    retrieval_df = retrieval_df.fillna('')
    retrieval_df['source'] = 'retrieval'
    merged_df = pd.concat([string_df, retrieval_df], ignore_index=True)
    merged_df['score'] = pd.to_numeric(merged_df['score'], errors='coerce').fillna(0)
    merged_df = merged_df.sort_values(['field_name', 'field_value', 'score'], ascending=[True, True, False])
    merged_df = merged_df.drop_duplicates(['field_name', 'field_value'], keep='first')
    merged_df.to_csv(merged_file, index=False, encoding='utf-8')
    print(f'  ✓ 保存合并结果: {merged_file} (共 {len(merged_df)} 条记录)')
    return merged_df
SPECIAL_FIELDS = {'inspire': ['prerisk_bench.feat_anes_vars.surg_admission_chars', 'prerisk_bench.feat_anes_vars.scheduled_anesthesia_type', 'data_process.process_cohort_medications_preop_non_ans.route', 'data_process.process_cohort_department.full_name'], 'mover': ['data_process.process_cohort_med_adm_preop_non_ans.med_route_nm', 'prerisk_bench.feat_anes_vars.scheduled_anesthesia_type', 'prerisk_bench.feat_anes_vars.surg_admission_chars'], 'vitaldb': ['prerisk_bench.feat_anes_vars.surg_admission_chars', 'prerisk_bench.feat_anes_vars.scheduled_anesthesia_type', 'prerisk_bench.feat_surg_vars.surgical_department', 'prerisk_bench.feat_surg_vars.scheduled_operative_type', 'prerisk_bench.feat_surg_vars.scheduled_surgical_approach', 'prerisk_bench.feat_surg_vars.scheduled_surgical_position']}
all_merged_data = []
for dataset in SPECIAL_FIELDS.keys():
    merged_df = merge_string_and_retrieval_mappings(dataset)
    all_merged_data.append(merged_df)
print(f'\\n开始处理特殊字段和低分匹配...')
special_rows = []
postpro_rows = []
for dataset, fields in SPECIAL_FIELDS.items():
    merged_file = os.path.join(MERGED_DIR, f'{dataset}_merged_mapping.csv')
    with open(merged_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            is_special = row['field_name'] in fields
            score_str = row.get('score', '').strip()
            try:
                score = float(score_str)
            except (ValueError, TypeError):
                score = 0.0
            if is_special:
                special_rows.append({'dataset': dataset, 'field_name': row['field_name'], 'field_value': row['field_value']})
            elif score < 0.85:
                postpro_rows.append({'dataset': dataset, 'field_name': row['field_name'], 'field_value': row['field_value']})
os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(SPECIAL_FILE, 'w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['dataset', 'field_name', 'field_value'])
    writer.writeheader()
    writer.writerows(special_rows)
with open(POSTPRO_FILE, 'w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['dataset', 'field_name', 'field_value'])
    writer.writeheader()
    writer.writerows(postpro_rows)
print(f'done:')
print(f'  特殊字段: {len(special_rows)} rows -> {SPECIAL_FILE}')
print(f'  低分(<0.85,非特殊字段): {len(postpro_rows)} rows -> {POSTPRO_FILE}')
for dataset in SPECIAL_FIELDS:
    s_count = sum((1 for r in special_rows if r['dataset'] == dataset))
    p_count = sum((1 for r in postpro_rows if r['dataset'] == dataset))
    print(f'  {dataset}: special={s_count}, postpro={p_count}')

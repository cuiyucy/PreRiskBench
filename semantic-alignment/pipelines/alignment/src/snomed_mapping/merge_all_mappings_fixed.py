import os as _release_os
from release_runtime import expand_path as _release_expand, expand_config as _release_expand_config
import os
import pandas as pd
import glob
from typing import List, Dict, Set

def clean_field_value(value):
    """清理字段值：去除开头和末尾的空格以及冒号"""
    if pd.isna(value) or value is None:
        return value
    cleaned = str(value).strip()
    cleaned = cleaned.strip(':')
    cleaned = cleaned.strip()
    return cleaned

def read_csv_safe(file_path: str) -> pd.DataFrame:
    """安全读取CSV文件，处理可能的编码问题"""
    try:
        return pd.read_csv(file_path, encoding='utf-8', dtype={'conceptId': str})
    except UnicodeDecodeError:
        try:
            return pd.read_csv(file_path, encoding='latin-1', dtype={'conceptId': str})
        except Exception as e:
            print(f'  读取失败: {e}')
            return pd.DataFrame()

def merge_all_mappings():
    """合并所有映射数据源，按数据集分组并正确去重"""
    output_dir = _release_expand('${ALIGNMENT_WORK_DIR}/data/mapping/final_mapping')
    os.makedirs(output_dir, exist_ok=True)
    print('=== 开始合并所有映射数据源（修复版） ===\n')
    print('1. 读取llm_mapping.final.csv（最高优先级）')
    llm_file = _release_expand('${ALIGNMENT_WORK_DIR}/data/mapping/llm_process/llm_mapping.final.csv')
    llm_df = read_csv_safe(llm_file)
    if llm_df.empty:
        print('  ❌ llm_mapping.final.csv 读取失败')
        return
    print(f'  读取成功: {len(llm_df)} 行')
    print('\n2. 读取retrieval_mapping中的keyword_rerank文件')
    retrieval_files = glob.glob(_release_expand('${ALIGNMENT_WORK_DIR}/data/mapping/retrieval_mapping/*_keyword_rerank.csv'))
    retrieval_dfs = []
    for file_path in retrieval_files:
        dataset_name = os.path.basename(file_path).split('_')[0]
        print(f'  读取: {os.path.basename(file_path)} (数据集: {dataset_name})')
        df = read_csv_safe(file_path)
        if not df.empty:
            if 'dataset' not in df.columns:
                df['dataset'] = dataset_name
            retrieval_dfs.append(df)
            print(f'    成功: {len(df)} 行')
    print('\n3. 读取string_mapping中的文件')
    string_files = glob.glob(_release_expand('${ALIGNMENT_WORK_DIR}/data/mapping/string_mapping/*_mapping.csv'))
    string_dfs = []
    for file_path in string_files:
        dataset_name = os.path.basename(file_path).split('_')[0]
        print(f'  读取: {os.path.basename(file_path)} (数据集: {dataset_name})')
        df = read_csv_safe(file_path)
        if not df.empty:
            if 'dataset' not in df.columns:
                df['dataset'] = dataset_name
            string_dfs.append(df)
            print(f'    成功: {len(df)} 行')
    print('\n4. 按数据集分组处理（正确优先级）')
    all_datasets = set()
    all_datasets.update(llm_df['dataset'].unique())
    for df in retrieval_dfs:
        all_datasets.update(df['dataset'].unique())
    for df in string_dfs:
        all_datasets.update(df['dataset'].unique())
    print(f'  发现的数据集: {list(all_datasets)}')
    total_final_rows = 0
    for dataset in all_datasets:
        print(f'\n  处理数据集: {dataset}')
        final_entries = {}
        llm_dataset_df = llm_df[llm_df['dataset'] == dataset].copy()
        if not llm_dataset_df.empty:
            print(f'    llm数据: {len(llm_dataset_df)} 行')
            for _, row in llm_dataset_df.iterrows():
                field_value_cleaned = clean_field_value(row['field_value'])
                key = f"{row['dataset']}|{row['field_name']}|{field_value_cleaned}"
                if key not in final_entries:
                    final_entries[key] = row
        retrieval_dataset_df = pd.concat([df for df in retrieval_dfs if not df.empty and 'dataset' in df.columns and df['dataset'].eq(dataset).any()])
        if not retrieval_dataset_df.empty:
            print(f'    retrieval数据: {len(retrieval_dataset_df)} 行')
            for _, row in retrieval_dataset_df.iterrows():
                field_value_cleaned = clean_field_value(row['field_value'])
                key = f"{row['dataset']}|{row['field_name']}|{field_value_cleaned}"
                if key not in final_entries:
                    final_entries[key] = row
        string_dataset_df = pd.concat([df for df in string_dfs if not df.empty and 'dataset' in df.columns and df['dataset'].eq(dataset).any()])
        if not string_dataset_df.empty:
            print(f'    string数据: {len(string_dataset_df)} 行')
            for _, row in string_dataset_df.iterrows():
                field_value_cleaned = clean_field_value(row['field_value'])
                key = f"{row['dataset']}|{row['field_name']}|{field_value_cleaned}"
                if key not in final_entries:
                    final_entries[key] = row
        final_df = pd.DataFrame(list(final_entries.values()))
        before_filter = len(final_df)
        final_df = final_df[final_df['conceptId'].notna() & (final_df['conceptId'] != '')]
        after_filter = len(final_df)
        print(f'    去重后: {before_filter} 行')
        print(f'    过滤空值后: {after_filter} 行')
        print(f'    移除空值: {before_filter - after_filter} 行')
        llm_count = len(final_df[final_df['source'] == 'llm'])
        retrieval_count = len(final_df[final_df['source'] == 'retrieval'])
        string_count = len(final_df[final_df['source'] == 'fuzzy'])
        print(f'    保留来源 - llm: {llm_count}, retrieval: {retrieval_count}, string: {string_count}')
        output_file = os.path.join(output_dir, f'{dataset}_mapping.csv')
        final_df.to_csv(output_file, index=False, encoding='utf-8')
        print(f'    保存到: {output_file}')
        total_final_rows += len(final_df)
    print('\n=== 合并完成 ===')
    print(f'总数据集数量: {len(all_datasets)}')
    print(f'最终总行数: {total_final_rows}')
    print(f'输出目录: {output_dir}')
    print('\n=== 验证结果 ===')
    for dataset in all_datasets:
        result_file = os.path.join(output_dir, f'{dataset}_mapping.csv')
        if os.path.exists(result_file):
            result_df = pd.read_csv(result_file)
            field_values_cleaned = result_df['field_value'].apply(clean_field_value)
            temp_df = result_df.copy()
            temp_df['field_value_temp'] = field_values_cleaned
            duplicates = temp_df.duplicated(subset=['dataset', 'field_name', 'field_value_temp']).sum()
            empty_conceptid = result_df['conceptId'].isna().sum()
            print(f'  {dataset}: {len(result_df)} 行, 重复: {duplicates}, 空conceptId: {empty_conceptid}')
        else:
            print(f'  {dataset}: 文件不存在')

def main():
    merge_all_mappings()
if __name__ == '__main__':
    main()

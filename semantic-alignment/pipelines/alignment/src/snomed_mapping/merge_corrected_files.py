import os as _release_os
from release_runtime import expand_path as _release_expand, expand_config as _release_expand_config
import os
import pandas as pd
import glob
from typing import List, Dict

def extract_version(filename: str) -> int:
    """从文件名中提取版本号"""
    import re
    match = re.search('mappingv(\\d+)_corrected\\.csv', filename)
    if match:
        return int(match.group(1))
    return 0

def merge_corrected_files(input_dir: str, output_file: str):
    """合并所有corrected文件，按版本优先级去重"""
    pattern = os.path.join(input_dir, 'mappingv*_corrected.csv')
    corrected_files = glob.glob(pattern)
    print(f'找到 {len(corrected_files)} 个corrected文件:')
    for file in corrected_files:
        version = extract_version(os.path.basename(file))
        print(f'  - {os.path.basename(file)} (版本: v{version})')
    corrected_files.sort(key=lambda x: extract_version(os.path.basename(x)), reverse=True)
    print('\n按版本优先级排序:')
    for file in corrected_files:
        version = extract_version(os.path.basename(file))
        print(f'  - {os.path.basename(file)} (版本: v{version})')
    all_data = []
    for file_path in corrected_files:
        version = extract_version(os.path.basename(file_path))
        print(f'\n正在读取: {os.path.basename(file_path)} (v{version})')
        try:
            df = pd.read_csv(file_path)
            df['version'] = version
            all_data.append(df)
            print(f'  读取成功: {len(df)} 行')
        except Exception as e:
            print(f'  读取失败: {e}')
    if not all_data:
        print('没有找到可合并的数据')
        return
    merged_df = pd.concat(all_data, ignore_index=True)
    print(f'\n合并后总行数: {len(merged_df)} 行')
    print('\n开始去重处理...')
    merged_df = merged_df.sort_values('version', ascending=False)
    deduplicated_df = merged_df.drop_duplicates(subset=['dataset', 'field_name', 'field_value'], keep='first')
    print(f'去重后行数: {len(deduplicated_df)} 行')
    print(f'去重移除行数: {len(merged_df) - len(deduplicated_df)} 行')
    print('\n各版本保留数量:')
    version_counts = deduplicated_df['version'].value_counts().sort_index()
    for version, count in version_counts.items():
        print(f'  v{version}: {count} 行')
    print(f'\n保存结果到: {output_file}')
    result_df = deduplicated_df.drop('version', axis=1)
    result_df.to_csv(output_file, index=False)
    print('合并完成!')

def main():
    input_dir = _release_expand('${ALIGNMENT_WORK_DIR}/data/mapping/llm_process')
    output_file = _release_expand('${ALIGNMENT_WORK_DIR}/data/mapping/llm_process/merged_corrected_final.csv')
    print('=== 开始合并corrected文件 ===')
    merge_corrected_files(input_dir, output_file)
    print('\n=== 验证合并结果 ===')
    try:
        result_df = pd.read_csv(output_file)
        print(f'最终文件行数: {len(result_df)} 行')
        print('字段列表:', list(result_df.columns))
        duplicates = result_df.duplicated(subset=['dataset', 'field_name', 'field_value']).sum()
        print(f'重复条目数量: {duplicates}')
        if duplicates == 0:
            print('✓ 去重验证通过')
        else:
            print('⚠ 存在重复条目')
    except Exception as e:
        print(f'验证失败: {e}')
if __name__ == '__main__':
    main()

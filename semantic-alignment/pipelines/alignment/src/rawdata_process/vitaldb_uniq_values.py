import os as _release_os
from release_runtime import expand_path as _release_expand, expand_config as _release_expand_config
import json
import re
from typing import Dict, List, Set, Tuple
import os

def convert_field_name_to_english(field_name: str) -> str:
    """
    将下划线格式的字段名转换为英文文本描述

    参数:
        field_name: 原始字段名

    返回:
        转换后的英文描述
    """
    if field_name.endswith('_flag'):
        return field_name
    field_mapping = {'_val': '', '_desc': '', '_detail': '', '_ens': '', '_cn': '', 'hx_': 'history_', 'preop_': 'preoperative_', 'surg_': 'surgical_', 'med_': 'medication_', 'rec_': 'record_', 'anes_': 'anesthesia_', 'pat_': 'patient_', 'demo_': 'demographic_', 'test_': 'test_', 'circ_': 'circulatory_', 'endo_': 'endocrine_', 'neuro_': 'neurological_', 'digest_': 'digestive_', 'immune_': 'immune_', 'urinary_': 'urinary_', 'admission_': 'admission_', 'department_': 'department_', 'diagnosis_': 'diagnosis_', 'operative_': 'operative_', 'procedure_': 'procedure_', 'approach_': 'approach_', 'position_': 'position_', 'summary': 'summary', 'details': 'details', 'chars': 'characteristics', 'type': 'type', 'name': 'name', 'time': 'time', 'route': 'route'}
    converted_name = field_name
    for pattern, replacement in field_mapping.items():
        if pattern in converted_name:
            converted_name = converted_name.replace(pattern, replacement)
    converted_name = re.sub('_+', ' ', converted_name).strip()
    return converted_name.title() if converted_name else field_name

def extract_field_names_from_data(data: Dict) -> List[str]:
    """
    从数据中提取字段名称，过滤掉以_flag结尾的字段

    参数:
        data: 数据字典

    返回:
        字段名称列表
    """
    unique_values = data.get('unique_values', {})
    field_names = list(unique_values.keys())
    filtered_field_names = [field for field in field_names if not field.endswith('_flag')]
    return filtered_field_names

def process_anes_vars_file(input_file: str, output_file: str) -> Tuple[Dict, List[str]]:
    """
    处理anes_vars文件，移除冗余的anes_vars_summary字段

    参数:
        input_file: 输入的JSON文件路径
        output_file: 输出的JSON文件路径

    返回:
        (输出数据, 字段名称列表)
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    original_unique_values = data.get('unique_values', {})
    filtered_unique_values = {}
    for field, values in original_unique_values.items():
        if field != 'anes_vars_summary':
            filtered_unique_values[field] = values
    output_data = {'dataset': data.get('dataset', ''), 'table_name': data.get('table_name', ''), 'unique_values': filtered_unique_values, 'statistics': {'total_fields': len(filtered_unique_values), 'total_values': sum((len(values) for values in filtered_unique_values.values()))}}
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f'已处理: {input_file}')
    print(f'生成: {output_file}')
    print(f'统计信息: 保留{len(filtered_unique_values)}个字段')
    field_names = extract_field_names_from_data(output_data)
    return (output_data, field_names)

def generate_field_summary_table(all_field_data: List[Dict]) -> Dict:
    """
    生成字段汇总表

    参数:
        all_field_data: 所有表的字段数据列表

    返回:
        字段汇总表数据
    """
    field_summary = {'dataset': 'vitaldb', 'tables': []}
    all_field_names = set()
    for table_data in all_field_data:
        if table_data:
            unique_values = table_data.get('unique_values', {})
            original_field_names = list(unique_values.keys())
            table_entry = {'table_name': table_data.get('table_name', ''), 'field_names': original_field_names}
            field_summary['tables'].append(table_entry)
            all_field_names.update(original_field_names)
    field_summary['all_field_names'] = sorted(list(all_field_names))
    field_summary['total_fields'] = len(all_field_names)
    return field_summary

def process_past_his_file(input_file: str, output_file: str) -> Tuple[Dict, List[str]]:
    """
    处理VitalDB数据库的past_his文件
    VitalDB的past_his字段结构与INSPIRE/MOVER不同，只有两种类型的病史

    参数:
        input_file: 输入的JSON文件路径
        output_file: 输出的JSON文件路径

    返回:
        (输出数据, 字段名称列表)
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    target_fields = ['hx_circ_details', 'hx_endo_details', 'past_his_summary']
    parsed_results = {}
    field_statistics = {}
    for field in target_fields:
        field_values = data.get('unique_values', {}).get(field, [])
        if not field_values:
            print(f'警告: {input_file} 中没有找到 {field} 字段')
            continue
        unique_values = sorted(list(set(field_values)))
        parsed_results[field] = unique_values
        field_statistics[field] = len(unique_values)
    output_data = {'dataset': data.get('dataset', ''), 'table_name': data.get('table_name', ''), 'unique_values': parsed_results, 'statistics': field_statistics, 'total_unique_values': sum(field_statistics.values())}
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f'已处理: {input_file}')
    print(f'生成: {output_file}')
    print(f'统计信息: 总唯一值数量={sum(field_statistics.values())}')
    for field, count in field_statistics.items():
        print(f'  {field}: {count}个值')
    field_names = extract_field_names_from_data(output_data)
    return (output_data, field_names)

def process_surg_vars_file(input_file: str, output_file: str) -> Tuple[Dict, List[str]]:
    """
    处理VitalDB数据库的surg_vars文件，保留指定字段的唯一值

    参数:
        input_file: 输入的JSON文件路径
        output_file: 输出的JSON文件路径

    返回:
        (输出数据, 字段名称列表)
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    target_fields = ['surg_admission_chars', 'surgical_department', 'preop_diagnosis', 'scheduled_operative_type', 'scheduled_procedure_name', 'scheduled_surgical_approach', 'scheduled_surgical_position']
    parsed_results = {}
    field_statistics = {}
    for field in target_fields:
        field_values = data.get('unique_values', {}).get(field, [])
        if not field_values:
            print(f'警告: {input_file} 中没有找到 {field} 字段')
            continue
        unique_values = sorted(list(set(field_values)))
        parsed_results[field] = unique_values
        field_statistics[field] = len(unique_values)
    output_data = {'dataset': data.get('dataset', ''), 'table_name': data.get('table_name', ''), 'unique_values': parsed_results, 'statistics': field_statistics, 'total_unique_values': sum(field_statistics.values())}
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f'已处理: {input_file}')
    print(f'生成: {output_file}')
    print(f'统计信息: 总唯一值数量={sum(field_statistics.values())}')
    for field, count in field_statistics.items():
        print(f'  {field}: {count}个值')
    field_names = extract_field_names_from_data(output_data)
    return (output_data, field_names)

def process_test_vars_file(input_file: str, output_file: str) -> Tuple[Dict, List[str]]:
    """
    处理VitalDB数据库的test_vars文件，提取以_val结尾的字段名称

    参数:
        input_file: 输入的JSON文件路径
        output_file: 输出的JSON文件路径

    返回:
        (输出数据, 字段名称列表)
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    all_fields = list(data.get('unique_values', {}).keys())
    test_name_fields = [field for field in all_fields if field.endswith('_val')]
    output_data = {'dataset': data.get('dataset', ''), 'table_name': data.get('table_name', ''), 'unique_values': {'test_name': sorted(test_name_fields)}, 'statistics': {'total_test_fields': len(test_name_fields), 'total_all_fields': len(all_fields)}}
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f'已处理: {input_file}')
    print(f'生成: {output_file}')
    print(f'统计信息: 检测字段数量={len(test_name_fields)} (总字段数={len(all_fields)})')
    print(f"检测字段列表: {', '.join(test_name_fields[:10])}{('...' if len(test_name_fields) > 10 else '')}")
    field_names = extract_field_names_from_data(output_data)
    return (output_data, field_names)

def process_pat_demo_file(input_file: str, output_file: str) -> Tuple[Dict, List[str]]:
    """
    处理VitalDB数据库的pat_demo文件，提取所有字段名称

    参数:
        input_file: 输入的JSON文件路径
        output_file: 输出的JSON文件路径

    返回:
        (输出数据, 字段名称列表)
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    all_fields = list(data.get('unique_values', {}).keys())
    output_data = {'dataset': data.get('dataset', ''), 'table_name': data.get('table_name', ''), 'unique_values': {'demo': sorted(all_fields)}, 'statistics': {'total_demo_fields': len(all_fields)}}
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f'已处理: {input_file}')
    print(f'生成: {output_file}')
    print(f'统计信息: 人口统计字段数量={len(all_fields)}')
    print(f"字段列表: {', '.join(all_fields[:10])}{('...' if len(all_fields) > 10 else '')}")
    field_names = extract_field_names_from_data(output_data)
    return (output_data, field_names)

def main():
    """主函数"""
    print('开始处理VitalDB数据库的特殊字段...')
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    input_dir = os.path.join(project_root, 'data', 'raw', 'unique_values')
    output_dir = os.path.join(project_root, 'data', 'processed', 'vitaldb')
    os.makedirs(output_dir, exist_ok=True)
    all_table_data = []
    print('\n按照新要求处理各表...')
    print('\n1. 处理vitaldb_feat_anes_vars...')
    anes_vars_input = os.path.join(input_dir, 'vitaldb_feat_anes_vars.json')
    anes_vars_output = os.path.join(output_dir, 'vitaldb_feat_anes_vars_parsed.json')
    if os.path.exists(anes_vars_input):
        output_data, field_names = process_anes_vars_file(anes_vars_input, anes_vars_output)
        if output_data:
            all_table_data.append(output_data)
    else:
        print(f'错误: 输入文件 {anes_vars_input} 不存在')
    print('\n2. 处理vitaldb_feat_past_his...')
    past_his_input = os.path.join(input_dir, 'vitaldb_feat_past_his.json')
    past_his_output = os.path.join(output_dir, 'vitaldb_feat_past_his_parsed.json')
    if os.path.exists(past_his_input):
        output_data, field_names = process_past_his_file(past_his_input, past_his_output)
        if output_data:
            all_table_data.append(output_data)
    else:
        print(f'错误: 输入文件 {past_his_input} 不存在')
    print('\n3. vitaldb_feat_pat_demo：跳过处理')
    print('\n4. 处理vitaldb_feat_surg_vars...')
    surg_vars_input = os.path.join(input_dir, 'vitaldb_feat_surg_vars.json')
    surg_vars_output = os.path.join(output_dir, 'vitaldb_feat_surg_vars_parsed.json')
    if os.path.exists(surg_vars_input):
        output_data, field_names = process_surg_vars_file(surg_vars_input, surg_vars_output)
        if output_data:
            all_table_data.append(output_data)
    else:
        print(f'错误: 输入文件 {surg_vars_input} 不存在')
    print('\n5. vitaldb_feat_test_vars：跳过处理')
    print('\n6. 生成字段汇总表...')
    field_summary = generate_field_summary_table(all_table_data)
    field_summary_output = os.path.join(output_dir, 'vitaldb_field_summary.json')
    with open(field_summary_output, 'w', encoding='utf-8') as f:
        json.dump(field_summary, f, ensure_ascii=False, indent=2)
    print(f'字段汇总表已保存到: {field_summary_output}')
    print(f"汇总了 {len(field_summary['tables'])} 个表的字段信息")
    for table in field_summary['tables']:
        print(f"  - {table['table_name']}: {len(table['field_names'])} 个字段")
    print('\nVitalDB数据库处理完成!')
if __name__ == '__main__':
    main()

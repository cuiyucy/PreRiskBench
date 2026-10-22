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
        field_name: 下划线格式的字段名，如"hemoglobin_val"

    返回:
        英文文本描述，如"hemoglobin"
    """
    if field_name.endswith('_flag'):
        return field_name
    field_mapping = {'_val': '', '_desc': '', '_detail': '', '_ens': '', '_cn': '', 'hx_': 'history_', 'preop_': 'preoperative_', 'surg_': 'surgical_', 'med_': 'medication_', 'rec_': 'record_', 'anes_': 'anesthesia_', 'test_': 'test_', 'demo_': 'demographic_', 'pat_': 'patient_', 'feat_': 'feature_', 'vars_': 'variables_', 'resp': 'respiratory', 'endo': 'endocrine', 'neuro': 'neurological', 'circ': 'circulatory', 'dige': 'digestive', 'immu': 'immune', 'urine': 'urinary', 'admission': 'admission', 'diagnosis': 'diagnosis', 'procedure': 'procedure', 'department': 'department', 'chars': 'characteristics', 'summary': 'summary', 'name': 'name', 'time': 'time', 'route': 'route'}
    converted_name = field_name
    for pattern, replacement in field_mapping.items():
        if pattern in converted_name:
            converted_name = converted_name.replace(pattern, replacement)
    converted_name = re.sub('_+', ' ', converted_name)
    converted_name = converted_name.strip()
    if not converted_name:
        return field_name
    converted_name = converted_name.title()
    return converted_name

def extract_field_names_from_data(data: Dict) -> List[str]:
    """
    从数据中提取所有字段名称（原始字段名）

    参数:
        data: 包含unique_values的字典

    返回:
        原始字段名称列表
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
    print(f'统计信息: 移除冗余字段anes_vars_summary，保留{len(filtered_unique_values)}个字段')
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
    field_summary = {'dataset': 'mover', 'tables': []}
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

def parse_med_rec_summary(med_rec_text: str) -> Dict[str, List[str]]:
    """
    解析MOVER数据库的med_rec_summary字段，提取三个字段的唯一值

    参数:
        med_rec_text: med_rec_summary字段的文本内容

    返回:
        包含三个字段唯一值的字典:
        {
            "用药时间": ["Long-term Use", "Recent Use (<1wk)", "Past History"],
            "用药名称": ["CEFAZOLIN SODIUM-DEXTROSE", "PLASMA-LYTE A", ...],
            "用药方式": ["IntraVENOUS", "Each Naris", ...]
        }
    """
    medication_times = set()
    medication_names = set()
    medication_routes = set()
    time_patterns = ['Long-term Use:', 'Recent Use \\(<1wk\\):', 'Past History:']
    for pattern in time_patterns:
        if re.search(pattern, med_rec_text):
            time_name = pattern.replace('\\\\', '').replace(':', '').replace('(', '').replace(')', '')
            medication_times.add(time_name)
    medication_pattern = '([A-Z][A-Z\\s\\d/\\-\\.]+?)\\s+\\d+[^()]*?\\(([^()]+?)\\)(?:[,\\.]|$)'
    matches = re.findall(medication_pattern, med_rec_text)
    for medication_full, medication_route in matches:
        medication_name = extract_medication_name(medication_full.strip())
        if medication_name:
            medication_names.add(medication_name)
        cleaned_route = medication_route.strip()
        if cleaned_route:
            medication_routes.add(cleaned_route)
    if not medication_names:
        route_pattern = '\\(([^()]+?)\\)(?:[,\\.]|$)'
        route_matches = re.findall(route_pattern, med_rec_text)
        for route in route_matches:
            cleaned_route = route.strip()
            if cleaned_route:
                medication_routes.add(cleaned_route)
        name_pattern = '(?:Long-term Use:|Recent Use \\(<1wk\\):|Past History:)\\s*([^(),]+?)(?:\\(|,|$)'
        name_matches = re.findall(name_pattern, med_rec_text)
        for name_match in name_matches:
            medication_name = extract_medication_name(name_match.strip())
            if medication_name:
                medication_names.add(medication_name)
    return {'medication_time': sorted(list(medication_times)), 'medication_name': sorted(list(medication_names)), 'medication_route': sorted(list(medication_routes))}

def process_med_rec_file(input_file: str, output_file: str) -> Tuple[Dict, List[str]]:
    """
    处理MOVER数据库的med_rec_summary文件，提取三个字段的唯一值

    参数:
        input_file: 输入的JSON文件路径
        output_file: 输出的JSON文件路径

    返回:
        (输出数据, 字段名称列表)
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    med_rec_summaries = data.get('unique_values', {}).get('med_rec_summary', [])
    if not med_rec_summaries:
        print(f'警告: {input_file} 中没有找到med_rec_summary字段')
        return
    all_medication_times = set()
    all_medication_names = set()
    all_medication_routes = set()
    for summary in med_rec_summaries:
        if summary:
            parsed_data = parse_med_rec_summary(summary)
            all_medication_times.update(parsed_data['medication_time'])
            all_medication_names.update(parsed_data['medication_name'])
            all_medication_routes.update(parsed_data['medication_route'])
    output_data = {'dataset': data.get('dataset', ''), 'table_name': data.get('table_name', ''), 'unique_values': {'medication_time': sorted(list(all_medication_times)), 'medication_name': sorted(list(all_medication_names)), 'medication_route': sorted(list(all_medication_routes))}, 'statistics': {'medication_time_count': len(all_medication_times), 'medication_name_count': len(all_medication_names), 'medication_route_count': len(all_medication_routes)}}
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f'已处理: {input_file}')
    print(f'生成: {output_file}')
    print(f'统计信息: medication_time={len(all_medication_times)}, medication_name={len(all_medication_names)}, medication_route={len(all_medication_routes)}')
    field_names = extract_field_names_from_data(output_data)
    return (output_data, field_names)

def extract_icd_codes(text: str) -> List[str]:
    """
    从文本中提取ICD编码

    参数:
        text: 包含ICD编码的文本

    返回:
        ICD编码列表，格式为"(前缀) 编码"，如"(ICD-10-CM) C32"
    """
    icd_codes = set()
    icd_pattern = '\\((ICD-[^)]+)\\)\\s*([A-Z0-9][A-Z0-9]+(?:\\s*,\\s*[A-Z0-9][A-Z0-9]+)*)'
    matches = re.findall(icd_pattern, text)
    for prefix, codes_str in matches:
        codes = re.split('\\s*,\\s*', codes_str)
        for code in codes:
            if code.strip():
                full_code = f'({prefix}) {code.strip()}'
                icd_codes.add(full_code)
    return sorted(list(icd_codes))

def extract_medication_name(medication_text: str) -> str:
    """
    从药物规格文本中提取药物名称
    格式示例: "CEFAZOLIN SODIUM-DEXTROSE 1-4 GM/50ML-% IV SOLN"
    提取结果: "CEFAZOLIN SODIUM-DEXTROSE"

    参数:
        medication_text: 药物规格文本

    返回:
        提取的药物名称
    """
    if not medication_text:
        return ''
    dose_patterns = ['\\d+', 'MG', 'MCG', 'ML', 'MEQ', 'UNIT', '%', 'GM', 'G', 'MG/ML', 'MCG/ML', 'MEQ/ML', 'UNIT/GM', 'UNIT/ML', 'MG/GM']
    first_dose_index = len(medication_text)
    for pattern in dose_patterns:
        match = re.search(pattern, medication_text)
        if match and match.start() < first_dose_index:
            first_dose_index = match.start()
    if first_dose_index < len(medication_text):
        medication_name = medication_text[:first_dose_index].strip()
        medication_name = re.sub('[\\s\\-]+$', '', medication_name)
        return medication_name
    return medication_text

def extract_surgery_name(surgery_text: str) -> str:
    """
    从hx_surgery_detail字段中提取手术名称
    格式示例: "0 days ago underwent ANGIOGRAM, VESSEL, CEREBRAL, WITH EMBOLIZATION"
    提取结果: "ANGIOGRAM, VESSEL, CEREBRAL, WITH EMBOLIZATION"

    参数:
        surgery_text: 手术描述文本

    返回:
        提取的手术名称
    """
    if not surgery_text:
        return ''
    underwent_index = surgery_text.find('underwent')
    if underwent_index != -1:
        surgery_name = surgery_text[underwent_index + len('underwent'):].strip()
        return surgery_name
    words = surgery_text.split()
    for i, word in enumerate(words):
        if word and word[0].isupper():
            return ' '.join(words[i:])
    return surgery_text

def process_past_his_file(input_file: str, output_file: str) -> Tuple[Dict, List[str]]:
    """
    处理MOVER数据库的past_his文件，处理分号分隔的病史和手术名称提取，移除past_his_summary

    参数:
        input_file: 输入的JSON文件路径
        output_file: 输出的JSON文件路径

    返回:
        (输出数据, 字段名称列表)
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    target_fields = ['hx_resp_detail', 'hx_endo_detail', 'hx_neuro_detail', 'hx_circ_detail', 'hx_dige_detail', 'hx_spine_detail', 'hx_immu_detail', 'hx_urine_detail', 'hx_other_detail', 'hx_surgery_detail']
    parsed_results = {}
    field_statistics = {}
    for field in target_fields:
        field_values = data.get('unique_values', {}).get(field, [])
        if not field_values:
            print(f'警告: {input_file} 中没有找到 {field} 字段')
            continue
        all_unique_items = set()
        for value in field_values:
            if value:
                items = [item.strip() for item in value.split(';') if item.strip()]
                if field == 'hx_surgery_detail':
                    processed_items = []
                    for item in items:
                        surgery_name = extract_surgery_name(item)
                        if surgery_name:
                            processed_items.append(surgery_name)
                    all_unique_items.update(processed_items)
                else:
                    all_unique_items.update(items)
        parsed_results[field] = sorted(list(all_unique_items))
        field_statistics[field] = len(all_unique_items)
    output_data = {'dataset': data.get('dataset', ''), 'table_name': data.get('table_name', ''), 'unique_values': parsed_results, 'statistics': field_statistics, 'total_unique_items': sum(field_statistics.values())}
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f'已处理: {input_file}')
    print(f'生成: {output_file}')
    print(f'统计信息: 总病史项目数量={sum(field_statistics.values())}')
    for field, count in field_statistics.items():
        print(f'  {field}: {count}个项目')
    field_names = extract_field_names_from_data(output_data)
    return (output_data, field_names)

def process_surg_vars_file(input_file: str, output_file: str) -> Tuple[Dict, List[str]]:
    """
    处理surg_vars文件，保留指定字段的唯一值

    参数:
        input_file: 输入的JSON文件路径
        output_file: 输出的JSON文件路径

    返回:
        (输出数据, 字段名称列表)
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    target_fields = ['surg_admission_chars', 'preop_diagnosis', 'scheduled_procedure_name']
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
    处理test_vars文件，提取以_val结尾的字段名称

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
    处理pat_demo文件，提取所有字段名称

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
    print('开始处理MOVER数据库的特殊字段...')
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    input_dir = os.path.join(project_root, 'data', 'raw', 'unique_values')
    output_dir = os.path.join(project_root, 'data', 'processed', 'mover')
    os.makedirs(output_dir, exist_ok=True)
    all_table_data = []
    print('\n按照新要求处理各表...')
    print('\n1. 处理feat_anes_vars...')
    anes_vars_input = os.path.join(input_dir, 'mover_feat_anes_vars.json')
    anes_vars_output = os.path.join(output_dir, 'mover_feat_anes_vars_parsed.json')
    if os.path.exists(anes_vars_input):
        output_data, field_names = process_anes_vars_file(anes_vars_input, anes_vars_output)
        if output_data:
            all_table_data.append(output_data)
    else:
        print(f'错误: 输入文件 {anes_vars_input} 不存在')
    print('\n2. 处理feat_med_rec...')
    med_rec_input = os.path.join(input_dir, 'mover_feat_med_rec.json')
    med_rec_output = os.path.join(output_dir, 'mover_feat_med_rec_parsed.json')
    if os.path.exists(med_rec_input):
        output_data, field_names = process_med_rec_file(med_rec_input, med_rec_output)
        if output_data:
            all_table_data.append(output_data)
    else:
        print(f'错误: 输入文件 {med_rec_input} 不存在')
    print('\n3. 处理feat_past_his...')
    past_his_input = os.path.join(input_dir, 'mover_feat_past_his.json')
    past_his_output = os.path.join(output_dir, 'mover_feat_past_his_parsed.json')
    if os.path.exists(past_his_input):
        output_data, field_names = process_past_his_file(past_his_input, past_his_output)
        if output_data:
            all_table_data.append(output_data)
    else:
        print(f'错误: 输入文件 {past_his_input} 不存在')
    print('\n4. feat_pat_demo：跳过处理')
    print('\n5. 处理feat_surg_vars...')
    surg_vars_input = os.path.join(input_dir, 'mover_feat_surg_vars.json')
    surg_vars_output = os.path.join(output_dir, 'mover_feat_surg_vars_parsed.json')
    if os.path.exists(surg_vars_input):
        output_data, field_names = process_surg_vars_file(surg_vars_input, surg_vars_output)
        if output_data:
            all_table_data.append(output_data)
    else:
        print(f'错误: 输入文件 {surg_vars_input} 不存在')
    print('\n6. feat_test_vars：跳过处理')
    print('\n7. 生成字段汇总表...')
    field_summary = generate_field_summary_table(all_table_data)
    field_summary_output = os.path.join(output_dir, 'mover_field_summary.json')
    with open(field_summary_output, 'w', encoding='utf-8') as f:
        json.dump(field_summary, f, ensure_ascii=False, indent=2)
    print(f'字段汇总表已保存到: {field_summary_output}')
    print(f"汇总了 {len(field_summary['tables'])} 个表的字段信息")
    for table in field_summary['tables']:
        print(f"  - {table['table_name']}: {len(table['field_names'])} 个字段")
    print('\nMOVER数据库处理完成!')
if __name__ == '__main__':
    main()

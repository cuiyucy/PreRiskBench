"""
FSN 修正器：用 SNOMED CT 全概念索引回查正确的 FSN

LLM 返回的 fsn 可能不规范或被截断，此脚本用 conceptId
从 SNOMED 全概念索引中获取标准 FSN（带后缀）。

用法:
  python3 5.fsn_corrector.py -i llm_mapping.csv -o llm_mapping_corrected.csv
  python3 5.fsn_corrector.py -i llm_mapping.csv -o llm_mapping_corrected.csv --min-score 0.85
"""
import os as _release_os
from release_runtime import expand_path as _release_expand, expand_config as _release_expand_config
import os
import csv
import json
import argparse
from pathlib import Path
from typing import Dict

def load_concept_fsn(index_path: str) -> Dict[str, str]:
    """加载 SNOMED 全概念 conceptId -> FSN 映射"""
    print(f'加载 SNOMED 索引: {index_path}')
    with open(index_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    mapping = data.get('concept_id_to_fsn', {})
    print(f'✓ 已加载 {len(mapping):,} 个概念')
    return mapping

def process(input_file: str, output_file: str, low_conf_file: str=None, not_found_file: str=None):
    """处理 llm_mapping.csv，回查正确的 FSN，分离低置信和未找到的记录"""
    project_root = Path(__file__).parent.parent.parent
    index_path = project_root / 'data' / 'processed' / 'snomed_allconcepts' / 'snomed_allconcepts_optimized_index.json'
    concept_fsn = load_concept_fsn(str(index_path))
    parent = Path(output_file).parent
    if low_conf_file is None:
        low_conf_file = str(parent / 'llm_mapping_low_confidence.csv')
    if not_found_file is None:
        not_found_file = str(parent / 'llm_mapping_not_found.csv')
    fieldnames = ['dataset', 'field_name', 'field_value', 'conceptId', 'fsn', 'source', 'score']
    post_fieldnames = ['dataset', 'field_name', 'field_value']
    total = 0
    corrected = 0
    not_found_count = 0
    low_conf_count = 0
    clean_count = 0
    input_path = Path(input_file)
    output_path = Path(output_file)
    low_conf_path = Path(low_conf_file)
    not_found_path = Path(not_found_file)
    print(f'输入: {input_path}')
    print(f'最终输出: {output_path}')
    print(f'低置信度输出: {low_conf_path}')
    print(f'未找到输出: {not_found_path}')
    print()
    with open(input_path, 'r', encoding='utf-8') as fin, open(output_path, 'w', encoding='utf-8', newline='') as fout_clean, open(low_conf_path, 'w', encoding='utf-8', newline='') as fout_low, open(not_found_path, 'w', encoding='utf-8', newline='') as fout_nf:
        reader = csv.DictReader(fin)
        writer_clean = csv.DictWriter(fout_clean, fieldnames=fieldnames)
        writer_low = csv.DictWriter(fout_low, fieldnames=post_fieldnames, extrasaction='ignore')
        writer_nf = csv.DictWriter(fout_nf, fieldnames=post_fieldnames, extrasaction='ignore')
        writer_clean.writeheader()
        writer_low.writeheader()
        writer_nf.writeheader()
        for row in reader:
            total += 1
            concept_id = row.get('conceptId', '').strip()
            score_str = row.get('score', '0')
            try:
                score = float(score_str) if score_str else 0.0
            except ValueError:
                score = 0.0
            if score < 0.85:
                low_conf_count += 1
                writer_low.writerow(row)
                continue
            if concept_id:
                correct_fsn = concept_fsn.get(concept_id)
                if correct_fsn:
                    old_fsn = row.get('fsn', '')
                    row['fsn'] = correct_fsn
                    if old_fsn != correct_fsn:
                        corrected += 1
                    clean_count += 1
                    writer_clean.writerow(row)
                else:
                    not_found_count += 1
                    writer_nf.writerow(row)
            else:
                not_found_count += 1
                writer_nf.writerow(row)
    print(f'处理完成!')
    print(f'  总记录: {total}')
    print(f'  最终保留(≥0.85): {clean_count}')
    print(f'  FSN已修正: {corrected}')
    print(f'  低置信度(<0.85): {low_conf_count} → {low_conf_path}')
    print(f'  conceptId未找到: {not_found_count} → {not_found_path}')
    print(f'  最终输出: {output_path}')

def main():
    parser = argparse.ArgumentParser(description='FSN 修正器 - 回查 SNOMED 标准 FSN')
    parser.add_argument('-i', '--input', default=_release_expand('${ALIGNMENT_WORK_DIR}/data/mapping/llm_process/llm_mapping.csv'), help='输入 CSV 文件 (默认: llm_mapping.csv)')
    parser.add_argument('-o', '--output', default=_release_expand('${ALIGNMENT_WORK_DIR}/data/mapping/llm_process/llm_mapping_corrected.csv'), help='输出 CSV 文件 (默认: llm_mapping_corrected.csv)')
    parser.add_argument('--low-conf', '-l', default=None, help='低置信度输出 (默认: llm_mapping_low_confidence.csv)')
    parser.add_argument('--not-found', '-n', default=None, help='未找到 conceptId 输出 (默认: llm_mapping_not_found.csv)')
    args = parser.parse_args()
    process(args.input, args.output, args.low_conf, args.not_found)
if __name__ == '__main__':
    main()

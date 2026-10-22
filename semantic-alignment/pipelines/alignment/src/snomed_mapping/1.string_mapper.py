import os as _release_os
from release_runtime import expand_path as _release_expand, expand_config as _release_expand_config
import os
import json
import csv
import re
import time
from typing import Dict, List, Optional, Tuple, Set

def clean_icd10_suffixes(term: str) -> str:
    """清理ICD-10/编码系统特有的无用后缀，提取核心医学术语

    清理规则:
      1. CMS-HCC 标记
      2. 手术入路 (approach): open/percutaneous/external 等
      3. 就诊类型 (encounter): initial/subsequent
      4. 后遗症 (sequela)
      5. 未特指属性: laterality/site/type/chronicity
      6. 左右侧: left/right/bilateral
      7. 孕期相关: antepartum/trimester/pregnant
      8. 年龄组: adult/child/pediatric
      9. NEC/NOS/not elsewhere classified
      10. 方括号注释 [scarring hair loss]
      11. 部位限定: hand/foot
    """
    if not term:
        return term
    t = term.strip()
    patterns = ['\\s*\\(CMS-HCC\\)\\s*$', ',\\s*(?:open|percutaneous|external|via natural|via percutaneous|transoral|transurethral|through natural or artificial opening(?:\\s+endoscopic)?|through dressing|through indwelling device)\\s+\\w*\\s*approach\\w*\\s*$', ',\\s*(initial|subsequent)\\s+encounter\\s*\\(cms-hcc\\)\\s*$', ',\\s*(initial|subsequent)\\s+encounter\\s*$', ',\\s*sequela\\s*$', ',\\s*unspecified\\s+(?:whether\\s+serious\\s+comorbidity\\s+present|laterality|site|type|chronicity|headache\\s+type|obesity\\s+type)\\s*\\(cms-hcc\\)\\s*$', ',\\s*unspecified\\s+(?:whether\\s+serious\\s+comorbidity\\s+present|laterality|site|type|chronicity|headache\\s+type|obesity\\s+type)\\s*$', ',\\s*(left|right|bilateral)\\s*\\(cms-hcc\\)\\s*$', ',\\s*(left|right|bilateral)\\s*$', ',\\s*(?:antepartum|(?:first|second|third)\\s+trimester|currently\\s+pregnant)\\s*$', ',\\s*(adult|child|pediatric)\\s*\\(cms-hcc\\)\\s*$', ',\\s*(adult|child|pediatric)\\s*$', ',\\s*not elsewhere classified\\s*$', ',\\s*not elsewhere specified\\s*$', ',\\s*NOS\\s*$', ',\\s*NEC\\s*$', ',\\s*unspecified\\s*$', '\\s*\\[.*?\\]\\s*$', ',\\s*hand\\s*$', ',\\s*foot\\s*$']
    for pat in patterns:
        new_t = re.sub(pat, '', t, flags=re.IGNORECASE).strip()
        if new_t != t:
            t = new_t
    return t.strip(' ,')

def load_field_mapping_config(config_path: str=None) -> Dict:
    """加载字段映射配置文件"""
    if config_path is None:
        project_root = _release_expand('${ALIGNMENT_WORK_DIR}')
        config_path = os.path.join(project_root, 'config/field_mapping.json')
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f'⚠ 配置文件加载失败: {e}')
        return {}

class StringSNOMEDMapper:
    """基于字符串的SNOMED映射器 (优化版)"""

    def __init__(self, optimized_index_path: str, config_path: str=None):
        """初始化并加载优化后的SNOMED索引"""
        self.concept_id_to_fsn = {}
        self.term_to_concept_ids = {}
        self.cache = {}
        self.field_mapping = load_field_mapping_config(config_path)
        self.field_type_mapping = self.field_mapping.get('field_type_mapping', {})
        self.load_optimized_index(optimized_index_path)

    def load_optimized_index(self, index_path: str):
        """加载优化后的SNOMED索引数据"""
        print(f'加载优化后的SNOMED索引: {index_path}...')
        try:
            with open(index_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.concept_id_to_fsn = data.get('concept_id_to_fsn', {})
            self.term_to_concept_ids = data.get('term_to_concept_ids', {})
            print(f'✓ 索引加载成功: {len(self.concept_id_to_fsn)} 个概念, {len(self.term_to_concept_ids)} 个术语')
        except Exception as e:
            print(f'❌ 索引加载失败: {e}')
            raise

    def map_term(self, term: str, field_type: str=None) -> Optional[Dict]:
        """映射一个术语：双重字符串检索 (原始值 + 清理后值)

        Args:
            term: 待映射的术语
            field_type: 字段类型 (disease/medication/procedure)，用于type:concept匹配
        """
        if not term:
            return None
        term_raw = term.strip().lower().replace('(cms-hcc)', '').strip()
        term_clean = clean_icd10_suffixes(term).lower()
        if not term_clean:
            return None
        cache_key = f'{term_raw}|{field_type}' if field_type else term_raw
        if cache_key in self.cache:
            return self.cache[cache_key]
        if field_type:
            for term_variant in [term_raw, term_clean]:
                if not term_variant:
                    continue
                type_prefixed = f'{field_type}:{term_variant}'
                if type_prefixed in self.term_to_concept_ids:
                    concept_id = self.term_to_concept_ids[type_prefixed][0]
                    result = {'conceptId': str(concept_id), 'fsn': self.concept_id_to_fsn.get(concept_id, ''), 'source': 'exact', 'score': 1.0}
                    self.cache[cache_key] = result
                    return result
        result = {'conceptId': '', 'fsn': '', 'source': 'fuzzy', 'score': ''}
        self.cache[cache_key] = result
        return result

def process_datasets_optimized(mapper: StringSNOMEDMapper, raw_unique_values_dir: str, output_dir: str):
    """优化后的数据集处理函数：从raw/unique_values目录读取JSON文件"""
    datasets = ['inspire', 'mover', 'vitaldb']
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for ds_name in datasets:
        json_file = os.path.join(raw_unique_values_dir, f'{ds_name}_unique_values.json')
        if not os.path.exists(json_file):
            print(f'⚠ 数据集文件不存在: {json_file}')
            continue
        output_csv = os.path.join(output_dir, f'{ds_name}_field_values_mapping.csv')
        with open(output_csv, 'w', encoding='utf-8', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['field_name', 'field_value', 'conceptId', 'fsn', 'source', 'score'])
            writer.writeheader()
            print(f'处理数据集文件: {json_file}')
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                unique_values = data.get('unique_values', {})
                rows = []
                for field_name, values in unique_values.items():
                    field_type = mapper.field_type_mapping.get(field_name)
                    for val in values:
                        if not val:
                            continue
                        mapping = mapper.map_term(val, field_type)
                        rows.append({'field_name': field_name, 'field_value': val, 'conceptId': mapping['conceptId'], 'fsn': mapping['fsn'], 'source': mapping['source'], 'score': mapping['score']})
                writer.writerows(rows)
        print(f'✓ {ds_name} 字段值映射完成: {output_csv}')
if __name__ == '__main__':
    project_root = _release_expand('${ALIGNMENT_WORK_DIR}')
    index_path = os.path.join(project_root, 'data/processed/snomed/snomed_optimized_index.json')
    config_path = os.path.join(project_root, 'config/field_mapping.json')
    raw_unique_values_dir = os.path.join(project_root, 'data/raw/unique_values')
    output_dir = os.path.join(project_root, 'data/mapping/string_mapping')
    start_time = time.time()
    mapper = StringSNOMEDMapper(index_path, config_path)
    process_datasets_optimized(mapper, raw_unique_values_dir, output_dir)
    end_time = time.time()
    print(f'\n全部映射任务完成！总耗时: {end_time - start_time:.2f}秒')

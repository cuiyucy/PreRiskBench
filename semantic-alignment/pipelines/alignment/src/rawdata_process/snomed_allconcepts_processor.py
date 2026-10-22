"""
SNOMED CT 全概念处理器
处理所有SNOMED CT概念，不进行类型筛选
"""
import os as _release_os
from release_runtime import expand_path as _release_expand, expand_config as _release_expand_config
import pandas as pd
import json
import os
import re
import argparse
import pickle
from pathlib import Path
from typing import List, Dict, Set

class SNOMEDAllConceptsProcessor:
    """SNOMED CT 全概念处理器"""
    DES_typeId = {'FSN': '900000000000003001', 'SYNONYM': '900000000000013009'}

    def __init__(self, data_dir: str=_release_expand('${ALIGNMENT_WORK_DIR}/data/raw/snomed_ct'), output_dir: str=_release_expand('${ALIGNMENT_WORK_DIR}/data/processed/snomed_allconcepts')):
        self.data_dir = Path(data_dir)
        self.data = {}
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            project_root = self.data_dir.parent.parent
            self.output_dir = project_root / 'processed' / 'snomed_allconcepts'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.all_concepts_cache_file = self.output_dir / 'all_concepts_cache.json'
        self.index_cache_file = self.output_dir / 'snomed_index_cache.pkl'
        self.fsn_index = {}
        self.synonym_index = {}

    def load_data(self, verbose: bool=True, use_index_cache: bool=True, force_rebuild_index: bool=False):
        """加载SNOMED CT descriptions文件并构建索引"""
        if use_index_cache and self.index_cache_file.exists() and (not force_rebuild_index):
            try:
                with open(self.index_cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                self.fsn_index = cached_data.get('fsn_index', {})
                self.synonym_index = cached_data.get('synonym_index', {})
                if verbose:
                    print('✓ 从缓存加载索引数据')
                    print(f'  ✓ FSN索引: {len(self.fsn_index)} 个概念')
                    print(f'  ✓ 同义词索引: {len(self.synonym_index)} 个概念')
                return
            except Exception as e:
                if verbose:
                    print(f'⚠ 索引缓存读取失败: {e}')
        if verbose:
            print('正在加载SNOMED CT数据文件...')
        desc_file = self.data_dir / 'sct2_Description_Snapshot-en_INT_20260201.txt'
        if desc_file.exists():
            self.data['descriptions'] = pd.read_csv(desc_file, sep='\t', dtype=str)
            if verbose:
                print(f"  ✓ descriptions: {len(self.data['descriptions'])} 条记录")
        else:
            if verbose:
                print(f'  ⚠ 文件不存在: {desc_file.name}')
            return
        if verbose:
            print('正在构建数据索引...')
        desc_df = self.data['descriptions']
        en_df = desc_df[desc_df['languageCode'] == 'en']
        fsn_df = en_df[(en_df['typeId'] == self.DES_typeId['FSN']) & (en_df['active'] == '1')]
        for _, row in fsn_df.iterrows():
            self.fsn_index[row['conceptId']] = row['term']
        fsn_inactive_df = en_df[(en_df['typeId'] == self.DES_typeId['FSN']) & (en_df['active'] == '0')]
        for concept_id, group in fsn_inactive_df.groupby('conceptId'):
            if concept_id not in self.synonym_index:
                self.synonym_index[concept_id] = []
            for term in group['term'].tolist():
                if term and (not (isinstance(term, float) and pd.isna(term))):
                    self.synonym_index[concept_id].append(term)
        synonym_df = en_df[en_df['typeId'] == self.DES_typeId['SYNONYM']]
        for concept_id, group in synonym_df.groupby('conceptId'):
            if concept_id not in self.synonym_index:
                self.synonym_index[concept_id] = []
            for term in group['term'].tolist():
                if term and (not (isinstance(term, float) and pd.isna(term))):
                    self.synonym_index[concept_id].append(term)
        if use_index_cache:
            try:
                cached_data = {'fsn_index': self.fsn_index, 'synonym_index': self.synonym_index}
                with open(self.index_cache_file, 'wb') as f:
                    pickle.dump(cached_data, f)
                if verbose:
                    print('✓ 索引缓存已保存')
            except Exception as e:
                if verbose:
                    print(f'⚠ 索引缓存保存失败: {e}')
        if verbose:
            print(f'  ✓ FSN索引: {len(self.fsn_index)} 个概念')
            print(f'  ✓ 同义词索引: {len(self.synonym_index)} 个概念')

    def get_all_concepts(self, use_cache: bool=True, force_rebuild: bool=False) -> Set[str]:
        """获取所有概念ID，支持缓存机制"""
        if use_cache and self.all_concepts_cache_file.exists() and (not force_rebuild):
            try:
                with open(self.all_concepts_cache_file, 'r', encoding='utf-8') as f:
                    cached_concepts = json.load(f)
                print(f'✓ 从缓存加载所有概念: {len(cached_concepts)} 个')
                return set(cached_concepts)
            except Exception as e:
                print(f'⚠ 缓存文件读取失败: {e}')
        print('正在构建所有概念列表...')
        all_concepts = set(self.fsn_index.keys())
        type_counts = {}
        for fsn_term in self.fsn_index.values():
            match = re.search('\\((.*?)\\)$', fsn_term)
            if match:
                suffix = match.group(1).strip()
                type_counts[suffix] = type_counts.get(suffix, 0) + 1
            else:
                type_counts['unknown'] = type_counts.get('unknown', 0) + 1
        print('所有概念类型分布:')
        for suffix, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True)[:20]:
            print(f'  {suffix}: {count:,}')
        if len(type_counts) > 20:
            print(f'  ... 还有 {len(type_counts) - 20} 种其他类型')
        if use_cache:
            try:
                with open(self.all_concepts_cache_file, 'w', encoding='utf-8') as f:
                    json.dump(list(all_concepts), f, ensure_ascii=False, indent=2)
                print(f'✓ 所有概念缓存已保存: {len(all_concepts)} 个')
            except Exception as e:
                print(f'⚠ 缓存保存失败: {e}')
        return all_concepts

    def clean_term(self, term: str) -> str:
        """清理术语，去除末尾括号后缀"""
        if term is None or (isinstance(term, float) and pd.isna(term)):
            return ''
        cleaned = re.sub('\\s*\\([^)]*\\)\\s*$', '', str(term))
        return cleaned.strip()

    def extract_type(self, fsn: str) -> str:
        """从FSN中提取概念类型（括号后缀）"""
        match = re.search('\\((.*?)\\)$', fsn)
        if match:
            return match.group(1).strip()
        return ''

    def process_all_concepts(self, batch_size: int=10000, verbose: bool=True):
        """处理所有概念，生成RAG文本和优化索引"""
        all_concepts = self.get_all_concepts()
        if verbose:
            print(f'开始处理 {len(all_concepts)} 个概念...')
        concept_list = list(all_concepts)
        total_batches = (len(concept_list) + batch_size - 1) // batch_size
        all_rag_texts = []
        concept_id_to_fsn = {}
        term_to_concept_ids = {}
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(concept_list))
            batch_concepts = concept_list[start_idx:end_idx]
            if verbose:
                print(f'处理批次 {batch_idx + 1}/{total_batches} ({start_idx + 1}-{end_idx})')
            batch_rag = []
            for concept_id in batch_concepts:
                if concept_id not in self.fsn_index:
                    continue
                original_fsn = self.fsn_index[concept_id]
                concept_type = self.extract_type(original_fsn)
                cleaned_fsn = self.clean_term(original_fsn)
                if not cleaned_fsn:
                    continue
                concept_id_to_fsn[concept_id] = original_fsn
                batch_rag.append({'conceptId': concept_id, 'text': cleaned_fsn, 'term_type': 'fsn', 'concept_type': concept_type})
                fsn_lower = cleaned_fsn.lower()
                if fsn_lower not in term_to_concept_ids:
                    term_to_concept_ids[fsn_lower] = []
                term_to_concept_ids[fsn_lower].append(concept_id)
                if concept_id in self.synonym_index:
                    for synonym in self.synonym_index[concept_id]:
                        if not synonym or (isinstance(synonym, float) and pd.isna(synonym)):
                            continue
                        cleaned_synonym = self.clean_term(synonym)
                        if not cleaned_synonym or cleaned_synonym == cleaned_fsn:
                            continue
                        batch_rag.append({'conceptId': concept_id, 'text': cleaned_synonym, 'term_type': 'synonym', 'concept_type': concept_type})
                        syn_lower = cleaned_synonym.lower()
                        if syn_lower not in term_to_concept_ids:
                            term_to_concept_ids[syn_lower] = []
                        term_to_concept_ids[syn_lower].append(concept_id)
            all_rag_texts.extend(batch_rag)
            if verbose:
                print(f'  本批次处理完成: {len(batch_rag)} 个RAG文本')
        rag_output = self.output_dir / 'snomed_allconcepts_rag_texts.json'
        with open(rag_output, 'w', encoding='utf-8') as f:
            json.dump(all_rag_texts, f, ensure_ascii=False, indent=2)
        optimized_index = {'concept_id_to_fsn': concept_id_to_fsn, 'term_to_concept_ids': term_to_concept_ids}
        index_output = self.output_dir / 'snomed_allconcepts_optimized_index.json'
        with open(index_output, 'w', encoding='utf-8') as f:
            json.dump(optimized_index, f, ensure_ascii=False, indent=2)
        if verbose:
            print(f'✓ 处理完成!')
            print(f'  ✓ RAG文本: {len(all_rag_texts)} 个文本')
            print(f'  ✓ 优化索引: {len(concept_id_to_fsn)} 个概念, {len(term_to_concept_ids)} 个术语')
            print(f'  ✓ 保存至: {self.output_dir}')

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='SNOMED CT 全概念处理器')
    parser.add_argument('--data_dir', required=True, help='SNOMED CT数据目录')
    parser.add_argument('--output_dir', help='输出目录（默认为data/processed/snomed_allconcepts）')
    parser.add_argument('--batch_size', type=int, default=10000, help='批处理大小')
    parser.add_argument('--force_rebuild', action='store_true', help='强制重建索引')
    args = parser.parse_args()
    processor = SNOMEDAllConceptsProcessor(data_dir=args.data_dir, output_dir=args.output_dir)
    processor.load_data(verbose=True, force_rebuild_index=args.force_rebuild)
    processor.process_all_concepts(batch_size=args.batch_size, verbose=True)
if __name__ == '__main__':
    main()

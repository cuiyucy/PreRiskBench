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

def load_field_mapping_config(config_path: str=None) -> Dict:
    if config_path is None:
        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent
        config_path = project_root / 'config' / 'field_mapping.json'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config

class SNOMEDRAGProcessor:
    """SNOMED CT RAG数据处理器"""
    DES_typeId = {'FSN': '900000000000003001', 'SYNONYM': '900000000000013009'}

    def __init__(self, data_dir: str, output_dir: str=None, config_path: str=None):
        self.data_dir = Path(data_dir)
        self.data = {}
        self.config = load_field_mapping_config(config_path)
        self.TARGET_SNOMED_TYPES = self.config.get('target_snomed_types', [])
        self.SNOMED_TYPE_TO_FIELD_TYPE = self.config.get('snomed_type_mapping', {})
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            project_root = self.data_dir.parent.parent
            self.output_dir = project_root / 'processed' / 'snomed'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.target_concepts_cache_file = self.output_dir / 'target_concepts_cache.json'
        self.index_cache_file = self.output_dir / 'snomed_index_cache.pkl'
        self.fsn_index = {}
        self.synonym_index = {}

    def load_data(self, verbose: bool=True, use_index_cache: bool=True, force_rebuild_index: bool=False):
        """加载SNOMED CT数据文件并构建索引"""
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
        if self.data['descriptions'] is not None:
            en_df = self.data['descriptions'][self.data['descriptions']['languageCode'] == 'en']
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

    def get_target_concepts(self, use_cache: bool=True, force_rebuild: bool=False) -> Set[str]:
        """获取目标概念ID，支持缓存机制"""
        if use_cache and self.target_concepts_cache_file.exists() and (not force_rebuild):
            try:
                with open(self.target_concepts_cache_file, 'r', encoding='utf-8') as f:
                    cached_concepts = json.load(f)
                print(f'✓ 从缓存加载目标概念: {len(cached_concepts)} 个')
                return set(cached_concepts)
            except Exception as e:
                print(f'⚠ 缓存文件读取失败: {e}')
        print('正在构建目标概念列表...')
        if len(self.fsn_index) == 0:
            return self._get_target_concepts_slow()
        target_types = self.TARGET_SNOMED_TYPES
        filtered_concepts = set()
        type_counts = {}
        for concept_id, fsn_term in self.fsn_index.items():
            match = re.search('\\((.*?)\\)$', fsn_term)
            if match:
                suffix = match.group(1).strip()
                if suffix in target_types:
                    filtered_concepts.add(concept_id)
                    type_counts[suffix] = type_counts.get(suffix, 0) + 1
        print('目标类型分布:')
        for suffix, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
            print(f'  {suffix}: {count:,}')
        if use_cache:
            try:
                with open(self.target_concepts_cache_file, 'w', encoding='utf-8') as f:
                    json.dump(list(filtered_concepts), f, ensure_ascii=False, indent=2)
                print(f'✓ 目标概念缓存已保存: {len(filtered_concepts)} 个')
            except Exception as e:
                print(f'⚠ 缓存保存失败: {e}')
        return filtered_concepts

    def _get_target_concepts_slow(self) -> Set[str]:
        """慢速方法：用于没有预索引的情况"""
        if 'descriptions' not in self.data or self.data['descriptions'] is None:
            return set()
        df = self.data['descriptions']
        active_fsn = df[(df['active'] == '1') & (df['languageCode'] == 'en') & (df['typeId'] == self.DES_typeId['FSN'])]
        target_types = self.TARGET_SNOMED_TYPES
        filtered_concepts = set()
        type_counts = {}
        for _, row in active_fsn.iterrows():
            term = row['term']
            match = re.search('\\((.*?)\\)$', term)
            if match:
                suffix = match.group(1).strip()
                if suffix in target_types:
                    filtered_concepts.add(row['conceptId'])
                    type_counts[suffix] = type_counts.get(suffix, 0) + 1
        print(f'目标类型过滤: 从 {len(active_fsn)} 个FSN概念中筛选出 {len(filtered_concepts)} 个目标概念')
        print('目标类型分布:')
        for suffix, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
            print(f'  {suffix}: {count:,}')
        return filtered_concepts

    def clean_term(self, term: str) -> str:
        """清理术语，去除括号后缀"""
        cleaned = re.sub('\\s*\\([^)]*\\)\\s*$', '', term)
        return cleaned.strip()

    def process_all_concepts(self, target_concepts: Set[str], batch_size: int=10000, verbose: bool=True):
        """处理所有目标概念，生成RAG文本和优化索引"""
        concept_list = list(target_concepts)
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
                match = re.search('\\((.*?)\\)$', original_fsn)
                if match:
                    snomed_type = match.group(1).strip()
                    field_type = self.SNOMED_TYPE_TO_FIELD_TYPE.get(snomed_type, 'other')
                else:
                    snomed_type = ''
                    field_type = 'other'
                cleaned_fsn = self.clean_term(original_fsn)
                if not cleaned_fsn:
                    continue
                concept_id_to_fsn[concept_id] = original_fsn
                enhanced_fsn = f'{field_type}:{cleaned_fsn}'
                batch_rag.append({'conceptId': concept_id, 'text': enhanced_fsn, 'term_type': 'fsn', 'concept_type': snomed_type, 'field_type': field_type})
                fsn_lower = cleaned_fsn.lower()
                fsn_with_type = f'{field_type}:{fsn_lower}'
                if fsn_with_type not in term_to_concept_ids:
                    term_to_concept_ids[fsn_with_type] = set()
                term_to_concept_ids[fsn_with_type].add(concept_id)
                if concept_id in self.synonym_index:
                    for synonym in self.synonym_index[concept_id]:
                        if not synonym or (isinstance(synonym, float) and pd.isna(synonym)):
                            continue
                        cleaned_synonym = self.clean_term(synonym)
                        if not cleaned_synonym or cleaned_synonym == cleaned_fsn:
                            continue
                        enhanced_synonym = f'{field_type}:{cleaned_synonym}'
                        batch_rag.append({'conceptId': concept_id, 'text': enhanced_synonym, 'term_type': 'synonym', 'concept_type': snomed_type, 'field_type': field_type})
                        syn_lower = cleaned_synonym.lower()
                        syn_with_type = f'{field_type}:{syn_lower}'
                        if syn_with_type not in term_to_concept_ids:
                            term_to_concept_ids[syn_with_type] = set()
                        term_to_concept_ids[syn_with_type].add(concept_id)
            all_rag_texts.extend(batch_rag)
            if verbose:
                print(f'  本批次处理完成: {len(batch_rag)} 个RAG文本')
        rag_output = self.output_dir / 'snomed_rag_texts.json'
        with open(rag_output, 'w', encoding='utf-8') as f:
            json.dump(all_rag_texts, f, ensure_ascii=False, indent=2)
        term_to_concept_ids_list = {}
        for term, ids in term_to_concept_ids.items():
            term_to_concept_ids_list[term] = list(ids)
        optimized_index = {'concept_id_to_fsn': concept_id_to_fsn, 'term_to_concept_ids': term_to_concept_ids_list}
        index_output = self.output_dir / 'snomed_optimized_index.json'
        with open(index_output, 'w', encoding='utf-8') as f:
            json.dump(optimized_index, f, ensure_ascii=False, indent=2)
        if verbose:
            print(f'✓ 处理完成!')
            print(f'  ✓ RAG文本: {len(all_rag_texts)} 个文本')
            print(f'  ✓ 优化索引: {len(concept_id_to_fsn)} 个概念, {len(term_to_concept_ids)} 个术语')
            print(f'  ✓ 保存至: {self.output_dir}')

def main():
    """主函数"""
    import time
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    snomed_dir = os.path.join(project_root, 'data', 'raw', 'snomed_ct')
    output_dir = os.path.join(project_root, 'data', 'processed', 'snomed')
    parser = argparse.ArgumentParser(description='SNOMED CT RAG数据处理器')
    parser.add_argument('--data-dir', default=snomed_dir, help='SNOMED数据目录')
    parser.add_argument('--output-dir', default=output_dir, help='输出目录')
    parser.add_argument('--config-path', default=None, help='字段映射配置文件路径')
    parser.add_argument('--limit', type=int, default=None, help='限制处理的概念数量')
    parser.add_argument('--batch-size', type=int, default=10000, help='批处理大小')
    parser.add_argument('--use-cache', action='store_true', default=True, help='使用目标概念缓存')
    parser.add_argument('--force-rebuild', action='store_true', default=False, help='强制重新构建目标概念缓存')
    parser.add_argument('--use-index-cache', action='store_true', default=True, help='使用索引缓存')
    parser.add_argument('--force-rebuild-index', action='store_true', default=False, help='强制重新构建索引缓存')
    args = parser.parse_args()
    processor = SNOMEDRAGProcessor(args.data_dir, args.output_dir, args.config_path)
    try:
        start_time = time.time()
        processor.load_data(verbose=True, use_index_cache=args.use_index_cache, force_rebuild_index=args.force_rebuild_index)
        load_time = time.time() - start_time
        start_time = time.time()
        target_concepts = processor.get_target_concepts(use_cache=args.use_cache, force_rebuild=args.force_rebuild)
        concept_time = time.time() - start_time
        if args.limit:
            target_concepts = set(list(target_concepts)[:args.limit])
        print(f'开始处理 {len(target_concepts)} 个目标概念...')
        print(f'数据加载时间: {load_time:.2f}秒')
        print(f'概念筛选时间: {concept_time:.2f}秒')
        start_time = time.time()
        processor.process_all_concepts(target_concepts, batch_size=args.batch_size, verbose=True)
        process_time = time.time() - start_time
        print(f'\n处理完成！')
        print(f'概念处理时间: {process_time:.2f}秒 ({process_time / 60:.1f}分钟)')
        total_time = load_time + concept_time + process_time
        print(f'\n性能统计:')
        print(f'数据加载: {load_time:.2f}秒 ({load_time / total_time * 100:.1f}%)')
        print(f'概念筛选: {concept_time:.2f}秒 ({concept_time / total_time * 100:.1f}%)')
        print(f'概念处理: {process_time:.2f}秒 ({process_time / total_time * 100:.1f}%)')
        print(f'总时间: {total_time:.2f}秒 ({total_time / 60:.1f}分钟)')
    except Exception as e:
        print(f'处理过程中出现错误: {e}')
        import traceback
        traceback.print_exc()
if __name__ == '__main__':
    main()

import os as _release_os
from release_runtime import expand_path as _release_expand, expand_config as _release_expand_config
import os
import json
import numpy as np
import pandas as pd
import faiss
import pickle
import yaml
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import csv
from typing import Set, List, Dict, Tuple
from collections import defaultdict
import re

class VectorRetrievalMapper:
    """SNOMED 向量检索与 BM25 融合映射器"""

    def __init__(self, config_path: str=None, embedding_dir: str=None):
        """初始化映射器

        Args:
            config_path: config.yaml路径
            embedding_dir: 嵌入数据目录，默认为 data/embeddings/snomed
        """
        if config_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(script_dir))
            config_path = os.path.join(project_root, 'config', 'config.yaml')
        self.embedding_dir = embedding_dir
        self.config = self._load_config(config_path)
        self.model = None
        self.faiss_index = None
        self.bm25 = None
        self.mapping_data = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.query_cache = {}
        self.field_mapping = {}
        self._load_resources()

    def _load_field_mapping(self, project_root: str):
        """加载字段映射配置"""
        field_mapping_path = os.path.join(project_root, 'config', 'field_mapping.json')
        try:
            with open(field_mapping_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.field_mapping = data.get('field_type_mapping', {})
                print(f'✓ 字段映射加载成功: {len(self.field_mapping)} 个字段')
        except Exception as e:
            print(f'⚠ 字段映射加载失败: {e}')
            self.field_mapping = {}

    def _save_query_cache(self, cache_path: str):
        """保存查询向量缓存到文件"""
        try:
            serializable_cache = {}
            for key, value in self.query_cache.items():
                serializable_cache[key] = value.tolist() if isinstance(value, np.ndarray) else value
            with open(cache_path, 'wb') as f:
                pickle.dump(serializable_cache, f)
            print(f'✓ 查询缓存已保存: {len(self.query_cache)} 个查询')
        except Exception as e:
            print(f'⚠ 查询缓存保存失败: {e}')

    def _load_query_cache(self, cache_path: str):
        """从文件加载查询向量缓存"""
        try:
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    serializable_cache = pickle.load(f)
                for key, value in serializable_cache.items():
                    if isinstance(value, list):
                        self.query_cache[key] = np.array(value, dtype=np.float32)
                    else:
                        self.query_cache[key] = value
                print(f'✓ 查询缓存已加载: {len(self.query_cache)} 个查询')
                return True
        except Exception as e:
            print(f'⚠ 查询缓存加载失败: {e}')
        return False

    def _load_config(self, config_path: str) -> dict:
        """加载配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return _release_expand_config(yaml.safe_load(f))

    def _load_resources(self):
        """加载模型、索引和映射数据"""
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self._load_field_mapping(project_root)
        cache_dir = os.path.join(project_root, 'data', 'mapping', 'retrieval_mapping')
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, 'query_vectors_cache.pkl')
        self._load_query_cache(cache_path)
        model_config = self.config['models']['clinical_bert']
        model_path = os.path.join(project_root, model_config['path'])
        print(f'正在加载模型: {model_path} (设备: {self.device})')
        self.model = SentenceTransformer(model_path, device=self.device)
        if self.embedding_dir:
            embedding_dir = self.embedding_dir
        else:
            embedding_dir = os.path.join(project_root, 'data', 'embeddings', 'snomed')
        index_path = os.path.join(embedding_dir, 'snomed_rag_faiss_index.index')
        print(f'正在加载 FAISS 索引: {index_path}')
        self.faiss_index = faiss.read_index(index_path)
        if hasattr(self.faiss_index, 'nprobe'):
            self.faiss_index.nprobe = 10
        print('✓ FAISS 索引加载完成')
        mapping_path = os.path.join(embedding_dir, 'snomed_rag_embedding_mapping.pkl')
        print(f'正在加载映射数据: {mapping_path}')
        with open(mapping_path, 'rb') as f:
            self.mapping_data = pickle.load(f)
        print('✓ FAISS 索引使用归一化向量，支持余弦相似度')

    def _clean_cms_suffix(self, text: str) -> str:
        return re.sub('\\(CMS-HCC\\)', '', text, flags=re.IGNORECASE).strip()

    @staticmethod
    def clean_icd10_suffixes(term: str) -> str:
        if not term:
            return term
        t = term.strip()
        patterns = ['\\s*\\(CMS-HCC\\)\\s*$', ',\\s*(?:open|percutaneous|external|via natural|via percutaneous|transoral|transurethral|through natural or artificial opening(?:\\s+endoscopic)?|through dressing|through indwelling device)\\s+\\w*\\s*approach\\w*\\s*$', ',\\s*(initial|subsequent)\\s+encounter\\s*\\(cms-hcc\\)\\s*$', ',\\s*(initial|subsequent)\\s+encounter\\s*$', ',\\s*sequela\\s*$', ',\\s*unspecified\\s+(?:whether\\s+serious\\s+comorbidity\\s+present|laterality|site|type|chronicity|headache\\s+type|obesity\\s+type)\\s*\\(cms-hcc\\)\\s*$', ',\\s*unspecified\\s+(?:whether\\s+serious\\s+comorbidity\\s+present|laterality|site|type|chronicity|headache\\s+type|obesity\\s+type)\\s*$', ',\\s*(left|right|bilateral)\\s*\\(cms-hcc\\)\\s*$', ',\\s*(left|right|bilateral)\\s*$', ',\\s*(?:antepartum|(?:first|second|third)\\s+trimester|currently\\s+pregnant)\\s*$', ',\\s*(adult|child|pediatric)\\s*\\(cms-hcc\\)\\s*$', ',\\s*(adult|child|pediatric)\\s*$', ',\\s*not elsewhere classified\\s*$', ',\\s*not elsewhere specified\\s*$', ',\\s*NOS\\s*$', ',\\s*NEC\\s*$', ',\\s*unspecified\\s*$', '\\s*\\[.*?\\]\\s*$', ',\\s*hand\\s*$', ',\\s*foot\\s*$']
        for pat in patterns:
            new_t = re.sub(pat, '', t, flags=re.IGNORECASE).strip()
            if new_t != t:
                t = new_t
        return t.strip(' ,')

    def _do_retrieve(self, query_text: str, top_k: int=50) -> dict:
        """执行单次向量检索"""
        if not query_text:
            return None
        query_lower = query_text.lower().strip()
        if query_lower in self.query_cache:
            query_vector = self.query_cache[query_lower]
        else:
            query_vector = self.model.encode([query_text], convert_to_numpy=True)
            if len(query_vector.shape) == 1:
                query_vector = query_vector.reshape(1, -1)
            query_vector = query_vector.astype(np.float32)
            query_vector = query_vector / np.linalg.norm(query_vector, axis=1, keepdims=True)
            self.query_cache[query_lower] = query_vector
        distances, indices = self.faiss_index.search(query_vector, top_k)
        cosine_similarities = 1.0 - distances[0] / 2.0
        candidates = []
        for idx, cos_sim in zip(indices[0], cosine_similarities):
            if idx == -1:
                continue
            concept_id = self.mapping_data['concept_ids'][idx]
            fsn = self.mapping_data['concept_to_fsn'].get(concept_id, 'Unknown FSN')
            original_text = self.mapping_data['texts'][idx]
            candidates.append({'conceptId': str(concept_id), 'fsn': fsn, 'original_text': original_text, 'score': float(cos_sim)})
        if candidates:
            best = self.rerank_by_keywords(query_text, candidates)
            return {'conceptId': best['conceptId'], 'fsn': best['fsn'], 'score': best['combined_score']}
        return None

    def retrieve(self, query: str, top_k: int=50) -> dict:
        """双重向量检索 (原始值 + 清理后值)，取相似度最高的"""
        if not query:
            return None
        raw_query = self._clean_cms_suffix(query)
        clean_query = self.clean_icd10_suffixes(query)
        result_raw = self._do_retrieve(raw_query, top_k) if raw_query else None
        result_clean = self._do_retrieve(clean_query, top_k) if clean_query and clean_query.lower() != raw_query.lower() else None
        if result_raw and result_clean:
            if result_raw.get('score', 0) >= result_clean.get('score', 0):
                return result_raw
            else:
                return result_clean
        elif result_raw:
            return result_raw
        elif result_clean:
            return result_clean
        return None

    def extract_keywords(self, text: str) -> Tuple[Set[str], Set[str]]:
        medical_stopwords = {'of', 'and', 'the', 'a', 'an', 'in', 'with', 'for', 'to', 'by', 'on', 'at', 'from', 'as', 'or', 'but', 'not', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'associated', 'related', 'due', 'during', 'after', 'before', 'primary', 'secondary', 'acute', 'chronic', 'history', 'female', 'male'}
        negation_words = {'without', 'no', 'not', 'non', 'never', 'none', 'absence', 'denied', 'negative'}
        text = re.sub('\\(CMS-HCC\\)', '', text, flags=re.IGNORECASE).strip()
        words = text.lower().split()
        split_words = []
        for word in words:
            if ':' in word:
                split_words.extend(word.split(':'))
            else:
                split_words.append(word)
        words = [re.sub('[^a-z0-9]', '', w) for w in split_words]
        keywords = set()
        negated_keywords = set()
        skip_next = False
        for word in words:
            if not word or word in medical_stopwords:
                skip_next = False
                continue
            if skip_next:
                negated_keywords.add(word)
                skip_next = False
                continue
            if word in negation_words:
                skip_next = True
                continue
            if len(word) > 2:
                keywords.add(word)
        return (keywords, negated_keywords)

    def calculate_keyword_score(self, query_keywords: Set[str], fsn_keywords: Set[str], query_negated: Set[str]=None) -> float:
        if not query_keywords:
            return 0.0
        if not fsn_keywords:
            return 0.0
        matched_count = 0
        for q_kw in query_keywords:
            if q_kw in fsn_keywords:
                matched_count += 1
            else:
                for f_kw in fsn_keywords:
                    if q_kw in f_kw or f_kw in q_kw:
                        matched_count += 1
                        break
        query_coverage = matched_count / len(query_keywords)
        candidate_coverage = matched_count / len(fsn_keywords)
        keyword_score = 0.6 * query_coverage + 0.4 * candidate_coverage
        if query_negated:
            negated_hit = 0
            for neg_kw in query_negated:
                if neg_kw in fsn_keywords:
                    negated_hit += 1
                else:
                    for f_kw in fsn_keywords:
                        if neg_kw in f_kw or f_kw in neg_kw:
                            negated_hit += 1
                            break
            if negated_hit > 0:
                negation_penalty = negated_hit / len(query_negated)
                keyword_score *= 1.0 - negation_penalty
        return min(1.0, keyword_score)

    def rerank_by_keywords(self, query: str, candidates: List[Dict]) -> Dict:
        """
        基于关键词匹配重新排序候选结果

        参数:
            query: 查询文本
            candidates: 候选结果列表，每个包含conceptId, fsn, score

        返回:
            重新排序后的最佳候选
        """
        if not candidates:
            return None
        query_keywords, query_negated = self.extract_keywords(query)
        reranked_candidates = []
        for candidate in candidates:
            text_for_matching = candidate.get('original_text', candidate['fsn'])
            vector_score = candidate['score']
            fsn_keywords, _ = self.extract_keywords(text_for_matching)
            keyword_score = self.calculate_keyword_score(query_keywords, fsn_keywords, query_negated)
            combined_score = vector_score * 0.8 + keyword_score * 0.2
            reranked_candidates.append({**candidate, 'keyword_score': round(keyword_score, 4), 'combined_score': round(combined_score, 4)})
        reranked_candidates.sort(key=lambda x: x['combined_score'], reverse=True)
        return reranked_candidates[0]

def process_all_datasets():
    """处理所有数据集的 fuzzy 项 - 双重向量检索"""
    project_root = _release_expand('${ALIGNMENT_WORK_DIR}')
    input_dir = os.path.join(project_root, 'data', 'mapping', 'string_mapping')
    output_dir = os.path.join(project_root, 'data', 'mapping', 'retrieval_mapping')
    cache_dir = output_dir
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    mapper = VectorRetrievalMapper()
    datasets = ['inspire', 'mover', 'vitaldb']
    for ds in datasets:
        input_file = os.path.join(input_dir, f'{ds}_field_values_mapping.csv')
        if not os.path.exists(input_file):
            print(f'跳过不存在的文件: {input_file}')
            continue
        print(f'\n正在处理数据集: {ds} ...')
        df = pd.read_csv(input_file)
        fuzzy_mask = df['source'] == 'fuzzy'
        fuzzy_df = df[fuzzy_mask].copy()
        if fuzzy_df.empty:
            print(f'  {ds} 没有 fuzzy 项，跳过。')
            continue
        fuzzy_items = fuzzy_df[['field_name', 'field_value']].drop_duplicates().values.tolist()
        value_to_field_names = defaultdict(list)
        for fname, val in fuzzy_items:
            value_to_field_names[val].append(fname)
        unique_fuzzy_values = fuzzy_df['field_value'].unique()
        print(f'  共有 {len(unique_fuzzy_values)} 个唯一的 fuzzy field_value 待检索...')
        batch_size = 128

        def build_query(val, field_names):
            field_type = ''
            for fn in field_names:
                ft = mapper.field_mapping.get(fn, '')
                if ft:
                    field_type = ft
                    break
            return field_type
        all_queries = []
        query_meta = {}
        for i, val in enumerate(unique_fuzzy_values):
            if pd.isna(val):
                continue
            field_names = value_to_field_names.get(val, [])
            field_type = build_query(val, field_names)
            raw_val = mapper._clean_cms_suffix(str(val))
            clean_val = mapper.clean_icd10_suffixes(str(val))
            raw_query = f'{field_type}:{raw_val}' if field_type else raw_val
            clean_query = f'{field_type}:{clean_val}' if field_type and clean_val != raw_val.lower() else clean_val if clean_val != raw_val.lower() else None
            all_queries.append((i, val, raw_query, 'raw'))
            if clean_query and clean_query.lower() != raw_query.lower():
                all_queries.append((i, val, clean_query, 'clean'))
        print(f'  总查询数: {len(all_queries)} (原始+清理)')
        uncached_queries = []
        uncached_meta = []
        for qi, (idx, val, qtext, qtype) in enumerate(all_queries):
            qlower = qtext.lower().strip()
            if qlower not in mapper.query_cache:
                uncached_queries.append(qtext)
                uncached_meta.append((qi, idx, val, qtype))
        if uncached_queries:
            print(f'    编码 {len(uncached_queries)} 个未缓存的查询...')
            new_vectors = mapper.model.encode(uncached_queries, batch_size=batch_size, convert_to_numpy=True)
            if len(new_vectors.shape) == 1:
                new_vectors = new_vectors.reshape(1, -1)
            new_vectors = new_vectors.astype(np.float32)
            new_vectors = new_vectors / np.linalg.norm(new_vectors, axis=1, keepdims=True)
            for i, (qi, idx, val, qtype) in enumerate(uncached_meta):
                qlower = uncached_queries[i].lower().strip()
                mapper.query_cache[qlower] = new_vectors[i:i + 1]
        print('  执行批量FAISS检索...')
        stacked_queries = []
        query_to_result_key = {}
        for qi, (idx, val, qtext, qtype) in enumerate(all_queries):
            qlower = qtext.lower().strip()
            if qlower in mapper.query_cache:
                stacked_queries.append(mapper.query_cache[qlower])
                query_to_result_key[len(stacked_queries) - 1] = (idx, val, qtype)
        value_to_best = {}
        if stacked_queries:
            stacked = np.vstack(stacked_queries)
            top_k = 50
            distances, indices = mapper.faiss_index.search(stacked, top_k)
            cosine_similarities = 1.0 - distances / 2.0
            for query_idx, (val_idx, val, qtype) in tqdm(query_to_result_key.items(), desc='  处理检索结果'):
                vector_indices = indices[query_idx]
                vector_sims = cosine_similarities[query_idx]
                candidates = []
                for vi, cos_sim in zip(vector_indices, vector_sims):
                    if vi == -1:
                        continue
                    concept_id = mapper.mapping_data['concept_ids'][vi]
                    fsn = mapper.mapping_data['concept_to_fsn'].get(concept_id, 'Unknown FSN')
                    original_text = mapper.mapping_data['texts'][vi]
                    candidates.append({'conceptId': str(concept_id), 'fsn': fsn, 'original_text': original_text, 'score': float(cos_sim)})
                if candidates:
                    best = mapper.rerank_by_keywords(all_queries[query_idx][2], candidates)
                    result_score = best['combined_score']
                    existing = value_to_best.get(val)
                    if existing is None or result_score > existing['score']:
                        value_to_best[val] = {'conceptId': str(best['conceptId']), 'fsn': best['fsn'], 'score': round(result_score, 4), 'source': qtype}
        raw_count = sum((1 for v in value_to_best.values() if v.get('source') == 'raw'))
        clean_count = sum((1 for v in value_to_best.values() if v.get('source') == 'clean'))
        print(f'  检索完成: 原始值最优={raw_count}, 清理后值最优={clean_count}')
        output_path = os.path.join(output_dir, f'{ds}_field_values_mapping_keyword_rerank.csv')
        print(f'  写入结果到: {output_path}')
        with open(output_path, 'w', encoding='utf-8', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['field_name', 'field_value', 'conceptId', 'fsn', 'source', 'score'])
            writer.writeheader()
            for _, row in df.iterrows():
                if row['source'] == 'exact':
                    writer.writerow({'field_name': row['field_name'], 'field_value': row['field_value'], 'conceptId': str(int(row['conceptId'])) if pd.notna(row['conceptId']) and isinstance(row['conceptId'], (int, float)) else str(row['conceptId']), 'fsn': row['fsn'], 'source': 'exact', 'score': row['score']})
                elif row['source'] == 'fuzzy':
                    mapping = value_to_best.get(row['field_value'])
                    if mapping:
                        writer.writerow({'field_name': row['field_name'], 'field_value': row['field_value'], 'conceptId': mapping['conceptId'], 'fsn': mapping['fsn'], 'source': 'retrieval', 'score': mapping['score']})
                    else:
                        writer.writerow({'field_name': row['field_name'], 'field_value': row['field_value'], 'conceptId': '', 'fsn': '', 'source': 'fuzzy', 'score': ''})
        print(f'✓ {ds} 处理完成')
    cache_path = os.path.join(cache_dir, 'query_vectors_cache.pkl')
    mapper._save_query_cache(cache_path)
    print('\n所有数据集处理完成！')
if __name__ == '__main__':
    process_all_datasets()

import os as _release_os
from release_runtime import expand_path as _release_expand, expand_config as _release_expand_config
import os
import json
import csv
import numpy as np
import faiss
import pickle
import torch
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Optional, Tuple, Set
import re

class HybridSNOMEDMapper:
    """融合字符串匹配和向量检索的SNOMED映射器"""

    def __init__(self, allconcepts_index_path: str, embedding_dir: str, model_path: str):
        """初始化映射器

        Args:
            allconcepts_index_path: allconcepts索引文件路径
            embedding_dir: 向量嵌入数据目录
            model_path: 模型路径
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.load_allconcepts_index(allconcepts_index_path)
        self.load_vector_resources(embedding_dir, model_path)
        self.cache = {}

    def load_allconcepts_index(self, index_path: str):
        """加载allconcepts索引数据"""
        print('加载allconcepts索引...')
        try:
            with open(index_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.concept_id_to_fsn = data.get('concept_id_to_fsn', {})
            self.term_to_concept_ids = data.get('term_to_concept_ids', {})
            self.fsn_to_concept_ids = {}
            for concept_id, fsn in self.concept_id_to_fsn.items():
                fsn_clean = re.sub('\\s*\\([^)]*\\)$', '', fsn)
                fsn_clean = fsn_clean.strip().strip('"\'').strip()
                fsn_clean = re.sub('\\s+', ' ', fsn_clean).lower()
                if fsn_clean not in self.fsn_to_concept_ids:
                    self.fsn_to_concept_ids[fsn_clean] = []
                self.fsn_to_concept_ids[fsn_clean].append(concept_id)
            print(f'✓ allconcepts索引加载成功: {len(self.concept_id_to_fsn)} 个概念, {len(self.term_to_concept_ids)} 个术语')
        except Exception as e:
            print(f'❌ allconcepts索引加载失败: {e}')
            raise

    def load_vector_resources(self, embedding_dir: str, model_path: str):
        """加载向量检索相关资源"""
        print('加载向量检索资源...')
        print(f'正在加载模型: {model_path}')
        self.model = SentenceTransformer(model_path, device=self.device)
        index_path = os.path.join(embedding_dir, 'snomed_rag_faiss_index.index')
        print(f'正在加载FAISS索引: {index_path}')
        self.faiss_index = faiss.read_index(index_path)
        mapping_path = os.path.join(embedding_dir, 'snomed_rag_embedding_mapping.pkl')
        print(f'正在加载映射数据: {mapping_path}')
        with open(mapping_path, 'rb') as f:
            self.mapping_data = pickle.load(f)
        if hasattr(self.faiss_index, 'nprobe'):
            self.faiss_index.nprobe = 10
        print('✓ 向量检索资源加载完成')

    def extract_keywords(self, text: str) -> Tuple[Set[str], Set[str]]:
        """提取关键词，返回(正向关键词, 否定关键词)"""
        negation_words = {'without', 'no', 'not', 'non', 'never', 'none', 'absence', 'denied', 'negative'}
        medical_stopwords = {'the', 'and', 'or', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'as', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'shall'}
        text_clean = re.sub('[^a-zA-Z0-9\\s]', ' ', text.lower())
        words = text_clean.split()
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
        """计算关键词匹配分数"""
        if not query_keywords:
            return 0.0
        coverage = 0
        for q_kw in query_keywords:
            if q_kw in fsn_keywords:
                coverage += 1
            else:
                for f_kw in fsn_keywords:
                    if q_kw in f_kw or f_kw in q_kw:
                        coverage += 0.5
                        break
        keyword_score = coverage / len(query_keywords)
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

    def string_match(self, fsn_text: str) -> Optional[Dict]:
        """字符串精确匹配（不区分FSN和同义词）

        Args:
            fsn_text: FSN文本（已去除括号后缀）
        """
        if not fsn_text:
            return None
        fsn_clean = fsn_text.strip().strip('"\'').strip()
        fsn_clean = re.sub('\\s+', ' ', fsn_clean).lower()
        if fsn_clean in self.term_to_concept_ids:
            concept_ids = self.term_to_concept_ids[fsn_clean]
            concept_id = concept_ids[0]
            return {'conceptId': concept_id, 'fsn': self.concept_id_to_fsn.get(concept_id, ''), 'source': 'exact', 'score': 1.0}
        return None

    def vector_retrieve(self, query_text: str, top_k: int=20) -> List[Dict]:
        """向量检索"""
        if not query_text:
            return []
        query_embedding = self.model.encode([query_text], convert_to_tensor=True, device=self.device)
        query_embedding = query_embedding.cpu().numpy().astype(np.float32)
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        distances, indices = self.faiss_index.search(query_embedding, top_k)
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < 0 or idx >= len(self.mapping_data['texts']):
                continue
            vector_score = 1.0 - distance / 2.0
            candidate_text = self.mapping_data['texts'][idx]
            concept_id = self.mapping_data['concept_ids'][idx]
            fsn = self.mapping_data['concept_to_fsn'].get(concept_id, '')
            query_keywords, query_negated = self.extract_keywords(query_text)
            candidate_keywords, _ = self.extract_keywords(candidate_text)
            keyword_score = self.calculate_keyword_score(query_keywords, candidate_keywords, query_negated)
            combined_score = vector_score * 0.8 + keyword_score * 0.2
            results.append({'conceptId': concept_id, 'fsn': fsn, 'source': 'retrieval', 'score': combined_score, 'vector_score': vector_score, 'keyword_score': keyword_score})
        return results

    def map_fsn(self, fsn_text: str) -> Dict:
        """映射FSN文本

        Args:
            fsn_text: FSN文本（需要去除括号后缀）
        """
        if not fsn_text:
            return {'conceptId': '', 'fsn': '', 'source': '', 'score': ''}
        fsn_clean = re.sub('\\s*\\([^)]*\\)$', '', fsn_text)
        fsn_clean = fsn_clean.strip().strip('"\'').strip()
        fsn_clean = re.sub('\\s+', ' ', fsn_clean)
        cache_key = fsn_clean.lower()
        if cache_key in self.cache:
            return self.cache[cache_key]
        exact_match = self.string_match(fsn_clean)
        if exact_match:
            self.cache[cache_key] = exact_match
            return exact_match
        retrieval_results = self.vector_retrieve(fsn_clean)
        if retrieval_results:
            best_match = max(retrieval_results, key=lambda x: x['score'])
            self.cache[cache_key] = best_match
            return best_match
        result = {'conceptId': '', 'fsn': '', 'source': '', 'score': ''}
        self.cache[cache_key] = result
        return result

def process_mapping_file(input_path: str, output_path: str, mapper: HybridSNOMEDMapper):
    """处理mapping.csv文件，忽略原始conceptId，重新匹配"""
    print(f'处理文件: {input_path}')
    rows = []
    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    print(f'读取到 {len(rows)} 行数据')
    processed_rows = []
    for i, row in enumerate(rows):
        if i % 100 == 0:
            print(f'处理进度: {i}/{len(rows)}')
        original_fsn = row['fsn']
        query_text = re.sub('\\s*\\([^)]*\\)$', '', original_fsn).strip() if original_fsn else row['field_value']
        result = mapper.map_fsn(query_text)
        processed_row = {'dataset': row['dataset'], 'field_name': row['field_name'], 'field_value': row['field_value'], 'conceptId': result.get('conceptId', ''), 'fsn': result.get('fsn', ''), 'source': result.get('source', 'hybrid'), 'score': str(result.get('score', ''))}
        processed_rows.append(processed_row)
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['dataset', 'field_name', 'field_value', 'conceptId', 'fsn', 'source', 'score'])
        writer.writeheader()
        writer.writerows(processed_rows)
    print(f'处理完成，结果保存到: {output_path}')
if __name__ == '__main__':
    allconcepts_index_path = _release_expand('${ALIGNMENT_WORK_DIR}/data/processed/snomed_allconcepts/snomed_allconcepts_optimized_index.json')
    embedding_dir = _release_expand('${ALIGNMENT_WORK_DIR}/data/embeddings/snomed_allconcepts')
    model_path = _release_expand('${ALIGNMENT_WORK_DIR}/models/transformers/models--medicalai--ClinicalBERT/snapshots/f7c7f65227cb311f33a79c24858d875876d478ac')
    input_path = _release_expand('${ALIGNMENT_WORK_DIR}/data/mapping/llm_process/mappingv4.csv')
    output_path = _release_expand('${ALIGNMENT_WORK_DIR}/data/mapping/llm_process/mappingv4_corrected.csv')
    print('初始化混合SNOMED映射器...')
    mapper = HybridSNOMEDMapper(allconcepts_index_path, embedding_dir, model_path)
    process_mapping_file(input_path, output_path, mapper)
    print('处理完成！')

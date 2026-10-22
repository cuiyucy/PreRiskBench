"""
DeepSeek API 批量处理器 v3 - 按字段名分组批量处理
核心优化:
  - 同一字段名的值合并为一次API调用 (利用上下文关系)
  - 输出精简: field_value, conceptId, fsn, source, score
"""
import os as _release_os
from release_runtime import expand_path as _release_expand, expand_config as _release_expand_config
import os
import csv
import json
import time
import logging
import argparse
import signal
import threading
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
try:
    import openai
except ImportError:
    openai = None
try:
    import yaml
except ImportError:
    yaml = None
try:
    import numpy as np
    import faiss
    HAS_VECTOR = True
except ImportError:
    HAS_VECTOR = False
try:
    from sentence_transformers import SentenceTransformer
    HAS_TRANSFORMER = True
except ImportError:
    HAS_TRANSFORMER = False

@dataclass
class ProcessingStats:
    total_records: int = 0
    processed_records: int = 0
    success_records: int = 0
    failed_records: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def update(self, processed=0, success=0, failed=0):
        with self._lock:
            self.processed_records += processed
            self.success_records += success
            self.failed_records += failed

def load_config(config_path: str=None) -> Dict:
    if yaml is None:
        return {}
    if config_path is None:
        script_dir = Path(__file__).parent
        project_root = script_dir.parent.parent
        config_path = project_root / 'config' / 'config.yaml'
    config_file = Path(config_path)
    if not config_file.exists():
        return {}
    with open(config_file, 'r', encoding='utf-8') as f:
        config = _release_expand_config(yaml.safe_load(f))
    return config or {}

class RAGRetriever:
    """SNOMED CT RAG 向量检索器"""

    def __init__(self, embedding_dir: str=None, logger: logging.Logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.model = None
        self.faiss_index = None
        self.texts = []
        self.embedding_dir = embedding_dir
        if not HAS_VECTOR or not HAS_TRANSFORMER:
            self.logger.error('faiss/sentence_transformers 未安装')
            return
        self._load_resources()

    def _load_resources(self):
        project_root = _release_expand('${ALIGNMENT_WORK_DIR}')
        if self.embedding_dir is None:
            self.embedding_dir = os.path.join(project_root, 'data', 'embeddings', 'snomed_allconcepts')
        texts_path = os.path.join(project_root, 'data', 'processed', 'snomed_allconcepts', 'snomed_allconcepts_rag_texts.json')
        index_path = os.path.join(self.embedding_dir, 'snomed_rag_faiss_index.index')
        self.logger.info(f'加载 RAG 文本: {texts_path}')
        with open(texts_path, 'r', encoding='utf-8') as f:
            self.texts = json.load(f)
        self.logger.info(f'  已加载 {len(self.texts):,} 条 SNOMED CT 记录')
        self.logger.info(f'加载 FAISS 索引: {index_path}')
        self.faiss_index = faiss.read_index(index_path)
        if hasattr(self.faiss_index, 'nprobe'):
            self.faiss_index.nprobe = 10
        model_path = os.path.join(project_root, 'models', 'transformers', 'models--medicalai--ClinicalBERT', 'snapshots', 'f7c7f65227cb311f33a79c24858d875876d478ac')
        self.logger.info('加载 ClinicalBERT 模型')
        self.model = SentenceTransformer(model_path)
        self.logger.info('✓ RAG 检索器初始化完成')

    def search_batch(self, queries: List[str], top_k: int=5) -> Dict[str, List[Dict]]:
        """批量向量检索"""
        if not self.model or not self.faiss_index or (not queries):
            return {q: [] for q in queries}
        valid = [(i, q.strip()) for i, q in enumerate(queries) if q and q.strip()]
        if not valid:
            return {q: [] for q in queries}
        try:
            q_texts = [q for _, q in valid]
            query_vectors = self.model.encode(q_texts, convert_to_numpy=True)
            if len(query_vectors.shape) == 1:
                query_vectors = query_vectors.reshape(1, -1)
            query_vectors = query_vectors.astype(np.float32)
            norms = np.linalg.norm(query_vectors, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            query_vectors = query_vectors / norms
            distances, indices = self.faiss_index.search(query_vectors, top_k)
            cosine_sims = 1.0 - distances / 2.0
            results = {q: [] for q in queries}
            for (orig_idx, _), dist_row, sim_row in zip(valid, indices, cosine_sims):
                candidates = []
                seen = set()
                for idx, cos_sim in zip(dist_row, sim_row):
                    if idx == -1 or idx >= len(self.texts):
                        continue
                    item = self.texts[idx]
                    cid = str(item.get('conceptId', ''))
                    if cid in seen:
                        continue
                    seen.add(cid)
                    candidates.append({'conceptId': cid, 'text': item.get('text', ''), 'score': round(float(cos_sim), 4)})
                results[queries[orig_idx]] = candidates[:top_k]
            return results
        except Exception as e:
            self.logger.error(f'批量RAG检索失败: {e}')
            return {q: [] for q in queries}

class DeepSeekBatchProcessorV3:
    """
    DeepSeek 批量处理器 v3 - 按字段名分组

    处理流程:
      1. 按字段名分组输入数据
      2. 每组分批 (默认20条/批)
      3. 对每批执行: RAG检索 → LLM批量判定 → 解析结果
    """
    OUTPUT_FIELDS = ['dataset', 'field_name', 'field_value', 'conceptId', 'fsn', 'source', 'score']

    def __init__(self, config: Dict, input_file: str, output_dir: str=None):
        llm_config = config.get('llm', {})
        self.api_key = llm_config.get('api_key', '')
        self.model = llm_config.get('model', 'deepseek-v4-pro')
        self.temperature = llm_config.get('temperature', 0.1)
        self.max_tokens = llm_config.get('max_tokens', 8192)
        self.timeout = llm_config.get('timeout', 300)
        self.reasoning_effort = llm_config.get('reasoning_effort', 'high')
        rag_config = config.get('rag', {})
        self.rag_top_k = rag_config.get('top_k', 10)
        self.rag_enabled = rag_config.get('enabled', True)
        self.batch_size = 20
        self.workers = 3
        self.max_retries = 3
        self.resume = True
        self.input_file = input_file
        self.output_dir = output_dir or _release_expand('${ALIGNMENT_WORK_DIR}/data/mapping/llm_process')
        self.output_file = os.path.join(self.output_dir, 'llm_mapping.csv')
        self.failed_file = os.path.join(self.output_dir, 'llm_failed.csv')
        self.progress_file = os.path.join(self.output_dir, '.v3_progress.json')
        os.makedirs(self.output_dir, exist_ok=True)
        self.stats = ProcessingStats()
        self.processed_keys = set()
        self._keys_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._setup_logging()
        if openai:
            api_url = llm_config.get('api_url')
            if not api_url:
                raise ValueError('Configure llm.api_url for the intended API endpoint')
            self.client = openai.OpenAI(api_key=self.api_key, base_url=api_url, timeout=self.timeout)
        else:
            self.client = None
        self.retriever = None
        if self.rag_enabled and HAS_VECTOR and HAS_TRANSFORMER:
            self.retriever = RAGRetriever(logger=self.logger)
            if self.retriever.model is None:
                self.retriever = None

    def _setup_logging(self):
        log_file = os.path.join(self.output_dir, f"processor_v3_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler(log_file, encoding='utf-8'), logging.StreamHandler()])
        self.logger = logging.getLogger(__name__)

    def load_and_group_data(self) -> Dict[str, List[Dict]]:
        """加载数据并按字段名分组"""
        groups = defaultdict(list)
        with open(self.input_file, 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                key = row.get('field_name', '')
                groups[key].append({'dataset': row.get('dataset', ''), 'field_name': key, 'field_value': row.get('field_value', ''), '_idx': len(groups[key])})
        total = sum((len(v) for v in groups.values()))
        self.stats.total_records = total
        self.logger.info(f'加载数据完成: {total} 条 | {len(groups)} 个字段组')
        for fn, records in sorted(groups.items(), key=lambda x: -len(x[1])):
            self.logger.info(f'  {fn}: {len(records)} 条')
        return dict(groups)

    def _call_llm(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """调用 DeepSeek API"""
        if not self.client:
            return None
        for attempt in range(self.max_retries):
            try:
                params = {'model': self.model, 'messages': [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}], 'temperature': self.temperature, 'max_tokens': self.max_tokens}
                if self.reasoning_effort:
                    params['reasoning_effort'] = self.reasoning_effort
                response = self.client.chat.completions.create(**params)
                content = (response.choices[0].message.content or '').strip()
                if content:
                    return content
            except Exception as e:
                self.logger.warning(f'  ✗ API失败(尝试{attempt + 1}): {str(e)[:100]}')
                if attempt < self.max_retries - 1:
                    time.sleep((attempt + 1) * 5)
        return None

    def _expand_queries_with_llm(self, field_name: str, field_values: List[str]) -> Dict[str, str]:
        """使用 LLM 生成 SNOMED 检索优化 query

        利用 LLM 对 SNOMED CT 的知识，将原始医学术语转换为更适合向量检索的查询
        """
        values_text = '\n'.join([f'{i + 1}. {v}' for i, v in enumerate(field_values)])
        prompt = f'字段类型: {field_name}\n\n待映射的医学术语:\n{values_text}\n\n请为每个术语生成1-2个适合SNOMED CT检索的英文query。\n要求:\n- 去除ICD-10特有后缀(not elsewhere classified, sequela, encounter等)\n- 使用SNOMED CT标准术语风格\n- 保留核心医学术语\n\n返回JSON格式:\n{{"queries": ["query1", "query2", ...]}}'
        response = self._call_llm('SNOMED CT术语专家。将ICD-10/临床术语转换为SNOMED CT检索用query。只返回简洁JSON。', prompt)
        if not response:
            return {}
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                data = json.loads(response[start:end])
                queries = data.get('queries', [])
                result = {}
                for i, v in enumerate(field_values):
                    if i < len(queries) and queries[i]:
                        result[v] = queries[i]
                    else:
                        result[v] = v
                return result
        except Exception as e:
            self.logger.warning(f'  Query expansion 解析失败: {e}')
        return {v: v for v in field_values}

    def process_field_batch(self, field_name: str, batch: List[Dict]) -> Tuple[List[Dict], List[str]]:
        """
        处理同一字段名的一批值 (三步流程):
          1. LLM Query Expansion - 生成 SNOMED 优化检索词
          2. RAG 批量检索 - 获取候选概念
          3. LLM 判定 - 选择最佳映射
        """
        field_values = [r['field_value'] for r in batch]
        self.logger.info(f'  Step1: LLM Query Expansion ({len(field_values)} 条)...')
        expanded_queries = self._expand_queries_with_llm(field_name, field_values)
        search_values = []
        for fv in field_values:
            expanded = expanded_queries.get(fv, fv)
            if expanded != fv:
                search_values.append(expanded)
            else:
                search_values.append(fv)
        self.logger.info(f'  Step2: RAG批量检索 (top_k={self.rag_top_k})...')
        rag_results = self.retriever.search_batch(search_values, top_k=self.rag_top_k) if self.retriever else {}
        cand_lines = []
        for i, fv in enumerate(field_values):
            cands = rag_results.get(search_values[i], [])
            if cands:
                cand_str = '; '.join([f"{c['conceptId']}({c['score']}){c['text'][:40]}" for c in cands[:8]])
                expanded_note = f' [→{search_values[i][:30]}]' if search_values[i] != fv else ''
                cand_lines.append(f'{i + 1}. [{fv}]{expanded_note} → {cand_str}')
            else:
                cand_lines.append(f'{i + 1}. [{fv}] → (无候选)')
        prompt = f'字段名: {field_name}\n\n待映射值及RAG候选(Top {self.rag_top_k}):\n{chr(10).join(cand_lines)}\n\n请为每个值选择最佳SNOMED CT概念。只返回JSON数组。\n即使不确定也给出最佳猜测，用低置信度(0.3-0.6)标记不确定的结果。\n\n返回格式:\n[\n  {{"v":"原值","id":"概念ID或空","fsn":"FSN或空","s":0.0-1.0}},\n  ...\n]\n\n规则:\n- id为数字SNOMED CT ID，必须与候选列表或SNOMED CT标准概念匹配\n- 完全无法确定时id留空字符串，但优先给出最佳猜测\n- s为置信度(0-1)，>0.85可靠，0.5-0.85可能，<0.5只是猜测\n- 保持原值不变，FSN含逗号不需要额外引号\n- 数组顺序与输入一致'
        response = self._call_llm('SNOMED CT医疗术语映射专家。根据RAG候选和医学知识，为每个字段值找到最准确的SNOMED CT概念映射。无法确定时也给出最佳猜测并用低置信度标记。只返回JSON数组。', prompt)
        results = []
        failed_values = []
        if response:
            try:
                start = response.find('[')
                end = response.rfind(']') + 1
                if start >= 0 and end > start:
                    parsed = json.loads(response[start:end])
                    for i, r in enumerate(batch):
                        if i < len(parsed):
                            p = parsed[i]
                            cid = str(p.get('id', '')).strip()
                            score = float(p.get('s', 0))
                            result = {'dataset': r['dataset'], 'field_name': r['field_name'], 'field_value': r['field_value'], 'conceptId': cid, 'fsn': p.get('fsn', ''), 'source': 'llm', 'score': score}
                            if cid and score >= 0.3:
                                results.append(result)
                            elif cid and score < 0.3:
                                results.append(result)
                                failed_values.append(r['field_value'])
                            else:
                                failed_values.append(r['field_value'])
                        else:
                            failed_values.append(r['field_value'])
                    return (results, failed_values)
            except Exception as e:
                self.logger.error(f'  JSON解析失败: {e}')
        for r in batch:
            failed_values.append(r['field_value'])
        return (results, failed_values)

    def process_field_group(self, field_name: str, records: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """处理一个字段组的所有记录"""
        all_results = []
        all_failed = []
        n_batches = (len(records) + self.batch_size - 1) // self.batch_size
        for b_idx in range(n_batches):
            start = b_idx * self.batch_size
            end = min(start + self.batch_size, len(records))
            batch = records[start:end]
            batch_keys = tuple(((r['field_name'], r['field_value']) for r in batch))
            with self._keys_lock:
                if self.resume and all((k in self.processed_keys for k in batch_keys)):
                    continue
            self.logger.info(f'\n  批次 {b_idx + 1}/{n_batches} ({len(batch)} 条)')
            t_start = time.time()
            results, failed_vals = self.process_field_batch(field_name, batch)
            elapsed = time.time() - t_start
            all_results.extend(results)
            with self._keys_lock:
                for r in batch:
                    if r['field_value'] in failed_vals:
                        all_failed.append(r)
                    self.processed_keys.add((r['field_name'], r['field_value']))
                self.stats.update(processed=len(batch), success=len(results), failed=len(failed_vals))
            status = f'成功={len(results)}, 失败={len(failed_vals)}'
            self.logger.info(f'  ✓ 完成 ({elapsed:.1f}s): {status}')
            self._save_progress()
            time.sleep(0.5)
        return (all_results, all_failed)

    def _write_csv(self, records: List[Dict], filepath: str, mode: str='a'):
        if not records:
            return
        with self._write_lock:
            write_header = not os.path.exists(filepath) or mode == 'w'
            with open(filepath, mode, newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.OUTPUT_FIELDS, extrasaction='ignore')
                if write_header:
                    writer.writeheader()
                writer.writerows(records)

    def _save_progress(self):
        try:
            with self._keys_lock:
                keys_snapshot = list(self.processed_keys)
                stats_snapshot = {'total_records': self.stats.total_records, 'processed_records': self.stats.processed_records, 'success_records': self.stats.success_records, 'failed_records': self.stats.failed_records}
            progress_data = {'processed_keys': [list(k) for k in keys_snapshot], 'stats': stats_snapshot, 'timestamp': datetime.now().isoformat()}
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(progress_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f'保存进度失败: {e}')

    def _load_progress(self):
        if self.resume and os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                keys = data.get('processed_keys', [])
                self.processed_keys = set((tuple(k) for k in keys))
                stats = data.get('stats', {})
                self.stats.processed_records = stats.get('processed_records', 0)
                self.stats.success_records = stats.get('success_records', 0)
                self.stats.failed_records = stats.get('failed_records', 0)
                self.logger.info(f'恢复进度: 已处理 {len(self.processed_keys)} 条')
            except Exception as e:
                self.logger.warning(f'加载进度失败: {e}')

    def process_all(self):
        mode_str = 'RAG+LLM批量' if self.retriever else '纯LLM'
        print('=' * 60)
        print('DeepSeek 批量处理器 v3.1 (并行处理)')
        print('=' * 60)
        print(f'模式: {mode_str} | 模型: {self.model} | 思考: {self.reasoning_effort}')
        print(f'每批大小: {self.batch_size} | 并发数: {self.workers} | RAG Top-K: {self.rag_top_k}' if self.retriever else f'每批大小: {self.batch_size} | 并发数: {self.workers}')
        print(f'输入: {self.input_file}')
        print(f'输出: {self.output_file}')
        print('=' * 60 + '\n')
        self._load_progress()
        groups = self.load_and_group_data()
        if not groups:
            self.logger.error('没有可处理的数据')
            return

        def signal_handler(signum, frame):
            self.logger.info('\n保存进度中...')
            self._save_progress()
            exit(0)
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        field_names = sorted(groups.keys(), key=lambda x: -len(groups[x]))

        def process_single_field(g_idx, field_name):
            try:
                records = groups[field_name]
                with self._keys_lock:
                    already_done = sum((1 for r in records if (r['field_name'], r['field_value']) in self.processed_keys))
                if self.resume and already_done == len(records):
                    self.logger.info(f'\n[{g_idx + 1}/{len(field_names)}] {field_name} (已跳过)')
                    return ([], [])
                self.logger.info(f"\n{'=' * 50}")
                self.logger.info(f'[{g_idx + 1}/{len(field_names)}] 字段: {field_name} ({len(records)} 条)')
                self.logger.info(f"{'=' * 50}")
                success, failed = self.process_field_group(field_name, records)
                self._write_csv(success, self.output_file)
                self._write_csv(failed, self.failed_file)
                self.logger.info(f'\n  字段完成: 成功={len(success)}, 失败={len(failed)}')
                return (success, failed)
            except Exception as e:
                self.logger.error(f'字段 [{field_name}] 处理异常: {e}', exc_info=True)
                return ([], [])
        pending_fields = [(i, fn) for i, fn in enumerate(field_names)]
        if self.workers > 1 and len(pending_fields) > 1:
            self.logger.info(f'\n启动并行处理: {self.workers} 个工作线程\n')
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                futures = {}
                for g_idx, field_name in pending_fields:
                    future = executor.submit(process_single_field, g_idx, field_name)
                    futures[future] = (g_idx, field_name)
                for future in as_completed(futures):
                    g_idx, field_name = futures[future]
                    try:
                        future.result()
                    except Exception as e:
                        self.logger.error(f'字段 [{field_name}] 处理异常: {e}')
        else:
            for g_idx, field_name in pending_fields:
                process_single_field(g_idx, field_name)
        self._print_summary()

    def _print_summary(self):
        self.logger.info('\n' + '=' * 60)
        self.logger.info('处理完成')
        self.logger.info('=' * 60)
        self.logger.info(f'总数: {self.stats.total_records} | 成功: {self.stats.success_records} | 失败: {self.stats.failed_records}')
        rate = self.stats.success_records / max(1, self.stats.processed_records) * 100
        self.logger.info(f'成功率: {rate:.1f}%')
        self.logger.info('-' * 40)
        self.logger.info(f'输出: {self.output_file}')
        self.logger.info(f'失败: {self.failed_file}')
        self.logger.info('=' * 60)

def main():
    parser = argparse.ArgumentParser(description='DeepSeek 批量处理器 v3.1 (并行处理)')
    parser.add_argument('--input-file', '-i', required=True, help='输入文件路径')
    parser.add_argument('--output-dir', '-o', default=None, help='输出目录')
    parser.add_argument('--config-path', '-c', default=None, help='配置文件路径')
    parser.add_argument('--no-resume', action='store_true', help='禁用断点续传')
    parser.add_argument('--batch-size', '-b', type=int, default=40, help='每批处理数量 (默认40)')
    parser.add_argument('--workers', '-w', type=int, default=3, help='并行工作线程数 (默认3)')
    args = parser.parse_args()
    config = load_config(args.config_path)
    if not config:
        print('错误: 无法加载配置文件')
        return 1
    processor = DeepSeekBatchProcessorV3(config=config, input_file=args.input_file, output_dir=args.output_dir)
    if args.no_resume:
        processor.resume = False
    if args.batch_size:
        processor.batch_size = args.batch_size
    if args.workers:
        processor.workers = args.workers
    processor.process_all()
    return 0
if __name__ == '__main__':
    exit(main())

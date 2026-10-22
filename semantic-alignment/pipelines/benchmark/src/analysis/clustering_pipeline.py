"""
SNOMED 概念语义聚类管道（v2）

改进:
  1. 保留 SNOMED 语义类型标签，按类型分层聚类
  2. extract_core 保留侧性(laterality)和临床限定词，减少过度剥离
  3. 代表概念按频率优先选择；同频时按概念 ID 字典序确定
  4. 支持 HDBSCAN / MiniBatchKMeans 两种聚类算法

输出:
  mapping/final_clustered/cid_to_centroid.json  — conceptId → 聚类代表conceptId
  mapping/final_clustered/cluster_stats.json    — 聚类统计信息
"""
import os as _release_os
from release_runtime import expand_path as _release_expand, expand_config as _release_expand_config
import sys, os, csv, re, json, pickle, argparse
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
SNOMED_DIR = Path(_release_expand('${ALIGNMENT_WORK_DIR}/data/raw/snomed_ct'))
MAPPING_DIR = Path(_release_expand('${ALIGNMENT_WORK_DIR}/data/mapping/final_output'))
OUTPUT_DIR = Path(_release_expand('${BENCHMARK_WORK_DIR}/mapping/final_clustered'))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BERT_CACHE = OUTPUT_DIR / 'core_embeddings_v2.pkl'
NER_CACHE = OUTPUT_DIR / 'core_cache_v3.pkl'
_ner_pipeline = None
_NER_KEEP_TYPES = {'Medication', 'Disease_disorder', 'Sign_symptom', 'Therapeutic_procedure', 'Diagnostic_procedure', 'Biological_structure'}

def _get_ner_pipeline():
    """懒加载模型；如需自定义下载端点，由调用者设置 HF_ENDPOINT。"""
    global _ner_pipeline
    if _ner_pipeline is None:
        from transformers import pipeline
        print('  加载 NER 模型 (d4data/biomedical-ner-all)...')
        _ner_pipeline = pipeline('ner', model='d4data/biomedical-ner-all', aggregation_strategy='simple')
    return _ner_pipeline
SEMANTIC_TYPE_PATTERN = re.compile('\\((disorder|procedure|substance|finding|medicinal product|clinical drug|product|morphologic abnormality|body structure|organism|qualifier value|regime/therapy|situation|event|observable entity|physical object|specimen|staging scale|cell|cell structure|tumor staging|person|environment|social concept|special concept|navigational concept|record artifact|assessment scale|occupation|ethnic group|religion|physical force)\\)', re.IGNORECASE)
TYPE_GROUP_MAP = {'disorder': 'DISORDER', 'finding': 'FINDING', 'procedure': 'PROCEDURE', 'regime/therapy': 'PROCEDURE', 'substance': 'SUBSTANCE', 'medicinal product': 'PRODUCT', 'clinical drug': 'PRODUCT', 'product': 'PRODUCT', 'morphologic abnormality': 'MORPH', 'body structure': 'BODY', 'cell': 'BODY', 'cell structure': 'BODY', 'organism': 'ORGANISM', 'qualifier value': 'OTHER', 'situation': 'OTHER', 'event': 'OTHER', 'observable entity': 'OTHER', 'physical object': 'OTHER', 'specimen': 'OTHER', 'staging scale': 'OTHER', 'tumor staging': 'OTHER', 'assessment scale': 'OTHER', 'person': 'OTHER', 'environment': 'OTHER', 'social concept': 'OTHER', 'special concept': 'OTHER', 'navigational concept': 'OTHER', 'record artifact': 'OTHER', 'occupation': 'OTHER', 'ethnic group': 'OTHER', 'religion': 'OTHER', 'physical force': 'OTHER'}
EXTRA_SYNONYMS = {'obesity': 'obesity', 'obese': 'obesity', 'overweight': 'obesity', 'carcinoma': 'cancer', 'cancer': 'cancer', 'malignant neoplasm': 'cancer', 'malignant tumor': 'cancer', 'sarcoma': 'cancer', 'renal failure': 'renal failure', 'kidney failure': 'renal failure', 'renal disease': 'renal disease', 'kidney disease': 'renal disease', 'chronic kidney disease': 'chronic kidney disease', 'congestive heart failure': 'heart failure', 'cardiac failure': 'heart failure', 'cerebrovascular accident': 'stroke', 'cva': 'stroke', 'coronary artery disease': 'coronary artery disease', 'coronary heart disease': 'coronary artery disease', 'urinary tract infection': 'urinary tract infection', 'uti': 'urinary tract infection', 'acetaminophen': 'paracetamol', 'paracetamol': 'paracetamol', 'adrenaline': 'epinephrine', 'epinephrine': 'epinephrine', 'lidocaine': 'lidocaine', 'lignocaine': 'lidocaine', 'frusemide': 'furosemide', 'operation': 'surgery', 'surgery': 'surgery', 'surgical procedure': 'surgery', 'excision': 'excision', 'resection': 'excision', 'removal': 'excision'}

def get_semantic_type(fsn: str) -> str:
    """从 FSN 中提取语义类型"""
    if not fsn:
        return 'UNKNOWN'
    m = SEMANTIC_TYPE_PATTERN.search(fsn)
    if m:
        raw_type = m.group(1).strip().lower()
        return TYPE_GROUP_MAP.get(raw_type, 'OTHER')
    return 'UNKNOWN'

def load_snomed_synonyms():
    path = SNOMED_DIR / 'sct2_Description_Snapshot-en_INT_20260201.txt'
    fsn_type = '900000000000003001'
    synonym_type = '900000000000013009'
    cid_to_terms = defaultdict(set)
    import sys as _sys
    csv.field_size_limit(_sys.maxsize)
    with open(path, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f, delimiter='\t'):
            if row['active'] == '1' and row['typeId'] in (fsn_type, synonym_type):
                term = re.sub('\\s*\\([^)]*\\)\\s*$', '', row['term']).strip().lower()
                cid_to_terms[row['conceptId']].add(term)
    synonym_map = {}
    for cid, terms in cid_to_terms.items():
        canonical = min(terms, key=len)
        for t in terms:
            if t not in synonym_map:
                synonym_map[t] = canonical
    return synonym_map

def extract_core(text: str) -> str:
    """
    用生物医学NER模型提取概念核心，替换正则方案。

    Step 1: 剥离语义类型后缀（如 "(disorder)"），后续分层聚类使用
    Step 2: 用 d4data/biomedical-ner-all 提取核心医学实体
    Step 3: 只保留 Medication/Disease_disorder/Sign_symptom/
            Therapeutic_procedure/Diagnostic_procedure/Biological_structure
            过滤 Dosage/Administration/Detailed_description 等模板词
    Step 4: 合并子词token (##cetamol → cetamol)
    Step 5: 加回语义类型后缀
    """
    if not text:
        return ''
    bare = text.strip().lower()
    type_suffix = ''
    tm = SEMANTIC_TYPE_PATTERN.search(bare)
    if tm:
        type_suffix = ' (' + tm.group(1).strip().lower() + ')'
        bare = bare[:tm.start()] + bare[tm.end():]
    bare = re.sub('\\s*\\([^)]*\\)\\s*', ' ', bare).strip()
    bare = re.sub('\\s+', ' ', bare)
    if not bare:
        return type_suffix if type_suffix else ''
    sem_group = TYPE_GROUP_MAP.get(tm.group(1).strip().lower(), 'OTHER') if tm else 'UNKNOWN'
    if sem_group == 'PRODUCT':
        keep_types = {'Medication'}
    else:
        keep_types = _NER_KEEP_TYPES | {'Detailed_description', 'Coreference'}
    try:
        ner = _get_ner_pipeline()
        entities = ner(bare)
    except Exception:
        result = bare + type_suffix if type_suffix else bare
        return result
    merged = []
    buf = ''
    buf_type = None
    for e in entities:
        if e.get('score', 0) < 0.3:
            continue
        word = e['word']
        etype = e['entity_group']
        if word.startswith('##'):
            buf += word[2:]
        else:
            if buf and buf_type in keep_types:
                merged.append(buf)
            buf = word
            buf_type = etype
    if buf and buf_type in keep_types:
        merged.append(buf)
    if merged:
        core = ' '.join(merged)
    else:
        core = bare
    if type_suffix:
        core += type_suffix
    return core

def normalize(text: str, snomed_syns: dict) -> str:
    """同义词归一化：将文本中的词替换为标准形式"""
    if not text:
        return text
    words = text.split()
    result = []
    for w in words:
        if w in snomed_syns:
            result.append(snomed_syns[w])
        elif w in EXTRA_SYNONYMS:
            result.append(EXTRA_SYNONYMS[w])
        else:
            result.append(w)
    return ' '.join(result)

def build_frequency_map(all_mapping_entries: list) -> dict:
    """
    统计每个 conceptId 在映射数据中的出现次数
    用于质心选择时偏向高频概念
    """
    freq = Counter()
    for entry in all_mapping_entries:
        freq[entry['conceptId']] += 1
    return dict(freq)

def select_centroid_by_frequency(indices, core_list, core_to_cids, freq_map, embeddings=None):
    """Choose the highest-frequency mapped concept within one cluster.

    Lexical concept-ID ordering makes ties deterministic for a new reconstruction.
    The frozen artifact hash remains authoritative for the paper run.
    """
    candidates = []
    for idx in indices:
        core = core_list[idx]
        for cid in core_to_cids.get(core, set()):
            candidates.append((-freq_map.get(cid, 0), str(cid), core, cid))
    if not candidates:
        return (None, None)
    _, _, best_core, best_cid = min(candidates)
    return (best_core, best_cid)

def cluster_one_type(core_list: list, embeddings: np.ndarray, core_to_cids: dict, freq_map: dict, mode: str, k_value: int=None, hdbscan_min_size: int=5) -> dict:
    """
    对单个语义类型的概念做聚类

    Args:
        core_list: 该类型的核心文本列表
        embeddings: 对应的 BERT 嵌入
        core_to_cids: core文本 -> conceptId集合
        freq_map: conceptId -> 频率
        mode: "kmeans", "hdbscan", 或 "hybrid"(大组kmeans,小组hdbscan)
        k_value: k-means 的k值, None则自动计算
        hdbscan_min_size: HDBSCAN 最小簇大小

    Returns:
        cid_to_centroid 映射 (仅此类型)
    """
    n = len(core_list)
    if n == 0:
        return {}
    resolved_mode = mode
    if mode == 'hybrid':
        resolved_mode = 'kmeans' if n > 1000 else 'hdbscan'
    cid_to_centroid = {}
    if resolved_mode == 'hdbscan':
        import hdbscan
        clusterer = hdbscan.HDBSCAN(min_cluster_size=hdbscan_min_size, min_samples=max(1, hdbscan_min_size // 2), metric='euclidean', cluster_selection_method='leaf', allow_single_cluster=False)
        labels = clusterer.fit_predict(embeddings)
        unique_labels = set(labels)
        n_clustered = sum((1 for lbl in unique_labels if lbl != -1))
        n_noise = sum((1 for l in labels if l == -1))
        noise_pct = n_noise / n * 100 if n else 0
        for lbl in unique_labels:
            indices = np.where(labels == lbl)[0]
            if lbl == -1:
                for idx in indices:
                    cids = core_to_cids.get(core_list[idx], set())
                    centroid_cid = next(iter(cids)) if cids else None
                    if centroid_cid:
                        for c in cids:
                            cid_to_centroid[c] = centroid_cid
            else:
                best_core, best_cid = select_centroid_by_frequency(indices, core_list, core_to_cids, freq_map, embeddings)
                if best_cid:
                    for idx in indices:
                        for c in core_to_cids.get(core_list[idx], set()):
                            cid_to_centroid[c] = best_cid
        if noise_pct > 50 and n > 10:
            print(f'      ⚠ noise_rate={noise_pct:.0f}%, 回退到kmeans...')
            return cluster_one_type(core_list, embeddings, core_to_cids, freq_map, mode='kmeans', k_value=k_value)
    else:
        from sklearn.cluster import MiniBatchKMeans
        if k_value is None:
            k_value = max(2, min(n // 3, 5000))
        k_value = min(k_value, n - 1) if n > 1 else 1
        if n <= 2:
            sorted_indices = sorted(range(n), key=lambda i: freq_map.get(next(iter(core_to_cids.get(core_list[i], set()))), 0), reverse=True)
            best_core = core_list[sorted_indices[0]]
            best_cid = next(iter(core_to_cids.get(best_core, set())))
            for c in set().union(*[core_to_cids.get(core_list[i], set()) for i in range(n)]):
                cid_to_centroid[c] = best_cid
        else:
            km = MiniBatchKMeans(n_clusters=k_value, random_state=42, batch_size=max(256, n // 4), n_init=3, max_iter=300)
            labels = km.fit_predict(embeddings)
            for cluster_id in np.unique(labels):
                indices = np.where(labels == cluster_id)[0]
                if len(indices) == 0:
                    continue
                best_core, best_cid = select_centroid_by_frequency(indices, core_list, core_to_cids, freq_map, embeddings)
                if best_cid:
                    for idx in indices:
                        for c in core_to_cids.get(core_list[idx], set()):
                            cid_to_centroid[c] = best_cid
    return cid_to_centroid
_HYBRID_KMEANS_TYPES = {'DISORDER', 'FINDING'}

def run_pipeline(mode: str='hdbscan', hdbscan_min_size: int=5, global_k: int=5000, force_recompute_embeddings: bool=False):
    """
    运行语义聚类管道

    Args:
        mode: 聚类算法 — "hdbscan" 或 "kmeans"
        hdbscan_min_size: HDBSCAN 最小簇大小
        global_k: k-means 总簇数(按类型比例分配)
        force_recompute_embeddings: 是否跳过embedding缓存重新计算
    """
    print('=' * 70)
    print(f'SNOMED 概念语义聚类管道 (v2, mode={mode})')
    print('=' * 70)
    print('\n[1] 加载同义词...')
    snomed_syns = load_snomed_synonyms()
    print(f'  {len(snomed_syns)} 个 term→规范名映射')
    print('\n[2] 收集概念并提取核心+归一化+分类...')
    core_to_cids = defaultdict(set)
    core_to_type = {}
    cid_to_fsn = {}
    all_mapping_entries = []
    fsn_to_core = {}
    if NER_CACHE.exists() and (not force_recompute_embeddings):
        print('  加载 NER 核心提取缓存...')
        with open(NER_CACHE, 'rb') as f:
            fsn_to_core = pickle.load(f)
    else:
        fsn_to_core = {}
    for ds in ['vitaldb', 'mover', 'inspire']:
        with open(MAPPING_DIR / f'{ds}_final_mapping.csv') as f:
            for row in csv.DictReader(f):
                fsn = row.get('fsn', '')
                cid = row['conceptId'].strip()
                cid_to_fsn[cid] = fsn
                if fsn in fsn_to_core:
                    core = fsn_to_core[fsn]
                else:
                    core = extract_core(fsn)
                    fsn_to_core[fsn] = core
                if core:
                    normalized = normalize(core, snomed_syns)
                    core_to_cids[normalized].add(cid)
                    sem_type = get_semantic_type(fsn)
                    core_to_type[normalized] = sem_type
                    all_mapping_entries.append({'dataset': ds, 'conceptId': cid, 'core': normalized, 'field_name': row['field_name'], 'field_value': row['field_value']})
    if fsn_to_core and (not NER_CACHE.exists() or force_recompute_embeddings):
        with open(NER_CACHE, 'wb') as f:
            pickle.dump(fsn_to_core, f)
    print(f'  总概念: {len(all_mapping_entries)} 条, 唯一核心: {len(core_to_cids)} 个')
    type_counts = Counter(core_to_type.values())
    print('  语义类型分布:')
    for t, cnt in type_counts.most_common():
        print(f'    {t:<14s}: {cnt:>5d}')
    freq_map = build_frequency_map(all_mapping_entries)
    print('\n[3] BERT 嵌入...')
    if BERT_CACHE.exists() and (not force_recompute_embeddings):
        print('  加载嵌入缓存...')
        with open(BERT_CACHE, 'rb') as f:
            bert_data = pickle.load(f)
        global_core_list = bert_data.get('core_list', list(core_to_cids.keys()))
        global_embeddings = bert_data.get('embeddings', None)
        mask = np.array([c in core_to_cids for c in global_core_list])
        if not np.all(mask):
            print(f'    缓存不完全匹配,重新计算...')
            bert_data = None
        else:
            print(f'    缓存核心数: {len(global_core_list)}')
    else:
        bert_data = None
    if bert_data is None:
        global_core_list = list(core_to_cids.keys())
        sys.path.insert(0, _release_expand('${BENCHMARK_WORK_DIR}'))
        sys.path.insert(0, os.path.join(_release_expand('${BENCHMARK_WORK_DIR}'), 'src'))
        from features.bert_embedder import BERTEmbedder
        embedder = BERTEmbedder(cache_dir=str(OUTPUT_DIR / 'bert_cache'))
        if not embedder.is_loaded:
            embedder.load()
        batch_size = 128
        all_embeddings = []
        for i in range(0, len(global_core_list), batch_size):
            batch = global_core_list[i:i + batch_size]
            embs = embedder.embed_texts(batch)
            all_embeddings.append(embs)
            if i // batch_size % 10 == 0:
                print(f'    嵌入 {i}/{len(global_core_list)}...')
        global_embeddings = np.vstack(all_embeddings) if len(all_embeddings) > 1 else all_embeddings[0]
        bert_data = {'core_list': global_core_list, 'embeddings': global_embeddings}
        with open(BERT_CACHE, 'wb') as f:
            pickle.dump(bert_data, f)
    print(f'\n[4] 分层聚类 (mode={mode})...')
    type_to_cores = defaultdict(list)
    type_to_indices = defaultdict(list)
    for i, core in enumerate(global_core_list):
        sem_type = core_to_type.get(core, 'UNKNOWN')
        type_to_cores[sem_type].append(core)
        type_to_indices[sem_type].append(i)
    if mode == 'kmeans':
        total = len(global_core_list)
        type_k_alloc = {}
        for sem_type, cores in type_to_cores.items():
            type_k_alloc[sem_type] = max(1, int(global_k * len(cores) / total))
        print(f'  K-means 总k={global_k}, 类型分配: {json.dumps(type_k_alloc)}')
    full_cid_to_centroid = {}
    cluster_id = 0
    cluster_stats = {}
    for sem_type in sorted(type_to_cores.keys()):
        cores = type_to_cores[sem_type]
        indices = type_to_indices[sem_type]
        embs = global_embeddings[indices]
        n = len(cores)
        if n == 0:
            continue
        if sem_type == 'PRODUCT':
            core_to_group = {}
            for core in cores:
                if core not in core_to_group:
                    cids = core_to_cids.get(core, set())
                    best_cid = None
                    best_freq = -1
                    for c in cids:
                        f = freq_map.get(c, 0)
                        if f > best_freq:
                            best_freq = f
                            best_cid = c
                    core_to_group[core] = best_cid if best_cid else next(iter(cids)) if cids else None
            partial_map = {}
            for core in cores:
                centroid = core_to_group.get(core)
                if centroid:
                    for c in core_to_cids.get(core, set()):
                        partial_map[c] = centroid
        else:
            type_k = type_k_alloc.get(sem_type, None) if mode == 'kmeans' else None
            partial_map = cluster_one_type(cores, embs, core_to_cids, freq_map, mode=mode, k_value=type_k, hdbscan_min_size=hdbscan_min_size)
        for cid, centroid in partial_map.items():
            full_cid_to_centroid[cid] = centroid
        unique_centroids = set(partial_map.values())
        n_clusters = len(unique_centroids)
        n_concepts = len(partial_map)
        cpr = round(n_concepts / n_clusters, 1) if n_clusters else 0
        print(f'    {sem_type:<14s}: {n} cores → {n_clusters} clusters (cpr={cpr})')
        centroid_to_members = defaultdict(list)
        for cid, centroid in partial_map.items():
            centroid_to_members[centroid].append(cid)
        for centroid, members in centroid_to_members.items():
            cluster_stats[cluster_id] = {'size': len(members), 'type': sem_type, 'centroid_cid': centroid}
            cluster_id += 1
    print(f'\n[5] 保存映射...')
    with open(OUTPUT_DIR / 'cid_to_centroid.json', 'w') as f:
        json.dump(full_cid_to_centroid, f, indent=2)
    total_clusters = len(cluster_stats)
    total_concepts = len(full_cid_to_centroid)
    stats_summary = {'total_concepts': total_concepts, 'num_clusters': total_clusters, 'compression_ratio': round(total_concepts / total_clusters, 1) if total_clusters else 0, 'mode': mode, 'type_distribution': dict(type_counts), 'cluster_sizes': {'min': min((s['size'] for s in cluster_stats.values())) if cluster_stats else 0, 'max': max((s['size'] for s in cluster_stats.values())) if cluster_stats else 0, 'mean': round(sum((s['size'] for s in cluster_stats.values())) / total_clusters, 1) if total_clusters else 0}}
    with open(OUTPUT_DIR / 'cluster_stats.json', 'w') as f:
        json.dump(stats_summary, f, indent=2)
    print(f'  cid_to_centroid.json: {total_concepts} 条映射')
    print(f'  cluster_stats.json: {stats_summary}')
    print(f'\n✅ 管道完成，映射已保存至: {OUTPUT_DIR}')
    return full_cid_to_centroid
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SNOMED 概念语义聚类管道 v2')
    parser.add_argument('--mode', choices=['hdbscan', 'kmeans', 'hybrid'], default='hybrid', help='聚类算法 (默认: hybrid = 大组kmeans + 小组hdbscan)')
    parser.add_argument('--hdbscan-min-size', type=int, default=5, help='HDBSCAN 最小簇大小 (默认: 5)')
    parser.add_argument('--k', type=int, default=5000, help='KMeans 总簇数 (默认: 5000)')
    parser.add_argument('--force-embeddings', action='store_true', help='强制重新计算 BERT 嵌入')
    args = parser.parse_args()
    run_pipeline(mode=args.mode, hdbscan_min_size=args.hdbscan_min_size, global_k=args.k, force_recompute_embeddings=args.force_embeddings)

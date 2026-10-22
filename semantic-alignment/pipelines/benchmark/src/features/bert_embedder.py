"""
ClinicalBERT 嵌入模块

从容器内的 /embedding/models/transformers/models--medicalai--ClinicalBERT 加载 ClinicalBERT，
为文本字段计算语义嵌入，替代独热编码用于 Exp4-7 实验。

在 build_benchmark.py 的 --bert 模式下自动调用，
也可独立使用进行嵌入计算。

用法:
    from features.bert_embedder import BERTEmbedder

    embedder = BERTEmbedder()
    embeddings = embedder.embed_texts(["Essential hypertension", "Type 2 diabetes"])
    # -> np.ndarray shape (2, 768)
"""
import os as _release_os
from release_runtime import expand_path as _release_expand, expand_config as _release_expand_config
import os
import sys
import logging
import hashlib
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
logger = logging.getLogger('bert_embedder')
_MODEL_CANDIDATES = [_release_expand('${ALIGNMENT_WORK_DIR}/models/transformers/models--medicalai--ClinicalBERT'), _release_expand('${ALIGNMENT_WORK_DIR}/models/transformers/models--medicalai--ClinicalBERT'), 'medicalai/ClinicalBERT']

def _find_model_path() -> str:
    """查找 ClinicalBERT 模型路径"""
    for p in _MODEL_CANDIDATES:
        if p.startswith('/') and Path(p).exists():
            path = Path(p)
            snapshots = path / 'snapshots'
            if snapshots.exists():
                ref_file = path / 'refs' / 'main'
                if ref_file.exists():
                    commit = ref_file.read_text().strip()
                    snapshot_path = snapshots / commit
                    if snapshot_path.exists():
                        return str(snapshot_path)
                snapshot_dirs = sorted(snapshots.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)
                if snapshot_dirs:
                    return str(snapshot_dirs[0])
            return p
        if not p.startswith('/'):
            try:
                from huggingface_hub import snapshot_download
                return p
            except Exception:
                pass
    raise FileNotFoundError(f'ClinicalBERT not found. Tried: {_MODEL_CANDIDATES}. Set CLINICALBERT_PATH env variable.')
BERT_TEXT_COLUMNS = {'feat_surg_vars': ['preop_diagnosis', 'scheduled_procedure_name'], 'feat_past_his': ['hx_circ_detail', 'hx_resp_detail', 'hx_endo_detail', 'hx_neuro_detail', 'hx_dige_detail', 'hx_spine_detail', 'hx_immu_detail', 'hx_urine_detail', 'hx_other_detail', 'hx_surgery_detail'], 'feat_med_rec': ['med_rec_summary']}

class BERTEmbedder:
    """
    ClinicalBERT 嵌入器

    为文本列计算稠密语义向量，替代稀疏独热编码。
    缓存机制: 唯一值 → 嵌入向量，避免重复计算。
    """
    EMBEDDING_DIM = 768

    def __init__(self, model_path: Optional[str]=None, batch_size: int=64, device: str='auto', cache_dir: Optional[str]=None):
        """
        Args:
            model_path: 模型路径, None 则自动查找
            batch_size: 批量编码大小
            device: 'auto'(优先GPU), 'cpu', 'cuda'
            cache_dir: 嵌入缓存目录
        """
        self.model_path = model_path or _find_model_path()
        self.batch_size = batch_size
        self.device = device
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._model = None
        self._tokenizer = None
        self._cache: Dict[str, np.ndarray] = {}
        self._loaded = False
        self._external_vocab = None

    def set_external_vocab(self, vocab: Dict[str, List[str]]):
        """设置外部onehot词汇表（迁移实验用，优先于数据集内唯一值）"""
        self._external_vocab = vocab

    def load(self):
        """加载模型和分词器"""
        if self._loaded:
            return
        import torch
        from transformers import AutoTokenizer, AutoModel
        logger.info(f'加载 ClinicalBERT: {self.model_path}')
        if self.device == 'auto':
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self._device = torch.device(self.device)
        try:
            from transformers import DistilBertTokenizer, DistilBertModel
            self._tokenizer = DistilBertTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            self._model = DistilBertModel.from_pretrained(self.model_path, trust_remote_code=True).to(self._device)
        except Exception as e:
            logger.warning(f'DistilBert加载失败: {e}, 尝试AutoTokenizer')
            from transformers import AutoTokenizer, AutoModel
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True, use_fast=False)
            self._model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True).to(self._device)
        self._model.eval()
        self._loaded = True
        if self.cache_dir:
            self._load_cache()
        logger.info(f'ClinicalBERT 就绪 (device={self._device})')

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @staticmethod
    def _mean_pooling(model_output, attention_mask) -> np.ndarray:
        """Mean pooling over token embeddings, ignoring padding"""
        import torch
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-09)
        return (sum_embeddings / sum_mask).cpu().numpy()

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        计算文本列表的 BERT 嵌入

        Args:
            texts: 文本列表

        Returns:
            np.ndarray shape (len(texts), 768)
        """
        if not self._loaded:
            self.load()
        import torch
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch = [str(t) if t and (not (isinstance(t, float) and np.isnan(t))) else '' for t in batch]
            encoded = self._tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors='pt').to(self._device)
            with torch.no_grad():
                output = self._model(**encoded)
            embeddings = self._mean_pooling(output, encoded['attention_mask'])
            all_embeddings.append(embeddings)
        return np.vstack(all_embeddings).astype(np.float32)

    def embed_unique_values(self, series: pd.Series, column_name: str) -> Dict[str, np.ndarray]:
        """
        为 Series 中的唯一值计算嵌入（优先用缓存）

        Returns:
            {value_string: embedding_vector} 字典
        """
        unique_vals = series.dropna().astype(str).unique()
        result = {}
        uncached = []
        for v in unique_vals:
            cache_key = self._make_cache_key(column_name, v)
            if cache_key in self._cache:
                result[v] = self._cache[cache_key]
            else:
                uncached.append(v)
        if uncached:
            logger.debug(f'  [{column_name}] 计算 {len(uncached)} 个嵌入...')
            embs = self.embed_texts(list(uncached))
            for v, emb in zip(uncached, embs):
                result[v] = emb
                cache_key = self._make_cache_key(column_name, v)
                self._cache[cache_key] = emb
        return result

    @staticmethod
    def _make_cache_key(column: str, value: str) -> str:
        """生成缓存键"""
        raw = f'{column}:{value}'
        return hashlib.md5(raw.encode()).hexdigest()

    def _load_cache(self):
        if not self.cache_dir:
            return
        cache_file = self.cache_dir / 'bert_embed_cache.pkl'
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    self._cache = pickle.load(f)
                logger.info(f'加载缓存: {len(self._cache)} 条')
            except Exception as e:
                logger.warning(f'缓存加载失败: {e}')
                self._cache = {}

    def save_cache(self):
        if not self.cache_dir:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = self.cache_dir / 'bert_embed_cache.pkl'
        with open(cache_file, 'wb') as f:
            pickle.dump(self._cache, f)
        logger.info(f'缓存已保存: {len(self._cache)} 条 → {cache_file}')

    def embed_feature_dataframe(self, df: pd.DataFrame, ft: str, dataset: str='') -> Optional[pd.DataFrame]:
        """
        对单个特征 DataFrame 计算特征矩阵

        Exp4-7 处理策略:
          1. 数值/布尔列 → 保留原值
          2. 枚举类(sex/race/anesthesia_type等) → One-Hot 编码
          3. 真正文本(诊断/手术名/药物/SNOMED描述) → BERT 768维嵌入

        Args:
            df: 含 surgery_id 的原始特征 DataFrame
            ft: 特征类型键 ('feat_pat_demo', 'feat_surg_vars', ...)
            dataset: 数据集名（用于日志）

        Returns:
            特征 DataFrame（surgery_id 为索引）
        """
        from features.feature_processor import FeatureProcessor as FP
        df = df.copy()
        df = df.set_index('surgery_id')
        if 'patient_id' in df.columns:
            df = df.drop(columns=['patient_id'])
        embedded_parts = []
        enum_onehot_cols = [c for c in FP.ONEHOT_COLUMNS.get(ft, []) if c not in self._get_text_targets(ft)]
        bert_text_targets = self._get_text_targets(ft)
        all_special = set(enum_onehot_cols) | set(bert_text_targets)
        for col in df.columns:
            if col in all_special:
                continue
            if col == 'med_rec_summary':
                continue
            try:
                numeric_col = pd.to_numeric(df[col], errors='coerce').fillna(0)
                embedded_parts.append(numeric_col.to_frame().astype(np.float32))
            except Exception:
                pass
        for col in enum_onehot_cols:
            if col not in df.columns:
                continue
            unique_vals = self._get_vocab_values(col, df)
            oh_df = self._make_onehot(df, col, unique_vals)
            if oh_df is not None:
                embedded_parts.append(oh_df)
        if 'med_rec_summary' in df.columns:
            drug_emb = self._embed_drug_column(df['med_rec_summary'])
            if drug_emb is not None:
                embedded_parts.append(drug_emb)
        for col in bert_text_targets:
            if col in df.columns:
                emb_df = self._embed_single_column(df[col], col)
                if emb_df is not None:
                    embedded_parts.append(emb_df)
        if not embedded_parts:
            return None
        result = pd.concat(embedded_parts, axis=1)
        logger.debug(f"  [{dataset}/{ft}] BERT模式: {df.shape[1]}原始列→{result.shape[1]}维 (数值{len([p for p in embedded_parts if p.shape[1] < 100])}+枚举OH{len(enum_onehot_cols)}+BERT文本{len(bert_text_targets) + int('med_rec_summary' in df.columns)})")
        return result

    def _embed_single_column(self, series: pd.Series, column_name: str) -> Optional[pd.DataFrame]:
        """为单列文本值计算 BERT 嵌入"""
        val_to_emb = self.embed_unique_values(series, column_name)
        if not val_to_emb:
            return None
        dim = next(iter(val_to_emb.values())).shape[0]
        cols = [f'{column_name}_bert_{i}' for i in range(dim)]
        data = np.zeros((len(series), dim), dtype=np.float32)
        for i, val in enumerate(series.fillna('').astype(str)):
            if val in val_to_emb:
                data[i] = val_to_emb[val]
        return pd.DataFrame(data, index=series.index, columns=cols, dtype=np.float32)

    def _embed_drug_column(self, series: pd.Series) -> Optional[pd.DataFrame]:
        """为 med_rec_summary 列提取药物列表, 计算平均嵌入作为行向量"""
        from features.feature_processor import FeatureProcessor
        drug_dict = FeatureProcessor.parse_med_rec_drugs(series)
        if not drug_dict:
            return None
        dim = self.EMBEDDING_DIM
        cols = [f'med_rec_bert_{i}' for i in range(dim)]
        data = np.zeros((len(series), dim), dtype=np.float32)
        for row_idx, idx in enumerate(series.index):
            drug_names = drug_dict.get(str(idx), [])
            if not drug_names:
                continue
            drug_embs = []
            for dn in drug_names:
                cache_key = self._make_cache_key('med_drug', dn)
                if cache_key in self._cache:
                    drug_embs.append(self._cache[cache_key])
            if not drug_embs:
                embs = self.embed_texts(drug_names)
                for dn, emb in zip(drug_names, embs):
                    cache_key = self._make_cache_key('med_drug', dn)
                    self._cache[cache_key] = emb
                drug_embs = list(embs)
            data[row_idx] = np.mean(drug_embs, axis=0)
        return pd.DataFrame(data, index=series.index, columns=cols, dtype=np.float32)

    def _get_vocab_values(self, col: str, df: pd.DataFrame) -> List[str]:
        """获取onehot词汇表：优先外部vocab，否则从数据中提取"""
        if self._external_vocab and col in self._external_vocab:
            return self._external_vocab[col]
        vals = sorted(df[col].dropna().unique().astype(str))
        if len(vals) > 500:
            vals = vals[:500]
        return vals

    @staticmethod
    def _make_onehot(df: pd.DataFrame, col: str, unique_vals: List[str]) -> Optional[pd.DataFrame]:
        """生成单列的One-Hot编码DataFrame"""
        if not unique_vals:
            return None
        oh_df = pd.DataFrame(0, index=df.index, columns=[f'{col}_{v}' for v in unique_vals], dtype=np.float32)
        for v in unique_vals:
            mask = df[col].astype(str) == v
            oh_df.loc[mask, f'{col}_{v}'] = 1.0
        return oh_df

    @staticmethod
    def _get_text_targets(ft: str) -> List[str]:
        """获取需要 BERT 嵌入的列名列表"""
        return BERT_TEXT_COLUMNS.get(ft, [])

def get_bert_embedder(cache_dir: Optional[str]=None) -> BERTEmbedder:
    """单例模式获取 BERT 嵌入器"""
    return BERTEmbedder(cache_dir=cache_dir)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print('BERTEmbedder 模块加载完成')
    print(f'模型候选路径: {_MODEL_CANDIDATES}')
    print(f'文本列配置: {list(BERT_TEXT_COLUMNS.keys())}')

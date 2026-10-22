"""
实验运行器 v5 — 基于预构建 benchmark 数据

实验设计:
  Baseline   - 单数据集训练(onehot)
  Exp1       - 未对齐联合训练(onehot)
  Exp2       - 聚类后联合训练(onehot)
  Exp3       - 未对齐模型迁移(onehot, Mover→Inspire + Mover→Vitaldb)
  Exp4       - 聚类后模型迁移(onehot, Mover→Inspire + Mover→Vitaldb)
  Exp5       - BERT联合训练(未对齐)
  Exp6       - BERT迁移(未对齐, Mover→Inspire + Mover→Vitaldb)
  Exp7       - BERT迁移(聚类后, Mover→Inspire + Mover→Vitaldb)
  Exp8       - BERT联合训练(聚类后)
  T1-T6      - 统一编号迁移 (onehot/bert × unaligned/aligned/clustered, 6方向双向)
  T13        - BERT/"概念+簇"双层表示 (aligned_dual), Inspire→Mover (medoid 簇代表)

用法:
  python -m src.experiments.run_experiments --exp baseline --task asa
  python -m src.experiments.run_experiments --exp all
"""
import os as _release_os
from release_runtime import expand_path as _release_expand, expand_config as _release_expand_config
import sys
import os
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime
import pandas as pd
import numpy as np
import yaml
SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
sys.path.insert(0, os.path.join(SRC_DIR, 'src'))
PROJECT_DIR = SRC_DIR
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('run_experiments')
TASK_LIST = ['asa', 'aki', 'icu_admit', 'icu_los', 'icu_los_prolonged', 'hospital_los', 'hospital_los_prolonged', 'ioht', 'ioh', 'inhospital_mortality']
TASK_LABEL_FILE_MAP = {'icu_los': 'icu_los', 'icu_los_prolonged': 'icu_los', 'hospital_los': 'hospital_los', 'hospital_los_prolonged': 'hospital_los'}
TASK_DATASET_AVAILABILITY = {'asa': ['vitaldb', 'mover', 'inspire'], 'aki': ['vitaldb', 'inspire'], 'icu_admit': ['vitaldb', 'mover'], 'icu_los': ['vitaldb', 'inspire'], 'icu_los_prolonged': ['vitaldb', 'inspire'], 'hospital_los': ['vitaldb', 'mover'], 'hospital_los_prolonged': ['vitaldb', 'mover'], 'ioht': ['vitaldb', 'mover', 'inspire'], 'ioh': ['vitaldb', 'mover', 'inspire'], 'inhospital_mortality': ['vitaldb', 'mover', 'inspire']}
BERT_CACHE_DIR = Path(PROJECT_DIR) / 'bert_cache'
UNIFIED_SINGLE = {'S1': 'baseline', 'S2': 'baseline_aligned', 'S3': 'baseline_clustered', 'S4': 'bert_baseline', 'S5': 'bert_baseline_aligned', 'S6': 'bert_baseline_clustered'}
UNIFIED_JOINT = {'J1': 'exp1', 'J2': 'exp14', 'J3': 'exp2', 'J4': 'exp5', 'J5': 'exp13', 'J6': 'exp8'}
UNIFIED_TRANSFER = {'T1': ('onehot', 'unaligned'), 'T2': ('onehot', 'aligned'), 'T3': ('onehot', 'aligned_clustered'), 'T4': ('bert', 'unaligned'), 'T5': ('bert', 'aligned'), 'T6': ('bert', 'aligned_clustered')}
ALL_TRANSFER_DIRECTIONS = [('vitaldb', 'mover'), ('mover', 'vitaldb'), ('vitaldb', 'inspire'), ('inspire', 'vitaldb'), ('mover', 'inspire'), ('inspire', 'mover')]

class ExperimentRunner:
    """
    实验运行器

    - Baseline/Exp1/Exp2: 读 benchmark/{task}/{mode}/ 下的预清洗数据
    - Exp3/Exp4: 迁移实验，训练集和测试集各用自身 benchmark + 训练集 vocab
    - Exp5-8: BERT 相关实验
    """

    def __init__(self, benchmark_dir: str='benchmark', config_path: str='configs/config.yaml', seed: int=42):
        self.benchmark_dir = Path(PROJECT_DIR) / benchmark_dir
        self.dataset_dir = Path(PROJECT_DIR) / 'dataset'
        self.config = self._load_config(config_path)
        self.seed = seed
        self._load_components()
        self._bert = None
        self._vocab_cache = {}

    def _load_components(self):
        from data.preprocessor import DataPreprocessor
        from models.base_models import ModelFactory
        from utils.metrics import ExperimentEvaluator
        self.DataPreprocessor = DataPreprocessor
        self.ModelFactory = ModelFactory
        self.ExperimentEvaluator = ExperimentEvaluator
        output_dir = Path(PROJECT_DIR) / self.config.get('paths', {}).get('output_dir', 'outputs')
        self.evaluator = self.ExperimentEvaluator(output_dir=str(output_dir))
        self.results = {}

    def _create_model(self, algo: str):
        """创建模型，并从 config hyperparameters 提取标量超参传入

        config 中的列表值（如 C: [0.01, 0.1, 1.0]）是网格搜索候选，直接传给
        sklearn/模型构造函数会报错，故只透传标量参数；其余用模型类默认值。
        random_state 统一注入当前实验 seed，保证多 seed 可复现。
        """
        hp = self.config.get('model', {}).get('hyperparameters', {}).get(algo, {}) or {}
        scalar_params = {k: v for k, v in hp.items() if isinstance(v, (int, float, str, bool))}
        scalar_params['random_state'] = self.seed
        if not scalar_params:
            return self.ModelFactory.create_model(algo)
        return self.ModelFactory.create_model(algo, scalar_params)

    def _load_config(self, config_path: str) -> Dict:
        with open(Path(PROJECT_DIR) / config_path, 'r', encoding='utf-8') as f:
            return _release_expand_config(yaml.safe_load(f))

    def _filter(self, items: List[str], filt: Optional[str]) -> List[str]:
        if filt and filt in items:
            return [filt]
        return items

    def _get_bert(self):
        """懒加载 BERT 嵌入器"""
        if self._bert is None:
            from features.bert_embedder import BERTEmbedder
            BERT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            self._bert = BERTEmbedder(cache_dir=str(BERT_CACHE_DIR))
        return self._bert

    def _get_or_build_vocab(self, task: str, mode: str, dataset: str) -> dict:
        """从 benchmark train.csv 构建 vocab，结果缓存"""
        cache_key = (task, mode, dataset)
        if cache_key in self._vocab_cache:
            return self._vocab_cache[cache_key]
        train_path = self.benchmark_dir / task / mode / dataset / 'train.csv'
        if not train_path.exists():
            logger.warning(f'  无 train 数据: {train_path}')
            return {}
        X_train_raw = pd.read_csv(train_path, index_col=0)
        vocab = self._build_transfer_vocab(X_train_raw, X_train_raw, top_n=500)
        self._vocab_cache[cache_key] = vocab
        return vocab

    @staticmethod
    def _encode_raw_features(X_raw: pd.DataFrame, vocab_data: Dict) -> pd.DataFrame:
        """
        对原始特征进行 one-hot 编码

        Args:
            X_raw: 原始特征 DataFrame (surgery_id 索引, 混合类型)
            vocab_data: 词汇表JSON (含 columns 字段)

        Returns:
            编码后的数值特征矩阵
        """
        import numpy as np
        vocab_columns = vocab_data.get('columns', {})
        generalized_mapping = vocab_data.get('generalized_mapping', {})
        is_generalized = bool(generalized_mapping)
        encoded_parts = []
        if '_source_dataset' in X_raw.columns:
            encoded_parts.append(X_raw[['_source_dataset']])
        col_vocab_lower_map = {x.lower(): x for col_name in vocab_columns for x in vocab_columns[col_name]}

        def _lookup_or_generalize(item: str, col_vocab: list) -> str:
            """匹配逻辑：优先直接匹配，再用 gen_map 泛化"""
            item_lower = item.strip().lower()
            direct = col_vocab_lower_map.get(item_lower)
            if direct and direct in col_vocab:
                return direct
            if item in col_vocab:
                return item
            if not is_generalized:
                return None
            parent_id = generalized_mapping.get(item_lower)
            if parent_id and parent_id in col_vocab:
                return parent_id
            import re
            bare = re.sub('\\s*\\([^)]*\\)\\s*$', '', item_lower).strip()
            if bare != item_lower:
                parent_id = generalized_mapping.get(bare)
                if parent_id and parent_id in col_vocab:
                    return parent_id
            return None
        for col in X_raw.columns:
            if col in ('_source_dataset',):
                continue
            if col in ('med_rec_summary', 'med_rec_summary_cluster') and col in vocab_columns:
                drug_vocab = vocab_columns[col]
                drug_matrix = pd.DataFrame(0, index=X_raw.index, columns=[f'{col}_{d}' for d in drug_vocab], dtype=np.float32)
                for idx, val in X_raw[col].items():
                    if pd.isna(val) or not isinstance(val, str) or (not val.strip()):
                        continue
                    drugs = [d.strip() for d in val.split(';') if d.strip()]
                    for d in drugs:
                        matched = _lookup_or_generalize(d, drug_vocab)
                        if matched:
                            drug_matrix.loc[idx, f'{col}_{matched}'] = 1.0
                encoded_parts.append(drug_matrix)
                continue
            if col.startswith('hx_') and (col.endswith('_detail') or col.endswith('_details')):
                if col in vocab_columns:
                    hist_vocab = vocab_columns[col]
                    hist_matrix = pd.DataFrame(0, index=X_raw.index, columns=[f'{col}_{item}' for item in hist_vocab], dtype=np.float32)
                    for idx, val in X_raw[col].items():
                        if pd.isna(val) or not isinstance(val, str) or (not val.strip()):
                            continue
                        items = [i.strip() for i in val.split(';') if i.strip()]
                        for item in items:
                            matched = _lookup_or_generalize(item, hist_vocab)
                            if matched:
                                col_name = f'{col}_{matched}'
                                if col_name in hist_matrix.columns:
                                    hist_matrix.loc[idx, col_name] = 1.0
                    encoded_parts.append(hist_matrix)
                continue
            if col in vocab_columns:
                categories = vocab_columns[col]
                if not categories:
                    continue
                dummies = pd.DataFrame(0, index=X_raw.index, columns=[f'{col}_{v}' for v in categories], dtype=np.float32)
                for val, group in X_raw[[col]].dropna().groupby(col):
                    val_str = str(val)
                    if is_generalized:
                        parent_id = generalized_mapping.get(val_str.strip().lower())
                        if parent_id and parent_id in categories:
                            dummies.loc[group.index, f'{col}_{parent_id}'] = 1.0
                        elif val_str in categories:
                            dummies.loc[group.index, f'{col}_{val_str}'] = 1.0
                    elif val_str in categories:
                        dummies.loc[group.index, f'{col}_{val_str}'] = 1.0
                encoded_parts.append(dummies)
            else:
                col_data = X_raw[col]
                if pd.api.types.is_numeric_dtype(col_data):
                    encoded_parts.append(col_data.astype(np.float32).to_frame())
        if not encoded_parts:
            return pd.DataFrame(index=X_raw.index)
        result = pd.concat(encoded_parts, axis=1)
        return result

    def _load_xy(self, task: str, mode: str, dataset: str, split: str) -> Tuple[pd.DataFrame, pd.Series]:
        """加载 benchmark 数据并 on-the-fly OneHot 编码

        mode 直接对应 benchmark 目录名: unaligned / aligned / aligned_clustered
        vocab 从该 dataset 的 train.csv 构建（缓存复用）
        """
        path_x = self.benchmark_dir / task / mode / dataset / f'{split}.csv'
        path_y = self.benchmark_dir / task / mode / dataset / f'{split}_y.csv'
        if not path_x.exists() or not path_y.exists():
            raise FileNotFoundError(f'Missing: {path_x} or {path_y}')
        X_raw = pd.read_csv(path_x, index_col=0)
        y = pd.read_csv(path_y, index_col=0).iloc[:, 0]
        vocab = self._get_or_build_vocab(task, mode, dataset)
        if not vocab:
            logger.warning(f'  [{task}/{mode}/{dataset}] 无vocab，返回数值特征')
            X = X_raw.select_dtypes(include=[np.number]).astype(np.float32)
        else:
            keep_cols = [c for c in X_raw.columns if c in vocab or pd.api.types.is_numeric_dtype(X_raw[c])]
            X_raw = X_raw[keep_cols]
            X = self._encode_raw_features(X_raw, {'columns': vocab})
        return (X, y)

    def _load_xy_val(self, task: str, mode: str, dataset: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
        """加载验证集 (可选，不存在则返回None)"""
        path_x = self.benchmark_dir / task / mode / dataset / f'val.csv'
        path_y = self.benchmark_dir / task / mode / dataset / f'val_y.csv'
        if not path_x.exists() or not path_y.exists():
            return (None, None)
        X_raw = pd.read_csv(path_x, index_col=0)
        y = pd.read_csv(path_y, index_col=0).iloc[:, 0]
        vocab = self._get_or_build_vocab(task, mode, dataset)
        if not vocab:
            X = X_raw.select_dtypes(include=[np.number]).astype(np.float32)
        else:
            keep_cols = [c for c in X_raw.columns if c in vocab or pd.api.types.is_numeric_dtype(X_raw[c])]
            X_raw = X_raw[keep_cols]
            X = self._encode_raw_features(X_raw, {'columns': vocab})
        return (X, y)

    def _get_common_columns(self, task: str, mode: str, datasets: List[str]) -> Set[str]:
        """
        获取多个数据集的共有原始列名（用于联合训练/迁移训练过滤）
        """
        common_cols = None
        for ds in datasets:
            path = self.benchmark_dir / task / mode / ds / 'train.csv'
            if not path.exists():
                continue
            header = set(pd.read_csv(path, nrows=0).columns)
            header -= {'surgery_id', '_source_dataset'}
            if common_cols is None:
                common_cols = header
            else:
                common_cols &= header
        return common_cols or set()

    def _filter_to_common_columns(self, X: pd.DataFrame, common_cols: Set[str]) -> pd.DataFrame:
        """过滤DataFrame到只有共有列的子集"""
        if not common_cols:
            return X
        keep_cols = [c for c in X.columns if c in common_cols or any((c.startswith(f'{cc}_') for cc in common_cols)) or c == '_source_dataset']
        return X[keep_cols] if keep_cols else X

    def _load_split_ids(self, task: str, mode: str, dataset: str, split: str) -> List[str]:
        """从 benchmark 中读取 split 的 surgery_id 列表"""
        path = self.benchmark_dir / task / mode / dataset / f'{split}.csv'
        if not path.exists():
            return []
        df = pd.read_csv(path, index_col=0, nrows=0)
        idx = pd.read_csv(path, usecols=[0]).iloc[:, 0]
        return idx.astype(str).tolist()

    def _check_exists(self, task: str, mode: str, dataset: str, split: str) -> bool:
        return (self.benchmark_dir / task / mode / dataset / f'{split}.csv').exists()

    def _load_raw_features(self, dataset: str, mode: str):
        """加载并清洗原始特征（不做 onehot/BERT 编码）"""
        from features.feature_processor import FeatureProcessor
        fp = FeatureProcessor(base_dir=str(self.dataset_dir), mode=mode, encode_text=False)
        features = fp.load_features(dataset)
        return fp.clean_features(features)

    def _load_benchmark_raw(self, task: str, mode: str, dataset: str, split: str) -> pd.DataFrame:
        """从 benchmark 加载原始（未编码）特征"""
        path = self.benchmark_dir / task / mode / dataset / f'{split}.csv'
        if not path.exists():
            raise FileNotFoundError(f'Missing: {path}')
        return pd.read_csv(path, index_col=0)

    def _build_bert_X(self, task: str, raw_mode: str, ds_splits: List[Tuple[str, str, List[str]]], enum_vocab_dataset: str='combined') -> Tuple[pd.DataFrame, pd.Series]:
        """
        从 benchmark 原始数据构建 BERT 嵌入特征矩阵

        raw_mode 支持三种模式（对应 benchmark 目录名）:
          - "unaligned":           原始临床文本 → BERT 嵌入
          - "aligned":             SNOMED FSN 文本 → BERT 嵌入
          - "aligned_clustered":   聚类后 centroid bare FSN → BERT 嵌入

        Args:
            ds_splits: [(dataset_name, split_name, ids_or_None), ...]
                       ids_or_None: List[str] 按ID过滤, 或 None 加载全部数据
            enum_vocab_dataset: 枚举列 One-Hot 编码使用哪个词汇表
                                "combined"=跨数据集联合, "mover"/"vitaldb"等=单数据集
                                迁移实验应使用源数据集名，确保不泄露目标集信息
        """
        bert = self._get_bert()
        if not bert.is_loaded:
            bert.load()
        bench_mode = raw_mode
        X_parts = []
        y_parts = []
        for ds_name, split, ids in ds_splits:
            if ids is not None and (not ids):
                continue
            try:
                X_raw = self._load_benchmark_raw(task, bench_mode, ds_name, split)
            except FileNotFoundError:
                logger.warning(f'  [{ds_name}/{split}] 无benchmark数据')
                continue
            y_path = self.benchmark_dir / task / bench_mode / ds_name / f'{split}_y.csv'
            if not y_path.exists():
                continue
            y_df = pd.read_csv(y_path, index_col=0)
            y_col = y_df.columns[0]
            if ids is None:
                X_filt = X_raw
                y_filt = y_df[y_col]
            else:
                ids_set = set((str(i) for i in ids))
                common_ids = ids_set & set((str(i) for i in X_raw.index))
                if not common_ids:
                    continue
                X_filt = X_raw.loc[X_raw.index.astype(str).isin(common_ids)]
                y_filt = y_df[y_df.index.astype(str).isin(common_ids)][y_col]
            text_cols = ['preop_diagnosis', 'scheduled_procedure_name', 'med_rec_summary', 'hx_circ_detail', 'hx_resp_detail', 'hx_endo_detail', 'hx_neuro_detail', 'hx_dige_detail', 'hx_spine_detail', 'hx_immu_detail', 'hx_urine_detail', 'hx_other_detail', 'hx_surgery_detail']
            text_cols = text_cols + [c for c in X_filt.columns if c.endswith('_cluster')]
            available_text = [c for c in text_cols if c in X_filt.columns]
            numeric_cols = [c for c in X_filt.columns if c not in text_cols and c != '_source_dataset' and pd.api.types.is_numeric_dtype(X_filt[c])]
            feat_parts = []
            if available_text:
                for col in available_text:
                    if col == 'med_rec_summary':
                        df_drug = X_filt[[col]].copy()
                        df_drug['surgery_id'] = df_drug.index
                        df_drug_emb = bert.embed_feature_dataframe(df_drug, 'feat_med_rec', ds_name)
                        if df_drug_emb is not None:
                            feat_parts.append(df_drug_emb)
                    else:
                        col_series = X_filt[col]
                        emb = bert._embed_single_column(col_series, col)
                        if emb is not None:
                            feat_parts.append(emb)
            if numeric_cols:
                feat_parts.append(X_filt[numeric_cols].astype(np.float32))
            enum_cols = [c for c in X_filt.columns if c not in text_cols and c != '_source_dataset' and (not pd.api.types.is_numeric_dtype(X_filt[c]))]
            if enum_cols:
                try:
                    vocab = self._get_or_build_vocab(task, bench_mode, enum_vocab_dataset)
                    if vocab:
                        encoded_enum = self._encode_raw_features(X_filt[enum_cols + (['_source_dataset'] if '_source_dataset' in X_filt.columns else [])].copy(), {'columns': vocab})
                        encoded_enum = encoded_enum.drop(columns=['_source_dataset'], errors='ignore')
                        if not encoded_enum.empty and encoded_enum.shape[1] > 0:
                            feat_parts.append(encoded_enum.astype(np.float32))
                except Exception as e:
                    logger.warning(f'  [{ds_name}] 枚举列 One-Hot 编码失败: {e}')
            if feat_parts:
                X_ds = pd.concat(feat_parts, axis=1) if len(feat_parts) > 1 else feat_parts[0]
                X_ds = X_ds.fillna(0)
                if '_source_dataset' in X_filt.columns:
                    X_ds['_source_dataset'] = X_filt['_source_dataset']
                X_parts.append(X_ds)
                y_parts.append(y_filt)
        bert.save_cache()
        if not X_parts or not y_parts:
            raise ValueError('无法构建 BERT 特征')
        X = pd.concat(X_parts, axis=0)
        X = X.groupby(X.index).first()
        y = pd.concat(y_parts, axis=0)
        y = y.groupby(y.index).first()
        common = set((str(i) for i in X.index)) & set((str(i) for i in y.index))
        X = X.loc[X.index.astype(str).isin(common)]
        y = y.loc[y.index.astype(str).isin(common)]
        return (X, y)

    def _load_raw_labels(self, task: str, dataset: str) -> Optional[pd.DataFrame]:
        """加载原始标签文件"""
        label_task = TASK_LABEL_FILE_MAP.get(task, task)
        label_dir = self.dataset_dir / dataset / 'prerisk_bench_labels'
        for fname in [f'labels_{label_task}.csv', f'label_{label_task}.csv']:
            path = label_dir / fname
            if path.exists():
                return pd.read_csv(path, low_memory=False)
        return None

    def _train_and_eval(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, algorithms: List[str], experiment: str, task: str, label: str, X_val: Optional[pd.DataFrame]=None, y_val: Optional[pd.Series]=None, prior_correct: bool=False) -> Dict:
        logger.info(f'  训练: {X_train.shape}, 测试: {X_test.shape}')
        for df in [X_train, X_test, X_val]:
            if df is not None and '_source_dataset' in df.columns:
                df.drop(columns=['_source_dataset'], inplace=True)
        common_cols = [c for c in X_train.columns if c in X_test.columns]
        X_train = X_train[common_cols]
        X_test = X_test[common_cols]
        preprocessor = self.DataPreprocessor()
        X_train_sc, y_train = preprocessor.fit_transform(X_train, y_train)
        X_test_sc = preprocessor.transform(X_test)
        X_val_sc = preprocessor.transform(X_val) if X_val is not None and len(X_val) > 0 else None
        if hasattr(X_train_sc, 'columns') and hasattr(X_test_sc, 'columns'):
            common = [c for c in X_train_sc.columns if c in X_test_sc.columns]
            X_train_sc = X_train_sc[common]
            X_test_sc = X_test_sc[common]
        if X_val_sc is not None and hasattr(X_val_sc, 'columns') and hasattr(X_train_sc, 'columns'):
            common_val = [c for c in X_train_sc.columns if c in X_val_sc.columns]
            X_val_sc = X_val_sc[common_val]
        REGRESSION_TASKS = {'icu_los', 'hospital_los'}
        is_regression = task.lower() in REGRESSION_TASKS
        r = {}
        for algo in algorithms:
            if is_regression and algo in ('mlp', 'cnn', 'tabtransformer'):
                logger.warning(f'  [{algo}] 回归任务不支持，跳过 (task={task})')
                continue
            model = self._create_model(algo)
            train_kwargs = {}
            if is_regression and algo in ('xgboost', 'lr', 'rf'):
                train_kwargs['is_regression'] = True
            if algo in ('xgboost', 'mlp', 'cnn', 'tabtransformer'):
                if X_val_sc is not None and y_val is not None:
                    train_kwargs['X_val'] = X_val_sc
                    train_kwargs['y_val'] = y_val
            model.train(X_train_sc, y_train, **train_kwargs)
            result = self.evaluator.evaluate_model(model, X_test_sc, y_test, experiment_name=experiment, task_name=task, dataset_name=label, algorithm=algo, X_val=X_val_sc, y_val=y_val, prior_correct=prior_correct, train_classes=np.unique(np.asarray(y_train).ravel()))
            r[f'{experiment}_{task}_{label}_{algo}'] = result
        return r

    def run_baseline(self, task_filter=None, dataset_filter=None, algo_filter=None):
        logger.info('\n' + '=' * 70)
        logger.info(' [BASELINE] 单数据集训练 (unaligned)')
        logger.info('=' * 70)
        tasks = self._filter(TASK_LIST, task_filter)
        datasets = self._filter(['vitaldb', 'mover', 'inspire'], dataset_filter)
        algs = self._filter(self.config['model']['algorithms'], algo_filter)
        for task in tasks:
            for ds in datasets:
                if not self._check_exists(task, 'unaligned', ds, 'train'):
                    continue
                try:
                    X_tr, y_tr = self._load_xy(task, 'unaligned', ds, 'train')
                    X_te, y_te = self._load_xy(task, 'unaligned', ds, 'test')
                    X_val, y_val = self._load_xy_val(task, 'unaligned', ds)
                    self.results.update(self._train_and_eval(X_tr, y_tr, X_te, y_te, algs, 'baseline', task, ds, X_val, y_val))
                except Exception as e:
                    logger.warning(f'  [{task}/{ds}] 失败: {e}')
        logger.info('\n[BASELINE] 完成')
        return self.results

    def _run_baseline_mode(self, mode: str, exp_name: str, task_filter=None, dataset_filter=None, algo_filter=None) -> Dict:
        """单数据集训练（指定模式），用于对齐前后对比实验"""
        logger.info('\n' + '=' * 70)
        logger.info(f' [{exp_name.upper()}] 单数据集训练 ({mode})')
        logger.info('=' * 70)
        tasks = self._filter(TASK_LIST, task_filter)
        datasets = self._filter(['vitaldb', 'mover', 'inspire'], dataset_filter)
        algs = self._filter(self.config['model']['algorithms'], algo_filter)
        for task in tasks:
            for ds in datasets:
                if not self._check_exists(task, mode, ds, 'train'):
                    continue
                try:
                    X_tr, y_tr = self._load_xy(task, mode, ds, 'train')
                    X_te, y_te = self._load_xy(task, mode, ds, 'test')
                    X_val, y_val = self._load_xy_val(task, mode, ds)
                    self.results.update(self._train_and_eval(X_tr, y_tr, X_te, y_te, algs, exp_name, task, ds, X_val, y_val))
                except Exception as e:
                    logger.warning(f'  [{task}/{ds}/{mode}] 失败: {e}')
        logger.info(f'\n[{exp_name.upper()}] 完成')
        return self.results

    def run_baseline_aligned(self, task_filter=None, dataset_filter=None, algo_filter=None):
        """单数据集训练 (SNOMED 对齐, 不聚类) — 对齐前后对比"""
        return self._run_baseline_mode('aligned', 'baseline_aligned', task_filter, dataset_filter, algo_filter)

    def run_baseline_clustered(self, task_filter=None, dataset_filter=None, algo_filter=None):
        """单数据集训练 (SNOMED 对齐 + 语义聚类) — 对齐前后对比"""
        return self._run_baseline_mode('aligned_clustered', 'baseline_clustered', task_filter, dataset_filter, algo_filter)

    def _run_bert_single(self, raw_mode: str, exp_name: str, task_filter=None, dataset_filter=None, algo_filter=None) -> Dict:
        """BERT 单数据集训练（指定模式），用于 BERT 三模式对比实验"""
        logger.info('\n' + '=' * 70)
        logger.info(f' [{exp_name.upper()}] BERT单数据集训练 ({raw_mode})')
        logger.info('=' * 70)
        tasks = self._filter(TASK_LIST, task_filter)
        datasets = self._filter(['vitaldb', 'mover', 'inspire'], dataset_filter)
        algs = self._filter(self.config['model']['algorithms'], algo_filter)
        for task in tasks:
            for ds in datasets:
                if not self._check_exists(task, raw_mode, ds, 'train'):
                    continue
                try:
                    X_tr, y_tr = self._build_bert_X(task, raw_mode, [(ds, 'train', None)], enum_vocab_dataset=ds)
                    X_te, y_te = self._build_bert_X(task, raw_mode, [(ds, 'test', None)], enum_vocab_dataset=ds)
                    X_val, y_val = (None, None)
                    if self._check_exists(task, raw_mode, ds, 'val'):
                        X_val, y_val = self._build_bert_X(task, raw_mode, [(ds, 'val', None)], enum_vocab_dataset=ds)
                    self.results.update(self._train_and_eval(X_tr, y_tr, X_te, y_te, algs, exp_name, task, ds, X_val, y_val))
                except Exception as e:
                    logger.warning(f'  [{task}/{ds}/{raw_mode}] 失败: {e}')
        logger.info(f'\n[{exp_name.upper()}] 完成')
        return self.results

    def run_bert_baseline(self, task_filter=None, dataset_filter=None, algo_filter=None):
        """BERT 单数据集训练 (unaligned)"""
        return self._run_bert_single('unaligned', 'bert_baseline', task_filter, dataset_filter, algo_filter)

    def run_bert_baseline_aligned(self, task_filter=None, dataset_filter=None, algo_filter=None):
        """BERT 单数据集训练 (aligned)"""
        return self._run_bert_single('aligned', 'bert_baseline_aligned', task_filter, dataset_filter, algo_filter)

    def run_bert_baseline_clustered(self, task_filter=None, dataset_filter=None, algo_filter=None):
        """BERT 单数据集训练 (aligned_clustered)"""
        return self._run_bert_single('aligned_clustered', 'bert_baseline_clustered', task_filter, dataset_filter, algo_filter)

    def run_exp1(self, task_filter=None, algo_filter=None):
        logger.info('\n' + '=' * 70)
        logger.info(' [EXP1] 未对齐联合训练')
        logger.info('=' * 70)
        tasks = self._filter(TASK_LIST, task_filter)
        algs = self._filter(self.config['model']['algorithms'], algo_filter)
        for task in tasks:
            if not self._check_exists(task, 'unaligned', 'combined', 'train'):
                continue
            try:
                datasets = TASK_DATASET_AVAILABILITY.get(task, [])
                common_cols = self._get_common_columns(task, 'unaligned', datasets) if len(datasets) > 1 else set()
                X_tr, y_tr = self._load_xy(task, 'unaligned', 'combined', 'train')
                X_te, y_te = self._load_xy(task, 'unaligned', 'combined', 'test')
                X_val, y_val = self._load_xy_val(task, 'unaligned', 'combined')
                if common_cols:
                    X_tr = self._filter_to_common_columns(X_tr, common_cols)
                    X_te = self._filter_to_common_columns(X_te, common_cols)
                    if X_val is not None:
                        X_val = self._filter_to_common_columns(X_val, common_cols)
                    logger.info(f'  [{task}] 共有特征: {X_tr.shape[1]}维 (原始过滤后)')
                self.results.update(self._train_and_eval(X_tr, y_tr, X_te, y_te, algs, 'exp1', task, 'combined', X_val, y_val))
            except Exception as e:
                logger.warning(f'  [{task}] 失败: {e}')
        logger.info('\n[EXP1] 完成')
        return self.results

    def run_exp2(self, task_filter=None, algo_filter=None):
        logger.info('\n' + '=' * 70)
        logger.info(' [EXP2] 聚类后联合训练 (One-Hot)')
        logger.info('=' * 70)
        tasks = self._filter(TASK_LIST, task_filter)
        algs = self._filter(self.config['model']['algorithms'], algo_filter)
        for task in tasks:
            if not self._check_exists(task, 'aligned_clustered', 'combined', 'train'):
                continue
            try:
                datasets = TASK_DATASET_AVAILABILITY.get(task, [])
                common_cols = self._get_common_columns(task, 'aligned_clustered', datasets) if len(datasets) > 1 else set()
                X_tr, y_tr = self._load_xy(task, 'aligned_clustered', 'combined', 'train')
                X_te, y_te = self._load_xy(task, 'aligned_clustered', 'combined', 'test')
                X_val, y_val = self._load_xy_val(task, 'aligned_clustered', 'combined')
                if common_cols:
                    X_tr = self._filter_to_common_columns(X_tr, common_cols)
                    X_te = self._filter_to_common_columns(X_te, common_cols)
                    if X_val is not None:
                        X_val = self._filter_to_common_columns(X_val, common_cols)
                    logger.info(f'  [{task}] 共有特征: {X_tr.shape[1]}维 (原始过滤后)')
                self.results.update(self._train_and_eval(X_tr, y_tr, X_te, y_te, algs, 'exp2', task, 'combined', X_val, y_val))
            except Exception as e:
                logger.warning(f'  [{task}] 失败: {e}')
        logger.info('\n[EXP2] 完成')
        return self.results

    def run_exp14(self, task_filter=None, algo_filter=None):
        """Exp14: One-Hot 联合训练 (aligned, 纯 SNOMED 对齐不聚类)"""
        logger.info('\n' + '=' * 70)
        logger.info(' [EXP14] 对齐联合训练 (One-Hot, aligned)')
        logger.info('=' * 70)
        tasks = self._filter(TASK_LIST, task_filter)
        algs = self._filter(self.config['model']['algorithms'], algo_filter)
        for task in tasks:
            if not self._check_exists(task, 'aligned', 'combined', 'train'):
                continue
            try:
                datasets = TASK_DATASET_AVAILABILITY.get(task, [])
                common_cols = self._get_common_columns(task, 'aligned', datasets) if len(datasets) > 1 else set()
                X_tr, y_tr = self._load_xy(task, 'aligned', 'combined', 'train')
                X_te, y_te = self._load_xy(task, 'aligned', 'combined', 'test')
                X_val, y_val = self._load_xy_val(task, 'aligned', 'combined')
                if common_cols:
                    X_tr = self._filter_to_common_columns(X_tr, common_cols)
                    X_te = self._filter_to_common_columns(X_te, common_cols)
                    if X_val is not None:
                        X_val = self._filter_to_common_columns(X_val, common_cols)
                    logger.info(f'  [{task}] 共有特征: {X_tr.shape[1]}维 (原始过滤后)')
                self.results.update(self._train_and_eval(X_tr, y_tr, X_te, y_te, algs, 'exp14', task, 'combined', X_val, y_val))
            except Exception as e:
                logger.warning(f'  [{task}] 失败: {e}')
        logger.info('\n[EXP14] 完成')
        return self.results

    def run_exp3(self, task_filter=None, algo_filter=None, global_top_n=None, keep_shared_only=False):
        """
        Exp3: 未对齐模型迁移 (One-Hot)
        - 训练: Mover, 测试: Inspire + Vitaldb
        """
        logger.info('\n' + '=' * 70)
        logger.info(' [EXP3] 未对齐模型迁移 (Mover→Inspire, Mover→Vitaldb)')
        logger.info('=' * 70)
        results = {}
        results.update(self._run_transfer(task_filter, algo_filter, 'unaligned', 'mover', 'inspire', 'exp3', global_top_n=global_top_n, keep_shared_only=keep_shared_only))
        results.update(self._run_transfer(task_filter, algo_filter, 'unaligned', 'mover', 'vitaldb', 'exp3', global_top_n=global_top_n, keep_shared_only=keep_shared_only))
        return results

    def run_exp4(self, task_filter=None, algo_filter=None, global_top_n=None, keep_shared_only=False):
        """
        Exp4: 聚类后模型迁移 (One-Hot)
        - 训练: Mover, 测试: Inspire + Vitaldb
        """
        logger.info('\n' + '=' * 70)
        logger.info(' [EXP4] 聚类后模型迁移 (Mover→Inspire, Mover→Vitaldb)')
        logger.info('=' * 70)
        results = {}
        results.update(self._run_transfer(task_filter, algo_filter, 'aligned_clustered', 'mover', 'inspire', 'exp4', global_top_n=global_top_n, keep_shared_only=keep_shared_only))
        results.update(self._run_transfer(task_filter, algo_filter, 'aligned_clustered', 'mover', 'vitaldb', 'exp4', global_top_n=global_top_n, keep_shared_only=keep_shared_only))
        return results

    def _run_transfer(self, task_filter, algo_filter, mode: str, source: str, target: str, exp_name: str, global_top_n: int=None, keep_shared_only: bool=False, asa_max: int=None) -> Dict:
        """
        迁移实验：读 source train 构建 vocab → OneHot 编码 source+target → 训练

        vocab 仅从 source train 构建（正确迁移做法），target 中不可见的值丢弃不编码。
        global_top_n: 全局特征预算模式，取全局频次 top N，同等维度下比较对齐效果。
        keep_shared_only: 只保留在 train 和 test 都非零的特征，去掉 src-only 噪声。
        asa_max: 若给定，训练/测试标签只保留 <= asa_max 的样本（统一 ASA I-IV）。
        mode 可以是 'unaligned' / 'aligned' / 'aligned_clustered'，
        三者在编码逻辑上完全一致，区别仅在于 benchmark 的文本内容不同。
        """
        from collections import Counter
        tasks = self._filter(TASK_LIST, task_filter)
        algs = self._filter(self.config['model']['algorithms'], algo_filter)
        for task in tasks:
            datasets = TASK_DATASET_AVAILABILITY.get(task, [])
            if source not in datasets or target not in datasets:
                continue
            try:
                common_raw = self._get_common_columns(task, mode, [source, target])
                logger.info(f'  [{source}→{target}] 共有列: {len(common_raw)}')
                X_src, y_src = self._load_xy_raw(task, mode, source, 'train', common_raw)
                X_tgt, y_tgt = self._load_xy_raw(task, mode, target, 'test', common_raw)
                X_val, y_val = self._load_xy_raw(task, mode, source, 'val', common_raw)
                if X_src.empty or X_tgt.empty:
                    logger.warning(f'  [{task}] 跳过: 训练集或测试集为空')
                    continue
                if asa_max is not None:
                    X_src, y_src = self._filter_y_max(X_src, y_src, asa_max)
                    X_tgt, y_tgt = self._filter_y_max(X_tgt, y_tgt, asa_max)
                    if not X_val.empty:
                        X_val, y_val = self._filter_y_max(X_val, y_val, asa_max)
                build_kwargs = {'top_n': 500}
                if global_top_n is not None:
                    build_kwargs['global_top_n'] = global_top_n
                vocab_cols = self._build_transfer_vocab(X_src, X_src, **build_kwargs)
                logger.info(f'  [{source}→{target}] 源域vocab: {len(vocab_cols)} 列' + (f' (global top {global_top_n})' if global_top_n else ''))
                X_tr = self._encode_raw_features(X_src, {'columns': vocab_cols})
                X_te = self._encode_raw_features(X_tgt, {'columns': vocab_cols})
                X_val_enc = self._encode_raw_features(X_val, {'columns': vocab_cols})
                if keep_shared_only:
                    shared_cols = [c for c in X_tr.columns if (X_tr[c] != 0).any() and (X_te[c] != 0).any()]
                    X_tr = X_tr[shared_cols]
                    X_te = X_te[shared_cols]
                    X_val_enc = X_val_enc[shared_cols]
                    logger.info(f'  [{source}→{target}] 共享特征过滤: {len(shared_cols)} 维 (src-only已丢弃)')
                if X_tr.shape[1] == 0 or X_te.shape[1] == 0:
                    logger.warning(f'  [{task}] 跳过: 编码后特征为空')
                    continue
                logger.info(f'  [{task}] 迁移特征: train={X_tr.shape[1]}维, test={X_te.shape[1]}维')
                label = f'{source}->{target}'
                self.results.update(self._train_and_eval(X_tr, y_src, X_te, y_tgt, algs, exp_name, task, label, X_val_enc, y_val, prior_correct=True))
            except Exception as e:
                logger.warning(f'  [{task}] 失败: {e}')
                import traceback
                traceback.print_exc()
        logger.info(f'\n[{exp_name}] 完成')
        return self.results

    def _load_xy_raw(self, task: str, mode: str, dataset: str, split: str, common_cols: set) -> Tuple[pd.DataFrame, pd.Series]:
        """加载原始数据，只保留共有列"""
        path_x = self.benchmark_dir / task / mode / dataset / f'{split}.csv'
        path_y = self.benchmark_dir / task / mode / dataset / f'{split}_y.csv'
        if not path_x.exists():
            return (pd.DataFrame(), pd.Series(dtype=float))
        X_raw = pd.read_csv(path_x, index_col=0)
        common = [c for c in X_raw.columns if c in common_cols]
        X_raw = X_raw[common]
        y = pd.read_csv(path_y, index_col=0).iloc[:, 0] if path_y.exists() else pd.Series(dtype=float)
        return (X_raw, y)

    @staticmethod
    def _build_transfer_vocab(X_src: pd.DataFrame, X_tgt: pd.DataFrame, top_n: int=500, global_top_n: int=None) -> dict:
        """
        构建 vocab

        两种模式:
          1. per-column top_n (默认):  每列独立取 top n 个值
          2. global_top_n:             所有列的值合并后取全局频次 top N，
                                       然后按列分组（同等预算下比较不同对齐的效果）
        """
        from collections import Counter
        vocab = {}
        col_items = []
        for col in X_src.columns:
            if col == '_source_dataset':
                continue
            if pd.api.types.is_numeric_dtype(X_src[col]):
                continue
            counter = Counter()
            split_items = col.startswith('hx_') or col == 'med_rec_summary' or col == 'med_rec_summary_cluster'
            for df in [X_src, X_tgt]:
                if col in df.columns:
                    for val in df[col].dropna():
                        val_str = str(val).strip()
                        if not val_str:
                            continue
                        if split_items:
                            for item in val_str.split(';'):
                                item = item.strip()
                                if item:
                                    counter[item] += 1
                        else:
                            counter[val_str] += 1
            if not counter:
                continue
            for item, freq in counter.most_common(top_n):
                col_items.append((col, item, freq))
        if global_top_n is not None and global_top_n > 0:
            col_items.sort(key=lambda x: x[2], reverse=True)
            col_items = col_items[:global_top_n]
        for col_name, item, freq in col_items:
            if col_name not in vocab:
                vocab[col_name] = []
            vocab[col_name].append(item)
        return vocab

    @staticmethod
    def _filter_y_max(X: pd.DataFrame, y: pd.Series, max_val: int) -> Tuple[pd.DataFrame, pd.Series]:
        """统一标签空间: 只保留标签 <= max_val 的样本 (如 ASA 1-4, 去掉 5)"""
        keep = y.astype(float) <= max_val
        keep_idx = set(keep[keep].index.astype(str))
        X_f = X.loc[X.index.astype(str).isin(keep_idx)]
        return (X_f, y[keep])

    def _run_bert_combined(self, raw_mode: str, exp_name: str, task_filter=None, algo_filter=None) -> Dict:
        """BERT 三库联合训练（模式: unaligned / aligned / aligned_clustered）"""
        logger.info('\n' + '=' * 70)
        logger.info(f' [{exp_name.upper()}] BERT联合训练 ({raw_mode})')
        logger.info('=' * 70)
        tasks = self._filter(TASK_LIST, task_filter)
        algs = self._filter(self.config['model']['algorithms'], algo_filter)
        for task in tasks:
            if not self._check_exists(task, raw_mode, 'combined', 'train'):
                continue
            try:
                train_ids = self._load_split_ids(task, raw_mode, 'combined', 'train')
                test_ids = self._load_split_ids(task, raw_mode, 'combined', 'test')
                val_ids = self._load_split_ids(task, raw_mode, 'combined', 'val')
                ds_train_splits = self._split_ids_by_dataset(train_ids, 'train')
                ds_test_splits = self._split_ids_by_dataset(test_ids, 'test')
                X_tr, y_tr = self._build_bert_X(task, raw_mode, ds_train_splits)
                X_te, y_te = self._build_bert_X(task, raw_mode, ds_test_splits)
                X_val, y_val = (None, None)
                if val_ids is not None:
                    ds_val_splits = self._split_ids_by_dataset(val_ids, 'val')
                    X_val, y_val = self._build_bert_X(task, raw_mode, ds_val_splits)
                datasets = TASK_DATASET_AVAILABILITY.get(task, [])
                common_cols = self._get_common_columns(task, raw_mode, datasets) if len(datasets) > 1 else set()
                if common_cols:
                    X_tr = self._filter_to_common_columns(X_tr, common_cols)
                    X_te = self._filter_to_common_columns(X_te, common_cols)
                    if X_val is not None:
                        X_val = self._filter_to_common_columns(X_val, common_cols)
                    logger.info(f'  [{task}] BERT共有特征: {X_tr.shape[1]}维')
                self.results.update(self._train_and_eval(X_tr, y_tr, X_te, y_te, algs, exp_name, task, 'combined', X_val, y_val))
            except Exception as e:
                logger.warning(f'  [{task}] 失败: {e}')
                import traceback
                traceback.print_exc()
        logger.info(f'\n[{exp_name.upper()}] 完成')
        return self.results

    def run_exp5(self, task_filter=None, algo_filter=None):
        """Exp5: BERT 联合训练 (unaligned)"""
        return self._run_bert_combined('unaligned', 'exp5', task_filter, algo_filter)

    def run_exp13(self, task_filter=None, algo_filter=None):
        """Exp13: BERT 联合训练 (aligned, 纯 SNOMED 对齐不聚类)"""
        return self._run_bert_combined('aligned', 'exp13', task_filter, algo_filter)

    def _run_bert_transfer(self, raw_mode: str, exp_name: str, source: str, target: str, task_filter=None, algo_filter=None, asa_max: int=None) -> Dict:
        """BERT 迁移：source train 训练 → target test 评估（含先验校正）
        asa_max: 若给定，训练/测试标签只保留 <= asa_max 的样本（统一 ASA I-IV）。
        """
        tasks = self._filter(TASK_LIST, task_filter)
        algs = self._filter(self.config['model']['algorithms'], algo_filter)
        for task in tasks:
            datasets = TASK_DATASET_AVAILABILITY.get(task, [])
            if source not in datasets or target not in datasets:
                continue
            try:
                X_tr, y_tr = self._build_bert_X(task, raw_mode, [(source, 'train', None)], enum_vocab_dataset=source)
                X_te, y_te = self._build_bert_X(task, raw_mode, [(target, 'test', None)], enum_vocab_dataset=source)
                if asa_max is not None:
                    X_tr, y_tr = self._filter_y_max(X_tr, y_tr, asa_max)
                    X_te, y_te = self._filter_y_max(X_te, y_te, asa_max)
                if len(X_tr) > 0 and len(X_te) > 0:
                    log_name = f'{source}->{target}'
                    logger.info(f'  [{task}] BERT迁移({log_name})特征: 训练={X_tr.shape[1]}维, 测试={X_te.shape[1]}维')
                    self.results.update(self._train_and_eval(X_tr, y_tr, X_te, y_te, algs, exp_name, task, log_name, None, None, prior_correct=True))
                else:
                    logger.warning(f'  [{task}] BERT迁移({log_name})跳过: 训练集或测试集为空')
            except Exception as e:
                logger.warning(f'  [{task}] 失败: {e}')
        return self.results

    def run_exp6(self, task_filter=None, algo_filter=None):
        """Exp6: BERT迁移 (unaligned, Mover→Inspire + Mover→Vitaldb)"""
        logger.info('\n' + '=' * 70)
        logger.info(' [EXP6] BERT迁移 (未对齐, Mover→Inspire + Mover→Vitaldb)')
        logger.info('=' * 70)
        self._run_bert_transfer('unaligned', 'exp6', 'mover', 'inspire', task_filter, algo_filter)
        self._run_bert_transfer('unaligned', 'exp6', 'mover', 'vitaldb', task_filter, algo_filter)
        logger.info('\n[EXP6] 完成')
        return self.results

    def run_exp7(self, task_filter=None, algo_filter=None):
        """Exp7: BERT迁移 (clustered, Mover→Inspire + Mover→Vitaldb)"""
        logger.info('\n' + '=' * 70)
        logger.info(' [EXP7] SNOMED+BERT迁移 (Mover→Inspire + Mover→Vitaldb)')
        logger.info('=' * 70)
        self._run_bert_transfer('aligned_clustered', 'exp7', 'mover', 'inspire', task_filter, algo_filter)
        self._run_bert_transfer('aligned_clustered', 'exp7', 'mover', 'vitaldb', task_filter, algo_filter)
        logger.info('\n[EXP7] 完成')
        return self.results

    def run_exp8(self, task_filter=None, algo_filter=None):
        """Exp8: BERT 联合训练 (aligned_clustered)"""
        return self._run_bert_combined('aligned_clustered', 'exp8', task_filter, algo_filter)

    def run_exp11(self, task_filter=None, algo_filter=None):
        """Exp11: One-Hot 纯 SNOMED 对齐迁移（不做概念聚类）"""
        logger.info('\n' + '=' * 70)
        logger.info(' [EXP11] One-Hot 纯对齐迁移 (aligned, Mover→Inspire + Mover→Vitaldb)')
        logger.info('=' * 70)
        results = {}
        results.update(self._run_transfer(task_filter, algo_filter, 'aligned', 'mover', 'inspire', 'exp11'))
        results.update(self._run_transfer(task_filter, algo_filter, 'aligned', 'mover', 'vitaldb', 'exp11'))
        return results

    def run_exp12(self, task_filter=None, algo_filter=None):
        """Exp12: BERT 纯 SNOMED 对齐迁移（不做概念聚类，SNOMED FSN 直送 ClinicalBERT）"""
        logger.info('\n' + '=' * 70)
        logger.info(' [EXP12] BERT 纯对齐迁移 (aligned, Mover→Inspire + Mover→Vitaldb)')
        logger.info('=' * 70)
        self._run_bert_transfer('aligned', 'exp12', 'mover', 'inspire', task_filter, algo_filter)
        self._run_bert_transfer('aligned', 'exp12', 'mover', 'vitaldb', task_filter, algo_filter)
        logger.info('\n[EXP12] 完成')
        return self.results

    def _run_unified_transfer(self, code: str, task_filter=None, algo_filter=None) -> Dict:
        """按统一编号运行 6 方向双向迁移（含先验校正）"""
        enc, mode = UNIFIED_TRANSFER[code]
        logger.info('\n' + '=' * 70)
        logger.info(f' [{code}] 迁移 {enc}/{mode} (6方向双向)')
        logger.info('=' * 70)
        if enc == 'onehot':
            for src, tgt in ALL_TRANSFER_DIRECTIONS:
                self._run_transfer(task_filter, algo_filter, mode, src, tgt, code)
        else:
            for src, tgt in ALL_TRANSFER_DIRECTIONS:
                self._run_bert_transfer(mode, code, src, tgt, task_filter, algo_filter)
        logger.info(f'\n[{code}] 完成')
        return self.results

    def run_t1(self, task_filter=None, algo_filter=None):
        """T1: OneHot/unaligned 6方向迁移"""
        return self._run_unified_transfer('T1', task_filter, algo_filter)

    def run_t2(self, task_filter=None, algo_filter=None):
        """T2: OneHot/aligned 6方向迁移"""
        return self._run_unified_transfer('T2', task_filter, algo_filter)

    def run_t3(self, task_filter=None, algo_filter=None):
        """T3: OneHot/clustered 6方向迁移"""
        return self._run_unified_transfer('T3', task_filter, algo_filter)

    def run_t4(self, task_filter=None, algo_filter=None):
        """T4: BERT/unaligned 6方向迁移"""
        return self._run_unified_transfer('T4', task_filter, algo_filter)

    def run_t5(self, task_filter=None, algo_filter=None):
        """T5: BERT/aligned 6方向迁移"""
        return self._run_unified_transfer('T5', task_filter, algo_filter)

    def run_t6(self, task_filter=None, algo_filter=None):
        """T6: BERT/clustered 6方向迁移"""
        return self._run_unified_transfer('T6', task_filter, algo_filter)

    def run_t13(self, task_filter=None, algo_filter=None):
        """T13: BERT/概念+簇双层 (aligned_dual), Inspire→Mover, ASA1-4"""
        logger.info('\n' + '=' * 70)
        logger.info(' [T13] BERT/概念+簇双层表示 (aligned_dual), Inspire→Mover, ASA1-4')
        logger.info('=' * 70)
        self._run_bert_transfer('aligned_dual', 'T13', 'inspire', 'mover', task_filter, algo_filter, asa_max=4)
        logger.info('\n[T13] 完成')
        return self.results

    def run_t14(self, task_filter=None, algo_filter=None):
        """T14: BERT/unaligned (原始), ASA1-4, 6方向双向迁移"""
        logger.info('\n' + '=' * 70)
        logger.info(' [T14] BERT/原始文本 (unaligned), ASA1-4, 6方向双向')
        logger.info('=' * 70)
        for src, tgt in ALL_TRANSFER_DIRECTIONS:
            self._run_bert_transfer('unaligned', 'T14', src, tgt, task_filter, algo_filter, asa_max=4)
        logger.info('\n[T14] 完成')
        return self.results

    def run_t15(self, task_filter=None, algo_filter=None):
        """T15: BERT/aligned (纯对齐), ASA1-4, 6方向双向迁移"""
        logger.info('\n' + '=' * 70)
        logger.info(' [T15] BERT/纯对齐 (aligned), ASA1-4, 6方向双向')
        logger.info('=' * 70)
        for src, tgt in ALL_TRANSFER_DIRECTIONS:
            self._run_bert_transfer('aligned', 'T15', src, tgt, task_filter, algo_filter, asa_max=4)
        logger.info('\n[T15] 完成')
        return self.results

    @staticmethod
    def _split_ids_by_dataset(all_ids: List[str], split: str='train') -> List[Tuple[str, str, List[str]]]:
        result = []
        for ds in ['vitaldb', 'mover', 'inspire']:
            result.append((ds, split, all_ids))
        return result

    @staticmethod
    def _split_bert_by_source(X: pd.DataFrame, y: pd.Series, train_sources: List[str], test_sources: List[str]) -> Tuple:
        if '_source_dataset' not in X.columns:
            logger.warning('_source_dataset 列不存在，使用 fallback')
            train_mask = pd.Series(True, index=X.index)
            test_mask = ~train_mask
        else:
            train_mask = X['_source_dataset'].isin(train_sources)
            test_mask = X['_source_dataset'].isin(test_sources)
        X_tr = X.loc[train_mask].drop(columns=['_source_dataset'], errors='ignore')
        y_tr = y.loc[y.index.astype(str).isin(X_tr.index.astype(str))]
        X_te = X.loc[test_mask].drop(columns=['_source_dataset'], errors='ignore')
        y_te = y.loc[y.index.astype(str).isin(X_te.index.astype(str))]
        return (X_tr, y_tr, X_te, y_te)

    def run_all(self, task_filter=None, dataset_filter=None, algo_filter=None):
        experiments = [('baseline', lambda: self.run_baseline(task_filter, dataset_filter, algo_filter)), ('baseline_aligned', lambda: self.run_baseline_aligned(task_filter, dataset_filter, algo_filter)), ('baseline_clustered', lambda: self.run_baseline_clustered(task_filter, dataset_filter, algo_filter)), ('exp1', lambda: self.run_exp1(task_filter, algo_filter)), ('exp2', lambda: self.run_exp2(task_filter, algo_filter)), ('exp3', lambda: self.run_exp3(task_filter, algo_filter)), ('exp4', lambda: self.run_exp4(task_filter, algo_filter)), ('exp5', lambda: self.run_exp5(task_filter, algo_filter)), ('exp6', lambda: self.run_exp6(task_filter, algo_filter)), ('exp7', lambda: self.run_exp7(task_filter, algo_filter)), ('exp8', lambda: self.run_exp8(task_filter, algo_filter)), ('exp11', lambda: self.run_exp11(task_filter, algo_filter)), ('exp12', lambda: self.run_exp12(task_filter, algo_filter))]
        for name, func in experiments:
            try:
                start = datetime.now()
                logger.info(f"\n{'#' * 70}\n # {name}\n{'#' * 70}")
                func()
                elapsed = (datetime.now() - start).total_seconds()
                logger.info(f'  [{name}] 耗时: {elapsed:.1f}s')
            except Exception as e:
                logger.error(f'  [{name}] 失败: {e}')
                import traceback
                traceback.print_exc()
        self._save_results()

    def _save_results(self):
        if not self.evaluator or not hasattr(self.evaluator, 'results'):
            return
        res = self.evaluator.results
        if not res:
            logger.warning('无结果可保存')
            return
        out_dir = Path(PROJECT_DIR) / self.config.get('paths', {}).get('output_dir', 'output')
        out_dir.mkdir(exist_ok=True)
        fp = out_dir / f'experiment_results_seed{self.seed}.csv'
        new_df = self.evaluator.get_results_summary()
        key_cols = ['experiment', 'task', 'dataset', 'algorithm']
        if fp.exists():
            try:
                old_df = pd.read_csv(fp)
                merged = pd.concat([old_df, new_df], ignore_index=True)
                merged = merged.drop_duplicates(subset=key_cols, keep='last')
            except Exception as e:
                logger.warning(f'合并旧结果失败({e})，直接覆盖')
                merged = new_df
        else:
            merged = new_df
        merged.to_csv(fp, index=False, encoding='utf-8-sig')
        logger.info(f'结果汇总已保存(合并): {fp} ({len(merged)} 条)')
        summary = new_df
        print('\n' + '=' * 80)
        print(' ' * 25 + 'RESULTS SUMMARY')
        print('=' * 80)
        print(summary.to_string(index=False))
        print('=' * 80)

    def show_status(self):
        print('\n' + '=' * 70)
        print(' Benchmark Data Status')
        print('=' * 70)
        for task in TASK_LIST:
            print(f'\n [{task}]')
            for mode in ['unaligned', 'aligned']:
                present = []
                for ds in ['vitaldb', 'mover', 'inspire', 'combined']:
                    if self._check_exists(task, mode, ds, 'train'):
                        present.append(ds)
                status = ', '.join(present) if present else '(none)'
                print(f'   {mode:12s}: {status}')
        print(f'\n BERT 缓存: {BERT_CACHE_DIR}')

def main():
    parser = argparse.ArgumentParser(description='实验运行器 v5')
    parser.add_argument('--exp', '-e', type=str, default='all', choices=['baseline', 'baseline_aligned', 'baseline_clustered', 'exp1', 'exp2', 'exp3', 'exp4', 'exp5', 'exp6', 'exp7', 'exp8', 'exp11', 'exp12', 'exp13', 'exp14', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T13', 'all', 'status'], help='实验 / status 查看数据状态')
    parser.add_argument('--task', '-t', type=str, default=None)
    parser.add_argument('--dataset', '-d', type=str, default=None)
    parser.add_argument('--algo', '-a', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42, help='随机种子（模型初始化/训练随机性, 数据划分固定不变）')
    parser.add_argument('--global_top', '-g', type=int, default=None, help='全局特征预算模式: 所有列值按频次合并后取 top N (迁移实验专用)')
    parser.add_argument('--shared_only', '-s', action='store_true', default=False, help='只保留 train/test 都非零的共享特征 (迁移实验专用)')
    args = parser.parse_args()
    runner = ExperimentRunner(seed=args.seed)
    if args.exp == 'status':
        runner.show_status()
        return
    dispatch = {'baseline': lambda: runner.run_baseline(args.task, args.dataset, args.algo), 'baseline_aligned': lambda: runner.run_baseline_aligned(args.task, args.dataset, args.algo), 'baseline_clustered': lambda: runner.run_baseline_clustered(args.task, args.dataset, args.algo), 'exp1': lambda: runner.run_exp1(args.task, args.algo), 'exp2': lambda: runner.run_exp2(args.task, args.algo), 'exp3': lambda: runner.run_exp3(args.task, args.algo, args.global_top, args.shared_only), 'exp4': lambda: runner.run_exp4(args.task, args.algo, args.global_top, args.shared_only), 'exp5': lambda: runner.run_exp5(args.task, args.algo), 'exp6': lambda: runner.run_exp6(args.task, args.algo), 'exp7': lambda: runner.run_exp7(args.task, args.algo), 'exp8': lambda: runner.run_exp8(args.task, args.algo), 'exp11': lambda: runner.run_exp11(args.task, args.algo), 'exp12': lambda: runner.run_exp12(args.task, args.algo), 'exp13': lambda: runner.run_exp13(args.task, args.algo), 'exp14': lambda: runner.run_exp14(args.task, args.algo), 'T1': lambda: runner.run_t1(args.task, args.algo), 'T2': lambda: runner.run_t2(args.task, args.algo), 'T3': lambda: runner.run_t3(args.task, args.algo), 'T4': lambda: runner.run_t4(args.task, args.algo), 'T5': lambda: runner.run_t5(args.task, args.algo), 'T6': lambda: runner.run_t6(args.task, args.algo), 'T13': lambda: runner.run_t13(args.task, args.algo), 'all': lambda: runner.run_all(args.task, args.dataset, args.algo)}
    dispatch[args.exp]()
    runner._save_results()
if __name__ == '__main__':
    main()

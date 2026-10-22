"""
一体化 Benchmark 构建脚本

合并原 sampler.py 的队列生成 + FeatureProcessor 的特征清洗，
产出可直接用于 ML 训练的数据文件。

流程:
  1. 分层采样生成 train/val/test 划分
  2. FeatureProcessor 清洗并编码特征
  3. 保存到 benchmark/{task}/{mode}/{dataset}/{split}.csv
  4. 保存 onehot 词汇表到 onehot_feature/{mode}/{dataset}.json

用法:
  python build_benchmark.py                          # 构建全部
  python build_benchmark.py --task asa                # 单个任务
  python build_benchmark.py --mode unaligned           # 仅未对齐
  python build_benchmark.py --task asa --dataset vitaldb
  python build_benchmark.py --dry-run                  # 只打印计划不执行

输出目录:
  benchmark/
  ├── asa/
  │   ├── unaligned/
  │   │   ├── vitaldb/
  │   │   │   ├── train.csv / train_y.csv
  │   │   │   ├── val.csv   / val_y.csv
  │   │   │   └── test.csv  / test_y.csv
  │   │   ├── mover/
  │   │   ├── inspire/
  │   │   └── combined/        (三数据集合并)
  │   └── aligned/  (同上)
  └── ...

  onehot_feature/
  ├── unaligned/
  │   ├── vitaldb.json
  │   ├── mover.json
  │   ├── inspire.json
  │   └── combined.json
  └── aligned/
      └── ...
"""
import os as _release_os
from release_runtime import expand_path as _release_expand, expand_config as _release_expand_config
import sys
import os
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set
from datetime import datetime
import pandas as pd
import numpy as np
import yaml
SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
PROJECT_DIR = os.path.join(SRC_DIR, '..')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('build_benchmark')
TASK_LIST = ['asa', 'aki', 'icu_admit', 'icu_los', 'icu_los_prolonged', 'hospital_los', 'hospital_los_prolonged', 'ioht', 'ioh', 'inhospital_mortality']
REGRESSION_TASKS = {'icu_los', 'hospital_los'}
BINARY_TASKS = {'asa', 'aki', 'icu_admit', 'icu_los_prolonged', 'hospital_los_prolonged', 'ioht', 'ioh', 'inhospital_mortality'}
TASK_DATASET_AVAILABILITY = {'asa': ['vitaldb', 'mover', 'inspire'], 'aki': ['vitaldb', 'inspire'], 'aki': ['vitaldb', 'inspire'], 'icu_admit': ['vitaldb', 'mover'], 'icu_los': ['vitaldb', 'inspire'], 'icu_los_prolonged': ['vitaldb', 'inspire'], 'hospital_los': ['vitaldb', 'mover'], 'hospital_los_prolonged': ['vitaldb', 'mover'], 'ioht': ['vitaldb', 'mover', 'inspire'], 'ioh': ['vitaldb', 'mover', 'inspire'], 'inhospital_mortality': ['vitaldb', 'mover', 'inspire']}
TASK_LABEL_FILE_MAP = {'icu_los': 'icu_los', 'icu_los_prolonged': 'icu_los', 'hospital_los': 'hospital_los', 'hospital_los_prolonged': 'hospital_los', 'asa_iv': 'asa'}
SMALL_THRESHOLD = 10000
SMALL_TEST_MIN_PER_CLASS = 20
LARGE_TEST_SIZE = 2000
LARGE_TRAIN_SIZE = 6000
LARGE_VAL_FROM_TRAIN = 2000
LARGE_MIN_PER_CLASS = 20
MAX_RESAMPLE_ATTEMPTS = 50

class StratifiedSampler:
    """
    配额制分层采样器

    核心逻辑:
      1. 优先保证测试集每类 >= min_per_class 个样本
      2. 其次保证训练集每类 >= min_per_class
      3. 验证集取剩余样本
      4. 各类别比例尽量接近全队列分布

    策略:
      - 先按配额分配正样本到各 split, 再从负样本池随机填充至目标大小
      - 小数据集 (≤10k): 6:2:2, min_per_class=20
      - 大数据集 (>10k): test=2000, train=8000, val=2000, min_per_class=20
    """

    def __init__(self, random_state: int=42):
        self.rs = random_state

    def _extract_label(self, labels_df: pd.DataFrame, task: str) -> pd.Series:
        candidates = [c for c in labels_df.columns if task in c.lower() and 'label' in c.lower()]
        if not candidates:
            candidates = [c for c in labels_df.columns if 'label' in c.lower()]
        if not candidates:
            raise ValueError(f'未找到 {task} 的标签列: {labels_df.columns.tolist()}')
        return labels_df[candidates[0]]

    def _stratify_numeric(self, y: pd.Series, n_bins: int=5) -> np.ndarray:
        clean = y.dropna()
        if len(clean) == 0:
            return np.array(['missing'] * len(y))
        try:
            bins = pd.qcut(clean, q=n_bins, labels=False, duplicates='drop')
            result = np.full(len(y), 'missing', dtype=object)
            result[clean.index] = bins.astype(str)
            return result
        except Exception:
            median = clean.median()
            result = np.where(y >= median, 'high', 'low')
            result[pd.isna(y)] = 'missing'
            return result

    def split(self, labels_df: pd.DataFrame, task: str, dataset_name: str, patient_map: Optional[Dict[str, str]]=None) -> Dict[str, List[str]]:
        """
        生成患者级 train/val/test 切分

        patient_map 提供 surgery_id -> patient_id 映射时，按患者分组切分，
        保证同一患者的所有手术只出现在一个 split（避免跨 split 数据泄露）。
        无映射时退化为样本级切分（兼容旧调用）。
        """
        from release_runtime import require_patient_map
        patient_map = require_patient_map(labels_df['surgery_id'], patient_map)
        y = self._extract_label(labels_df, task)
        valid_mask = y.notna()
        data = labels_df[valid_mask].copy()
        y_valid = y[valid_mask].copy()
        is_regression = task in REGRESSION_TASKS
        lb_col = '_label'
        if is_regression:
            strat_col = self._stratify_numeric(y_valid)
            data[lb_col] = strat_col
        else:
            data[lb_col] = y_valid.astype(int)
        if patient_map:
            data['_patient'] = data['surgery_id'].astype(str).map(patient_map).fillna(data['surgery_id'].astype(str))
            unit = data.groupby('_patient')[lb_col].agg(lambda s: s.value_counts().idxmax()).to_frame(name=lb_col)
        else:
            data['_patient'] = data['surgery_id'].astype(str)
            unit = data[[lb_col]].copy()
            unit.index = data['_patient']
        n_total = len(unit)
        if n_total <= SMALL_THRESHOLD:
            unit_splits = self._split_small(unit, lb_col)
        else:
            unit_splits = self._split_large(unit, lb_col)
        result: Dict[str, List[str]] = {}
        for k, udf in unit_splits.items():
            unit_ids = set(udf.index.astype(str))
            result[k] = data[data['_patient'].astype(str).isin(unit_ids)]['surgery_id'].tolist()
        return result

    def _quota_split(self, data: pd.DataFrame, label_col: str, target_sizes: Dict[str, int], min_per_class: int, max_total: Optional[int]=None) -> Dict[str, pd.DataFrame]:
        """
        配额制分层划分

        Args:
            data: 含 surgery_id 和 label 列
            label_col: 标签列名
            target_sizes: {'test': N_test, 'train': N_train, 'val': N_val}
            min_per_class: 每个 split 中各类别最少样本数

        算法:
          1. 对每个类别, 按比例计算各 split 需要的样本数
          2. 若期望数 < min_per_class, 提升到 min_per_class (优先 test > train > val)
          3. 从各类别样本池中随机取出配额
          4. 用负样本/剩余类别样本填充到目标大小
        """
        import random
        random.seed(self.rs)
        np.random.seed(self.rs)
        label_vals = sorted(data[label_col].unique())
        pools: Dict[str, pd.DataFrame] = {}
        for lv in label_vals:
            pool = data[data[label_col] == lv]
            if len(pool) > 0:
                pools[str(lv)] = pool.sample(frac=1, random_state=self.rs)
        total_per_class = {k: len(v) for k, v in pools.items()}
        total = sum(total_per_class.values())
        proportions = {k: v / total for k, v in total_per_class.items()}
        quotas: Dict[str, Dict[str, int]] = {sp: {} for sp in ['test', 'train', 'val']}
        expected = {}
        for sp in ['test', 'train', 'val']:
            expected[sp] = {}
            for lv in proportions:
                expected[sp][lv] = int(round(target_sizes.get(sp, 0) * proportions[lv]))
        remaining_pool = {k: len(v) for k, v in pools.items()}
        for sp in ['test', 'train', 'val']:
            tgt = target_sizes.get(sp, 0)
            alloc = {}
            allocated_so_far = 0
            for lv in label_vals:
                lv_s = str(lv)
                need = max(expected[sp].get(lv_s, 0), min_per_class)
                avail = remaining_pool.get(lv_s, 0)
                take = min(need, avail)
                alloc[lv_s] = take
                remaining_pool[lv_s] -= take
                allocated_so_far += take
            slack = tgt - allocated_so_far
            if slack > 0:
                rem_total = sum(remaining_pool.values())
                if rem_total > 0:
                    for lv in label_vals:
                        lv_s = str(lv)
                        rp = remaining_pool[lv_s]
                        if rp <= 0:
                            continue
                        prop = rp / rem_total
                        extra = min(int(round(slack * prop)), rp)
                        alloc[lv_s] = alloc.get(lv_s, 0) + extra
                        remaining_pool[lv_s] -= extra
                        allocated_so_far += extra
                        slack -= extra
                        if slack <= 0:
                            break
            quotas[sp] = alloc
        if max_total:
            total_allocated = sum((sum(q.values()) for q in quotas.values()))
            if total_allocated > max_total:
                excess = total_allocated - max_total
                n_classes = len(label_vals)
                min_keep = min_per_class * n_classes * 3
                reducible = total_allocated - min_keep
                if reducible > 0 and excess > 0:
                    scale = (reducible - excess) / reducible
                    for sp in ['train', 'val', 'test']:
                        for lv_s in quotas[sp]:
                            base = min_per_class
                            extra = quotas[sp][lv_s] - base
                            if extra > 0:
                                quotas[sp][lv_s] = base + int(extra * scale)
            total_after_scale = sum((sum(q.values()) for q in quotas.values()))
            if total_after_scale > max_total:
                final_excess = total_after_scale - max_total
                for sp in ['train', 'val']:
                    if final_excess <= 0:
                        break
                    for lv_s in list(quotas[sp].keys()):
                        if final_excess <= 0:
                            break
                        removable = quotas[sp][lv_s] - min_per_class
                        if removable > 0:
                            take = min(removable, final_excess)
                            quotas[sp][lv_s] -= take
                            final_excess -= take
        used_ids: Set[str] = set()
        result: Dict[str, pd.DataFrame] = {}
        for sp in ['test', 'train', 'val']:
            parts = []
            for lv_s, q in quotas[sp].items():
                if q <= 0:
                    continue
                lv = int(lv_s) if lv_s.lstrip('-').isdigit() else lv_s
                pool = pools[lv_s]
                available = pool[~pool.index.isin(used_ids)]
                take = min(q, len(available))
                if take > 0:
                    selected = available.head(take)
                    used_ids.update(selected.index.tolist())
                    parts.append(selected)
            if parts:
                result[sp] = pd.concat(parts)
            else:
                result[sp] = pd.DataFrame(columns=data.columns)
        leftover = data[~data.index.isin(used_ids)]
        if len(leftover) > 0 and len(result.get('train', pd.DataFrame())) < target_sizes.get('train', 0):
            used_set = set(result['train'].index) if 'train' in result else set()
            remaining = leftover[~leftover.index.isin(used_set)]
            if len(remaining) > 0:
                need = target_sizes.get('train', 0) - len(result.get('train', pd.DataFrame()))
                take = min(need, len(remaining))
                extra = remaining.sample(n=take, random_state=self.rs)
                if 'train' not in result or result['train'].empty:
                    result['train'] = extra
                else:
                    result['train'] = pd.concat([result['train'], extra])
        return result

    def _split_small(self, data: pd.DataFrame, label_col: str) -> Dict[str, pd.DataFrame]:
        n_total = len(data)
        n_test = max(int(n_total * 0.2), SMALL_TEST_MIN_PER_CLASS * data[label_col].nunique())
        n_train = max(int(n_total * 0.6), SMALL_TEST_MIN_PER_CLASS * data[label_col].nunique())
        n_val = n_total - n_test - n_train
        n_val = max(n_val, SMALL_TEST_MIN_PER_CLASS * data[label_col].nunique())
        if n_train + n_val + n_test > n_total:
            excess = n_train + n_val + n_test - n_total
            n_train -= excess // 2
            n_val -= excess - excess // 2
            n_train = max(n_train, SMALL_TEST_MIN_PER_CLASS * data[label_col].nunique())
            n_val = max(n_val, SMALL_TEST_MIN_PER_CLASS * data[label_col].nunique())
        splits = self._quota_split(data, label_col, target_sizes={'test': n_test, 'train': n_train, 'val': n_val}, min_per_class=SMALL_TEST_MIN_PER_CLASS)
        for sp, df in splits.items():
            if len(df) > 0:
                dist = df[label_col].value_counts().to_dict()
                logger.info(f'  [{sp}] n={len(df)} dist={dist}')
        return splits

    def _split_large(self, data: pd.DataFrame, label_col: str) -> Dict[str, pd.DataFrame]:
        n_total = len(data)
        n_test = min(LARGE_TEST_SIZE, int(n_total * 0.15))
        n_train = min(LARGE_TRAIN_SIZE, n_total - n_test - LARGE_MIN_PER_CLASS * data[label_col].nunique())
        n_val = min(LARGE_VAL_FROM_TRAIN, n_total - n_test - n_train)
        n_test = max(n_test, LARGE_MIN_PER_CLASS * data[label_col].nunique())
        n_train = max(n_train, LARGE_MIN_PER_CLASS * data[label_col].nunique())
        n_val = max(n_val, LARGE_MIN_PER_CLASS * data[label_col].nunique())
        MAX_TOTAL = LARGE_TRAIN_SIZE + LARGE_TEST_SIZE + LARGE_VAL_FROM_TRAIN
        splits = self._quota_split(data, label_col, target_sizes={'test': n_test, 'train': n_train, 'val': n_val}, min_per_class=LARGE_MIN_PER_CLASS, max_total=MAX_TOTAL)
        for sp, df in splits.items():
            if len(df) > 0:
                dist = df[label_col].value_counts().to_dict()
                logger.info(f'  [{sp}] n={len(df)} dist={dist}')
        return splits

class BenchmarkBuilder:
    """
    一体化 benchmark 构建器

    目录结构:
      benchmark/{task}/{mode}/{dataset}/
        train.csv, train_y.csv
        val.csv,   val_y.csv
        test.csv,  test_y.csv
        combined_*.csv (三数据集合并)

      onehot_feature/{mode}/
        {dataset}.json
        combined.json

    csv 格式: surgery_id 作为 index, 其余列为特征 (float32)
    y 文件: surgery_id 作为 index, 列名为 label_{task}
    """

    def __init__(self, data_root: str='dataset', output_dir: str='benchmark', vocab_dir: str='onehot_feature', max_onehot_categories: int=500, cache_dir: str='.benchmark_cache', label_filters: Optional[Dict[str, callable]]=None):
        self.data_root = Path(data_root)
        self.output_dir = Path(output_dir)
        self.vocab_dir = Path(vocab_dir)
        self.max_onehot_categories = max_onehot_categories
        self.cache_dir = Path(cache_dir)
        self.label_filters = label_filters or {}

    def build(self, tasks: Optional[List[str]]=None, modes: Optional[List[str]]=None, datasets: Optional[List[str]]=None, encode_text: bool=True, dry_run: bool=False):
        """
        主构建入口

        Args:
            tasks:     任务列表, None=全部
            modes:     模式列表 ['unaligned', 'aligned'], None=全部
            datasets:  数据集列表, None=按可用性自动选择
            encode_text: 是否编码文本字段
            dry_run:   只打印计划不执行
        """

    def _get_split_cache_path(self, task: str, mode: str) -> Path:
        return self.cache_dir / f'splits_{task}_{mode}.json'

    def _load_patient_map(self, dataset_name: str) -> Dict[str, str]:
        """加载 surgery_id -> patient_id 映射（用于患者级切分）

        从 feat_anes_vars.csv 读取（该列在 clean_and_merge 中被丢弃）。
        """
        path = self.data_root / dataset_name / 'prerisk_bench_feat' / 'feat_anes_vars.csv'
        if not path.exists():
            logger.warning(f'  无 patient 映射文件: {path}, 退化为样本级切分')
            return {}
        df = pd.read_csv(path, usecols=['surgery_id', 'patient_id'])
        mapping = {str(s): str(p) for s, p in zip(df['surgery_id'], df['patient_id'])}
        logger.info(f"  [{dataset_name}] patient 映射: {len(mapping)} 条 ({df['patient_id'].nunique()} 个患者)")
        return mapping

    def _load_split_cache(self, task: str, mode: str) -> Optional[Dict]:
        path = self._get_split_cache_path(task, mode)
        if path.exists():
            import json
            return json.loads(path.read_text())
        return None

    def _save_split_cache(self, task: str, mode: str, all_splits: Dict):
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self._get_split_cache_path(task, mode)
        import json
        path.write_text(json.dumps(all_splits, ensure_ascii=False))
        logger.info(f'    采样缓存已保存: {path}')

    def build(self, tasks: Optional[List[str]]=None, modes: Optional[List[str]]=None, datasets: Optional[List[str]]=None, encode_text: bool=True, dry_run: bool=False):
        if tasks is None:
            tasks = TASK_LIST
        if modes is None:
            modes = ['unaligned', 'aligned']
        if datasets is None:
            datasets = ['vitaldb', 'mover', 'inspire']
        print('=' * 70)
        print(f' Benchmark Builder')
        print(f' 数据根目录: {self.data_root.resolve()}')
        print(f' 输出目录:   {self.output_dir.resolve()}')
        print(f' 词汇表目录: {self.vocab_dir.resolve()}')
        print(f' 模式:       {modes}')
        print(f' 任务:       {tasks}')
        print(f' Dry Run:    {dry_run}')
        print('=' * 70)
        total = 0
        errors = 0
        for mode in modes:
            is_aligned = mode == 'aligned'
            logger.info(f"\n{'#' * 50}")
            logger.info(f' # MODE: {mode}')
            logger.info(f"{'#' * 50}")
            processor = self._create_processor(mode, encode_text)
            available_ds = self._get_available_datasets(mode, datasets)
            if not dry_run:
                logger.info(f'  fit() 学习词汇表: {available_ds}')
                processor.fit(available_ds)
            for task in tasks:
                task_ds = [d for d in TASK_DATASET_AVAILABILITY.get(task, []) if d in available_ds]
                if not task_ds:
                    logger.warning(f'  [{task}] 无可用的{mode}数据集, 跳过')
                    continue
                logger.info(f'\n  [{task}] 数据集: {task_ds}')
                try:
                    X_dict = {}
                    quality_ids = {}
                    for ds_name in task_ds:
                        X_raw = processor.clean_and_merge(ds_name)
                        if X_raw is None or len(X_raw) == 0:
                            logger.warning(f'    [{ds_name}] 无特征数据, 跳过')
                            continue
                        X_filtered = self._apply_quality_filter(X_raw, ds_name)
                        if len(X_filtered) == 0:
                            logger.warning(f'    [{ds_name}] 质量过滤后无样本, 跳过')
                            continue
                        X_dict[ds_name] = X_filtered
                        quality_ids[ds_name] = set(X_filtered.index.astype(str))
                        logger.info(f'    [{ds_name}] 过滤后特征: {X_filtered.shape}')
                    task_ds = [d for d in task_ds if d in X_dict]
                    if not task_ds:
                        logger.warning(f'    [{task}] 所有数据集均被过滤, 跳过')
                        continue
                    all_splits = self._load_split_cache(task, mode)
                    if all_splits is not None:
                        cache_stale = False
                        for ds_name in list(all_splits.keys()):
                            if ds_name == 'combined':
                                continue
                            if ds_name not in quality_ids:
                                cache_stale = True
                                break
                            cached_ids = set()
                            for sp in ['train', 'val', 'test']:
                                cached_ids |= set((str(s) for s in all_splits[ds_name].get(sp, [])))
                            if cached_ids - quality_ids[ds_name] or quality_ids[ds_name] - cached_ids:
                                cache_stale = True
                                break
                        if cache_stale:
                            logger.info(f'    采样缓存已过期（质量过滤后ID变化），重新采样')
                            all_splits = None
                    if all_splits is None:
                        all_splits: Dict[str, Dict[str, List[str]]] = {}
                        combined_train = []
                        combined_val = []
                        combined_test = []
                        for ds_name in task_ds:
                            labels = self._load_labels(task, ds_name)
                            if labels.empty:
                                logger.warning(f'    [{task}/{ds_name}] 无标签, 跳过')
                                continue
                            qids = quality_ids[ds_name]
                            labels = labels[labels['surgery_id'].astype(str).isin(qids)]
                            if labels.empty:
                                logger.warning(f'    [{task}/{ds_name}] 质量过滤后无标签, 跳过')
                                continue
                            if task in self.label_filters:
                                y_candidates = [c for c in labels.columns if 'label' in c.lower()]
                                if not y_candidates:
                                    logger.warning(f'    [{task}/{ds_name}] 无标签列, 跳过')
                                    continue
                                ycol = y_candidates[0]
                                filt = self.label_filters[task]
                                keep = labels[ycol].map(lambda v: filt(v)).fillna(False)
                                labels = labels[keep]
                                logger.info(f'    [{task}/{ds_name}] 标签过滤后: {len(labels)} (列 {ycol})')
                            sampler = StratifiedSampler()
                            pid_map = self._load_patient_map(ds_name)
                            splits = sampler.split(labels, task, ds_name, patient_map=pid_map)
                            all_splits[ds_name] = splits
                            combined_train.extend(splits['train'])
                            combined_val.extend(splits['val'])
                            combined_test.extend(splits['test'])
                        if not all_splits:
                            logger.warning(f'    [{task}] 无标签数据, 跳过')
                            continue
                        if len(task_ds) >= 2:
                            all_splits['combined'] = {'train': combined_train, 'val': combined_val, 'test': combined_test}
                        self._save_split_cache(task, mode, all_splits)
                        logger.info(f'    采样完成: {len(all_splits)} 数据集 + combined')
                    else:
                        logger.info(f'    从缓存加载采样: {len(all_splits)} 数据集')
                    if dry_run:
                        for ds in all_splits:
                            for sp in ['train', 'val', 'test']:
                                n = len(all_splits[ds][sp])
                                path = self._benchmark_path(task, mode, ds, sp)
                                logger.info(f'    [DRY] {path} ({n} 样本)')
                        total += sum((len(all_splits[ds][sp]) for ds in all_splits for sp in ['train', 'val', 'test']))
                        continue
                    for ds_name, splits in all_splits.items():
                        if ds_name == 'combined':
                            self._save_combined(X_dict, task_ds, splits, task, mode)
                        else:
                            X = X_dict.get(ds_name)
                            if X is None:
                                continue
                            self._save_dataset_splits(X, ds_name, splits, task, mode)
                    if not dry_run:
                        self._save_vocab(processor, mode, task_ds)
                    total += sum((len(ids) for splits in all_splits.values() for ids in splits.values()))
                    logger.info(f'    [{task}] 完成')
                except Exception as e:
                    logger.error(f'    [{task}] 失败: {e}')
                    import traceback
                    traceback.print_exc()
                    errors += 1
        print(f"\n{'=' * 70}")
        print(f' 完成. 处理 {total} 个样本, {errors} 个错误')
        print(f' 输出: {self.output_dir.resolve()}')
        print(f' 词汇表: {self.vocab_dir.resolve()}')
        print(f"{'=' * 70}")

    def _benchmark_path(self, task: str, mode: str, dataset: str, split: str, is_y: bool=False) -> Path:
        """benchmark/{task}/{mode}/{dataset}/{split}{_y}.csv"""
        fname = f"{split}{('_y' if is_y else '')}.csv"
        return self.output_dir / task / mode / dataset / fname

    def _create_processor(self, mode: str, encode_text: bool):
        from features.feature_processor import FeatureProcessor
        return FeatureProcessor(base_dir=str(self.data_root), mode=mode, max_onehot_categories=self.max_onehot_categories, encode_text=encode_text)

    def _get_available_datasets(self, mode: str, requested: List[str]) -> List[str]:
        """检查哪些数据集在指定 mode 下有特征文件"""
        subdir = 'alignment_feat' if mode == 'aligned' else 'prerisk_bench_feat'
        available = []
        for ds in requested:
            feat_dir = self.data_root / ds / subdir
            if feat_dir.exists() and any(feat_dir.iterdir()):
                available.append(ds)
            else:
                logger.warning(f'  {ds}/{subdir} 不存在或为空')
        return available

    def _load_labels(self, task: str, dataset: str) -> pd.DataFrame:
        """加载标签文件, 支持同一标签文件对应多个 task"""
        label_task = TASK_LABEL_FILE_MAP.get(task, task)
        p = self.data_root / dataset / 'prerisk_bench_labels' / f'labels_{label_task}.csv'
        if not p.exists():
            return pd.DataFrame()
        return pd.read_csv(p)

    def _filter_X(self, X: pd.DataFrame, surgery_ids: List[str]) -> pd.DataFrame:
        """按 surgery_id 列表过滤特征矩阵"""
        if X.index.name is None or X.index.name != 'surgery_id':
            X = X.reset_index()
            if 'surgery_id' not in X.columns:
                raise ValueError('特征矩阵无 surgery_id')
            X = X.set_index('surgery_id')
        ids_str = set((str(sid) for sid in surgery_ids))
        return X.loc[X.index.astype(str).isin(ids_str)]

    @staticmethod
    def _is_nonempty(series: pd.Series) -> pd.Series:
        """判断值非空（非 NaN、非空字符串、非 'nan' 字符串）"""
        return series.notna() & (series.astype(str).str.strip() != '') & (series.astype(str).str.strip().str.lower() != 'nan')

    def _apply_quality_filter(self, X: pd.DataFrame, ds_name: str) -> pd.DataFrame:
        """
        应用数据质量过滤:
          1. 硬性: sex, age, height, weight, preop_diagnosis, scheduled_procedure_name 非空
          2. 质量: >=1 个 hx_detail 非空 或 >=5 个检验值非空

        Returns:
            过滤后的 X (仅包含通过质量检查的 surgery_id)
        """
        n_before = len(X)
        mandatory = ['sex', 'age', 'height', 'weight', 'preop_diagnosis', 'scheduled_procedure_name']
        mask = pd.Series(True, index=X.index)
        for col in mandatory:
            if col in X.columns:
                mask &= self._is_nonempty(X[col])
            else:
                mask &= False
                logger.warning(f"    [{ds_name}] 质量过滤: 缺少必选列 '{col}'")
        hx_detail_cols = [c for c in X.columns if c.startswith('hx_') and ('detail' in c or 'details' in c) and ('_cn' not in c)]
        has_hx = pd.Series(False, index=X.index)
        for col in hx_detail_cols:
            has_hx |= self._is_nonempty(X[col])
        lab_val_cols = [c for c in X.columns if c.endswith('_val')]
        n_lab = pd.Series(0, index=X.index)
        for col in lab_val_cols:
            n_lab += self._is_nonempty(X[col]).astype(int)
        has_lab5 = n_lab >= 5
        quality_ok = has_hx | has_lab5
        mask = mask & quality_ok
        X_filtered = X[mask].copy()
        n_after = len(X_filtered)
        n_excluded = n_before - n_after
        hx_pass = has_hx[mask].sum()
        lab_pass = has_lab5[mask].sum()
        logger.info(f'    [{ds_name}] 质量过滤: {n_before} → {n_after} (-{n_excluded}, {n_excluded / n_before * 100:.1f}%)  [hx={hx_pass}, lab5={lab_pass}, hx_cols={len(hx_detail_cols)}, lab_cols={len(lab_val_cols)}]')
        return X_filtered

    @staticmethod
    def _identify_numeric_cols(X: pd.DataFrame) -> list:
        """识别需要中位数填充的数值列（*_val, age, height, weight, bmi 等）"""
        numeric_cols = []
        for col in X.columns:
            if col in ('surgery_id', 'patient_id', '_source_dataset'):
                continue
            if col.endswith('_flag') or col.endswith('_summary'):
                continue
            if col.endswith('_val') or col in ('age', 'height', 'weight', 'bmi'):
                numeric_cols.append(col)
        return numeric_cols

    def _apply_median_imputation(self, X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
        """
        用训练集中位数填充所有 split 的数值列缺失值。

        Returns:
            (X_train_imputed, X_val_imputed, X_test_imputed)
        """
        X_tr = X_train.copy()
        X_v = X_val.copy()
        X_te = X_test.copy()
        numeric_cols = self._identify_numeric_cols(X_tr)
        if not numeric_cols:
            return (X_tr, X_v, X_te)
        imputed_count = 0
        for col in numeric_cols:
            if col not in X_tr.columns:
                continue
            train_vals = pd.to_numeric(X_tr[col], errors='coerce')
            median_val = train_vals.median()
            if pd.isna(median_val):
                continue
            for X_sp in (X_tr, X_v, X_te):
                if col in X_sp.columns:
                    col_numeric = pd.to_numeric(X_sp[col], errors='coerce')
                    nan_before = col_numeric.isna().sum()
                    X_sp[col] = col_numeric.fillna(median_val)
                    if X_sp is X_tr:
                        imputed_count += nan_before
        if imputed_count > 0:
            logger.info(f'      中位数填充: {len(numeric_cols)} 个数值列, 训练集填充了 {imputed_count} 个缺失值 (median={median_val})')
        return (X_tr, X_v, X_te)

    @staticmethod
    def _get_source_dataset(surgery_id: str, ds_list: List[str]) -> str:
        """根据 surgery_id 前缀判断来源数据集（用于 combined splits 过滤）"""
        sid = str(surgery_id)
        for ds in ds_list:
            if ds == 'vitaldb' and sid.startswith('SNUH'):
                return ds
            elif ds == 'mover' and (sid.startswith('UCSF') or sid.startswith('MVR')):
                return ds
            elif ds == 'inspire' and (sid.startswith('WCH') or sid.startswith('INSP')):
                return ds
        return 'unknown'

    def _get_label_col(self, labels_df: pd.DataFrame, task: str) -> str:
        """获取精确的标签列名，优先精确匹配 label_{task}"""
        exact_col = f'label_{task}'
        if exact_col in labels_df.columns:
            return exact_col
        candidates = [c for c in labels_df.columns if 'label' in c.lower() and task in c.lower()]
        if candidates:
            return candidates[0]
        candidates = [c for c in labels_df.columns if 'label' in c.lower()]
        if candidates:
            return candidates[0]
        raise ValueError(f'标签文件中找不到 label 列: task={task}')

    def _save_dataset_splits(self, X: pd.DataFrame, ds_name: str, splits: Dict[str, List[str]], task: str, mode: str):
        """保存单个数据集的三分数据（含中位数填充）"""
        labels_df = self._load_labels(task, ds_name)
        label_col = self._get_label_col(labels_df, task)
        labels = labels_df.set_index('surgery_id')
        X_splits = {}
        for sp in ['train', 'val', 'test']:
            ids = splits.get(sp, [])
            if not ids:
                continue
            ids_set = set((str(s) for s in ids)) & set((str(s) for s in labels.index))
            X_sp = self._filter_X(X, ids)
            X_sp = X_sp.loc[X_sp.index.astype(str).isin(ids_set)]
            if len(X_sp) > 0:
                X_splits[sp] = X_sp
        if 'train' not in X_splits:
            logger.warning(f'      {ds_name}: 训练集为空, 跳过')
            return
        X_train, X_val, X_test = self._apply_median_imputation(X_splits.get('train', pd.DataFrame()), X_splits.get('val', pd.DataFrame()), X_splits.get('test', pd.DataFrame()))
        for sp, X_sp in [('train', X_train), ('val', X_val), ('test', X_test)]:
            if len(X_sp) == 0:
                continue
            ids_set = set((str(s) for s in splits.get(sp, []))) & set((str(s) for s in labels.index))
            labels_sp = labels.loc[labels.index.astype(str).isin(ids_set)][[label_col]]
            labels_sp.columns = [f'label_{task}']
            X_sp = X_sp.loc[X_sp.index.astype(str).isin(ids_set)]
            self._write_df(X_sp, self._benchmark_path(task, mode, ds_name, sp))
            self._write_df(labels_sp, self._benchmark_path(task, mode, ds_name, sp, is_y=True))
            logger.info(f'      {ds_name}_{sp}: X={X_sp.shape}, y={labels_sp.shape}')

    def _save_combined(self, X_dict: Dict[str, pd.DataFrame], ds_list: List[str], splits: Dict[str, List[str]], task: str, mode: str):
        """保存合并(cross-dataset)的三分数据（含中位数填充）"""
        all_y_parts = {}
        all_x_parts = {}
        for ds_name in ds_list:
            if ds_name not in X_dict:
                continue
            labels_df = self._load_labels(task, ds_name)
            try:
                label_col = self._get_label_col(labels_df, task)
            except ValueError:
                logger.warning(f'      [combined/{ds_name}] 无标签列, 跳过')
                continue
            labels = labels_df.set_index('surgery_id')
            for sp in ['train', 'val', 'test']:
                ids = splits.get(sp, [])
                if not ids:
                    continue
                ids_set = set((str(s) for s in ids)) & set((str(s) for s in labels.index))
                part = labels.loc[labels.index.astype(str).isin(ids_set)][[label_col]]
                part.columns = [f'label_{task}']
                if sp not in all_y_parts:
                    all_y_parts[sp] = []
                all_y_parts[sp].append(part)
                X_sp = self._filter_X(X_dict[ds_name], ids)
                X_sp = X_sp.loc[X_sp.index.astype(str).isin(ids_set)]
                if len(X_sp) > 0:
                    if sp not in all_x_parts:
                        all_x_parts[sp] = []
                    all_x_parts[sp].append(X_sp)
        if 'train' not in all_x_parts or len(all_x_parts['train']) == 0:
            logger.warning(f'      combined: 训练集为空, 跳过')
            return
        X_train = pd.concat(all_x_parts['train'])
        X_val = pd.concat(all_x_parts.get('val', [])) if 'val' in all_x_parts else pd.DataFrame()
        X_test = pd.concat(all_x_parts.get('test', [])) if 'test' in all_x_parts else pd.DataFrame()
        X_train, X_val, X_test = self._apply_median_imputation(X_train, X_val, X_test)
        for X_sp in [X_train, X_val, X_test]:
            if len(X_sp) == 0:
                continue
            source_map = {}
            for ds_name in ds_list:
                if ds_name in X_dict:
                    X_ds = self._filter_X(X_dict[ds_name], list(X_sp.index.astype(str)))
                    for sid in X_ds.index.astype(str):
                        source_map[sid] = ds_name
            X_sp['_source_dataset'] = X_sp.index.astype(str).map(source_map)
        for sp, X_sp in [('train', X_train), ('val', X_val), ('test', X_test)]:
            if len(X_sp) == 0 or sp not in all_y_parts:
                continue
            y_comb = pd.concat(all_y_parts[sp])
            common = set((str(s) for s in X_sp.index)) & set((str(s) for s in y_comb.index))
            X_sp = X_sp.loc[X_sp.index.astype(str).isin(common)]
            y_comb = y_comb.loc[y_comb.index.astype(str).isin(common)]
            self._write_df(X_sp, self._benchmark_path(task, mode, 'combined', sp))
            self._write_df(y_comb, self._benchmark_path(task, mode, 'combined', sp, is_y=True))
            logger.info(f'      combined_{sp}: X={X_sp.shape}, y={y_comb.shape}')

    @staticmethod
    def _write_df(df: pd.DataFrame, path: Path):
        """写入 CSV"""
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path)

    def _save_vocab(self, processor, mode: str, datasets: List[str]):
        """保存 onehot 词汇表到 onehot_feature/{mode}/{dataset}.json

        每个数据集的词汇表只包含该数据集独有的列值。
        """
        if not processor.fitted:
            return
        for ds in datasets:
            from features.feature_processor import FeatureProcessor
            ds_processor = FeatureProcessor(base_dir=str(self.data_root), mode=mode, max_onehot_categories=self.max_onehot_categories, encode_text=processor.encode_text)
            ds_processor.fit([ds])
            self._save_single_vocab(ds_processor, mode, ds)
        combined_processor = FeatureProcessor(base_dir=str(self.data_root), mode=mode, max_onehot_categories=self.max_onehot_categories, encode_text=processor.encode_text)
        combined_processor.fit(datasets)
        self._save_single_vocab(combined_processor, mode, 'combined')

    def _save_single_vocab(self, processor, mode: str, dataset: str):
        """保存单个数据集的 onehot 词汇表"""
        path = self.vocab_dir / mode / f'{dataset}.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        vocab = {'mode': mode, 'dataset': dataset, 'built_at': datetime.now().isoformat(), 'columns': processor.onehot_vocabularies}
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)
        total_cats = sum((len(v) for v in processor.onehot_vocabularies.values()))
        logger.info(f'    词汇表: {path} ({len(processor.onehot_vocabularies)} 列, {total_cats} 类)')

def main():
    parser = argparse.ArgumentParser(description='Benchmark 数据构建器')
    parser.add_argument('--task', '-t', type=str, default=None, help='指定任务 (如 asa, aki)')
    parser.add_argument('--mode', '-m', type=str, default=None, choices=['unaligned', 'aligned'], help='数据模式')
    parser.add_argument('--dataset', '-d', type=str, default=None, help='指定数据集 (如 vitaldb)')
    parser.add_argument('--encode-text', dest='encode_text', action='store_true', default=True, help='编码文本字段 (默认)')
    parser.add_argument('--no-encode-text', dest='encode_text', action='store_false', help='不编码文本 (仅数值特征)')
    parser.add_argument('--max-categories', type=int, default=500, help='独热编码最大类别数')
    parser.add_argument('--dry-run', action='store_true', help='只打印计划不执行')
    parser.add_argument('--data-root', type=str, default='dataset', help='数据集根目录')
    parser.add_argument('--output-dir', type=str, default='benchmark', help='输出目录')
    args = parser.parse_args()
    tasks = [args.task] if args.task else None
    modes = [args.mode] if args.mode else None
    datasets = [args.dataset] if args.dataset else None
    builder = BenchmarkBuilder(data_root=args.data_root, output_dir=args.output_dir, max_onehot_categories=args.max_categories)
    builder.build(tasks=tasks, modes=modes, datasets=datasets, encode_text=args.encode_text, dry_run=args.dry_run)
if __name__ == '__main__':
    main()

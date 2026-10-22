import os as _release_os
from release_runtime import expand_path as _release_expand, expand_config as _release_expand_config
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
import logging
import json
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SamplingQueueManager:
    """
    分层采样队列管理器

    每个任务一个文件夹，包含各数据集的train/val/test三分队列文件。

    目录结构:
        sample_queues/
        ├── asa/
        │   ├── vitaldb_train.csv / vitaldb_val.csv / vitaldb_test.csv
        │   ├── mover_train.csv / mover_val.csv / mover_test.csv
        │   ├── inspire_train.csv / inspire_val.csv / inspire_test.csv
        │   └── combined_train.csv / combined_val.csv / combined_test.csv
        ├── aki/
        │   └── ...
        └── inhospital_mortality/
            └── ...

    **采样策略 - 分层抽样:**
    规则1: 可用样本 ≤ 10000 → 训练:验证:测试 = 6:2:2
    规则2: 可用样本 > 10000 → 测试2000, 训练8000(从剩余抽), 验证2000(从训练抽)
    约束: 测试集每类最少10个样本(≤10k时)或20个样本(>10k时), 不满足则重新抽样
    回归任务: 对数值分桶后分层抽样
    """
    QUEUE_DIR = 'sample_queues'
    SMALL_THRESHOLD = 10000
    SMALL_TEST_MIN_PER_CLASS = 10
    LARGE_TEST_SIZE = 2000
    LARGE_TRAIN_SIZE = 8000
    LARGE_VAL_FROM_TRAIN = 2000
    LARGE_MIN_PER_CLASS = 20
    MAX_RESAMPLE_ATTEMPTS = 50
    TASK_LIST = ['asa', 'aki', 'icu_admit', 'icu_los', 'hospital_los', 'ioht', 'ioh', 'inhospital_mortality']
    REGRESSION_TASKS = {'icu_los', 'hospital_los'}
    BINARY_TASKS = {'asa', 'aki', 'icu_admit', 'ioht', 'ioh', 'inhospital_mortality'}
    TASK_DATASET_AVAILABILITY = {'asa': ['vitaldb', 'mover', 'inspire'], 'aki': ['vitaldb', 'inspire'], 'icu_admit': ['vitaldb', 'mover'], 'icu_los': ['vitaldb', 'mover'], 'hospital_los': ['vitaldb', 'mover'], 'ioht': ['vitaldb', 'mover', 'inspire'], 'ioh': ['vitaldb', 'mover', 'inspire'], 'inhospital_mortality': ['vitaldb', 'mover', 'inspire']}

    def __init__(self, queue_dir: str=QUEUE_DIR):
        self.queue_dir = Path(queue_dir)
        self.queue_dir.mkdir(parents=True, exist_ok=True)

    def _get_queue_path(self, dataset_name: str, task_name: str, split: str) -> Path:
        """返回: sample_queues/{task_name}/{dataset_name}_{split}.csv"""
        return self.queue_dir / task_name / f'{dataset_name}_{split}.csv'

    def _all_splits_exist(self, dataset_name: str, task_name: str) -> bool:
        for split in ['train', 'val', 'test']:
            if not self._get_queue_path(dataset_name, task_name, split).exists():
                return False
        return True

    def _extract_label_column(self, labels_df: pd.DataFrame, task_name: str) -> pd.Series:
        if labels_df.empty:
            raise ValueError('标签DataFrame为空')
        candidates = [col for col in labels_df.columns if task_name in col.lower() and 'label' in col.lower()]
        if not candidates:
            candidates = [col for col in labels_df.columns if 'label' in col.lower()]
        if not candidates:
            raise ValueError(f'未找到 {task_name} 的标签列, 可用列: {labels_df.columns.tolist()}')
        return labels_df[candidates[0]]

    def _is_regression_task(self, task_name: str) -> bool:
        return task_name in self.REGRESSION_TASKS

    def _stratify_numeric(self, y: pd.Series, n_bins: int=5) -> np.ndarray:
        """回归任务: 将连续值分桶用于分层抽样"""
        clean_y = y.dropna()
        if len(clean_y) == 0:
            return np.array(['missing'] * len(y))
        try:
            bins = pd.qcut(clean_y, q=n_bins, labels=False, duplicates='drop')
            result = np.full(len(y), 'missing', dtype=object)
            result[clean_y.index] = bins.astype(str)
            return result
        except Exception:
            median = clean_y.median()
            result = np.where(y >= median, 'high', 'low')
            result[pd.isna(y)] = 'missing'
            return result

    def _check_class_minimums(self, sampled_labels: pd.Series, min_per_class: int) -> Tuple[bool, Dict]:
        """检查每类是否满足最少样本数约束"""
        class_counts = sampled_labels.value_counts().to_dict()
        violations = {k: v for k, v in class_counts.items() if v < min_per_class}
        return (len(violations) == 0, violations)

    def _stratified_split_small(self, data: pd.DataFrame, label_col: str, min_per_class: int=SMALL_TEST_MIN_PER_CLASS, random_state: int=42) -> Dict[str, pd.DataFrame]:
        """
        小数据集(≤10k): 6:2:2 分层划分

        保证测试集每类至少 min_per_class 个样本
        """
        n_total = len(data)
        n_test = max(int(n_total * 0.2), min_per_class * data[label_col].nunique())
        n_val = max(int(n_total * 0.2), min_per_class * data[label_col].nunique())
        n_train = n_total - n_test - n_val
        for attempt in range(self.MAX_RESAMPLE_ATTEMPTS):
            rs = random_state + attempt
            shuffled = data.sample(frac=1, random_state=rs).reset_index(drop=True)
            test_size = min(n_test, len(shuffled) - n_val)
            val_size = min(n_val, len(shuffled) - test_size)
            test_frac = test_size / len(shuffled)
            val_frac = val_size / (len(shuffled) - test_size)
            from sklearn.model_selection import train_test_split
            temp, test_df = train_test_split(shuffled, test_size=test_frac, stratify=shuffled[label_col], random_state=rs)
            ok, violations = self._check_class_minimums(test_df[label_col], min_per_class)
            if ok and len(test_df) >= min_per_class * data[label_col].nunique():
                train_df, val_df = train_test_split(temp, test_size=val_frac, stratify=temp[label_col], random_state=rs + 1)
                logger.info(f'  小数据集6:2:2划分成功 (第{attempt + 1}次尝试)')
                return {'train': train_df.reset_index(drop=True), 'val': val_df.reset_index(drop=True), 'test': test_df.reset_index(drop=True)}
        logger.warning(f'  经{self.MAX_RESAMPLE_ATTEMPTS}次尝试仍未完全满足约束, 使用最佳结果')
        temp, test_df = train_test_split(data, test_size=0.2, stratify=data[label_col], random_state=42)
        train_df, val_df = train_test_split(temp, test_size=0.25, stratify=temp[label_col], random_state=43)
        return {'train': train_df.reset_index(drop=True), 'val': val_df.reset_index(drop=True), 'test': test_df.reset_index(drop=True)}

    def _stratified_split_large(self, data: pd.DataFrame, label_col: str, min_per_class: int=LARGE_MIN_PER_CLASS, random_state: int=42) -> Dict[str, pd.DataFrame]:
        """
        大数据集(>10k): 测试2000 + 训练8000 + 验证2000(从训练中取)

        约束:
        - 测试集每类≥min_per_class, 标签分布与全量一致
        - 训练集8000, 所有类别都有覆盖且≥min_per_class
        - 验证集2000从训练集中分层抽取
        """
        from sklearn.model_selection import train_test_split
        unique_classes = data[label_col].nunique()
        for attempt in range(self.MAX_RESAMPLE_ATTEMPTS):
            rs = random_state + attempt
            test_size = min(self.LARGE_TEST_SIZE, len(data) - self.LARGE_TRAIN_SIZE - 100)
            test_frac = test_size / len(data)
            remaining, test_df = train_test_split(data, test_size=test_frac, stratify=data[label_col], random_state=rs)
            ok, violations = self._check_class_minimums(test_df[label_col], min_per_class)
            if not ok or len(test_df) < min_per_class * unique_classes:
                continue
            train_target = min(self.LARGE_TRAIN_SIZE, len(remaining) - 100)
            if len(remaining) <= train_target:
                train_df = remaining.copy()
            else:
                train_df, _ = train_test_split(remaining, train_size=train_target / len(remaining), stratify=remaining[label_col], random_state=rs + 1)
            train_ok, train_violations = self._check_class_minimums(train_df[label_col], min_per_class)
            if not train_ok:
                continue
            val_size = min(self.LARGE_VAL_FROM_TRAIN, len(train_df) - min_per_class * unique_classes)
            if val_size < min_per_class * unique_classes:
                val_size = int(len(train_df) * 0.2)
            if len(train_df) <= val_size + min_per_class * unique_classes:
                val_df = pd.DataFrame(columns=data.columns)
            else:
                train_df, val_df = train_test_split(train_df, test_size=val_size / len(train_df), stratify=train_df[label_col], random_state=rs + 2)
            logger.info(f'  大数据集8k+2k+2k划分成功 (第{attempt + 1}次尝试)')
            return {'train': train_df.reset_index(drop=True), 'val': val_df.reset_index(drop=True), 'test': test_df.reset_index(drop=True)}
        logger.warning(f'  经{self.MAX_RESAMPLE_ATTEMPTS}次尝试, 使用宽松策略')
        _, test_df = train_test_split(data, test_size=min(2000, len(data) * 0.15), stratify=data[label_col], random_state=42)
        remaining = data[~data.index.isin(test_df.index)]
        train_df = remaining.sample(n=min(8000, len(remaining)), random_state=42)
        if len(train_df) > 400:
            train_df, val_df = train_test_split(train_df, test_size=0.25, stratify=train_df[label_col], random_state=43)
        else:
            val_df = pd.DataFrame(columns=data.columns)
        return {'train': train_df.reset_index(drop=True), 'val': val_df.reset_index(drop=True), 'test': test_df.reset_index(drop=True)}

    def generate_queue(self, labels_df: pd.DataFrame, dataset_name: str, task_name: str, force_regenerate: bool=False) -> Dict[str, pd.DataFrame]:
        """
        为指定数据集和任务生成train/val/test三分采样队列

        Args:
            labels_df: 包含surgery_id和标签的DataFrame
            dataset_name: 数据集名称
            task_name: 任务名称
            force_regenerate: 是否强制重新生成

        Returns:
            {'train': DataFrame, 'val': DataFrame, 'test': DataFrame}
            每个DataFrame仅含surgery_id列
        """
        if self._all_splits_exist(dataset_name, task_name) and (not force_regenerate):
            result = {}
            for split in ['train', 'val', 'test']:
                path = self._get_queue_path(dataset_name, task_name, split)
                result[split] = pd.read_csv(path)
            logger.info(f'已存在队列文件: {dataset_name}/{task_name}')
            return result
        try:
            y = self._extract_label_column(labels_df, task_name)
            valid_mask = y.notna()
            valid_data = pd.DataFrame({'surgery_id': labels_df.loc[valid_mask, 'surgery_id'], 'label': y[valid_mask]}).dropna(subset=['surgery_id'])
            n_valid = len(valid_data)
            if n_valid == 0:
                logger.warning(f'[{dataset_name}/{task_name}] 无有效样本')
                empty_df = pd.DataFrame(columns=['surgery_id'])
                return {'train': empty_df, 'val': empty_df, 'test': empty_df}
            is_regression = self._is_regression_task(task_name)
            if is_regression:
                valid_data['stratify_col'] = self._stratify_numeric(valid_data['label'])
                label_col = 'stratify_col'
            else:
                label_col = 'label'
            original_dist = valid_data[label_col].value_counts(normalize=True).to_dict()
            if n_valid <= self.SMALL_THRESHOLD:
                splits = self._stratified_split_small(valid_data, label_col, min_per_class=self.SMALL_TEST_MIN_PER_CLASS)
                mode_desc = f'小数据集({n_valid}) 6:2:2'
            else:
                splits = self._stratified_split_large(valid_data, label_col, min_per_class=self.LARGE_MIN_PER_CLASS)
                mode_desc = f'大数据集({n_valid}) 8k+2k+2k'
            result = {}
            for split_name, split_df in splits.items():
                queue_df = split_df[['surgery_id']].reset_index(drop=True)
                queue_path = self._get_queue_path(dataset_name, task_name, split_name)
                queue_path.parent.mkdir(parents=True, exist_ok=True)
                queue_df.to_csv(queue_path, index=False)
                result[split_name] = queue_df
            log_lines = [f'[{dataset_name}/{task_name}] 队列生成完成 ({mode_desc}):', f'  原始数据: {n_valid} 样本, {valid_data[label_col].nunique()} 类/桶']
            for split_name, split_df in splits.items():
                split_dist = split_df[label_col].value_counts().to_dict()
                log_lines.append(f"  {split_name}: {len(split_df)} 样本 | 分布: {dict(((k, v) for k, v in list(split_dist.items())[:5]))}{('...' if len(split_dist) > 5 else '')}")
            logger.info('\n'.join(log_lines))
            return result
        except Exception as e:
            logger.error(f'生成队列失败 [{dataset_name}/{task_name}]: {e}')
            import traceback
            traceback.print_exc()
            empty_df = pd.DataFrame(columns=['surgery_id'])
            return {'train': empty_df, 'val': empty_df, 'test': empty_df}

    def generate_combined_queue(self, all_labels: Dict[str, pd.DataFrame], task_name: str, force_regenerate: bool=False) -> Dict[str, pd.DataFrame]:
        """
        生成联合训练用的合并队列 (train/val/test)

        从每个可用数据集合并后统一分层划分
        """
        if self._all_splits_exist('combined', task_name) and (not force_regenerate):
            result = {}
            for split in ['train', 'val', 'test']:
                path = self._get_queue_path('combined', task_name, split)
                result[split] = pd.read_csv(path)
            logger.info(f'已存在联合队列: combined/{task_name}')
            return result
        available_datasets = self.TASK_DATASET_AVAILABILITY.get(task_name, [])
        all_valid_data = []
        ds_sizes = {}
        for ds_name in available_datasets:
            if ds_name not in all_labels or all_labels[ds_name].empty:
                continue
            try:
                y = self._extract_label_column(all_labels[ds_name], task_name)
                valid_data = pd.DataFrame({'surgery_id': all_labels[ds_name].loc[y.notna(), 'surgery_id'], 'label': y[y.notna()], '_source_ds': ds_name}).dropna(subset=['surgery_id'])
                if len(valid_data) > 0:
                    all_valid_data.append(valid_data)
                    ds_sizes[ds_name] = len(valid_data)
            except Exception as e:
                logger.warning(f'  {ds_name} 数据准备失败: {e}')
        if not all_valid_data:
            logger.warning(f'[combined/{task_name}] 无可用数据')
            empty_df = pd.DataFrame(columns=['surgery_id'])
            return {'train': empty_df, 'val': empty_df, 'test': empty_df}
        combined_valid = pd.concat(all_valid_data, ignore_index=True)
        is_regression = self._is_regression_task(task_name)
        if is_regression:
            combined_valid['stratify_col'] = self._stratify_numeric(combined_valid['label'])
            label_col = 'stratify_col'
        else:
            label_col = 'label'
        n_total = len(combined_valid)
        if n_total <= self.SMALL_THRESHOLD:
            splits = self._stratified_split_small(combined_valid, label_col, min_per_class=self.SMALL_TEST_MIN_PER_CLASS)
            mode_desc = f'联合小数据集({n_total}) 6:2:2'
        else:
            splits = self._stratified_split_large(combined_valid, label_col, min_per_class=self.LARGE_MIN_PER_CLASS)
            mode_desc = f'联合大数据集({n_total}) 8k+2k+2k'
        result = {}
        for split_name, split_df in splits.items():
            queue_df = split_df[['surgery_id']].reset_index(drop=True)
            queue_path = self._get_queue_path('combined', task_name, split_name)
            queue_path.parent.mkdir(parents=True, exist_ok=True)
            queue_df.to_csv(queue_path, index=False)
            result[split_name] = queue_df
        source_dist_all = combined_valid['_source_ds'].value_counts().to_dict()
        log_lines = [f'[combined/{task_name}] 联合队列生成完成 ({mode_desc}):', f'  总可用: {n_total} 样本 (来源: {ds_sizes})']
        for split_name, split_df in splits.items():
            src_dist = split_df['_source_ds'].value_counts().to_dict() if '_source_ds' in split_df.columns else {}
            lbl_dist = split_df[label_col].value_counts().to_dict()
            log_lines.append(f'  {split_name}: {len(split_df)} 样本 | 来源: {src_dist} | 分布: {list(lbl_dist.items())[:5]}')
        logger.info('\n'.join(log_lines))
        return result

    def load_queue(self, dataset_name: str, task_name: str, split: str='train') -> Set[str]:
        """加载指定split的采样队列"""
        queue_path = self._get_queue_path(dataset_name, task_name, split)
        if not queue_path.exists():
            logger.warning(f'队列文件不存在: {queue_path}')
            return set()
        df = pd.read_csv(queue_path)
        return set(df['surgery_id'].astype(str).tolist())

    def load_all_splits(self, dataset_name: str, task_name: str) -> Dict[str, Set[str]]:
        """加载某任务的所有split队列"""
        result = {}
        for split in ['train', 'val', 'test']:
            result[split] = self.load_queue(dataset_name, task_name, split)
        return result

    def filter_by_queue(self, df: pd.DataFrame, dataset_name: str, task_name: str, split: str='train') -> pd.DataFrame:
        """用指定split的采样队列过滤DataFrame"""
        queue = self.load_queue(dataset_name, task_name, split)
        if not queue:
            logger.warning(f'[{dataset_name}/{task_name}] {split}队列为空, 返回全部数据')
            return df
        mask = df['surgery_id'].astype(str).isin(queue)
        filtered = df[mask].copy()
        logger.info(f'队列过滤 [{dataset_name}/{task_name}/{split}]: {len(df)} -> {len(filtered)} 样本')
        return filtered

    def get_all_queues_info(self) -> Dict:
        """获取所有队列的信息摘要"""
        info = {}
        for task_dir in self.queue_dir.iterdir():
            if not task_dir.is_dir():
                continue
            task_name = task_dir.name
            info[task_name] = {}
            for queue_file in task_dir.glob('*.csv'):
                stem = queue_file.stem
                parts = stem.rsplit('_', 1)
                if len(parts) == 2 and parts[1] in ['train', 'val', 'test']:
                    ds_name, split_name = parts
                    if ds_name not in info[task_name]:
                        info[task_name][ds_name] = {}
                    df = pd.read_csv(queue_file)
                    info[task_name][ds_name][split_name] = len(df)
        return info

def create_sampler(queue_dir: str='sample_queues') -> SamplingQueueManager:
    return SamplingQueueManager(queue_dir=queue_dir)

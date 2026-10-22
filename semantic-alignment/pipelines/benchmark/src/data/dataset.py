import os as _release_os
from release_runtime import expand_path as _release_expand, expand_config as _release_expand_config
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:

    def __init__(self, data_root: str, dataset_name: str, data_mode: str='unaligned', use_sampling: bool=True):
        self.data_root = Path(data_root)
        self.dataset_name = dataset_name
        self.dataset_path = self.data_root / dataset_name
        self.data_mode = data_mode
        self.use_sampling = use_sampling
        from data.sampler import SamplingQueueManager
        self.sampler = SamplingQueueManager()
        if data_mode == 'unaligned':
            self.feat_path = self.dataset_path / 'prerisk_bench_feat'
            self.label_path = self.dataset_path / 'prerisk_bench_labels'
            logger.info(f'使用未对齐数据 (prerisk_bench): {dataset_name}')
        elif data_mode == 'aligned':
            self.feat_path = self.dataset_path / 'alignment_feat'
            self.label_path = self.dataset_path / 'alignment_labels'
            logger.info(f'使用已对齐数据 (alignment): {dataset_name}')
        else:
            raise ValueError(f"不支持的数据模式: {data_mode}. 支持: ['unaligned', 'aligned']")
        self.ID_COLUMNS = {'surgery_id', 'patient_id'}
        self.SUMMARY_PATTERN = '_summary'

    def load_features(self, feature_type: str) -> pd.DataFrame:
        if self.data_mode == 'aligned':
            file_path = self.feat_path / f'{feature_type}_aligned.csv'
        else:
            file_path = self.feat_path / f'{feature_type}.csv'
        if not file_path.exists():
            logger.warning(f'特征文件不存在: {file_path}')
            return pd.DataFrame()
        df = pd.read_csv(file_path, low_memory=False)
        logger.info(f'[{self.data_mode}] 加载特征: {feature_type}, 形状: {df.shape}')
        return df

    def load_all_features(self) -> Dict[str, pd.DataFrame]:
        feature_types = ['feat_pat_demo', 'feat_anes_vars', 'feat_surg_vars', 'feat_test_vars', 'feat_past_his', 'feat_med_rec']
        features = {}
        for feat_type in feature_types:
            df = self.load_features(feat_type)
            if not df.empty:
                features[feat_type] = df
        logger.info(f'[{self.data_mode}] {self.dataset_name} 加载了 {len(features)} 个特征文件')
        return features

    def load_label(self, label_name: str) -> pd.DataFrame:
        if self.data_mode == 'aligned':
            label_file = self.label_path / 'labels.csv'
        else:
            label_file = self.label_path / f'labels_{label_name}.csv'
        if not label_file.exists():
            logger.warning(f'标签文件不存在: {label_file}')
            return pd.DataFrame()
        df = pd.read_csv(label_file, low_memory=False)
        logger.info(f'[{self.data_mode}] 加载标签: {label_name}, 形状: {df.shape}')
        return df

    def merge_features_and_labels(self, label_name: str, split: str='train', feature_types: Optional[List[str]]=None) -> Tuple[pd.DataFrame, pd.Series]:
        labels_df = self.load_label(label_name)
        if labels_df.empty:
            raise ValueError(f'无法加载标签: {label_name}')
        if self.use_sampling:
            labels_df = self.sampler.filter_by_queue(labels_df, self.dataset_name, label_name, split=split)
            if labels_df.empty:
                raise ValueError(f'采样队列为空 [{self.dataset_name}/{label_name}/{split}], 请先运行 python main.py --generate-queues')
        if feature_types is None:
            features_dict = self.load_all_features()
        else:
            features_dict = {}
            for feat_type in feature_types:
                df = self.load_features(feat_type)
                if not df.empty:
                    features_dict[feat_type] = df
        merged_features = None
        for feat_type, df in features_dict.items():
            if merged_features is None:
                merged_features = df.copy()
            else:
                on_cols = ['surgery_id', 'patient_id']
                available_on_cols = [col for col in on_cols if col in merged_features.columns and col in df.columns]
                if available_on_cols:
                    merged_features = merged_features.merge(df, on=available_on_cols, how='inner', suffixes=('', f'_{feat_type}'))
        if self.data_mode == 'aligned':
            label_col = [col for col in labels_df.columns if 'label' in col.lower() and label_name in col.lower()]
        else:
            label_col = [col for col in labels_df.columns if 'label' in col.lower()]
        if not label_col:
            raise ValueError(f'未找到标签列: {label_name}, 可用列: {labels_df.columns.tolist()}')
        final_merged = merged_features.merge(labels_df[['surgery_id', label_col[0]]], on='surgery_id', how='inner')
        columns_to_exclude = set(self.ID_COLUMNS)
        columns_to_exclude.update([col for col in final_merged.columns if self.SUMMARY_PATTERN in col.lower()])
        columns_to_exclude.add(label_col[0])
        X = final_merged.drop(columns=list(columns_to_exclude), errors='ignore')
        y = final_merged[label_col[0]]
        logger.info(f'[{self.data_mode}/{split}] 合并完成 - 特征维度: {X.shape[1]}, 样本数: {X.shape[0]}, 标签分布: {dict(y.value_counts())}')
        return (X, y)

    def load_train_val_test(self, label_name: str, feature_types: Optional[List[str]]=None) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
        """一次性加载train/val/test三分数据"""
        result = {}
        for split in ['train', 'val', 'test']:
            try:
                X, y = self.merge_features_and_labels(label_name, split=split, feature_types=feature_types)
                result[split] = (X, y)
            except Exception as e:
                logger.warning(f'[{split}] 加载失败: {e}')
        return result

    def get_actual_feature_info(self) -> Dict:
        features_dict = self.load_all_features()
        merged = None
        for feat_type, df in features_dict.items():
            if merged is None:
                merged = df.copy()
            else:
                on_cols = list(self.ID_COLUMNS & set(merged.columns) & set(df.columns))
                if on_cols:
                    merged = merged.merge(df, on=on_cols, how='inner', suffixes=('', f'_{feat_type}'))
        if merged is None:
            return {'total_features': 0, 'numeric_features': [], 'categorical_features': [], 'duplicate_columns': []}
        exclude_cols = set(self.ID_COLUMNS)
        exclude_cols.update((col for col in merged.columns if self.SUMMARY_PATTERN in col.lower()))
        all_features = [col for col in merged.columns if col not in exclude_cols]
        numeric_features = [f for f in all_features if pd.api.types.is_numeric_dtype(merged[f])]
        categorical_features = [f for f in all_features if pd.api.types.is_object_dtype(merged[f])]
        duplicate_columns = [f for f in all_features if any((f.endswith(f'_{ft}') for ft in features_dict.keys()))]
        info = {'total_features': len(all_features), 'numeric_features': numeric_features, 'categorical_features': categorical_features, 'duplicate_columns': duplicate_columns, 'all_feature_names': all_features}
        logger.info(f'[{self.data_mode}] {self.dataset_name} 实际特征统计:')
        logger.info(f"  总特征数: {info['total_features']}")
        logger.info(f'  数值型 ({len(numeric_features)}): {numeric_features[:5]}...')
        logger.info(f'  类别/文本型 ({len(categorical_features)}): {categorical_features[:8]}...')
        logger.info(f'  跨表重复字段 ({len(duplicate_columns)}): {duplicate_columns[:5]}...')
        return info

class MultiDatasetLoader:

    def __init__(self, data_root: str, use_sampling: bool=True, sample_size: int=500):
        self.data_root = data_root
        self.use_sampling = use_sampling
        self.sample_size = sample_size
        self.loaders = {}

    def get_loader(self, dataset_name: str, data_mode: str='unaligned') -> DataLoader:
        key = f'{dataset_name}_{data_mode}'
        if key not in self.loaders:
            self.loaders[key] = DataLoader(self.data_root, dataset_name, data_mode, use_sampling=self.use_sampling)
        return self.loaders[key]

    def load_dataset_for_task(self, dataset_name: str, task_name: str, data_mode: str='unaligned', split: str='train', feature_types: Optional[List[str]]=None) -> Tuple[pd.DataFrame, pd.Series]:
        loader = self.get_loader(dataset_name, data_mode)
        return loader.merge_features_and_labels(task_name, split=split, feature_types=feature_types)

    def get_available_labels(self, dataset_name: str, data_mode: str='unaligned') -> List[str]:
        loader = self.get_loader(dataset_name, data_mode)
        label_files = list(loader.label_path.glob('labels_*.csv'))
        if data_mode == 'aligned' and (not label_files):
            labels_file = loader.label_path / 'labels.csv'
            if labels_file.exists():
                df = pd.read_csv(labels_file, nrows=1)
                label_cols = [col.replace('labels_', '').split('_label_')[0] for col in df.columns if 'label' in col.lower()]
                return list(set(label_cols))
        labels = [f.stem.replace('labels_', '') for f in label_files]
        return labels

    def load_aligned_datasets_for_joint_training(self, task_name: str, split: str='train', datasets: Optional[List[str]]=None) -> Tuple[pd.DataFrame, pd.Series]:
        if datasets is None:
            datasets = ['vitaldb', 'mover', 'inspire']
        all_X = []
        all_y = []
        for ds_name in datasets:
            try:
                X, y = self.load_dataset_for_task(ds_name, task_name, data_mode='aligned', split=split)
                all_X.append(X)
                all_y.append(y)
                logger.info(f'已加载 {ds_name}[{split}]: {X.shape[0]} 个样本')
            except Exception as e:
                logger.warning(f'无法加载 {ds_name}/{task_name}: {e}')
                continue
        if not all_X:
            raise ValueError(f'任务 {task_name} 没有可用数据')
        X_combined = pd.concat(all_X, ignore_index=True)
        y_combined = pd.concat(all_y, ignore_index=True)
        logger.info(f'联合训练数据[{split}] - 总样本: {X_combined.shape[0]}, 特征维度: {X_combined.shape[1]}')
        return (X_combined, y_combined)

    def load_unaligned_for_joint_training(self, task_name: str, common_fields: List[str], split: str='train', datasets: Optional[List[str]]=None) -> Tuple[pd.DataFrame, pd.Series]:
        from data.sampler import SamplingQueueManager
        sampler = SamplingQueueManager()
        if datasets is None:
            datasets = ['vitaldb', 'mover', 'inspire']
        all_X = []
        all_y = []
        for ds_name in datasets:
            try:
                loader = self.get_loader(ds_name, 'unaligned')
                labels_df = loader.load_label(task_name)
                if self.use_sampling:
                    labels_df = sampler.filter_by_queue(labels_df, 'combined', task_name, split=split)
                if labels_df.empty:
                    continue
                label_col = [col for col in labels_df.columns if 'label' in col.lower()]
                if not label_col:
                    continue
                features_dict = loader.load_all_features()
                merged_feat = None
                for feat_type, df in features_dict.items():
                    if merged_feat is None:
                        merged_feat = df.copy()
                    else:
                        on_cols = ['surgery_id', 'patient_id']
                        avail = [c for c in on_cols if c in merged_feat.columns and c in df.columns]
                        if avail:
                            merged_feat = merged_feat.merge(df, on=avail, how='inner')
                available_fields = [f for f in common_fields if f in merged_feat.columns]
                id_cols = ['surgery_id', 'patient_id']
                keep_cols = id_cols + available_fields
                feat_filtered = merged_feat[keep_cols] if available_fields else merged_feat
                merged = feat_filtered.merge(labels_df[['surgery_id', label_col[0]]], on='surgery_id', how='inner')
                X = merged.drop(columns=id_cols + [label_col[0]], errors='ignore')
                y = merged[label_col[0]]
                all_X.append(X)
                all_y.append(y)
                logger.info(f'  {ds_name}[{split}]: {X.shape[0]} 样本, {X.shape[1]} 特征')
            except Exception as e:
                logger.warning(f'  {ds_name} 加载失败: {e}')
                continue
        if not all_X:
            raise ValueError(f'任务 {task_name} 无可用数据[{split}]')
        X_combined = pd.concat(all_X, ignore_index=True)
        y_combined = pd.concat(all_y, ignore_index=True)
        logger.info(f'[EXP1-未对齐联合][{split}] 合并后: {X_combined.shape[0]} 样本, {X_combined.shape[1]} 特征')
        return (X_combined, y_combined)

    def generate_all_sample_queues(self, force: bool=False) -> Dict:
        """生成所有采样队列(train/val/test三分)。返回: {task: {dataset: {split: n}}}"""
        from data.sampler import SamplingQueueManager
        sampler = SamplingQueueManager()
        results = {}
        for task_name in sampler.TASK_LIST:
            available_ds = sampler.TASK_DATASET_AVAILABILITY.get(task_name, [])
            results[task_name] = {}
            all_labels = {}
            for ds_name in available_ds:
                try:
                    loader = self.get_loader(ds_name, 'unaligned')
                    labels_df = loader.load_label(task_name)
                    if not labels_df.empty:
                        all_labels[ds_name] = labels_df
                        queue_result = sampler.generate_queue(labels_df, ds_name, task_name, force_regenerate=force)
                        results[task_name][ds_name] = {s: len(df) for s, df in queue_result.items()}
                except Exception as e:
                    logger.warning(f'跳过 {ds_name}/{task_name}: {e}')
            if len(all_labels) >= 2:
                combined_result = sampler.generate_combined_queue(all_labels, task_name, force_regenerate=force)
                results[task_name]['combined'] = {s: len(df) for s, df in combined_result.items()}
        total_queues = sum((len(ds_splits) for task_results in results.values() for ds_splits in task_results.values()))
        logger.info(f'\n队列生成完成: {total_queues} 个队列文件')
        return results
if __name__ == '__main__':
    loader = MultiDatasetLoader('dataset', use_sampling=False)
    print('\n' + '=' * 70)
    print(' [TEST] Data Loading Test')
    print('=' * 70)
    print('\n--- 未对齐数据 (prerisk_bench) ---')
    unaligned_labels = loader.get_available_labels('vitaldb', 'unaligned')
    print(f'VitalDB可用标签: {unaligned_labels}')
    try:
        X_unaligned, y_unaligned = loader.load_dataset_for_task('vitaldb', 'asa', 'unaligned')
        print(f'\nVitalDB ASA (未对齐):')
        print(f'  特征形状: {X_unaligned.shape}')
        print(f'  前5个特征: {list(X_unaligned.columns[:5])}')
    except Exception as e:
        print(f'错误: {e}')

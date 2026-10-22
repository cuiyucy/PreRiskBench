import os as _release_os
from release_runtime import expand_path as _release_expand, expand_config as _release_expand_config
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Dict, List, Optional
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:

    def __init__(self, test_size: float=0.2, val_size: float=0.1, random_state: int=42, cv_folds: int=5):
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.cv_folds = cv_folds
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.numeric_columns = []
        self.binary_columns = []
        self.categorical_columns = []

    def identify_column_types(self, X: pd.DataFrame) -> None:
        binary_cols = []
        numeric_cols = []
        for col in X.select_dtypes(include=[np.number]).columns:
            uniq = set(X[col].dropna().unique())
            if uniq <= {0, 1}:
                binary_cols.append(col)
            else:
                numeric_cols.append(col)
        self.numeric_columns = numeric_cols
        self.binary_columns = binary_cols
        self.categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
        logger.info(f'数值列: {len(self.numeric_columns)}, OneHot列(不标准化): {len(self.binary_columns)}, 分类列: {len(self.categorical_columns)}')

    def encode_categorical(self, X: pd.DataFrame) -> pd.DataFrame:
        X_encoded = X.copy()
        for col in self.categorical_columns:
            if col in X_encoded.columns:
                le = LabelEncoder()
                X_encoded[col] = X_encoded[col].astype(str).fillna('missing')
                X_encoded[col] = le.fit_transform(X_encoded[col])
                self.label_encoders[col] = le
        return X_encoded

    def handle_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in X.columns:
            if X[col].dtype in ['object', 'category']:
                X[col] = X[col].fillna('missing')
            else:
                if X[col].isna().all():
                    fill_val = 0
                else:
                    med = X[col].median()
                    fill_val = med if not pd.isna(med) else 0
                X[col] = X[col].fillna(fill_val)
        nan_counts = X.isna().sum()
        if nan_counts.sum() > 0:
            cols_with_nan = nan_counts[nan_counts > 0].index.tolist()
            X[cols_with_nan] = X[cols_with_nan].fillna(0)
        return X

    def remove_id_and_summary_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        id_cols = ['patient_id', 'surgery_id', 'mrn']
        columns_to_drop = [col for col in X.columns if col.lower() in id_cols]
        if columns_to_drop:
            X = X.drop(columns=columns_to_drop)
            logger.info(f'删除 ID 列: {columns_to_drop}')
        return X

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        X = self.remove_id_and_summary_columns(X)
        self.identify_column_types(X)
        X = self.handle_missing_values(X)
        X = self.encode_categorical(X)
        if self.numeric_columns:
            X[self.numeric_columns] = self.scaler.fit_transform(X[self.numeric_columns])
        return (X, y)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = self.remove_id_and_summary_columns(X)
        X = self.handle_missing_values(X)
        X = self.encode_categorical(X)
        if self.numeric_columns:
            X[self.numeric_columns] = self.scaler.transform(X[self.numeric_columns])
        return X

    def split_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state, stratify=y)
        logger.info(f'训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}')
        return (X_train, X_test, y_train, y_test)

    def get_cross_validation_splits(self, X: pd.DataFrame, y: pd.Series):
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        return skf.split(X, y)

class TaskDataHandler:

    def __init__(self):
        self.preprocessors = {}

    def get_preprocessor(self, task_name: str) -> DataPreprocessor:
        if task_name not in self.preprocessors:
            self.preprocessors[task_name] = DataPreprocessor()
        return self.preprocessors[task_name]

    def prepare_task_data(self, X: pd.DataFrame, y: pd.Series, task_name: str) -> Dict:
        preprocessor = self.get_preprocessor(task_name)
        X_processed, y_processed = preprocessor.fit_transform(X, y)
        X_train, X_test, y_train, y_test = preprocessor.split_data(X_processed, y_processed)
        return {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test, 'preprocessor': preprocessor, 'feature_names': X_processed.columns.tolist(), 'n_features': X_processed.shape[1]}
if __name__ == '__main__':
    from data.dataset import MultiDatasetLoader
    loader = MultiDatasetLoader('dataset')
    handler = TaskDataHandler()
    print('\n=== 测试数据预处理流程 ===')
    try:
        X, y = loader.load_dataset_for_task('vitaldb', 'asa')
        data_dict = handler.prepare_task_data(X, y, 'asa')
        print(f'\n任务: ASA预测')
        print(f"特征数量: {data_dict['n_features']}")
        print(f"训练集: {data_dict['X_train'].shape}")
        print(f"测试集: {data_dict['X_test'].shape}")
        print(f"标签分布 (训练集):\n{data_dict['y_train'].value_counts()}")
    except Exception as e:
        print(f'错误: {e}')

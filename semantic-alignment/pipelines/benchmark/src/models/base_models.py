import os as _release_os
from release_runtime import expand_path as _release_expand, expand_config as _release_expand_config
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.utils.class_weight import compute_class_weight
from typing import Dict, Any, Optional, Tuple
import xgboost as xgb
import logging
import joblib
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseModel:

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.best_params = None
        self.is_fitted = False

    def train(self, X_train, y_train, **kwargs) -> 'BaseModel':
        raise NotImplementedError

    def predict(self, X) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError('模型尚未训练')
        return self.model.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError('模型尚未训练')
        return self.model.predict_proba(X)

    def save_model(self, path: str) -> None:
        joblib.dump({'model': self.model, 'model_name': self.model_name, 'best_params': self.best_params, 'is_fitted': self.is_fitted}, path)
        logger.info(f'模型已保存: {path}')

    def load_model(self, path: str) -> 'BaseModel':
        data = joblib.load(path)
        self.model = data['model']
        self.best_params = data['best_params']
        self.is_fitted = data['is_fitted']
        logger.info(f'模型已加载: {path}')
        return self

class LogisticRegressionModel(BaseModel):

    def __init__(self, params: Optional[Dict]=None):
        super().__init__('Logistic Regression')
        default_params = {'C': 1.0, 'max_iter': 1000, 'class_weight': 'balanced', 'random_state': 42, 'solver': 'lbfgs'}
        self.params = {**default_params, **(params or {})}
        self.model = LogisticRegression(**self.params)

    def train(self, X_train, y_train, use_grid_search: bool=False, param_grid: Optional[Dict]=None, is_regression: bool=False) -> 'LogisticRegressionModel':
        if is_regression:
            from sklearn.linear_model import Ridge
            self.params = {'alpha': 1.0, 'random_state': 42}
            self.model = Ridge(**self.params)
            self.model.fit(X_train, y_train)
            self._is_regressor = True
        elif use_grid_search and param_grid:
            grid_param_grid = param_grid or {'C': [0.01, 0.1, 1.0, 10.0], 'max_iter': [1000]}
            grid_search = GridSearchCV(self.model, grid_param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            self.model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
            logger.info(f'Grid Search最佳参数: {self.best_params}')
        else:
            self.model.fit(X_train, y_train)
        self.is_fitted = True
        logger.info(f'{self.model_name} 训练完成')
        return self

class RandomForestModel(BaseModel):

    def __init__(self, params: Optional[Dict]=None):
        super().__init__('Random Forest')
        default_params = {'n_estimators': 200, 'max_depth': 20, 'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1}
        self.params = {**default_params, **(params or {})}
        self.model = RandomForestClassifier(**self.params)

    def train(self, X_train, y_train, use_grid_search: bool=False, param_grid: Optional[Dict]=None, is_regression: bool=False) -> 'RandomForestModel':
        if is_regression:
            from sklearn.ensemble import RandomForestRegressor
            self.model = RandomForestRegressor(n_estimators=self.params.get('n_estimators', 200), max_depth=self.params.get('max_depth', 20), random_state=self.params.get('random_state', 42), n_jobs=-1)
            self.model.fit(X_train, y_train)
            self._is_regressor = True
        elif use_grid_search and param_grid:
            grid_param_grid = param_grid or {'n_estimators': [100, 200, 300], 'max_depth': [10, 20, 30]}
            grid_search = GridSearchCV(self.model, grid_param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            self.model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
            logger.info(f'Grid Search最佳参数: {self.best_params}')
        else:
            self.model.fit(X_train, y_train)
        self.is_fitted = True
        logger.info(f'{self.model_name} 训练完成')
        return self

class XGBoostModel(BaseModel):

    def __init__(self, params: Optional[Dict]=None):
        super().__init__('XGBoost')
        default_params = {'n_estimators': 1000, 'max_depth': 8, 'learning_rate': 0.05, 'objective': 'multi:softmax', 'eval_metric': 'mlogloss', 'random_state': 42, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 0.1, 'reg_lambda': 1.0, 'min_child_weight': 3}
        self.params = {**default_params, **(params or {})}
        self.model = xgb.XGBClassifier(**self.params)
        self._label_offset = 0

    def train(self, X_train, y_train, use_grid_search: bool=False, param_grid: Optional[Dict]=None, X_val=None, y_val=None, is_regression: bool=False) -> 'XGBoostModel':
        import numpy as np
        y_arr = np.asarray(y_train).ravel()
        n_unique = len(np.unique(y_arr[~pd.isna(y_arr)]))
        n_classes = n_unique
        is_multiclass = n_classes > 2 and (not is_regression)
        label_offset = 0
        sample_weight = None
        if is_regression or n_unique > 20:
            self.params['objective'] = 'reg:squarederror'
            self.params.pop('num_class', None)
            self.params.pop('eval_metric', None)
            y_train = y_arr
            self.model = xgb.XGBRegressor(**self.params)
            self._is_regressor = True
            self._label_offset = 0
        elif is_multiclass:
            self.params['objective'] = 'multi:softprob'
            self.params['num_class'] = n_classes
            self.params['eval_metric'] = 'mlogloss'
            min_label = int(np.min(y_arr[~pd.isna(y_arr)]))
            if min_label != 0:
                label_offset = min_label
                self._label_offset = min_label
                y_train = y_arr - min_label
            else:
                self._label_offset = 0
                y_train = y_arr
            classes = np.unique(y_train)
            class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
            sample_weight = np.array([class_weights[list(classes).index(c)] for c in y_train])
            self.model = xgb.XGBClassifier(**self.params)
            self._is_regressor = False
        else:
            self.params['objective'] = 'binary:logistic'
            self.params['eval_metric'] = 'auc'
            self.params.pop('num_class', None)
            y_train = y_arr
            self._label_offset = 0
            classes = np.unique(y_train)
            if len(classes) == 2:
                neg_count = np.sum(y_train == classes[0])
                pos_count = np.sum(y_train == classes[1])
                if pos_count > neg_count:
                    neg_count, pos_count = (pos_count, neg_count)
                scale_pos_weight = neg_count / max(pos_count, 1)
                self.params['scale_pos_weight'] = scale_pos_weight
                logger.info(f'    二分类 scale_pos_weight={scale_pos_weight:.2f} (neg={neg_count}, pos={pos_count})')
            self.model = xgb.XGBClassifier(**self.params)
            self._is_regressor = False
        import re

        def _sanitize_names(df):
            bad = re.compile('[\\[\\],<>]')
            new_cols = [bad.sub('_', str(c)) for c in df.columns]
            if list(new_cols) != list(df.columns):
                df = df.copy()
                df.columns = new_cols
            return df
        X_train = _sanitize_names(X_train)
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        if isinstance(y_train, (pd.Series, pd.DataFrame)):
            y_train = np.asarray(y_train).ravel()
        if use_grid_search and param_grid:
            grid_param_grid = param_grid or {'n_estimators': [100, 200, 300], 'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.05, 0.1]}
            grid_search = GridSearchCV(self.model, grid_param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            self.model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
            logger.info(f'Grid Search最佳参数: {self.best_params}')
        else:
            eval_set = [(X_train, y_train)]
            fit_kwargs = {'sample_weight': sample_weight, 'verbose': False}
            if X_val is not None and y_val is not None:
                X_val = _sanitize_names(X_val)
                if isinstance(X_val, pd.DataFrame):
                    X_val = X_val.values
                if isinstance(y_val, (pd.Series, pd.DataFrame)):
                    y_val = np.asarray(y_val).ravel()
                y_val = y_val - label_offset
                eval_set.append((X_val, y_val))
                from xgboost import callback as xgb_callback
                self.model.set_params(callbacks=[xgb_callback.EarlyStopping(rounds=50)])
            self.model.fit(X_train, y_train, eval_set=eval_set, **fit_kwargs)
        self.is_fitted = True
        logger.info(f'{self.model_name} 训练完成')
        return self

    def predict(self, X) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError('模型尚未训练')
        preds = self.model.predict(X)
        if self._label_offset:
            preds = preds + self._label_offset
        return preds

    def get_feature_importance(self) -> Dict[str, float]:
        if not self.is_fitted:
            raise ValueError('模型尚未训练')
        importance = self.model.feature_importances_
        feature_names = self.model.get_booster().feature_names
        return dict(zip(feature_names, importance))

class SVMModel(BaseModel):
    """支持向量机分类器 (SVM)

    使用 LinearSVC + CalibratedClassifierCV 实现概率输出。
    LinearSVC 比 SVC(kernel='linear') 快很多，特别是大样本场景。
    默认使用 linear kernel（适合高维稀疏的临床 one-hot 特征）。
    """

    def __init__(self, params: Optional[Dict]=None):
        super().__init__('SVM')
        self.params_dict = {**(params or {})}
        self._model_calibrated = None
        self._model_raw = None

    def train(self, X_train, y_train, **kwargs) -> 'SVMModel':
        from sklearn.svm import LinearSVC
        from sklearn.calibration import CalibratedClassifierCV
        params = {'C': self.params_dict.get('C', 1.0), 'class_weight': self.params_dict.get('class_weight', 'balanced'), 'random_state': self.params_dict.get('random_state', 42), 'max_iter': self.params_dict.get('max_iter', 2000), 'dual': 'auto'}
        self._model_raw = LinearSVC(**params).fit(X_train, y_train)
        self._model_calibrated = CalibratedClassifierCV(self._model_raw, method='sigmoid', cv='prefit').fit(X_train, y_train)
        self.is_fitted = True
        logger.info(f'{self.model_name} 训练完成')
        return self

    def predict(self, X) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError('模型尚未训练')
        return self._model_raw.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError('模型尚未训练')
        return self._model_calibrated.predict_proba(X)

    def save_model(self, path: str) -> None:
        import joblib
        joblib.dump({'_model_raw': self._model_raw, '_model_calibrated': self._model_calibrated, 'model_name': self.model_name, 'params_dict': self.params_dict, 'is_fitted': self.is_fitted}, path)
        logger.info(f'SVM模型已保存: {path}')

    def load_model(self, path: str) -> 'SVMModel':
        import joblib
        data = joblib.load(path)
        self._model_raw = data['_model_raw']
        self._model_calibrated = data['_model_calibrated']
        self.model_name = data['model_name']
        self.params_dict = data['params_dict']
        self.is_fitted = data['is_fitted']
        logger.info(f'SVM模型已加载: {path}')
        return self

class KNNModel(BaseModel):
    """K近邻分类器"""

    def __init__(self, params: Optional[Dict]=None):
        super().__init__('KNN')
        from sklearn.neighbors import KNeighborsClassifier
        default_params = {'n_neighbors': 5, 'weights': 'distance', 'metric': 'cosine', 'n_jobs': -1}
        self.params = {**default_params, **(params or {})}
        self.model = KNeighborsClassifier(**self.params)

    def train(self, X_train, y_train, **kwargs) -> 'KNNModel':
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        logger.info(f'{self.model_name} 训练完成')
        return self

    def predict_proba(self, X) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError('模型尚未训练')
        return self.model.predict_proba(X)

class NaiveBayesModel(BaseModel):
    """朴素贝叶斯分类器 (GaussianNB, 适合标准化后的连续特征)"""

    def __init__(self, params: Optional[Dict]=None):
        super().__init__('Naive Bayes')
        from sklearn.naive_bayes import GaussianNB
        default_params = {'var_smoothing': 1e-09}
        self.params = {**default_params, **(params or {})}
        self.model = GaussianNB(**self.params)

    def train(self, X_train, y_train, **kwargs) -> 'NaiveBayesModel':
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        logger.info(f'{self.model_name} 训练完成')
        return self

    def predict_proba(self, X) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError('模型尚未训练')
        return self.model.predict_proba(X)

class ModelFactory:

    @staticmethod
    def create_model(model_type: str, params: Optional[Dict]=None) -> BaseModel:
        models = {'lr': LogisticRegressionModel, 'rf': RandomForestModel, 'xgboost': XGBoostModel, 'mlp': _lazy_mlp, 'cnn': _lazy_cnn, 'tabtransformer': _lazy_tabtransformer}
        if model_type not in models:
            raise ValueError(f'不支持的模型类型: {model_type}. 支持的类型: {list(models.keys())}')
        lazy_fns = (_lazy_mlp, _lazy_cnn, _lazy_tabtransformer)
        creator = models[model_type]
        if callable(creator) and creator in lazy_fns:
            return creator(params)
        return models[model_type](params)

    @staticmethod
    def get_all_models() -> Dict[str, BaseModel]:
        return {'lr': ModelFactory.create_model('lr'), 'rf': ModelFactory.create_model('rf'), 'xgboost': ModelFactory.create_model('xgboost'), 'svm': ModelFactory.create_model('svm'), 'knn': ModelFactory.create_model('knn'), 'nb': ModelFactory.create_model('nb'), 'mlp': ModelFactory.create_model('mlp'), 'gnn': ModelFactory.create_model('gnn')}

def _lazy_mlp(params=None):
    """延迟导入 MLPModel"""
    from src.models.neural_models import MLPModel
    return MLPModel(params)

def _lazy_gnn(params=None):
    """延迟导入 GNNModel"""
    from src.models.neural_models import GNNModel
    return GNNModel(params)

def _lazy_tabresnet(params=None):
    """延迟导入 TabResNetModel"""
    from src.models.neural_models import TabResNetModel
    return TabResNetModel(params)

def _lazy_tabtransformer(params=None):
    """延迟导入 TabTransformerModel"""
    from src.models.neural_models import TabTransformerModel
    return TabTransformerModel(params)

def _lazy_cnn(params=None):
    """延迟导入 CNNModel"""
    from src.models.neural_models import CNNModel
    return CNNModel(params)
if __name__ == '__main__':
    import numpy as np
    print('\n=== 测试模型训练 ===')
    np.random.seed(42)
    X_train = np.random.randn(100, 10)
    y_train = np.random.randint(0, 2, 100)
    X_test = np.random.randn(20, 10)
    for model_type in ['lr', 'rf', 'xgboost']:
        print(f'\n--- {model_type.upper()} ---')
        model = ModelFactory.create_model(model_type)
        model.train(X_train, y_train)
        predictions = model.predict(X_test)
        proba = model.predict_proba(X_test)
        print(f'预测形状: {predictions.shape}')
        print(f'概率形状: {proba.shape}')
        print(f'预测分布: {np.bincount(predictions)}')

import os as _release_os
from release_runtime import expand_path as _release_expand, expand_config as _release_expand_config
import numpy as np
import pandas as pd
import json
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, cohen_kappa_score, mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
from typing import Dict, List, Optional, Tuple
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent

class MetricsCalculator:

    @staticmethod
    def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray]=None, task_type: str='binary') -> Dict:
        metrics = {}
        n_classes = len(np.unique(y_true))
        if task_type == 'regression':
            metrics['MAE'] = float(mean_absolute_error(y_true, y_pred))
            metrics['RMSE'] = float(mean_squared_error(y_true, y_pred) ** 0.5)
            metrics['R²'] = float(r2_score(y_true, y_pred))
            metrics['ExplainedVar'] = float(explained_variance_score(y_true, y_pred))
        elif task_type == 'binary' or n_classes == 2:
            metrics['Macro-F1'] = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
            metrics['Accuracy'] = float(accuracy_score(y_true, y_pred))
            metrics['Precision'] = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
            metrics['Recall'] = float(recall_score(y_true, y_pred, average='macro', zero_division=0))
            metrics['Kappa'] = float(cohen_kappa_score(y_true, y_pred))
            if y_proba is not None:
                try:
                    if len(y_proba.shape) > 1 and y_proba.shape[1] > 1:
                        y_prob = y_proba[:, 1]
                    else:
                        y_prob = np.asarray(y_proba).ravel()
                    metrics['AUROC'] = float(roc_auc_score(y_true, y_prob))
                    metrics['AUPRC'] = float(average_precision_score(y_true, y_prob))
                except Exception:
                    metrics['AUROC'] = 0.0
                    metrics['AUPRC'] = 0.0
        else:
            metrics['Macro-F1'] = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
            metrics['Kappa'] = float(cohen_kappa_score(y_true, y_pred))
            metrics['Accuracy'] = float(accuracy_score(y_true, y_pred))
            try:
                from sklearn.metrics import cohen_kappa_score as qwk
                metrics['QWK'] = float(qwk(y_true, y_pred, weights='quadratic'))
            except Exception:
                metrics['QWK'] = 0.0
            metrics['MAE'] = float(mean_absolute_error(y_true, y_pred))
            metrics['RMSE'] = float(mean_squared_error(y_true, y_pred) ** 0.5)
            if y_proba is not None:
                try:
                    metrics['AUROC'] = float(roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro'))
                    from sklearn.preprocessing import label_binarize
                    classes = np.unique(y_true)
                    y_bin = label_binarize(y_true, classes=classes)
                    if y_bin.shape[1] == 1:
                        y_bin = np.hstack([1 - y_bin, y_bin])
                    metrics['AUPRC'] = float(average_precision_score(y_bin, y_proba, average='macro'))
                except Exception:
                    metrics['AUROC'] = 0.0
                    metrics['AUPRC'] = 0.0
        if task_type == 'regression':
            metrics['Confusion Matrix'] = []
        else:
            cm = confusion_matrix(y_true, y_pred)
            metrics['Confusion Matrix'] = cm.tolist()
        return metrics

    @staticmethod
    def render_confusion_matrix_md(cm: List, dataset_name: str) -> str:
        n = len(cm)
        header = '| True \\ Pred | ' + ' | '.join([f'Pred {i}' for i in range(n)]) + ' |'
        sep = '|' + '---|' * (n + 1)
        rows = []
        for i in range(n):
            row_vals = ' | '.join([str(x) for x in cm[i]])
            rows.append(f'| **True {i}** | {row_vals} |')
        return f'# Confusion Matrix for {dataset_name}\n\n{header}\n{sep}\n' + '\n'.join(rows) + '\n'

class ExperimentEvaluator:

    def __init__(self, output_dir: Optional[str]=None):
        self.calculator = MetricsCalculator()
        self.results = {}
        self.output_base = Path(output_dir) if output_dir else PROJECT_DIR / 'outputs'

    def evaluate_model(self, model, X_test: pd.DataFrame, y_test: pd.Series, experiment_name: str, task_name: str, dataset_name: str, algorithm: str='unknown', X_val: Optional[pd.DataFrame]=None, y_val: Optional[pd.Series]=None, prior_correct: bool=False, train_classes: Optional[np.ndarray]=None) -> Dict:
        y_pred = model.predict(X_test)
        try:
            y_proba = model.predict_proba(X_test)
        except Exception:
            y_proba = None
        n_classes = len(np.unique(y_test.values))
        REGRESSION_TASKS = {'icu_los', 'hospital_los'}
        is_regression = task_name.lower() in REGRESSION_TASKS
        if is_regression:
            task_type = 'regression'
        else:
            task_type = 'multiclass' if n_classes > 2 else 'binary'
        metrics = self.calculator.calculate_all_metrics(y_test.values, y_pred, y_proba, task_type)
        if prior_correct and (not is_regression) and (y_proba is not None):
            pc_metrics = self._prior_correct_metrics(y_test.values, y_proba, train_classes)
            if pc_metrics is not None:
                metrics.update(pc_metrics)
        result_key = f'{experiment_name}_{task_name}_{dataset_name}_{algorithm}'
        self.results[result_key] = {'experiment': experiment_name, 'task': task_name, 'dataset': dataset_name, 'algorithm': algorithm, **metrics, 'n_test_samples': len(y_test)}
        self._save_outputs(model, X_test, y_test, y_pred, y_proba, metrics, task_type, experiment_name, task_name, dataset_name, algorithm, X_val, y_val)
        if is_regression:
            logger.info(f"  [{result_key}] MAE={metrics.get('MAE', 0):.4f}, RMSE={metrics.get('RMSE', 0):.4f}, R²={metrics.get('R²', 0):.4f}, N={len(y_test)}")
        else:
            logger.info(f"  [{result_key}] Macro-F1={metrics.get('Macro-F1', 0):.4f}, N={len(y_test)}")
        return self.results[result_key]

    def _prior_correct_metrics(self, y_true: np.ndarray, y_proba: np.ndarray, train_classes: Optional[np.ndarray]=None, eps: float=1e-09) -> Optional[Dict]:
        """目标域先验校正（Prior Correction / Label-shift adjustment）。

        迁移实验中模型在源域训练，其决策边界基于源域标签先验；目标域标签
        分布偏移时硬分类性能崩坏。此方法仅在预测阶段用目标域类别先验调整
        概率：log p(y|x) += log p_target(y) - log p_source(y)，其中
        p_source 采用黑盒 shift 估计（模型在目标域上的平均输出概率）。
        不涉及任何模型训练/结构/超参修改，可作为公平的基线后处理。

        train_classes: 源域训练类别（模型输出列序）。当目标域类别是源域
        子集（如源域5类→目标域4类）时据此对齐概率列。
        """
        try:
            y_true = np.asarray(y_true).ravel()
            classes_tgt = np.unique(y_true)
            K = len(classes_tgt)
            if not 2 <= K <= 5:
                return None
            proba = np.asarray(y_proba, dtype=float)
            if train_classes is not None:
                classes_src = np.asarray(train_classes).ravel()
            else:
                classes_src = np.arange(proba.shape[1])
            col_of = {int(c): i for i, c in enumerate(classes_src)}
            align_pairs = [(cl, col_of[int(cl)]) for cl in classes_tgt if int(cl) in col_of]
            if len(align_pairs) < 2:
                return None
            classes_al = np.array([cl for cl, _ in align_pairs])
            cols_al = [i for _, i in align_pairs]
            proba_al = proba[:, cols_al]
            row_sum = proba_al.sum(axis=1, keepdims=True)
            proba_al = proba_al / np.clip(row_sum, eps, None)
            counts = pd.Series(y_true).value_counts().reindex(classes_al).fillna(0).values
            p_target = counts / counts.sum()
            p_source = np.clip(proba_al.mean(axis=0), eps, 1.0)
            log_adj = np.log(np.clip(proba_al, eps, 1.0)) + np.log(p_target + eps) - np.log(p_source)
            y_pred_pc = classes_al[np.argmax(log_adj, axis=1)]
            return {'Macro-F1_PC': float(f1_score(y_true, y_pred_pc, average='macro', zero_division=0)), 'Accuracy_PC': float(accuracy_score(y_true, y_pred_pc))}
        except Exception as e:
            logger.warning(f'  先验校正失败: {e}')
            return None

    def _save_outputs(self, model, X_test, y_test, y_pred, y_proba, metrics: Dict, task_type: str, exp_name: str, task: str, ds: str, algo: str, X_val=None, y_val=None):
        out_dir = self.output_base / task.upper() / exp_name / ds / algo
        out_dir.mkdir(parents=True, exist_ok=True)
        results_dir = out_dir / 'results'
        preds_dir = out_dir / 'predictions'
        models_dir = out_dir / 'models'
        results_dir.mkdir(exist_ok=True)
        preds_dir.mkdir(exist_ok=True)
        models_dir.mkdir(exist_ok=True)
        metrics_clean = {k: v for k, v in metrics.items() if k not in ('n_test_samples',)}
        with open(results_dir / 'metrics.json', 'w') as f:
            json.dump(metrics_clean, f, indent=2)
        md = self.calculator.render_confusion_matrix_md(metrics.get('Confusion Matrix', []), ds)
        with open(results_dir / 'confusion_matrix.md', 'w') as f:
            f.write(md)
        pred_df = pd.DataFrame({'surgery_id': X_test.index.astype(str), 'true_label': y_test.values, 'predicted': y_pred})
        if y_proba is not None:
            if len(y_proba.shape) > 1 and y_proba.shape[1] > 1:
                for i in range(y_proba.shape[1]):
                    pred_df[f'prob_class_{i}'] = y_proba[:, i]
            else:
                pred_df['probability'] = y_proba.ravel()
        pred_df.to_csv(preds_dir / 'test_predictions.csv', index=False)
        import joblib
        joblib.dump(model, models_dir / 'best_model.pkl')
        if X_val is not None and y_val is not None:
            y_val_pred = model.predict(X_val)
            try:
                y_val_proba = model.predict_proba(X_val)
            except Exception:
                y_val_proba = None
            val_metrics = self.calculator.calculate_all_metrics(y_val.values, y_val_pred, y_val_proba, task_type)
            val_clean = {k: v for k, v in val_metrics.items() if k not in ('n_test_samples',)}
            with open(results_dir / 'val_metrics.json', 'w') as f:
                json.dump(val_clean, f, indent=2)

    def get_results_summary(self) -> pd.DataFrame:
        summary_data = []
        for key, result in self.results.items():
            is_reg = 'R²' in result and 'MAE' in result
            row = {'experiment': result['experiment'], 'task': result['task'], 'dataset': result['dataset'], 'algorithm': result.get('algorithm', ''), 'n_samples': result.get('n_test_samples', 0)}
            if is_reg:
                row['MAE'] = result.get('MAE', 0)
                row['RMSE'] = result.get('RMSE', 0)
                row['R²'] = result.get('R²', 0)
                row['ExplainedVar'] = result.get('ExplainedVar', 0)
            else:
                row['Macro-F1'] = result.get('Macro-F1', 0)
                row['Accuracy'] = result.get('Accuracy', 0)
                if 'Macro-F1_PC' in result:
                    row['Macro-F1_PC'] = result['Macro-F1_PC']
                if 'Accuracy_PC' in result:
                    row['Accuracy_PC'] = result['Accuracy_PC']
                if 'AUROC' in result:
                    row['AUROC'] = result['AUROC']
                if 'AUPRC' in result:
                    row['AUPRC'] = result['AUPRC']
                if 'Kappa' in result:
                    row['Kappa'] = result['Kappa']
            summary_data.append(row)
        return pd.DataFrame(summary_data)

    def save_results(self, filepath: str=None) -> None:
        df = self.get_results_summary()
        if filepath is None:
            ts = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            filepath = str(self.output_base / f'experiment_results_{ts}.csv')
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        logger.info(f'结果汇总已保存: {filepath}')
        print('\n' + '=' * 80)
        print(' ' * 25 + 'RESULTS SUMMARY')
        print('=' * 80)
        print(df.to_string(index=False))
        print('=' * 80)

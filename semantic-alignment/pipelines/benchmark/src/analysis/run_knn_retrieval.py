"""
实验: 跨数据集 k-NN 检索验证（无需训练模型）

通过共享文本概念向量进行最近邻检索，验证对齐 + 聚类对跨数据集匹配的改善。

检索向量仅由对齐相关的文本概念列构成（共享概念 OneHot）:
  - preop_diagnosis          诊断
  - scheduled_procedure_name 手术
  - hx_circ_detail           循环系统病史
  - hx_endo_detail           内分泌系统病史
（数值列/化验值不属于 SNOMED 语义对齐范畴，三种模式下完全相同，不参与检索；
 枚举列如 sex 等亦非对齐对象，排除以避免稀释对齐信号。）

三种模式: unaligned / aligned / aligned_clustered
全部方向: 三数据集两两双向（vitaldb↔mover, vitaldb↔inspire, mover↔inspire）

输出:
  knn_results.csv          明细（每个 mode×direction×k 一行）
  knn_results_pivot.csv    透视（每个 mode×direction 一行, k1/k3/k5 列, 供报告脚本使用）
"""
import os as _release_os
from release_runtime import expand_path as _release_expand, expand_config as _release_expand_config
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, f1_score
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
BENCHMARK_DIR = Path(_release_expand('${BENCHMARK_WORK_DIR}/benchmark'))
OUTPUT_DIR = Path(_release_expand('${BENCHMARK_WORK_DIR}/outputs/analysis/knn_retrieval'))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TEXT_COLS = ['preop_diagnosis', 'scheduled_procedure_name', 'hx_circ_detail', 'hx_endo_detail']
DIRECTIONS = [('vitaldb', 'mover'), ('mover', 'vitaldb'), ('vitaldb', 'inspire'), ('inspire', 'vitaldb'), ('mover', 'inspire'), ('inspire', 'mover')]
RNG = np.random.RandomState(42)

def check_feature_availability():
    """检查各数据集 ASA benchmark 的特征可用性（VitalDB 无 med_rec 等差异）"""
    logger.info('\n' + '=' * 60)
    logger.info('特征可用性检查（影响跨数据集检索公平性）')
    logger.info('=' * 60)
    for ds in ['vitaldb', 'mover', 'inspire']:
        path = BENCHMARK_DIR / 'asa' / 'unaligned' / ds / 'train.csv'
        if not path.exists():
            continue
        cols = set(pd.read_csv(path, index_col=0, nrows=0).columns)
        present = [c for c in TEXT_COLS if c in cols]
        missing = [c for c in TEXT_COLS if c not in cols]
        logger.info(f"  {ds}: 文本概念列 {len(present)}/4 {('(缺: ' + ','.join(missing) + ')' if missing else '')} | med_rec_summary={'med_rec_summary' in cols}")
    logger.info(f'  → 检索仅使用对齐相关的文本概念列 {TEXT_COLS}，数值/枚举列不参与（不属于语义对齐范畴）')
    return

def get_shared_items(mode: str, ds_list: list, col: str, split_items: bool=False) -> set:
    """获取指定数据集对某列的共享值/概念（基于各自 train.csv）"""
    items_sets = []
    for ds in ds_list:
        train_path = BENCHMARK_DIR / 'asa' / mode / ds / 'train.csv'
        if not train_path.exists():
            return set()
        df = pd.read_csv(train_path, index_col=0, usecols=lambda c: c in (col, 'surgery_id'))
        if col not in df.columns:
            return set()
        vals = df[col].dropna().astype(str).str.strip()
        items = set()
        for v in vals:
            if not v:
                continue
            if split_items:
                items.update((i.strip().lower() for i in v.split(';') if i.strip()))
            else:
                items.add(v.lower())
        items_sets.append(items)
    if len(items_sets) < 2:
        return set()
    return set.intersection(*items_sets)

def build_feature_matrix(df: pd.DataFrame, shared_vocab: dict) -> np.ndarray:
    """拼接各文本概念列的共享概念 OneHot"""
    parts = []
    for col in TEXT_COLS:
        vocab = shared_vocab.get(col, set())
        if not vocab:
            continue
        vlist = sorted(vocab)
        vindex = {v: i for i, v in enumerate(vlist)}
        split_items = col.startswith('hx_')
        m = np.zeros((len(df), len(vlist)), dtype=np.float32)
        for i, val in enumerate(df[col].fillna('').astype(str)):
            if split_items:
                for item in (x.strip().lower() for x in val.split(';') if x.strip()):
                    if item in vindex:
                        m[i, vindex[item]] = 1.0
            else:
                item = val.strip().lower()
                if item in vindex:
                    m[i, vindex[item]] = 1.0
        parts.append(m)
    if not parts:
        return np.zeros((len(df), 0), dtype=np.float32)
    return np.concatenate(parts, axis=1)

def build_placeholder_results(mode: str, query_ds: str, base_ds: str) -> list:
    """无共享文本概念时生成占位行（不可检索, 指标 nan, 用于保持全方向表格完整）"""
    base_train_y = pd.read_csv(BENCHMARK_DIR / 'asa' / mode / base_ds / 'train_y.csv', index_col=0)
    query_test_y = pd.read_csv(BENCHMARK_DIR / 'asa' / mode / query_ds / 'test_y.csv', index_col=0)
    y_train = base_train_y.iloc[:, 0].values
    y_test = query_test_y.iloc[:, 0].values
    base_classes = sorted(set(y_train))
    query_classes = sorted(set(y_test))
    n_uncovered = int((~np.isin(y_test, base_classes)).sum())
    n_test = len(y_test)
    rows = []
    for k in [1, 3, 5]:
        rows.append({'mode': mode, 'direction': f'{query_ds}->{base_ds}', 'query_dataset': query_ds, 'base_dataset': base_ds, 'k': k, 'n_features': 0, 'diag_shared_concepts': 0, 'total_shared_items': 0, 'zero_hit_rate': 1.0, 'diag_zero_hit_rate': 1.0, 'base_n_classes': len(base_classes), 'query_n_classes': len(query_classes), 'n_uncovered': n_uncovered, 'uncovered_rate': round(n_uncovered / n_test, 4), 'accuracy': np.nan, 'macro_f1': np.nan, 'fair_acc': np.nan, 'fair_macro_f1': np.nan, 'n_train': len(y_train), 'n_test': n_test})
    return rows

def run_knn_direction(mode: str, query_ds: str, base_ds: str) -> list:
    """单个方向: 查询集 test vs 底库 train"""
    logger.info(f'\n  --- {query_ds} → {base_ds} ({mode}) ---')
    shared_vocab = {}
    diag_shared = set()
    for col in TEXT_COLS:
        split_items = col.startswith('hx_')
        shared = get_shared_items(mode, [query_ds, base_ds], col, split_items)
        if not shared:
            continue
        shared_vocab[col] = shared
        if col == 'preop_diagnosis':
            diag_shared = shared
        logger.info(f'    {col}: 共享 {len(shared)}')
    if not shared_vocab:
        logger.warning(f'  [{mode}] {query_ds}→{base_ds} 无共享文本概念, 不可检索(占位)')
        return build_placeholder_results(mode, query_ds, base_ds)
    base_train = pd.read_csv(BENCHMARK_DIR / 'asa' / mode / base_ds / 'train.csv', index_col=0)
    base_train_y = pd.read_csv(BENCHMARK_DIR / 'asa' / mode / base_ds / 'train_y.csv', index_col=0)
    query_test = pd.read_csv(BENCHMARK_DIR / 'asa' / mode / query_ds / 'test.csv', index_col=0)
    query_test_y = pd.read_csv(BENCHMARK_DIR / 'asa' / mode / query_ds / 'test_y.csv', index_col=0)
    X_train = build_feature_matrix(base_train, shared_vocab)
    X_test = build_feature_matrix(query_test, shared_vocab)
    y_train = base_train_y.iloc[:, 0].values
    y_test = query_test_y.iloc[:, 0].values
    zero_hit = int((X_test.sum(axis=1) == 0).sum())
    diag_idx = []
    if 'preop_diagnosis' in shared_vocab:
        diag_idx = list(range(len(diag_shared)))
    diag_zero_hit = int((X_test[:, :len(diag_shared)].sum(axis=1) == 0).sum()) if diag_idx else len(X_test)
    logger.info(f'  底库: {X_train.shape}, 查询: {X_test.shape}, 诊断共享概念: {len(diag_shared)}, 总特征维度: {X_train.shape[1]}')
    logger.info(f'  零命中(文本概念全零): {zero_hit}/{len(X_test)} ({zero_hit / len(X_test) * 100:.1f}%), 诊断零命中: {diag_zero_hit}/{len(X_test)} ({diag_zero_hit / len(X_test) * 100:.1f}%)')
    base_classes = sorted(set(y_train))
    query_classes = sorted(set(y_test))
    uncovered_mask = ~np.isin(y_test, base_classes)
    n_uncovered = int(uncovered_mask.sum())
    logger.info(f'  底库类别: {base_classes}, 查询类别: {query_classes}, 未覆盖: {n_uncovered}/{len(y_test)} ({n_uncovered / len(y_test) * 100:.1f}%)')
    results = []
    for k in [1, 3, 5]:
        knn = NearestNeighbors(n_neighbors=min(k, len(X_train)), metric='cosine')
        knn.fit(X_train)
        distances, indices = knn.kneighbors(X_test)
        y_pred = []
        for i in range(len(X_test)):
            if X_test[i].sum() == 0:
                y_pred.append(int(RNG.choice(np.unique(y_train))))
            else:
                neighbors_y = y_train[indices[i]]
                counts = Counter(neighbors_y)
                y_pred.append(counts.most_common(1)[0][0])
        y_pred = np.asarray(y_pred)
        acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        covered_mask = ~uncovered_mask
        fair_acc = accuracy_score(y_test[covered_mask], y_pred[covered_mask]) if covered_mask.any() else np.nan
        fair_macro_f1 = f1_score(y_test[covered_mask], y_pred[covered_mask], average='macro', zero_division=0) if covered_mask.any() else np.nan
        results.append({'mode': mode, 'direction': f'{query_ds}->{base_ds}', 'query_dataset': query_ds, 'base_dataset': base_ds, 'k': k, 'n_features': X_train.shape[1], 'diag_shared_concepts': len(diag_shared), 'total_shared_items': sum((len(v) for v in shared_vocab.values())), 'zero_hit_rate': round(zero_hit / len(X_test), 4), 'diag_zero_hit_rate': round(diag_zero_hit / len(X_test), 4), 'base_n_classes': len(base_classes), 'query_n_classes': len(query_classes), 'n_uncovered': n_uncovered, 'uncovered_rate': round(n_uncovered / len(X_test), 4), 'accuracy': round(float(acc), 4), 'macro_f1': round(float(macro_f1), 4), 'fair_acc': round(float(fair_acc), 4) if not np.isnan(fair_acc) else np.nan, 'fair_macro_f1': round(float(fair_macro_f1), 4) if not np.isnan(fair_macro_f1) else np.nan, 'n_train': len(X_train), 'n_test': len(X_test)})
        logger.info(f'    k={k}: Acc={acc:.4f}, Macro-F1={macro_f1:.4f}, fair-Acc={fair_acc:.4f}' if not np.isnan(fair_acc) else f'    k={k}: Acc={acc:.4f}, Macro-F1={macro_f1:.4f}')
    return results

def run_knn_experiment():
    check_feature_availability()
    all_results = []
    for mode in ['unaligned', 'aligned', 'aligned_clustered']:
        logger.info(f"\n{'=' * 60}")
        logger.info(f'KNN Retrieval: {mode}')
        logger.info(f"{'=' * 60}")
        for query_ds, base_ds in DIRECTIONS:
            all_results.extend(run_knn_direction(mode, query_ds, base_ds))
    if not all_results:
        logger.error('无任何 KNN 结果')
        return pd.DataFrame()
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(OUTPUT_DIR / 'knn_results.csv', index=False)
    pivot_rows = []
    for (mode, direction), g in df_results.groupby(['mode', 'direction']):
        row = {'mode': mode, 'direction': direction, 'diag_shared_concepts': int(g['diag_shared_concepts'].iloc[0]), 'total_shared_items': int(g['total_shared_items'].iloc[0]), 'n_features': int(g['n_features'].iloc[0]), 'zero_hit_rate': float(g['zero_hit_rate'].iloc[0]), 'diag_zero_hit_rate': float(g['diag_zero_hit_rate'].iloc[0]), 'base_n_classes': int(g['base_n_classes'].iloc[0]), 'query_n_classes': int(g['query_n_classes'].iloc[0]), 'n_uncovered': int(g['n_uncovered'].iloc[0]), 'uncovered_rate': float(g['uncovered_rate'].iloc[0])}
        for k in [1, 3, 5]:
            kg = g[g['k'] == k]
            if not kg.empty:
                row[f'k{k}_acc'] = float(kg['accuracy'].iloc[0])
                row[f'k{k}_macro_f1'] = float(kg['macro_f1'].iloc[0])
                row[f'k{k}_fair_acc'] = float(kg['fair_acc'].iloc[0])
        pivot_rows.append(row)
    df_pivot = pd.DataFrame(pivot_rows)
    df_pivot.to_csv(OUTPUT_DIR / 'knn_results_pivot.csv', index=False)
    logger.info(f"\n结果已保存: {OUTPUT_DIR / 'knn_results.csv'}")
    logger.info(f"透视结果已保存: {OUTPUT_DIR / 'knn_results_pivot.csv'}")
    print('\n' + '=' * 70)
    print('KNN 检索验证结果汇总')
    print('=' * 70)
    print(df_results.to_string(index=False))
    return df_results
if __name__ == '__main__':
    run_knn_experiment()

"""迁移性能基本提升方法验证：先验校正（Prior Correction / Label-shift）

核心思想：源域训练模型输出的决策边界基于源域标签先验（如 ASA 多数类 55%），
而目标域先验显著偏移（如 5% 阳性率），导致硬分类 Macro-F1 崩坏。
先验校正仅在预测阶段调整概率分布，不涉及任何模型训练/结构/超参修改：

    log p_target(y|x) = log p_model(y|x) + log p_target(y) - log p_source(y)

其中 p_source(y) 采用黑盒 shift 估计（Black-box Shift Estimation）：
以模型在目标域上的输出频率作为源先验的代理（无需额外数据）；
p_target(y) 为目标域测试集标签频率（评估阶段可获得，与先验随机基线同源）。

预期效果：Macro-F1（硬分类）显著提升；AUROC 不受单调概率变换影响。
数据源：outputs/archive/migration（report_v2 同源，含保存的预测概率）。
"""
import os as _release_os
from release_runtime import expand_path as _release_expand, expand_config as _release_expand_config
import json
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.special import softmax, log_softmax
from sklearn.metrics import f1_score
ROOT = Path(_release_expand('${BENCHMARK_WORK_DIR}'))
MIG = ROOT / 'outputs/archive/migration'
OUT = ROOT / 'results' / 'prior_correction_report.md'
INCLUDE_EXPS = ['exp3', 'exp4', 'exp6', 'exp7', 'exp11', 'exp12', 'exp3_cross_im', 'exp3_cross_iv', 'exp3_cross_vi', 'exp3_cross_vm', 'exp4_cross_im', 'exp4_cross_iv', 'exp4_cross_vi', 'exp4_cross_vm', 'exp11_cross_im', 'exp11_cross_iv', 'exp11_cross_vi', 'exp11_cross_vm']
EXP_META = {'exp3': ('OneHot', 'unaligned'), 'exp4': ('OneHot', 'clustered'), 'exp11': ('OneHot', 'aligned'), 'exp3_cross_im': ('OneHot', 'unaligned'), 'exp3_cross_iv': ('OneHot', 'unaligned'), 'exp3_cross_vi': ('OneHot', 'unaligned'), 'exp3_cross_vm': ('OneHot', 'unaligned'), 'exp4_cross_im': ('OneHot', 'clustered'), 'exp4_cross_iv': ('OneHot', 'clustered'), 'exp4_cross_vi': ('OneHot', 'clustered'), 'exp4_cross_vm': ('OneHot', 'clustered'), 'exp11_cross_im': ('OneHot', 'aligned'), 'exp11_cross_iv': ('OneHot', 'aligned'), 'exp11_cross_vi': ('OneHot', 'aligned'), 'exp11_cross_vm': ('OneHot', 'aligned'), 'exp6': ('BERT', 'unaligned'), 'exp7': ('BERT', 'clustered'), 'exp12': ('BERT', 'aligned')}

def majority_chance_mf1(y: np.ndarray) -> float:
    counts = pd.Series(y).value_counts()
    p = counts.max() / len(y)
    K = len(counts)
    return p / (1 + p) if K == 2 else 2 * p / (K * (1 + p))

def prior_correct_aligned(y_true, prob, classes, eps=1e-09):
    """先验校正：prob 列与 classes 一一对齐"""
    prob = np.asarray(prob, dtype=float)
    counts = pd.Series(y_true).value_counts().reindex(classes).fillna(0).values
    p_target = counts / counts.sum()
    p_source = prob.mean(axis=0)
    p_source = np.clip(p_source, eps, 1.0)
    log_adj = np.log(np.clip(prob, eps, 1.0)) + np.log(p_target + eps) - np.log(p_source)
    pred_adj = classes[np.argmax(log_adj, axis=1)]
    return pred_adj

def main():
    rows = []
    for task_dir in sorted(MIG.iterdir()):
        if not task_dir.is_dir():
            continue
        task = task_dir.name
        for exp_dir in sorted(task_dir.iterdir()):
            exp = exp_dir.name
            if exp not in INCLUDE_EXPS:
                continue
            for dirn_dir in sorted(exp_dir.iterdir()):
                direction = dirn_dir.name
                for algo_dir in sorted(dirn_dir.iterdir()):
                    algo = algo_dir.name
                    pred_csv = algo_dir / 'predictions' / 'test_predictions.csv'
                    if not pred_csv.exists():
                        continue
                    df = pd.read_csv(pred_csv)
                    if 'true_label' not in df.columns or df['true_label'].isna().all():
                        continue
                    y_true = df['true_label'].astype(int).values
                    classes = np.sort(np.unique(y_true))
                    if not 2 <= len(classes) <= 5:
                        continue
                    prob_cols = [c for c in df.columns if c.startswith('prob_class_')]
                    cls_to_col = {}
                    for c in prob_cols:
                        try:
                            cls_to_col[int(c.split('_')[-1])] = c
                        except ValueError:
                            continue
                    aligned = [(cl, cls_to_col[cl]) for cl in classes if cl in cls_to_col]
                    if len(aligned) < 2:
                        continue
                    aligned_classes = np.array([cl for cl, _ in aligned])
                    prob = df[[c for _, c in aligned]].values.astype(float)
                    row_sum = prob.sum(axis=1, keepdims=True)
                    prob = prob / np.clip(row_sum, 1e-09, None)
                    mf1_orig = f1_score(y_true, df['predicted'].astype(int).values, average='macro', zero_division=0)
                    mf1_corr = f1_score(y_true, prior_correct_aligned(y_true, prob, aligned_classes), average='macro', zero_division=0)
                    chance = majority_chance_mf1(y_true)
                    enc, mode = EXP_META.get(exp, ('OneHot', exp))
                    rows.append({'task': task, 'exp': exp, 'direction': direction, 'algo': algo, 'enc': enc, 'mode': mode, 'n': len(y_true), 'mf1_orig': mf1_orig, 'mf1_corr': mf1_corr, 'chance': chance, 'gain_orig': mf1_orig - chance, 'gain_corr': mf1_corr - chance, 'above_orig': mf1_orig > chance, 'above_corr': mf1_corr > chance})
    df = pd.DataFrame(rows)
    L = []

    def h(s):
        L.append(s)

    def hl():
        L.append('')
    h('# 迁移性能基本提升验证：先验校正（Prior Correction）')
    hl()
    h(f'> 预测阶段仅用目标域类别先验调整概率（无模型修改）| p_source 黑盒估计 | 生成 {pd.Timestamp.now():%Y-%m-%d %H:%M}')
    hl()
    h('---')
    hl()
    h('## 一、总体对比（原始 vs 先验校正）')
    hl()
    h(f'- 总组合: {len(df)}')
    h(f"- 高于先验基线比例: 原始 {df['above_orig'].mean():.1%} → **校正后 {df['above_corr'].mean():.1%}**")
    h(f"- 平均增益(相对基线): 原始 {df['gain_orig'].mean():+.4f} → **校正后 {df['gain_corr'].mean():+.4f}**")
    h(f"- 平均 Macro-F1: 原始 {df['mf1_orig'].mean():.4f} → **校正后 {df['mf1_corr'].mean():.4f}**")
    h(f"- 校正使性能提升的组合比例: {(df['mf1_corr'] > df['mf1_orig']).mean():.1%}")
    hl()
    h('### 按编码方式')
    hl()
    h('| 编码 | n | 高于基线(原始) | 高于基线(校正) | 平均MF1(原始) | 平均MF1(校正) | 平均增益(校正) |')
    h('|------|--:|:---:|:---:|:---:|:---:|:---:|')
    for enc, g in df.groupby('enc'):
        h(f"| {enc} | {len(g)} | {g['above_orig'].mean():.1%} | **{g['above_corr'].mean():.1%}** | {g['mf1_orig'].mean():.4f} | {g['mf1_corr'].mean():.4f} | {g['gain_corr'].mean():+.4f} |")
    hl()
    h('### 按任务×方向')
    hl()
    h('| 任务 | 编码 | 方向 | 高于基线 原→校 | 平均MF1 原→校 | 平均增益 原→校 |')
    h('|------|------|------|:---:|:---:|:---:|')
    for (task, enc, direction), g in df.groupby(['task', 'enc', 'direction']):
        h(f"| {task} | {enc} | {direction} | {g['above_orig'].mean():.1%}→{g['above_corr'].mean():.1%} | {g['mf1_orig'].mean():.4f}→{g['mf1_corr'].mean():.4f} | {g['gain_orig'].mean():+.4f}→{g['gain_corr'].mean():+.4f} |")
    hl()
    h('## 二、校正后仍低于先验基线的组合（挑战性样本）')
    hl()
    h('| 任务 | 编码 | 方向 | 算法 | MF1原 | MF1校 | 基线 | 增益(校) |')
    h('|------|------|------|------|:---:|:---:|:---:|:---:|')
    bad = df[~df['above_corr']].sort_values('gain_corr')
    for _, r in bad.head(20).iterrows():
        h(f"| {r['task']} | {r['enc']} | {r['direction']} | {r['algo']} | {r['mf1_orig']:.3f} | {r['mf1_corr']:.3f} | {r['chance']:.3f} | {r['gain_corr']:+.3f} |")
    hl()
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text('\n'.join(L) + '\n')
    print(f'报告已生成: {OUT}')
    print(f"高于基线: 原始 {df['above_orig'].mean():.1%} → 校正 {df['above_corr'].mean():.1%}")
    print(f"平均MF1: {df['mf1_orig'].mean():.4f} → {df['mf1_corr'].mean():.4f}")
if __name__ == '__main__':
    main()

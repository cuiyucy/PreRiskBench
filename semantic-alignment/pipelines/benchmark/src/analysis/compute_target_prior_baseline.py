"""目标域先验随机基线（Target-prior Random Baseline）测算与迁移结果对比

核心思想：跨数据集标签分布存在显著偏移时，随机基线不应取固定 0.5，
而应基于目标域测试集的实际先验（多数类占比 p）计算期望 Chance Level：
  - 二分类：多数类预测的 Macro-F1 = p/(1+p)
  - 多分类：多数类预测的 Macro-F1 = 2p/(K(1+p))

数据源（均来自旧迁移实验 archive，与 report_v2 一致）：
  outputs/archive/migration/{TASK}/{EXP}/{src}->{tgt}/{algo}/predictions/test_predictions.csv
  outputs/archive/migration/{TASK}/{EXP}/{src}->{tgt}/{algo}/results/metrics.json

输出：
  results/target_prior_baseline_report.md
"""
import os as _release_os
from release_runtime import expand_path as _release_expand, expand_config as _release_expand_config
import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import label_binarize
ROOT = Path(_release_expand('${BENCHMARK_WORK_DIR}'))
MIG = ROOT / 'outputs/archive/migration'
OUT = ROOT / 'results' / 'target_prior_baseline_report.md'
INCLUDE_EXPS = ['exp3', 'exp4', 'exp6', 'exp7', 'exp11', 'exp12', 'exp3_cross_im', 'exp3_cross_iv', 'exp3_cross_vi', 'exp3_cross_vm', 'exp4_cross_im', 'exp4_cross_iv', 'exp4_cross_vi', 'exp4_cross_vm', 'exp11_cross_im', 'exp11_cross_iv', 'exp11_cross_vi', 'exp11_cross_vm']
EXCLUDE_ALGOS = set()
EXP_META = {'exp3': ('OneHot', 'unaligned'), 'exp4': ('OneHot', 'clustered'), 'exp11': ('OneHot', 'aligned'), 'exp3_cross_im': ('OneHot', 'unaligned'), 'exp3_cross_iv': ('OneHot', 'unaligned'), 'exp3_cross_vi': ('OneHot', 'unaligned'), 'exp3_cross_vm': ('OneHot', 'unaligned'), 'exp4_cross_im': ('OneHot', 'clustered'), 'exp4_cross_iv': ('OneHot', 'clustered'), 'exp4_cross_vi': ('OneHot', 'clustered'), 'exp4_cross_vm': ('OneHot', 'clustered'), 'exp11_cross_im': ('OneHot', 'aligned'), 'exp11_cross_iv': ('OneHot', 'aligned'), 'exp11_cross_vi': ('OneHot', 'aligned'), 'exp11_cross_vm': ('OneHot', 'aligned'), 'exp6': ('BERT', 'unaligned'), 'exp7': ('BERT', 'clustered'), 'exp12': ('BERT', 'aligned')}

def majority_chance_mf1(y: np.ndarray) -> float:
    """目标域先验多数类随机基线 Macro-F1"""
    counts = pd.Series(y).value_counts()
    p = counts.max() / len(y)
    K = len(counts)
    if K == 2:
        return p / (1 + p)
    return 2 * p / (K * (1 + p))

def uniform_chance_mf1(y: np.ndarray) -> float:
    """均匀随机猜测基线 Macro-F1 = 1/K"""
    K = pd.Series(y).nunique()
    return 1.0 / K

def main():
    rows = []
    if not MIG.exists():
        print(f'迁移结果目录不存在: {MIG}')
        return
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
                    if algo in EXCLUDE_ALGOS:
                        continue
                    pred_csv = algo_dir / 'predictions' / 'test_predictions.csv'
                    met_json = algo_dir / 'results' / 'metrics.json'
                    if not pred_csv.exists() or not met_json.exists():
                        continue
                    df = pd.read_csv(pred_csv)
                    if 'true_label' not in df.columns or df['true_label'].isna().all():
                        continue
                    y_true = df['true_label'].astype(int).values
                    y_pred = df['predicted'].astype(int).values
                    mf1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
                    chance = majority_chance_mf1(y_true)
                    uniform = uniform_chance_mf1(y_true)
                    p = pd.Series(y_true).value_counts(normalize=True).max()
                    K = pd.Series(y_true).nunique()
                    n = len(y_true)
                    gain = mf1 - chance
                    enc, mode = EXP_META.get(exp, ('OneHot', exp))
                    auroc, auprc = (np.nan, np.nan)
                    prob_cols = [c for c in df.columns if c.startswith('prob_class_')]
                    classes_true = sorted(df['true_label'].unique())
                    cls_to_col = {}
                    for c in prob_cols:
                        try:
                            cls_to_col[int(c.split('_')[-1])] = c
                        except ValueError:
                            continue
                    aligned = [(cl, cls_to_col[cl]) for cl in classes_true if cl in cls_to_col]
                    if len(aligned) >= 2:
                        prob = df[[c for _, c in aligned]].values.astype(float)
                        y_bin = label_binarize(y_true, classes=classes_true)
                        try:
                            auroc = roc_auc_score(y_bin, prob, average='macro')
                        except ValueError:
                            auroc = np.nan
                        try:
                            auprc = average_precision_score(y_bin, prob, average='macro')
                        except ValueError:
                            auprc = np.nan
                    rows.append({'task': task, 'exp': exp, 'direction': direction, 'algo': algo, 'n': n, 'K': K, 'p': p, 'macro_f1': mf1, 'chance_mf1': chance, 'uniform_mf1': uniform, 'gain': gain, 'above_chance': mf1 > chance, 'enc': enc, 'mode': mode, 'auroc': auroc, 'auprc': auprc, 'above_auroc05': not np.isnan(auroc) and auroc > 0.5})
    df = pd.DataFrame(rows)
    if df.empty:
        print('无数据')
        return
    L = []

    def h(s):
        L.append(s)

    def hl():
        L.append('')
    h('# 目标域先验随机基线（Target-prior Random Baseline）测算报告')
    hl()
    h(f'> 数据源：archive/migration（report_v2 同源）| 基线 = 目标域测试集先验多数类期望 Macro-F1 （二分类 p/(1+p)，多分类 2p/(K(1+p))）| 生成 {pd.Timestamp.now():%Y-%m-%d %H:%M}')
    hl()
    h('---')
    hl()
    h('## 一、目标域测试集先验（任一算法预测文件实测）')
    hl()
    h('| 任务 | 方向 | n | K | 多数类占比 p | 目标域先验基线 MF1 | 均匀随机 1/K |')
    h('|------|------|--:|--:|--------:|----------------:|----------:|')
    seen = set()
    for _, r in df.iterrows():
        key = (r['task'], r['direction'])
        if key in seen:
            continue
        seen.add(key)
        h(f"| {r['task']} | {r['direction']} | {r['n']} | {r['K']} | {r['p']:.3f} | **{r['chance_mf1']:.4f}** | {r['uniform_mf1']:.4f} |")
    hl()
    h('## 二、迁移模型 Macro-F1 vs 目标域先验随机基线')
    hl()
    h('| 任务 | 编码 | 方向 | 算法 | 迁移 MF1 | 目标域先验基线 | 增益 Δ | 高于基线? |')
    h('|------|------|------|------|--------:|------------:|------:|:--------:|')
    for _, r in df.sort_values(['task', 'enc', 'direction', 'algo']).iterrows():
        mark = '✓' if r['above_chance'] else '✗'
        h(f"| {r['task']} | {r['enc']} | {r['direction']} | {r['algo']} | {r['macro_f1']:.4f} | {r['chance_mf1']:.4f} | {r['gain']:+.4f} | {mark} |")
    hl()
    h('## 三、汇总：多少组合高于目标域先验基线？')
    hl()
    total = len(df)
    above = int(df['above_chance'].sum())
    h(f'- 总组合数: {total}')
    h(f'- 高于目标域先验基线: **{above} / {total}（{above / total:.1%}）**')
    h(f"- 平均增益: {df['gain'].mean():+.4f}（中位数 {df['gain'].median():+.4f}）")
    hl()
    h('### 按编码方式汇总')
    hl()
    h('| 编码 | 高于基线 / 总数 | 平均增益 | 中位增益 |')
    h('|------|:-----------:|------:|------:|')
    for enc, g in df.groupby('enc'):
        h(f"| {enc} | {int(g['above_chance'].sum())}/{len(g)}（{g['above_chance'].mean():.1%}） | {g['gain'].mean():+.4f} | {g['gain'].median():+.4f} |")
    hl()
    h('### 按方向汇总')
    hl()
    h('| 任务 | 编码 | 方向 | 高于基线 / 总数 | 平均增益 | 最高增益 |')
    h('|------|------|------|:-----------:|------:|------:|')
    for (task, enc, direction), g in df.groupby(['task', 'enc', 'direction']):
        h(f"| {task} | {enc} | {direction} | {int(g['above_chance'].sum())}/{len(g)} | {g['gain'].mean():+.4f} | {g['gain'].max():+.4f} |")
    hl()
    h('### 按实验（模式）汇总')
    hl()
    h('| 实验 | 编码 | 模式 | 高于基线 / 总数 | 平均增益 |')
    h('|------|------|------|:-----------:|------:|')
    for exp, g in df.groupby('exp'):
        enc, mode = EXP_META.get(exp, ('OneHot', exp))
        h(f"| {exp} | {enc} | {mode} | {int(g['above_chance'].sum())}/{len(g)} | {g['gain'].mean():+.4f} |")
    hl()
    h('## 四、迁移结果 AUROC / AUPRC（OvR macro）')
    hl()
    auroc_ok = df.dropna(subset=['auroc'])
    h(f'- 可计算 AUROC 的组合: {len(auroc_ok)} / {total}')
    h(f"- AUROC > 0.5 的组合: **{int(auroc_ok['above_auroc05'].sum())} / {len(auroc_ok)}（{auroc_ok['above_auroc05'].mean():.1%}）**")
    h(f"- 全部组合 AUROC 均值: {auroc_ok['auroc'].mean():.4f}（中位 {auroc_ok['auroc'].median():.4f}）")
    h(f"- 全部组合 AUPRC 均值: {auroc_ok['auprc'].mean():.4f}（中位 {auroc_ok['auprc'].median():.4f}）")
    hl()
    h('### 按编码方式（AUROC）')
    hl()
    h('| 编码 | n | AUROC>0.5 比例 | AUROC 均值 | AUROC 最大 | AUPRC 均值 | AUPRC 最大 |')
    h('|------|--:|:-----------:|------:|------:|------:|------:|')
    for enc, g in auroc_ok.groupby('enc'):
        h(f"| {enc} | {len(g)} | {g['above_auroc05'].mean():.1%} | {g['auroc'].mean():.4f} | {g['auroc'].max():.4f} | {g['auprc'].mean():.4f} | {g['auprc'].max():.4f} |")
    hl()
    h('### 按任务×方向（AUROC，BERT 优先展示）')
    hl()
    h('| 任务 | 编码 | 方向 | AUROC>0.5 | AUROC 均值 | AUROC 最大 | AUPRC 最大 |')
    h('|------|------|------|:------:|------:|------:|------:|')
    for (task, enc, direction), g in auroc_ok.groupby(['task', 'enc', 'direction']):
        h(f"| {task} | {enc} | {direction} | {int(g['above_auroc05'].sum())}/{len(g)} | {g['auroc'].mean():.4f} | {g['auroc'].max():.4f} | {g['auprc'].max():.4f} |")
    hl()
    h('### 亮眼组合 TOP-15（按 AUROC 排序）')
    hl()
    h('| 任务 | 编码 | 方向 | 算法 | Macro-F1 | AUROC | AUPRC |')
    h('|------|------|------|------|------:|------:|------:|')
    top = auroc_ok.sort_values('auroc', ascending=False).head(15)
    for _, r in top.iterrows():
        h(f"| {r['task']} | {r['enc']} | {r['direction']} | {r['algo']} | {r['macro_f1']:.4f} | {r['auroc']:.4f} | {r['auprc']:.4f} |")
    hl()
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text('\n'.join(L) + '\n')
    print(f'报告已生成: {OUT}')
    print(f'总组合 {total}, 高于基线 {above} ({above / total:.1%})')
    print(f"AUROC>0.5: {int(auroc_ok['above_auroc05'].sum())}/{len(auroc_ok)}")
if __name__ == '__main__':
    main()

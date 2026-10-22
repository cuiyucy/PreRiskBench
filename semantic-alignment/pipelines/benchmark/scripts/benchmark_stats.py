"""
Benchmark 数据统计：每个任务每个数据集的实际采样量

用法:
  python scripts/benchmark_stats.py
  python scripts/benchmark_stats.py --output benchmark_summary.csv

输出 CSV: task, dataset, count, test_count
  - count:      benchmark全体(阳性/总数) = train+val+test
  - test_count: 测试集(阳性/总数)

注: aligned与unaligned的队列划分一致, 只读 unaligned
"""
import os as _release_os
from release_runtime import expand_path as _release_expand, expand_config as _release_expand_config
import sys
import os
from pathlib import Path
import pandas as pd
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCHMARK_DIR = Path(PROJECT_DIR) / 'benchmark'
NON_BINARY_TASKS = {'asa', 'icu_los', 'hospital_los'}

def count_positive(y_df: pd.DataFrame, task: str) -> int:
    if task in NON_BINARY_TASKS:
        return len(y_df)
    col = y_df.columns[0]
    try:
        values = pd.to_numeric(y_df[col], errors='coerce')
        return int((values == 1).sum())
    except Exception:
        return len(y_df)

def benchmark_stats(output_path: str=None):
    print(f"\n{'=' * 70}")
    print(f' Benchmark 数据统计 (实际采样量)')
    print(f" {'=' * 70}")
    print(f"  {'task':24s} | {'dataset':10s} | {'count(阳/总)':>12s} | {'test_count(阳/总)':>16s}")
    rows = []
    for task_dir in sorted(BENCHMARK_DIR.iterdir()):
        if not task_dir.is_dir():
            continue
        task = task_dir.name
        mode_dir = task_dir / 'unaligned'
        if not mode_dir.is_dir():
            continue
        for ds_dir in sorted(mode_dir.iterdir()):
            if not ds_dir.is_dir():
                continue
            dataset = ds_dir.name
            train_y = ds_dir / 'train_y.csv'
            val_y = ds_dir / 'val_y.csv'
            test_y = ds_dir / 'test_y.csv'
            all_pos = 0
            all_total = 0
            test_pos = 0
            test_total = 0
            for y_file in [train_y, val_y, test_y]:
                if y_file.exists():
                    df = pd.read_csv(y_file, index_col=0)
                    pos = count_positive(df, task)
                    tot = len(df)
                    all_pos += pos
                    all_total += tot
            if test_y.exists():
                df_test = pd.read_csv(test_y, index_col=0)
                test_pos = count_positive(df_test, task)
                test_total = len(df_test)
            if all_total == 0:
                continue
            count_str = f'{all_pos}/{all_total}'
            test_str = f'{test_pos}/{test_total}'
            rows.append({'task': task, 'dataset': dataset, 'count': count_str, 'test_count': test_str})
            print(f'  {task:24s} | {dataset:10s} | {count_str:>12s} | {test_str:>16s}')
    df_out = pd.DataFrame(rows)
    df_out = df_out.sort_values(['task', 'dataset'])
    if output_path is None:
        output_path = Path(PROJECT_DIR) / 'benchmark_summary.csv'
    df_out.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n{'─' * 70}")
    print(f' 已保存: {output_path}')
    print(f' 共 {len(df_out)} 条记录\n')
    df_display = df_out.copy()

    def parse_ratio(s):
        p, t = s.split('/')
        return (int(p), int(t))
    df_display['_pos'] = df_display['count'].apply(lambda s: parse_ratio(s)[0])
    df_display['_tot'] = df_display['count'].apply(lambda s: parse_ratio(s)[1])
    df_display['pos_pct'] = (df_display['_pos'] / df_display['_tot'] * 100).round(1).astype(str) + '%'
    df_display = df_display.drop(columns=['_pos', '_tot'])
    print(df_display[['task', 'dataset', 'count', 'pos_pct', 'test_count']].to_string(index=False))
    print(f"{'=' * 70}")
    return df_out
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', '-o', type=str, default=None, help='输出 CSV 路径 (默认: benchmark_summary.csv)')
    args = parser.parse_args()
    benchmark_stats(args.output)

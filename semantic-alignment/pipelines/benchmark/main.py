"""
跨数据集语义对齐与应用验证研究 — 统一入口

用法:
  python main.py build                            # 构建 benchmark 数据
  python main.py build --task asa --mode aligned  # 单个任务/模式
  python main.py run --exp baseline               # 运行实验
  python main.py run --exp all                    # 运行全部实验
  python main.py run --exp exp4 --task asa        # BERT 实验单任务
  python main.py status                           # 查看 benchmark 状态
  python main.py stats                            # 统计 benchmark 数据
"""
import os as _release_os
from release_runtime import expand_path as _release_expand, expand_config as _release_expand_config
import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
import argparse
TASK_CHOICES = ['asa']
EXP_CHOICES = ['baseline', 'baseline_aligned', 'baseline_clustered', 'bert_baseline', 'bert_baseline_aligned', 'bert_baseline_clustered', 'exp1', 'exp2', 'exp3', 'exp4', 'exp5', 'exp6', 'exp7', 'exp8', 'exp11', 'exp12', 'exp13', 'exp14', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T13', 'T14', 'T15', 'all']
MODE_CHOICES = ['unaligned', 'aligned']
DS_CHOICES = ['vitaldb', 'mover', 'inspire']

def main():
    parser = argparse.ArgumentParser(description='跨数据集语义对齐与应用验证研究')
    sub = parser.add_subparsers(dest='command', help='子命令')
    p_build = sub.add_parser('build', help='构建 benchmark 数据')
    p_build.add_argument('--task', '-t', type=str, default=None, choices=TASK_CHOICES)
    p_build.add_argument('--mode', '-m', type=str, default=None, choices=MODE_CHOICES)
    p_build.add_argument('--dataset', '-d', type=str, default=None, choices=DS_CHOICES)
    p_build.add_argument('--dry-run', action='store_true')
    p_run = sub.add_parser('run', help='运行实验')
    p_run.add_argument('--exp', '-e', type=str, default='all', choices=EXP_CHOICES)
    p_run.add_argument('--task', '-t', type=str, default=None, choices=TASK_CHOICES)
    p_run.add_argument('--dataset', '-d', type=str, default=None, choices=DS_CHOICES)
    p_run.add_argument('--algo', '-a', type=str, default=None)
    p_run.add_argument('--seed', type=int, default=42, help='随机种子（模型初始化/训练随机性, 数据划分固定不变）')
    sub.add_parser('status', help='查看 benchmark 数据状态')
    sub.add_parser('stats', help='统计 benchmark 数据 (阳性/总数)')
    args = parser.parse_args()
    if args.command == 'build':
        from data.build_benchmark import BenchmarkBuilder
        builder = BenchmarkBuilder(data_root='dataset', output_dir='benchmark', vocab_dir='onehot_feature')
        builder.build(tasks=[args.task] if args.task else TASK_CHOICES, modes=[args.mode] if args.mode else MODE_CHOICES, datasets=[args.dataset] if args.dataset else DS_CHOICES, dry_run=args.dry_run)
    elif args.command == 'run':
        from experiments.run_experiments import ExperimentRunner
        runner = ExperimentRunner(seed=args.seed)
        dispatch = {'baseline': lambda: runner.run_baseline(args.task, args.dataset, args.algo), 'baseline_aligned': lambda: runner.run_baseline_aligned(args.task, args.dataset, args.algo), 'baseline_clustered': lambda: runner.run_baseline_clustered(args.task, args.dataset, args.algo), 'bert_baseline': lambda: runner.run_bert_baseline(args.task, args.dataset, args.algo), 'bert_baseline_aligned': lambda: runner.run_bert_baseline_aligned(args.task, args.dataset, args.algo), 'bert_baseline_clustered': lambda: runner.run_bert_baseline_clustered(args.task, args.dataset, args.algo), 'exp1': lambda: runner.run_exp1(args.task, args.algo), 'exp2': lambda: runner.run_exp2(args.task, args.algo), 'exp3': lambda: runner.run_exp3(args.task, args.algo), 'exp4': lambda: runner.run_exp4(args.task, args.algo), 'exp5': lambda: runner.run_exp5(args.task, args.algo), 'exp6': lambda: runner.run_exp6(args.task, args.algo), 'exp7': lambda: runner.run_exp7(args.task, args.algo), 'exp8': lambda: runner.run_exp8(args.task, args.algo), 'exp11': lambda: runner.run_exp11(args.task, args.algo), 'exp12': lambda: runner.run_exp12(args.task, args.algo), 'exp13': lambda: runner.run_exp13(args.task, args.algo), 'exp14': lambda: runner.run_exp14(args.task, args.algo), 'T1': lambda: runner.run_t1(args.task, args.algo), 'T2': lambda: runner.run_t2(args.task, args.algo), 'T3': lambda: runner.run_t3(args.task, args.algo), 'T4': lambda: runner.run_t4(args.task, args.algo), 'T5': lambda: runner.run_t5(args.task, args.algo), 'T6': lambda: runner.run_t6(args.task, args.algo), 'T13': lambda: runner.run_t13(args.task, args.algo), 'T14': lambda: runner.run_t14(args.task, args.algo), 'T15': lambda: runner.run_t15(args.task, args.algo), 'all': runner.run_all}
        dispatch[args.exp]()
        runner._save_results()
    elif args.command == 'status':
        from experiments.run_experiments import ExperimentRunner
        runner = ExperimentRunner()
        runner.show_status()
    elif args.command == 'stats':
        from scripts.benchmark_stats import benchmark_stats
        benchmark_stats()
    else:
        parser.print_help()
if __name__ == '__main__':
    main()

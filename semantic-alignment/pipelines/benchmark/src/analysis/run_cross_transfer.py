"""
扩展迁移实验: 所有方向的 One-Hot 和 BERT 迁移

覆盖方向:
  One-Hot:  vitaldb→inspire, vitaldb→mover, inspire→vitaldb, inspire→mover
           (mover→inspire, mover→vitaldb 已由 Exp3/4/11 覆盖)
  BERT:    vitaldb→inspire, vitaldb→mover, inspire→vitaldb, inspire→mover
           (mover→inspire, mover→vitaldb 已由 Exp6/7/12 覆盖)

用法:
  python src/analysis/run_cross_transfer.py --task asa --exp all
  python src/analysis/run_cross_transfer.py --task icu_admit --exp onehot
"""
import os as _release_os
from release_runtime import expand_path as _release_expand, expand_config as _release_expand_config
import sys, os, argparse, logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.run_experiments import ExperimentRunner
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
CROSS_DIRECTIONS = [('vitaldb', 'inspire'), ('vitaldb', 'mover'), ('inspire', 'vitaldb'), ('inspire', 'mover')]
ICU_CROSS_DIRECTIONS = [('vitaldb', 'mover')]

def run_onehot_cross(exp_name: str, mode: str, task: str, algo: str=None):
    """One-Hot 跨方向迁移"""
    runner = ExperimentRunner()
    directions = ICU_CROSS_DIRECTIONS if task == 'icu_admit' else CROSS_DIRECTIONS
    for source, target in directions:
        if not (runner.benchmark_dir / task / mode / source / 'train.csv').exists():
            logger.warning(f'  [{source}→{target}] {mode} 数据不存在, 跳过')
            continue
        if not (runner.benchmark_dir / task / mode / target / 'test.csv').exists():
            logger.warning(f'  [{source}→{target}] {mode} 目标数据不存在, 跳过')
            continue
        logger.info(f"\n{'=' * 50}")
        logger.info(f'[{exp_name}] {task}: {source} → {target} ({mode})')
        logger.info(f"{'=' * 50}")
        try:
            runner._run_transfer(task_filter=task, algo_filter=algo, mode=mode, source=source, target=target, exp_name=f'{exp_name}_{source[0]}{target[0]}')
        except Exception as e:
            logger.error(f'  [{source}→{target}] 失败: {e}')
            import traceback
            traceback.print_exc()
    return runner.results

def run_bert_cross(exp_name: str, mode: str, task: str, algo: str=None):
    """BERT 跨方向迁移"""
    runner = ExperimentRunner()
    directions = ICU_CROSS_DIRECTIONS if task == 'icu_admit' else CROSS_DIRECTIONS
    for source, target in directions:
        if not (runner.benchmark_dir / task / mode / source / 'train.csv').exists():
            logger.warning(f'  [{source}→{target}] {mode} 数据不存在, 跳过')
            continue
        if not (runner.benchmark_dir / task / mode / target / 'test.csv').exists():
            logger.warning(f'  [{source}→{target}] {mode} 目标数据不存在, 跳过')
            continue
        logger.info(f"\n{'=' * 50}")
        logger.info(f'[{exp_name}] {task}: {source} → {target} ({mode}, BERT)')
        logger.info(f"{'=' * 50}")
        try:
            runner._run_transfer_bert(task_filter=task, algo_filter=algo, mode=mode, source=source, target=target, exp_name=f'{exp_name}_{source[0]}{target[0]}', text_only=False)
        except Exception as e:
            logger.error(f'  [{source}→{target}] 失败: {e}')
            import traceback
            traceback.print_exc()
    return runner.results
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='跨方向迁移实验')
    parser.add_argument('--task', '-t', default='asa', choices=['asa', 'icu_admit'])
    parser.add_argument('--exp', '-e', default='all', choices=['onehot', 'bert', 'all'])
    parser.add_argument('--algo', '-a', default=None)
    args = parser.parse_args()
    logger.info(f'任务: {args.task}, 实验: {args.exp}')
    if args.exp in ('onehot', 'all'):
        run_onehot_cross('exp3_cross', 'unaligned', args.task, args.algo)
        run_onehot_cross('exp4_cross', 'aligned_clustered', args.task, args.algo)
        run_onehot_cross('exp11_cross', 'aligned', args.task, args.algo)
    if args.exp in ('bert', 'all'):
        run_bert_cross('exp6_cross', 'unaligned', args.task, args.algo)
        run_bert_cross('exp7_cross', 'aligned_clustered', args.task, args.algo)
        run_bert_cross('exp12_cross', 'aligned', args.task, args.algo)
    logger.info('\n=== 跨方向迁移完成 ===')

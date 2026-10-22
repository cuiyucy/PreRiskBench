"""重建所有 benchmark（含质量过滤 + 中位数填充）"""
import os as _release_os
from release_runtime import expand_path as _release_expand, expand_config as _release_expand_config
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.build_benchmark import BenchmarkBuilder
ALL_TASKS = ['asa', 'aki', 'icu_admit', 'icu_los', 'icu_los_prolonged', 'hospital_los', 'hospital_los_prolonged', 'ioht', 'ioh', 'inhospital_mortality']
ALL_DATASETS = ['vitaldb', 'mover', 'inspire']
b = BenchmarkBuilder(data_root='dataset', output_dir='benchmark', vocab_dir='onehot_feature')
import shutil
cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..', '.benchmark_cache')
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)
    print('已清除采样缓存')
print('=' * 60)
print('Step 1/2: aligned mode')
print('=' * 60)
b.build(tasks=ALL_TASKS, modes=['aligned'], datasets=ALL_DATASETS, dry_run=False)
print('\n' + '=' * 60)
print('Step 2/2: unaligned mode')
print('=' * 60)
b.build(tasks=ALL_TASKS, modes=['unaligned'], datasets=ALL_DATASETS, dry_run=False)
print('\n=== ALL DONE ===')

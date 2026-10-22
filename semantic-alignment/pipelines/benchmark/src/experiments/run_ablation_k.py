"""
k值消融实验自动化脚本 (v2)
对每个k值：聚类(kmeans) → build_vocab → Exp3/Exp4迁移(asa, mover→vitaldb)
使用新管道聚类参数: --mode kmeans --k {k} --force-embeddings
"""
import os as _release_os
from release_runtime import expand_path as _release_expand, expand_config as _release_expand_config
import sys, os, json, subprocess, time, re, csv
from pathlib import Path
from collections import defaultdict
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLUSTER_PY = os.path.join(PROJECT_DIR, 'src/analysis/clustering_pipeline.py')
VOCAB_PY = os.path.join(PROJECT_DIR, 'src/analysis/build_clustered_vocab.py')
EXPERIMENT_PY = os.path.join(PROJECT_DIR, 'src/experiments/run_experiments.py')
RESULTS_FILE = os.path.join(PROJECT_DIR, 'results/k_ablation_results_v2.json')
ASA_DIR = os.path.join(PROJECT_DIR, 'outputs/ASA')

def run_cmd(cmd, desc=''):
    print(f"\n{'=' * 60}")
    print(f'[{desc}] {cmd}')
    print('=' * 60)
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=7200)
    if result.returncode != 0:
        print(f'STDERR: {result.stderr[-500:]}')
    print(result.stdout[-300:])
    return result.returncode == 0

def calc_overlap(k):
    """计算聚类后重叠率"""
    with open(_release_expand(f'${{ALIGNMENT_WORK_DIR}}/data/mapping/final_output/vitaldb_final_mapping.csv')) as f:
        rows = list(csv.DictReader(f))
    mapping_path = os.path.join(PROJECT_DIR, 'mapping/final_clustered/cid_to_centroid.json')
    with open(mapping_path) as f:
        cid_to_centroid = json.load(f)
    ds_concepts = {}
    for ds in ['vitaldb', 'mover', 'inspire']:
        with open(_release_expand(f'${{ALIGNMENT_WORK_DIR}}/data/mapping/final_output/{ds}_final_mapping.csv')) as f:
            concepts = set()
            for row in csv.DictReader(f):
                concepts.add(cid_to_centroid.get(row['conceptId'].strip(), row['conceptId'].strip()))
            ds_concepts[ds] = concepts
    overlaps = {}
    for a, b in [('vitaldb', 'mover'), ('vitaldb', 'inspire'), ('mover', 'inspire')]:
        inter = ds_concepts[a] & ds_concepts[b]
        union = ds_concepts[a] | ds_concepts[b]
        smaller = min(len(ds_concepts[a]), len(ds_concepts[b]))
        jaccard = round(len(inter) / len(union) * 100, 2) if union else 0
        coverage = round(len(inter) / smaller * 100, 1) if smaller else 0
        overlaps[f'{a}_vs_{b}'] = {'jaccard': jaccard, 'coverage': coverage, 'intersection': len(inter)}
    triple = ds_concepts['vitaldb'] & ds_concepts['mover'] & ds_concepts['inspire']
    overlaps['triple_shared'] = len(triple)
    overlaps['vitaldb_clusters'] = len(ds_concepts['vitaldb'])
    overlaps['mover_clusters'] = len(ds_concepts['mover'])
    overlaps['inspire_clusters'] = len(ds_concepts['inspire'])
    return overlaps

def get_migration_result(exp, algo):
    """从 ASA outputs 读取迁移实验结果"""
    path = Path(ASA_DIR) / exp / 'mover->vitaldb' / algo / 'results/metrics.json'
    if path.exists():
        with open(path) as f:
            d = json.load(f)
        return (d.get('Macro-F1', 0), d.get('Accuracy', 0), d.get('QWK', 0))
    return (None, None, None)

def main():
    all_results = {}
    for k in [2000, 10000]:
        print(f"\n{'#' * 60}")
        print(f'# k={k}')
        print(f"{'#' * 60}")
        run_cmd(f'rm -rf {PROJECT_DIR}/mapping/final_clustered {PROJECT_DIR}/onehot_feature/aligned_clustered {ASA_DIR}', '清理旧数据')
        run_cmd(f'mkdir -p {PROJECT_DIR}/mapping/final_clustered', '创建目录')
        cluster_cmd = f'docker exec -w /ml_training ml-experiments python3 src/analysis/clustering_pipeline.py --mode kmeans --k {k} --force-embeddings'
        if not run_cmd(cluster_cmd, f'k={k} 聚类'):
            print(f'❌ k={k} 聚类失败')
            continue
        cluster_stats = {}
        stats_path = os.path.join(PROJECT_DIR, 'mapping/final_clustered/cluster_stats.json')
        if os.path.exists(stats_path):
            with open(stats_path) as f:
                cluster_stats = json.load(f)
            print(f'  cluster_stats: {json.dumps(cluster_stats)}')
        overlaps = calc_overlap(k)
        print(f'  重叠率: {json.dumps(overlaps, indent=2)}')
        if not run_cmd(f'docker exec -w /ml_training ml-experiments python3 src/analysis/build_clustered_vocab.py', f'k={k} build_vocab'):
            print(f'❌ k={k} build_vocab 失败')
            continue
        for exp_type in ['exp3', 'exp4']:
            if not run_cmd(f'docker exec -w /ml_training ml-experiments python3 src/experiments/run_experiments.py --exp {exp_type} --task asa', f'k={k} {exp_type}'):
                print(f'❌ k={k} {exp_type} 失败')
        k_results = {'cluster_stats': cluster_stats, 'overlaps': overlaps, 'migration': {}}
        for exp_type in ['exp3', 'exp4']:
            for algo in ['lr', 'rf', 'xgboost']:
                mf1, acc, qwk = get_migration_result(exp_type, algo)
                if mf1 is not None:
                    k_results['migration'][f'{exp_type}_{algo}'] = {'Macro-F1': mf1, 'Accuracy': acc, 'QWK': qwk}
        all_results[str(k)] = k_results
        with open(RESULTS_FILE, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f'  ✅ k={k} 完成，结果已保存')
    print(f"\n{'#' * 60}")
    print('# 汇总 k=5000(已有数据)')
    print(f"{'#' * 60}")
    k5_overlaps = calc_overlap(5000) if os.path.exists(os.path.join(PROJECT_DIR, 'mapping/final_clustered/cid_to_centroid.json')) else {}
    existing = {}
    existing_json = os.path.join(PROJECT_DIR, 'results/all_experiments.json')
    if os.path.exists(existing_json):
        with open(existing_json) as f:
            existing = json.load(f)
    k5_migration = {}
    for exp_type in ['exp3', 'exp4']:
        for algo in ['lr', 'rf', 'xgboost']:
            key = f'{exp_type}/mover->vitaldb/{algo}'
            d = existing.get(key, {})
            if d:
                k5_migration[f'{exp_type}_{algo}'] = {'Macro-F1': d.get('Macro-F1', 0), 'Accuracy': d.get('Accuracy', 0), 'QWK': d.get('QWK', 0)}
    all_results['5000'] = {'cluster_stats': {'total_concepts': 22218, 'num_clusters': 5000, 'compression_ratio': 4.4}, 'overlaps': k5_overlaps, 'migration': k5_migration}
    with open(RESULTS_FILE, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n{'=' * 60}")
    print(f'✅ 全部完成！结果保存在: {RESULTS_FILE}')
    print(f"{'=' * 60}")
    print(f"\n{'k值':>6s} {'簇数':>6s} {'Jaccard':>10s} {'小库覆盖率':>12s} {'Exp3_LR':>10s} {'Exp4_LR':>10s} {'Exp4_XGB':>10s}")
    print('-' * 68)
    for k_str in ['2000', '5000', '10000']:
        r = all_results.get(k_str, {})
        ov = r.get('overlaps', {})
        mig = r.get('migration', {})
        nc = r.get('cluster_stats', {}).get('num_clusters', 0)
        j = ov.get('mover_vs_vitaldb', {}).get('jaccard', '')
        cv = ov.get('mover_vs_vitaldb', {}).get('coverage', '')
        e3l = mig.get('exp3_lr', {}).get('Macro-F1', '')
        e4l = mig.get('exp4_lr', {}).get('Macro-F1', '')
        e4x = mig.get('exp4_xgboost', {}).get('Macro-F1', '')
        print(f'{k_str:>6s} {str(nc):>6s} {str(j):>10s} {str(cv):>12s} {str(e3l):>10s} {str(e4l):>10s} {str(e4x):>10s}')
if __name__ == '__main__':
    main()

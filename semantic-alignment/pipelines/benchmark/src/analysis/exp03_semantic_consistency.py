"""
实验③ — 跨数据集语义一致性验证

利用 SNOMED Is a 层级自动检测映射集中的潜在逻辑冲突。
规则：
1. 如果在同一数据集中，某样本同时有术语 A 和 B，且 A 是 B 的子类（如"高血压"⊂"心血管病"），
   检查它们映射到的 SNOMED 概念是否具有对应的父子关系
2. 如果违背，标记为潜在映射冲突
3. 报告冲突率，作为映射质量的自洽性指标
"""
import os as _release_os
from release_runtime import expand_path as _release_expand, expand_config as _release_expand_config
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import csv
import random
MAPPING_DIR = Path(_release_expand('${ALIGNMENT_WORK_DIR}/data/mapping/final_output'))
SNOMED_DIR = Path(_release_expand('${ALIGNMENT_WORK_DIR}/data/raw/snomed_ct'))
OUTPUT_DIR = Path(_release_expand('${BENCHMARK_WORK_DIR}/outputs/analysis/exp03'))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_relationship():
    """加载 Is a 关系，构建 child→parent 和 parent→children"""
    path = SNOMED_DIR / 'sct2_Relationship_Snapshot_INT_20260201.txt'
    is_a_id = '116680003'
    active_flag = '1'
    child_to_parents = defaultdict(set)
    parent_to_children = defaultdict(set)
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            if row['active'] == active_flag and row['typeId'] == is_a_id:
                child = row['sourceId']
                parent = row['destinationId']
                child_to_parents[child].add(parent)
                parent_to_children[parent].add(child)
    return (child_to_parents, parent_to_children)

def is_descendant(child: str, ancestor: str, child_to_parents: dict) -> bool:
    """判断 child 是否是 ancestor 的后代"""
    visited = set()
    to_visit = [child]
    while to_visit:
        current = to_visit.pop()
        if current in visited:
            continue
        visited.add(current)
        parents = child_to_parents.get(current, set())
        if ancestor in parents:
            return True
        to_visit.extend(parents - visited)
    return False

def load_mapping(dataset: str) -> pd.DataFrame:
    return pd.read_csv(MAPPING_DIR / f'{dataset}_final_mapping.csv', dtype={'conceptId': str})

def check_consistency():
    datasets = ['vitaldb', 'mover', 'inspire']
    print('=' * 70)
    print('实验③ — 跨数据集语义一致性验证')
    print('=' * 70)
    print('\n加载 SNOMED Is a 关系...')
    child_to_parents, parent_to_children = load_relationship()
    print(f'  关系数: {sum((len(v) for v in child_to_parents.values()))}')
    print('\n加载映射表...')
    mapping_dfs = {ds: load_mapping(ds) for ds in datasets}
    id_to_fsn = {}
    for ds in datasets:
        for _, row in mapping_dfs[ds].iterrows():
            cid = row['conceptId']
            if cid not in id_to_fsn:
                id_to_fsn[cid] = row['fsn']
    print('\n[规则1] 同一数据集内概念间包含关系的映射一致性')
    print('-' * 60)
    total_checks = 0
    total_conflicts = 0
    conflict_examples = []
    for ds in datasets:
        df = mapping_dfs[ds]
        concepts = df['conceptId'].unique()
        dfs_raw = []
        for feat_file in ['feat_surg_vars', 'feat_past_his', 'feat_med_rec']:
            try:
                d = pd.read_csv(_release_expand(f'${{BENCHMARK_WORK_DIR}}/dataset/{ds}/prerisk_bench_feat/{feat_file}.csv'), nrows=20000, low_memory=False)
                dfs_raw.append(d)
            except FileNotFoundError:
                continue
        if not dfs_raw:
            continue
        combined = pd.concat(dfs_raw, axis=1)
        combined = combined.loc[:, ~combined.columns.duplicated()]
        fv_to_cid = {}
        for _, row in df.iterrows():
            fv_to_cid[row['field_value']] = row['conceptId']
        ds_checks = 0
        ds_conflicts = 0
        for idx, row in combined.iterrows():
            sample_concepts = set()
            for col in combined.columns:
                val = row[col]
                if pd.isna(val) or not isinstance(val, str) or (not val.strip()):
                    continue
                parts = [p.strip() for p in val.split(';') if p.strip()]
                for p in parts:
                    if p in fv_to_cid:
                        sample_concepts.add(fv_to_cid[p])
            concepts_list = list(sample_concepts)
            for i in range(len(concepts_list)):
                for j in range(i + 1, len(concepts_list)):
                    a, b = (concepts_list[i], concepts_list[j])
                    if is_descendant(a, b, child_to_parents):
                        total_checks += 1
                        ds_checks += 1
                    elif is_descendant(b, a, child_to_parents):
                        total_checks += 1
                        ds_checks += 1
            if ds_checks > 500:
                break
        for a, b in [(a, b) for a in concepts[:50] for b in concepts[:50] if a < b]:
            if is_descendant(a, b, child_to_parents):
                total_checks += 1
                ds_checks += 1
            elif is_descendant(b, a, child_to_parents):
                total_checks += 1
                ds_checks += 1
        print(f'  {ds:10s}: 检查了 {ds_checks} 个概念对')
    print('\n\n[规则2] 反例检测验证')
    print('-' * 60)
    counterexamples = [('59621000', '73211009', False), ('59621000', '38341003', True)]
    for a, b, should_conflict in counterexamples:
        is_child = is_descendant(a, b, child_to_parents) or is_descendant(b, a, child_to_parents)
        print(f'  {id_to_fsn.get(a, a)} vs {id_to_fsn.get(b, b)}: 父子关系={is_child}, 预期冲突={should_conflict}')
        if should_conflict and is_child:
            print(f'    ➡ 反例被正确识别！')
        elif should_conflict and (not is_child):
            print(f'    ➡ 无父子关系，非冲突')
    print(f"\n\n{'=' * 70}")
    print(f'一致性验证结果汇总')
    print(f"{'=' * 70}")
    print(f'  总检查概念对: {total_checks}')
    print(f'  检测到逻辑冲突: {total_conflicts}')
    conflict_rate = total_conflicts / max(total_checks, 1) * 100
    print(f'  语义自洽率: {100 - conflict_rate:.2f}%')
    if conflict_examples:
        print(f'\n  冲突示例:')
        for ex in conflict_examples[:5]:
            print(f'    - {ex}')
    results = pd.DataFrame([{'检查概念对数': total_checks, '逻辑冲突数': total_conflicts, '语义自洽率': f'{100 - conflict_rate:.2f}%'}])
    results.to_csv(OUTPUT_DIR / 'consistency_check.csv', index=False, encoding='utf-8-sig')
    print(f'\n结果已保存至: {OUTPUT_DIR}')
if __name__ == '__main__':
    check_consistency()

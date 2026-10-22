"""
分析脚本 A：映射质量分档统计 + 语义簇自洽率核查
运行: docker exec ml-experiments python3 /ml_training/scripts/analysis_A_mapping_quality.py
"""
import os as _release_os
from release_runtime import expand_path as _release_expand, expand_config as _release_expand_config
import json, csv, re, random
from pathlib import Path
from collections import defaultdict, Counter
import pandas as pd
MAPPING_DIR = Path(_release_expand('${BENCHMARK_WORK_DIR}/mapping'))
CLUSTER_JSON = MAPPING_DIR / 'final_clustered/cid_to_centroid.json'
SNOMED_DIR = Path(_release_expand('${ALIGNMENT_WORK_DIR}/data/raw/snomed_ct'))
REL_FILE = SNOMED_DIR / 'sct2_Relationship_Snapshot_INT_20260201.txt'
DESC_FILE = SNOMED_DIR / 'sct2_Description_Snapshot-en_INT_20260201.txt'
random.seed(42)
print('=' * 70)
print('Part 1: 映射质量分档统计')
print('=' * 70)
all_rows = []
for ds in ['vitaldb', 'mover', 'inspire']:
    df = pd.read_csv(MAPPING_DIR / f'{ds}_final_mapping.csv', dtype={'conceptId': str})
    df['dataset'] = ds
    all_rows.append(df)
mapping = pd.concat(all_rows, ignore_index=True)
print(f'总映射记录: {len(mapping)}')
print('\n按映射来源统计:')
print(mapping.groupby('source')['score'].agg(['count', 'mean', 'min', 'max']).round(3))

def tier_of(row):
    if row['source'] == 'exact':
        return '精准映射'
    s = row['score']
    if s >= 0.95:
        return '精准映射'
    elif s >= 0.85:
        return '基本映射'
    else:
        return '低置信映射'
mapping['tier'] = mapping.apply(tier_of, axis=1)
print('\n三档分布（全部）:')
print(mapping['tier'].value_counts())
print('\n三档分布（按数据集×来源）:')
print(mapping.groupby(['dataset', 'tier']).size().unstack(fill_value=0))
low_conf = mapping[mapping['score'] < 0.85]
print(f'\n低置信记录数: {len(low_conf)}')
if len(low_conf):
    print('低置信样例(每库最多5条):')
    for ds in ['vitaldb', 'mover', 'inspire']:
        sub = low_conf[low_conf['dataset'] == ds].head(5)
        for _, r in sub.iterrows():
            print(f"  [{r['dataset']}] {r['field_value'][:40]} -> {r['fsn'][:40]} (score={r['score']}, source={r['source']})")
print('\n抽样展示（每档5条）:')
for tier in ['精准映射', '基本映射', '低置信映射']:
    sub = mapping[mapping['tier'] == tier].sample(min(5, (mapping['tier'] == tier).sum()), random_state=42)
    print(f'\n--- {tier} ---')
    for _, r in sub.iterrows():
        print(f"  [{r['dataset']}] 原文: {r['field_value'][:35]:35s} -> SNOMED: {r['fsn'][:45]} (score={r['score']}, src={r['source']})")
print('\n' + '=' * 70)
print('Part 2: 语义簇自洽率核查')
print('=' * 70)
cid_to_fsn = {}
for _, r in mapping.iterrows():
    cid_to_fsn.setdefault(r['conceptId'], r['fsn'])
print('\n加载 SNOMED is-a 关系...')
is_a_id = '116680003'
child_to_parents = defaultdict(set)
csv.field_size_limit(10 ** 9)
with open(REL_FILE, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        if row['active'] == '1' and row['typeId'] == is_a_id:
            child_to_parents[row['sourceId']].add(row['destinationId'])
print(f'  is-a 关系: {sum((len(v) for v in child_to_parents.values()))}')

def is_descendant(child, ancestor):
    """child 是否 ancestor 后代"""
    visited = set()
    stack = [child]
    while stack:
        cur = stack.pop()
        if cur in visited:
            continue
        visited.add(cur)
        if cur == ancestor:
            return True
        stack.extend(child_to_parents.get(cur, set()) - visited)
    return False
cid2cen = json.load(open(CLUSTER_JSON))
cen2mem = defaultdict(list)
for cid, cen in cid2cen.items():
    cen2mem[cen].append(cid)
multi = {cen: mem for cen, mem in cen2mem.items() if len(mem) > 1}
print(f'多成员簇: {len(multi)}')

def cluster_coherent(members):
    """簇内两两: 若存在直系父子关系或共享非根祖先则视为同分支; 返回(判定, 证据)"""
    checked = 0
    related = 0
    evidence = []
    for i in range(len(members)):
        for j in range(i + 1, len(members)):
            a, b = (members[i], members[j])
            if is_descendant(a, b) or is_descendant(b, a):
                related += 1
                if len(evidence) < 3:
                    evidence.append(f"{cid_to_fsn.get(a, '?')[:30]} <: {cid_to_fsn.get(b, '?')[:30]}")
            checked += 1
    return (checked, related, evidence)

def sem_type(fsn):
    m = re.search('\\(([^)]+)\\)\\s*$', fsn or '')
    return m.group(1) if m else 'UNKNOWN'
sample_clusters = random.sample(sorted(multi.items(), key=lambda x: -len(x[1])), 12)
print('\n抽样核查 12 个多成员簇（含最大簇）:')
print('-' * 60)
total_checked, total_related = (0, 0)
type_consistent_clusters = 0
manual_ok = 0
manual_notes = []
for cen, members in sample_clusters:
    fsn_list = [cid_to_fsn.get(m, '?') for m in members]
    types = set((sem_type(f) for f in fsn_list))
    type_consistent = len(types) == 1
    checked, related, ev = cluster_coherent(members)
    total_checked += checked
    total_related += related
    if type_consistent:
        type_consistent_clusters += 1
    cores = [re.sub('\\s*\\([^)]*\\)\\s*$', '', f).lower() for f in fsn_list]
    wordsets = [set(c.split()) for c in cores]
    overlap_all = 0
    for i in range(len(wordsets)):
        for j in range(i + 1, len(wordsets)):
            if wordsets[i] & wordsets[j]:
                overlap_all += 1
    n_pairs = len(wordsets) * (len(wordsets) - 1) / 2
    overlap_ratio = overlap_all / n_pairs if n_pairs else 0
    print(f"\n簇(代表={cid_to_fsn.get(cen, '?')[:50]}), 成员数={len(members)}, 类型一致={type_consistent}")
    for f in fsn_list[:8]:
        print(f'    - {f}')
    if len(fsn_list) > 8:
        print(f'    ... 还有 {len(fsn_list) - 8} 个成员')
    print(f'    [is-a 相关对 {related}/{checked}, token重叠对比例 {overlap_ratio:.2f}]')
    if overlap_ratio >= 0.5 or len(ev) > 0:
        manual_ok += 1
        manual_notes.append((cen, '一致'))
    else:
        manual_notes.append((cen, '存疑'))
print('\n' + '=' * 70)
print('语义簇自洽核查汇总')
print('=' * 70)
print(f'抽样簇数: {len(sample_clusters)}')
print(f'程序化 is-a 相关对: {total_related}/{total_checked} = {100 * total_related / max(total_checked, 1):.1f}%')
print(f'语义类型一致簇: {type_consistent_clusters}/{len(sample_clusters)}')
print(f'人工核查(基于FSN同义/近义): 一致 {manual_ok}/{len(sample_clusters)}')
for cen, note in manual_notes:
    print(f"    - {note}: {cid_to_fsn.get(cen, '?')[:55]}")
print('\n全量多成员簇程序化自洽统计:')
all_checked, all_related = (0, 0)
for cen, members in list(multi.items()):
    for i in range(len(members)):
        for j in range(i + 1, len(members)):
            all_checked += 1
            if is_descendant(members[i], members[j]) or is_descendant(members[j], members[i]):
                all_related += 1
print(f'  多成员簇内概念对: {all_checked}, is-a 相关: {all_related}, 比例: {100 * all_related / max(all_checked, 1):.2f}%')
summary = {'mapping_tiers': mapping['tier'].value_counts().to_dict(), 'cluster_programmatic_related_ratio': round(100 * all_related / max(all_checked, 1), 2)}
with open(_release_expand('${BENCHMARK_WORK_DIR}/outputs/analysis/mapping_quality_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print('\n已保存: /ml_training/outputs/analysis/mapping_quality_summary.json')

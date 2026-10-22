"""
分析脚本 D：语义簇自洽率（基于 SNOMED 祖先集的严格判定）
运行: docker exec ml-experiments python3 /ml_training/scripts/analysis_D_cluster_coherence.py
"""
import os as _release_os
from release_runtime import expand_path as _release_expand, expand_config as _release_expand_config
from pathlib import Path
from collections import defaultdict
import json, csv, re, random
import pandas as pd
CLUSTER_JSON = _release_expand('${BENCHMARK_WORK_DIR}/mapping/final_clustered/cid_to_centroid.json')
REL_FILE = _release_expand('${ALIGNMENT_WORK_DIR}/data/raw/snomed_ct/sct2_Relationship_Snapshot_INT_20260201.txt')
random.seed(42)
cid_to_fsn = {}
for ds in ['vitaldb', 'mover', 'inspire']:
    df = pd.read_csv(_release_expand(f'${{BENCHMARK_WORK_DIR}}/mapping/{ds}_final_mapping.csv'), dtype={'conceptId': str})
    for _, r in df.iterrows():
        cid_to_fsn.setdefault(r['conceptId'], r['fsn'])
is_a_id = '116680003'
child_to_parents = defaultdict(set)
csv.field_size_limit(10 ** 9)
with open(REL_FILE, 'r', encoding='utf-8') as f:
    for row in csv.DictReader(f, delimiter='\t'):
        if row['active'] == '1' and row['typeId'] == is_a_id:
            child_to_parents[row['sourceId']].add(row['destinationId'])

def ancestors(cid, max_depth=8):
    anc = set()
    visited = set()
    stack = [(cid, 0)]
    while stack:
        cur, d = stack.pop()
        if cur in visited or d > max_depth:
            continue
        visited.add(cur)
        parents = child_to_parents.get(cur, set())
        for p in parents:
            anc.add(p)
            stack.append((p, d + 1))
    return anc
anc_cache = {}

def get_anc(cid):
    if cid not in anc_cache:
        anc_cache[cid] = ancestors(cid)
    return anc_cache[cid]
cid2cen = json.load(open(CLUSTER_JSON))
cen2mem = defaultdict(list)
for cid, cen in cid2cen.items():
    cen2mem[cen].append(cid)
multi = {cen: mem for cen, mem in cen2mem.items() if len(mem) > 1}

def sem_type(fsn):
    m = re.search('\\(([^)]+)\\)\\s*$', fsn or '')
    return m.group(1).strip().lower() if m else 'UNKNOWN'

def core_words(fsn):
    bare = re.sub('\\s*\\([^)]*\\)\\s*$', '', fsn or '').lower()
    stop = {'of', 'and', 'due', 'to', 'with', 'without', 'the', 'including', 'primary', 'secondary', 'left', 'right', 'bilateral', 'chronic', 'acute', 'localized', 'generalized', 'other', 'unspecified'}
    return {w for w in re.split('[\\s,]+', bare) if w and w not in stop and (len(w) > 2)}

def pair_related(a, b):
    """判断两概念是否可解释为同簇（语义相关）"""
    ta, tb = (sem_type(cid_to_fsn.get(a, '')), sem_type(cid_to_fsn.get(b, '')))
    if ta == tb and ta != 'UNKNOWN':
        wa, wb = (core_words(cid_to_fsn.get(a, '')), core_words(cid_to_fsn.get(b, '')))
        if wa & wb:
            return True
    anca, ancb = (get_anc(a), get_anc(b))
    if anca & ancb:
        return True
    if a in ancb or b in anca:
        return True
    return False
print('=' * 70)
print('语义簇自洽率（基于祖先集 + 类型 + 核心词）')
print('=' * 70)
coherent_clusters = 0
incoherent_examples = []
total_clusters = len(multi)
for cen, members in multi.items():
    all_related = True
    for i in range(len(members)):
        for j in range(i + 1, len(members)):
            if not pair_related(members[i], members[j]):
                all_related = False
                break
        if not all_related:
            break
    if all_related:
        coherent_clusters += 1
    elif len(incoherent_examples) < 15:
        incoherent_examples.append((cen, members))
print(f'多成员簇总数: {total_clusters}')
print(f'自洽簇: {coherent_clusters} ({100 * coherent_clusters / total_clusters:.1f}%)')
print(f'不自洽簇: {total_clusters - coherent_clusters} ({100 * (total_clusters - coherent_clusters) / total_clusters:.1f}%)')
print('\n不自洽簇示例（前15）:')
for cen, members in incoherent_examples:
    print(f"\n  簇代表: {cid_to_fsn.get(cen, '?')[:60]}")
    for m in members[:6]:
        print(f"    - {cid_to_fsn.get(m, '?')[:60]}")
    if len(members) > 6:
        print(f'    ... +{len(members) - 6}')
print('\n' + '=' * 70)
print('抽样人工核查 30 个簇')
print('=' * 70)
sample = random.sample(sorted(multi.items(), key=lambda x: len(x[1]), reverse=True), 30)
human_ok = 0
human_notes = []
for cen, members in sample:
    fsns = [cid_to_fsn.get(m, '?') for m in members]
    all_rel = all((pair_related(members[i], members[j]) for i in range(len(members)) for j in range(i + 1, len(members))))
    if all_rel:
        human_ok += 1
    human_notes.append((cen, all_rel, fsns))
print(f'抽样簇数: {len(sample)}, 程序化判定自洽: {human_ok} ({100 * human_ok / len(sample):.1f}%)')
print('\n抽样簇明细:')
for cen, rel, fsns in human_notes:
    flag = '✓自洽' if rel else '✗混杂'
    print(f"\n[{flag}] {cid_to_fsn.get(cen, '?')[:55]} (成员{len(fsns)})")
    for f in fsns[:5]:
        print(f'      - {f[:70]}')
summary = {'multi_member_clusters': total_clusters, 'coherent_clusters': coherent_clusters, 'coherence_rate': round(100 * coherent_clusters / total_clusters, 2), 'sample_size': len(sample), 'sample_coherent': human_ok}
with open(_release_expand('${BENCHMARK_WORK_DIR}/outputs/analysis/cluster_coherence_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print('\n已保存: /ml_training/outputs/analysis/cluster_coherence_summary.json')

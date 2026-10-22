"""
构建 aligned_clustered benchmark：将 aligned CSV 中的 FSN 值替换为代表概念的 bare FSN

处理规则：
- 文本字段（预定义列表）：FSN → conceptId → centroid conceptId → centroid bare FSN
- 枚举字段：保留 aligned 原值不变
- 数值字段：保留不变
- 映射失败的值：保留原值
"""
import os as _release_os
from release_runtime import expand_path as _release_expand, expand_config as _release_expand_config
import sys, os, csv, json, re
from pathlib import Path
from collections import defaultdict
import pandas as pd
BENCHMARK_DIR = Path(_release_expand('${BENCHMARK_WORK_DIR}/benchmark'))
MAPPING_DIR = Path(_release_expand('${ALIGNMENT_WORK_DIR}/data/mapping/final_output'))
CLUSTER_DIR = Path(_release_expand('${BENCHMARK_WORK_DIR}/mapping/final_clustered'))
_DEFAULT_TASKS = ['asa', 'icu_admit', 'hospital_los', 'hospital_los_prolonged', 'aki', 'icu_los', 'icu_los_prolonged', 'inhospital_mortality', 'ioh', 'ioht']
if os.environ.get('CLUSTER_TASKS'):
    TASKS = [t.strip() for t in os.environ['CLUSTER_TASKS'].split(',') if t.strip()]
else:
    TASKS = _DEFAULT_TASKS
DATASETS = ['vitaldb', 'mover', 'inspire']
TEXT_COLUMNS = {'preop_diagnosis', 'scheduled_procedure_name', 'med_rec_summary', 'hx_circ_detail', 'hx_resp_detail', 'hx_endo_detail', 'hx_neuro_detail', 'hx_dige_detail', 'hx_spine_detail', 'hx_immu_detail', 'hx_urine_detail', 'hx_other_detail', 'hx_surgery_detail'}
SPLIT_CLUSTER_COLUMNS = {'med_rec_summary', 'hx_circ_detail', 'hx_resp_detail', 'hx_endo_detail', 'hx_neuro_detail', 'hx_dige_detail', 'hx_spine_detail', 'hx_immu_detail', 'hx_urine_detail', 'hx_other_detail', 'hx_surgery_detail'}

def load_mappings():
    """加载 FSN→cid, cid→centroid, centroid→bare_FSN 映射"""
    with open(CLUSTER_DIR / 'cid_to_centroid.json') as f:
        cid_to_centroid = json.load(f)
    print(f'  聚类映射: {len(cid_to_centroid)} 条')
    fsn_to_cid = {}
    cid_to_fsn = {}
    for ds in DATASETS:
        with open(MAPPING_DIR / f'{ds}_final_mapping.csv') as f:
            for row in csv.DictReader(f):
                fsn = row.get('fsn', '')
                cid = row['conceptId'].strip()
                bare = re.sub('\\s*\\([^)]*\\)\\s*$', '', fsn).strip() if fsn else ''
                if bare:
                    fsn_to_cid[bare.lower()] = cid
                    if cid not in cid_to_fsn:
                        cid_to_fsn[cid] = bare
    print(f'  FSN→cid: {len(fsn_to_cid)}, cid→FSN: {len(cid_to_fsn)}')
    return (fsn_to_cid, cid_to_centroid, cid_to_fsn)

def get_centroid_fsn(value: str, fsn_to_cid: dict, cid_to_centroid: dict, cid_to_fsn: dict) -> str:
    """将 FSN 值映射到其聚类代表概念的 bare FSN"""
    vl = value.strip().lower()
    cid = fsn_to_cid.get(vl)
    if not cid:
        bare = re.sub('\\s*\\([^)]*\\)\\s*$', '', vl).strip()
        if bare != vl:
            cid = fsn_to_cid.get(bare)
    if not cid:
        return value
    centroid = cid_to_centroid.get(cid, cid)
    fsn = cid_to_fsn.get(centroid)
    if fsn:
        return fsn
    return value

def main():
    print('=' * 60)
    print('构建 aligned_clustered benchmark')
    print('=' * 60)
    fsn_to_cid, cid_to_centroid, cid_to_fsn = load_mappings()
    for task in TASKS:
        task_dir = BENCHMARK_DIR / task / 'aligned_clustered'
        task_dir.mkdir(parents=True, exist_ok=True)
        for ds in DATASETS:
            src_dir = BENCHMARK_DIR / task / 'aligned' / ds
            if not src_dir.exists():
                continue
            dst_dir = task_dir / ds
            dst_dir.mkdir(parents=True, exist_ok=True)
            stats = {'converted': 0, 'kept': 0}
            for split_file in sorted(src_dir.glob('*.csv')):
                df = pd.read_csv(split_file, index_col=0)
                for col in df.columns:
                    if col in TEXT_COLUMNS and col in df.columns:
                        is_split_col = col in SPLIT_CLUSTER_COLUMNS
                        new_vals = []
                        for v in df[col]:
                            if pd.isna(v) or not str(v).strip():
                                new_vals.append(v)
                            elif is_split_col:
                                items = str(v).split(';')
                                clustered_items = []
                                for item in items:
                                    item = item.strip()
                                    if not item:
                                        continue
                                    new_item = get_centroid_fsn(item, fsn_to_cid, cid_to_centroid, cid_to_fsn)
                                    if new_item != item:
                                        stats['converted'] += 1
                                    else:
                                        stats['kept'] += 1
                                    clustered_items.append(new_item)
                                new_vals.append('; '.join(clustered_items) if clustered_items else v)
                            else:
                                new_val = get_centroid_fsn(str(v), fsn_to_cid, cid_to_centroid, cid_to_fsn)
                                if new_val != str(v):
                                    stats['converted'] += 1
                                else:
                                    stats['kept'] += 1
                                new_vals.append(new_val)
                        df[col] = new_vals
                df.to_csv(dst_dir / split_file.name)
            print(f"  {task}/{ds}: converted={stats['converted']}, kept={stats['kept']}")
    print('\n构建 combined benchmark...')
    for task in TASKS:
        combined_dir = BENCHMARK_DIR / task / 'aligned_clustered' / 'combined'
        combined_dir.mkdir(parents=True, exist_ok=True)
        for split in ['train', 'test', 'val']:
            dfs = []
            for ds in DATASETS:
                p = BENCHMARK_DIR / task / 'aligned_clustered' / ds / f'{split}.csv'
                if not p.exists():
                    continue
                df = pd.read_csv(p, index_col=0)
                df['_source_dataset'] = ds
                dfs.append(df)
            if not dfs:
                continue
            combined = pd.concat(dfs)
            combined.to_csv(combined_dir / f'{split}.csv')
            y_dfs = []
            for ds in DATASETS:
                yp = BENCHMARK_DIR / task / 'aligned' / ds / f'{split}_y.csv'
                if yp.exists():
                    y_dfs.append(pd.read_csv(yp, index_col=0))
            if y_dfs:
                pd.concat(y_dfs).to_csv(combined_dir / f'{split}_y.csv')
    print('\n✅ aligned_clustered benchmark 构建完成')
if __name__ == '__main__':
    main()

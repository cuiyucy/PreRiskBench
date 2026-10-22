"""
构建 "概念+簇" 双层表示 benchmark (aligned_dual)

思想: 不再用簇代表直接替换概念, 而是同时保留两层:
  - 概念层: aligned 的 SNOMED FSN 名称 (原始列名)
  - 簇层:   每列的簇代表概念 bare FSN (新增 {col}_cluster 列)

从 aligned benchmark 生成, 只处理 asa 任务 (测试用)。
"""
import os as _release_os
from release_runtime import expand_path as _release_expand, expand_config as _release_expand_config
import sys
import os
import csv
import json
import re
from pathlib import Path
import pandas as pd
SRC = Path(_release_expand('${BENCHMARK_WORK_DIR}'))
BENCH_DIR = SRC / 'benchmark'
MAPPING_DIR = Path(_release_expand('${ALIGNMENT_WORK_DIR}/data/mapping/final_output'))
CLUSTER_DIR = SRC / 'mapping' / 'final_clustered'
TASKS = ['asa']
DATASETS = ['vitaldb', 'mover', 'inspire']
SPLITS = ['train', 'test', 'val']
TEXT_COLUMNS = {'preop_diagnosis', 'scheduled_procedure_name', 'med_rec_summary', 'hx_circ_detail', 'hx_resp_detail', 'hx_endo_detail', 'hx_neuro_detail', 'hx_dige_detail', 'hx_spine_detail', 'hx_immu_detail', 'hx_urine_detail', 'hx_other_detail', 'hx_surgery_detail'}
SPLIT_COLUMNS = {'med_rec_summary', 'hx_circ_detail', 'hx_resp_detail', 'hx_endo_detail', 'hx_neuro_detail', 'hx_dige_detail', 'hx_spine_detail', 'hx_immu_detail', 'hx_urine_detail', 'hx_other_detail', 'hx_surgery_detail'}

def load_mappings():
    """加载 FSN→cid, cid→centroid, centroid→bare_FSN 映射"""
    with open(CLUSTER_DIR / 'cid_to_centroid.json') as f:
        cid_to_centroid = json.load(f)
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
    return (fsn_to_cid, cid_to_centroid, cid_to_fsn)

def get_centroid_fsn(value, fsn_to_cid, cid_to_centroid, cid_to_fsn):
    """FSN 文本 → 簇代表概念 bare FSN"""
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

def build():
    fsn_to_cid, cid_to_centroid, cid_to_fsn = load_mappings()
    print(f'映射加载: cid_to_centroid={len(cid_to_centroid)}, fsn_to_cid={len(fsn_to_cid)}')
    for task in TASKS:
        src_dir = BENCH_DIR / task / 'aligned'
        dst_dir = BENCH_DIR / task / 'aligned_dual'
        if not src_dir.exists():
            print(f'跳过: 无 {src_dir}')
            continue
        for ds in DATASETS:
            sds = src_dir / ds
            dds = dst_dir / ds
            if not sds.exists():
                continue
            dds.mkdir(parents=True, exist_ok=True)
            for split in SPLITS:
                fx = sds / f'{split}.csv'
                fy = sds / f'{split}_y.csv'
                if not fx.exists():
                    continue
                if fy.exists():
                    pd.read_csv(fy, index_col=0).to_csv(dds / f'{split}_y.csv')
                df = pd.read_csv(fx, index_col=0, low_memory=False)
                n_conv = 0
                n_kept = 0
                for col in list(df.columns):
                    if col not in TEXT_COLUMNS:
                        continue
                    new_col = f'{col}_cluster'
                    new_vals = []
                    for v in df[col]:
                        if pd.isna(v) or not str(v).strip():
                            new_vals.append(v)
                        elif col in SPLIT_COLUMNS:
                            items = [i.strip() for i in str(v).split(';') if i.strip()]
                            clustered = []
                            for item in items:
                                new_item = get_centroid_fsn(item, fsn_to_cid, cid_to_centroid, cid_to_fsn)
                                if new_item != item:
                                    n_conv += 1
                                else:
                                    n_kept += 1
                                clustered.append(new_item)
                            new_vals.append('; '.join(clustered) if clustered else v)
                        else:
                            new_item = get_centroid_fsn(str(v), fsn_to_cid, cid_to_centroid, cid_to_fsn)
                            if new_item != str(v):
                                n_conv += 1
                            else:
                                n_kept += 1
                            new_vals.append(new_item)
                    df[new_col] = new_vals
                df.to_csv(dds / f'{split}.csv')
                print(f'  {task}/{ds}/{split}: 列={df.shape[1]} (概念+簇), 转换={n_conv}, 保留={n_kept}')
        dcomb = dst_dir / 'combined'
        dcomb.mkdir(parents=True, exist_ok=True)
        for split in SPLITS:
            parts = []
            y_parts = []
            for ds in DATASETS:
                fx = dst_dir / ds / f'{split}.csv'
                fy = dst_dir / ds / f'{split}_y.csv'
                if fx.exists():
                    d = pd.read_csv(fx, index_col=0, low_memory=False)
                    d['_source_dataset'] = ds
                    parts.append(d)
                if fy.exists():
                    y_parts.append(pd.read_csv(fy, index_col=0))
            if parts:
                pd.concat(parts, axis=0).to_csv(dcomb / f'{split}.csv')
            if y_parts:
                pd.concat(y_parts, axis=0).to_csv(dcomb / f'{split}_y.csv')
        print(f'[{task}] aligned_dual 构建完成')
if __name__ == '__main__':
    build()

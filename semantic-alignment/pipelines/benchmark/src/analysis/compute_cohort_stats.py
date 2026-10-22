"""
统计全队列和基准队列的人口学/临床特征表格。
全队列   = 质量过滤（六字段非空 + 病史/检验纳排）后的队列（与论文表1『质量过滤后』一致）
基准队列 = 全队列经患者级采样（固定 seed=42）得到的 ASA 任务基准样本
输出: outputs/cohort_analysis/cohort_characteristics.tex
"""
import os as _release_os
from release_runtime import expand_path as _release_expand, expand_config as _release_expand_config
import pandas as pd
import numpy as np
from pathlib import Path
import json
ROOT = Path(_release_expand('${BENCHMARK_WORK_DIR}'))
DATASET_DIR = ROOT / 'dataset'
BENCH_DIR = ROOT / 'benchmark'
OUT_DIR = ROOT / 'outputs' / 'cohort_analysis'
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATASETS = ['vitaldb', 'mover', 'inspire']
BENCH_TASK = 'asa'

def classify_anesthesia(val: str) -> str:
    if pd.isna(val):
        return 'unknown'
    v = val.lower().strip()
    if 'general' in v:
        return 'GA'
    if any((t in v for t in ['neuraxial', 'epidural', 'spinal', 'intrathecal', 'subarachnoid'])):
        return 'neuraxial'
    if any((t in v for t in ['regional', 'nerve block', 'plexus'])):
        return 'regional'
    if any((t in v for t in ['mac', 'monitored', 'sedation'])):
        return 'MAC'
    if 'local' in v and 'anesthetic' in v:
        return 'regional'
    return 'other'

def classify_admission(val: str) -> str:
    if pd.isna(val):
        return 'unknown'
    v = val.lower().strip()
    if any((t in v for t in ['elective', 'optional', 'scheduled'])):
        return 'elective'
    if 'emergency' in v:
        return 'emergency'
    if 'outpatient' in v or 'ambulatory' in v:
        return 'outpatient'
    if 'inpatient' in v:
        return 'inpatient'
    return 'other'
LAB_RANGES = {'hemoglobin': (12.0, 16.0), 'hematocrit': (36.0, 48.0), 'wbc': (4.0, 11.0), 'platelet': (150, 400), 'ptinr': (0.9, 1.2), 'aptt': (25, 38), 'fibrinogen': (200, 400), 'sodium': (135, 145), 'potassium': (3.5, 5.2), 'chloride': (96, 106), 'bicarbonate': (22, 29), 'ionized_calcium': (1.15, 1.33), 'bun': (7, 20), 'creatinine': (0.6, 1.2), 'albumin': (3.5, 5.0), 'total_protein': (6.0, 8.3), 'total_bilirubin': (0.2, 1.2), 'alt': (7, 56), 'ast': (10, 40), 'glucose': (70, 140), 'lactate': (0.5, 2.2), 'crp': (0, 10), 'ph': (7.35, 7.45), 'pao2': (75, 100), 'paco2': (35, 45), 'sao2': (95, 100), 'base_excess': (-2, 2)}
LAB_GROUPS = {'全血细胞计数': ['hemoglobin', 'hematocrit', 'wbc', 'platelet'], '凝血功能': ['ptinr', 'aptt', 'fibrinogen'], '电解质': ['sodium', 'potassium', 'chloride', 'bicarbonate', 'ionized_calcium'], '肾功能': ['bun', 'creatinine'], '肝功能': ['albumin', 'total_protein', 'total_bilirubin', 'alt', 'ast'], '代谢/炎症': ['glucose', 'lactate', 'crp'], '动脉血气': ['ph', 'pao2', 'paco2', 'sao2', 'base_excess']}

def col_to_lab(col_name: str) -> str:
    name = col_name.lower().replace('_val', '').replace('_flag', '').replace('_', '')
    for std in LAB_RANGES:
        if std in name or name in std:
            return std
    return ''

def is_abnormal(val, lab_name: str) -> bool:
    if pd.isna(val) or lab_name not in LAB_RANGES:
        return False
    lo, hi = LAB_RANGES[lab_name]
    return val < lo or val > hi

def load_full_cohort(dataset: str) -> pd.DataFrame:
    """全队列特征: dataset/{ds}/alignment_feat/*.csv 按 surgery_id 合并"""
    feat_dir = DATASET_DIR / dataset / 'alignment_feat'
    csvs = sorted(feat_dir.glob('*.csv'))
    merged = pd.read_csv(csvs[0])
    for fp in csvs[1:]:
        df = pd.read_csv(fp)
        new_cols = {'surgery_id'} | set(df.columns) - set(merged.columns)
        df_sub = df[list(new_cols & set(df.columns))]
        merged = merged.merge(df_sub, on='surgery_id', how='outer')
    return merged.drop_duplicates(subset='surgery_id', keep='first')

def _is_nonempty(series: pd.Series) -> pd.Series:
    """与 build_benchmark 一致的非空判定（非 NaN、非空串、非 'nan' 串）"""
    s = series.astype(str).str.strip().str.lower()
    return series.notna() & (s != '') & (s != 'nan')

def load_quality_filtered_cohort(dataset: str) -> pd.DataFrame:
    """全队列 = 质量过滤后的队列（与论文表1『质量过滤后』口径一致）。

    过滤规则与 build_benchmark 相同:
      1. 硬性: sex, age, height, weight, preop_diagnosis, scheduled_procedure_name 非空
      2. 质量: >=1 个 hx_*_detail(s) 非空 或 >=5 个 *_val 检验值非空
    """
    df = load_full_cohort(dataset)
    mandatory = ['sex', 'age', 'height', 'weight', 'preop_diagnosis', 'scheduled_procedure_name']
    mask = pd.Series(True, index=df.index)
    for col in mandatory:
        if col in df.columns:
            mask &= _is_nonempty(df[col])
        else:
            return pd.DataFrame()
    hx_cols = [c for c in df.columns if c.startswith('hx_') and ('detail' in c or 'details' in c) and ('_cn' not in c)]
    has_hx = pd.Series(False, index=df.index)
    for col in hx_cols:
        has_hx |= _is_nonempty(df[col])
    lab_cols = [c for c in df.columns if c.endswith('_val')]
    n_lab = pd.Series(0, index=df.index)
    for col in lab_cols:
        n_lab += _is_nonempty(df[col]).astype(int)
    mask &= has_hx | (n_lab >= 5)
    return df[mask].copy()

def load_benchmark_cohort(dataset: str) -> pd.DataFrame:
    """基准队列 = 全队列经患者级采样后的 ASA 任务样本（train+val+test 并集去重）"""
    rows = []
    seen = set()
    ds_dir = BENCH_DIR / BENCH_TASK / 'aligned' / dataset
    if not ds_dir.exists():
        return pd.DataFrame()
    for split in ['train.csv', 'test.csv', 'val.csv']:
        fp = ds_dir / split
        if fp.exists():
            df = pd.read_csv(fp)
            for _, row in df.iterrows():
                sid = row['surgery_id']
                if sid not in seen:
                    seen.add(sid)
                    rows.append(row.to_dict())
    return pd.DataFrame(rows)
HX_COL_MAP = {'hx_circ_detail': '循环系统', 'hx_circ_details': '循环系统', 'hx_resp_detail': '呼吸系统', 'hx_resp_details': '呼吸系统', 'hx_endo_detail': '内分泌系统', 'hx_endo_details': '内分泌系统', 'hx_neuro_detail': '神经系统', 'hx_neuro_details': '神经系统', 'hx_dige_detail': '消化系统', 'hx_dige_details': '消化系统', 'hx_spine_detail': '脊柱', 'hx_spine_details': '脊柱', 'hx_immu_detail': '免疫系统', 'hx_immu_details': '免疫系统', 'hx_urine_detail': '泌尿系统', 'hx_urine_details': '泌尿系统', 'hx_other_detail': '其他', 'hx_other_details': '其他', 'hx_surgery_detail': '既往手术', 'hx_surgery_details': '既往手术'}

def compute_cohort_stats(df: pd.DataFrame) -> dict:
    s = {}
    n = len(df)
    if n == 0:
        return {'_n': 0}
    s['_n'] = n
    s['_patients'] = df['surgery_id'].nunique()
    sex_col = df.get('sex', pd.Series(dtype=str)).str.lower()
    s['male'] = int((sex_col == 'male').sum())
    s['female'] = int((sex_col == 'female').sum())
    for col in ['age', 'height', 'weight', 'bmi']:
        series = pd.to_numeric(df.get(col, pd.Series()), errors='coerce').dropna()
        s[f'{col}_mean'] = float(series.mean()) if len(series) > 0 else None
        s[f'{col}_std'] = float(series.std()) if len(series) > 0 else None
    for col in df.columns:
        cat = HX_COL_MAP.get(col)
        if cat is None:
            continue
        series = df[col]
        has_data = series.notna() & (series != '') & (series.astype(str).str.lower() != 'none')
        s[f'hx_{cat}'] = int(has_data.sum())
        s[f'hx_{cat}_avail'] = True
    for cat in set(HX_COL_MAP.values()):
        if f'hx_{cat}_avail' not in s:
            s[f'hx_{cat}_avail'] = False
    col_lab_map = {}
    for c in df.columns:
        if not c.endswith('_val'):
            continue
        std = col_to_lab(c)
        if std:
            col_lab_map[c] = std
    for group_name, labs in LAB_GROUPS.items():
        grp_cols = [c for c, std in col_lab_map.items() if std in labs]
        if not grp_cols:
            s[f'abn_{group_name}'] = 'N/A'
            s[f'abn_{group_name}_avail'] = False
            continue
        abn = pd.Series(False, index=df.index)
        for col in grp_cols:
            std = col_lab_map[col]
            series = pd.to_numeric(df[col], errors='coerce')
            abn = abn | series.apply(lambda x: is_abnormal(x, std))
        s[f'abn_{group_name}'] = int(abn.sum())
        s[f'abn_{group_name}_avail'] = True
    adm = df.get('surg_admission_chars', pd.Series(dtype=str))
    adm_class = adm.apply(classify_admission)
    s['elective'] = int((adm_class == 'elective').sum())
    s['emergency'] = int((adm_class == 'emergency').sum())
    s['outpatient'] = int((adm_class == 'outpatient').sum())
    s['inpatient'] = int((adm_class == 'inpatient').sum())
    s['dx_avail'] = int((df.get('preop_diagnosis', pd.Series()).notna() & (df.get('preop_diagnosis', pd.Series()) != '')).sum())
    s['proc_avail'] = int((df.get('scheduled_procedure_name', pd.Series()).notna() & (df.get('scheduled_procedure_name', pd.Series()) != '')).sum())
    dept = df.get('surgical_department', pd.Series(dtype=str))
    s['dept_avail'] = int(dept.notna().sum()) if not dept.empty and dept.notna().sum() > 0 else 0
    s['dept_has_col'] = 'surgical_department' in df.columns
    anes = df.get('scheduled_anesthesia_type', pd.Series(dtype=str)).apply(classify_anesthesia)
    s['anes_GA'] = int((anes == 'GA').sum())
    s['anes_neuraxial'] = int((anes == 'neuraxial').sum())
    s['anes_regional'] = int((anes == 'regional').sum())
    s['anes_MAC'] = int((anes == 'MAC').sum())
    s['anes_other'] = int((anes == 'other').sum())
    med_col = df.get('med_rec_summary', pd.Series(dtype=str))
    s['med_avail'] = int(med_col.notna().sum()) if not med_col.empty else 0
    s['med_has_col'] = 'med_rec_summary' in df.columns
    return s
FULL_LABEL_SPECS = {'asa': ('labels_asa.csv', 'label_asa'), 'ioh': ('labels_ioh.csv', 'label_ioh'), 'ioht': ('labels_ioht.csv', 'label_ioht'), 'aki': ('labels_aki.csv', 'label_aki'), 'icu_admit': ('labels_icu_admit.csv', 'label_icu_admit'), 'icu_los': ('labels_icu_los.csv', 'label_icu_los'), 'icu_los_prolonged': ('labels_icu_los.csv', 'label_icu_los_prolonged'), 'hospital_los': ('labels_hospital_los.csv', 'label_hospital_los'), 'hospital_los_prolonged': ('labels_hospital_los.csv', 'label_hospital_los_prolonged'), 'inhospital_mortality': ('labels_inhospital_mortality.csv', 'label_inhospital_mortality')}

def load_full_task_stats():
    """从 dataset/{ds}/prerisk_bench_labels/ 统计全队列任务样本量"""
    result = {}
    for task, (csv_name, label_col) in FULL_LABEL_SPECS.items():
        result[task] = {}
        for ds in DATASETS:
            fp = DATASET_DIR / ds / 'prerisk_bench_labels' / csv_name
            if not fp.exists():
                result[task][ds] = (0, 0)
                continue
            y = pd.read_csv(fp)
            if label_col not in y.columns:
                result[task][ds] = (0, 0)
                continue
            y = y.dropna(subset=[label_col])
            n_tot = len(y)
            if task in ['hospital_los', 'icu_los']:
                n_pos = n_tot
            elif task == 'asa':
                n_pos = int((y[label_col] >= 3).sum())
            else:
                n_pos = int((y[label_col] > 0).sum())
            result[task][ds] = (n_pos, n_tot)
    return result

def load_bench_task_stats():
    """从 benchmark/{task}/aligned/ 统计基准队列任务样本量"""
    result = {}
    for td in sorted(BENCH_DIR.iterdir()):
        if not td.is_dir():
            continue
        task = td.name
        result[task] = {}
        for ds in DATASETS:
            ds_dir = td / 'aligned' / ds
            if not ds_dir.exists():
                result[task][ds] = (0, 0)
                continue
            n_tot = 0
            n_pos = 0
            for split in ['train.csv', 'test.csv', 'val.csv']:
                fp = ds_dir / split
                yp = ds_dir / split.replace('.csv', '_y.csv')
                if not (fp.exists() and yp.exists()):
                    continue
                y = pd.read_csv(yp)
                n_tot += len(y)
                label_col = [c for c in y.columns if c != 'surgery_id'][0]
                if task in ['hospital_los', 'icu_los']:
                    n_pos = n_tot
                elif task == 'asa':
                    n_pos += int((y[label_col] >= 3).sum())
                else:
                    n_pos += int((y[label_col] > 0).sum())
            result[task][ds] = (n_pos, n_tot)
    return result

def fmt_pct(n_val, total: int) -> str:
    if total == 0:
        return '0 (0.0\\%)'
    return f'{int(n_val):,} ({n_val / total * 100:.1f}\\%)'

def fmt_mstd(mean, std) -> str:
    if mean is None or std is None or np.isnan(mean) or np.isnan(std):
        return 'NA'
    return f'{mean:.1f} $\\pm$ {std:.1f}'

def gen_latex(full_stats: dict, bench_stats: dict, full_task: dict, bench_task: dict):
    ds_order = ['vitaldb', 'inspire', 'mover']

    def row6(key, fmt='int'):
        cells = []
        for ds in ds_order:
            for src in [full_stats, bench_stats]:
                d = src.get(ds, {})
                n = d.get('_n', 0)
                if fmt == 'int':
                    v = d.get(key)
                    cells.append(str(int(v)) if isinstance(v, (int, float)) and (not isinstance(v, str)) else 'NA')
                elif fmt == 'pct':
                    v = d.get(key)
                    cells.append(fmt_pct(int(v), n) if isinstance(v, (int, float)) and (not isinstance(v, str)) and (n > 0) else 'NA')
                elif fmt == 'mstd':
                    cells.append(fmt_mstd(d.get(f'{key}_mean'), d.get(f'{key}_std')))
        return ' & '.join(cells)

    def row6_custom(fn):
        cells = []
        for ds in ds_order:
            for src in [full_stats, bench_stats]:
                cells.append(fn(ds, src))
        return ' & '.join(cells)
    lines = []
    L = lines.append
    L('\\begin{table*}[t]')
    L(' \\centering')
    L(' \\small')
    L(' \\setlength{\\tabcolsep}{3pt}')
    L(' \\caption{三个中心全队列与基准队列的结构化特征。全队列指经质量过滤（六项关键字段非空且病史/检验满足纳排）后的队列；基准队列为全队列经患者级分层采样（固定种子 42）得到的 ASA 任务样本。按七个术前特征模块组织：人口统计学、既往病史、术前检验、拟行手术、拟行麻醉、术前用药和补充信息。最后部分报告了各中心的各任务样本量。二元变量报告为 $n$~(\\%)，以队列总数为分母；缺失字段视为未记录。连续变量使用非缺失值报告为 平均值~$\\pm$~标准差。数据划分固定种子 42，模型重复实验采用 5 个随机种子（1--5）并报告均值$\\pm$标准差。}')
    L(' \\label{tab:cohort_characteristics}')
    L(' \\resizebox{\\textwidth}{!}{%')
    L(' \\begin{tabular}{lcccccc}')
    L(' \\toprule')
    L(' \\multirow{2}{*}{\\textbf{特征}} & \\multicolumn{2}{c}{\\textbf{VitalDB}} & \\multicolumn{2}{c}{\\textbf{INSPIRE}} & \\multicolumn{2}{c}{\\textbf{MOVER}} \\\\')
    L(' \\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}')
    L('  & \\textbf{全队列} & \\textbf{基准队列} & \\textbf{全队列} & \\textbf{基准队列} & \\textbf{全队列} & \\textbf{基准队列} \\\\')
    L(' \\midrule')
    L('         \\addlinespace[2pt]')
    L('         \\multicolumn{7}{l}{\\textbf{队列概览}} \\\\')
    L(f"         手术总数 & {row6('_n')} \\\\")
    L(f"         独立患者数 & {row6('_patients')} \\\\")
    L('         \\addlinespace[2pt]')
    L('         \\multicolumn{7}{l}{\\textbf{人口统计学特征}} \\\\')
    for item in [('male', '男性'), ('female', '女性')]:
        L(f"         {item[1]}, $n$~(\\%) & {row6(item[0], 'pct')} \\\\")
    for item in [('age', '年龄, 岁'), ('height', '身高, cm'), ('weight', '体重, kg'), ('bmi', 'BMI, kg/m$^2$')]:
        L(f"         {item[1]} (平均$\\pm$SD) & {row6(item[0], 'mstd')} \\\\")
    L('         \\addlinespace[2pt]')
    L('         \\multicolumn{7}{l}{\\textbf{既往病史}} \\\\')
    for cat in ['循环系统', '呼吸系统', '内分泌系统', '神经系统', '消化系统', '脊柱', '免疫系统', '泌尿系统', '其他', '既往手术']:
        row = row6_custom(lambda ds, src: fmt_pct(src[ds].get(f'hx_{cat}', 0), src[ds]['_n']) if src[ds].get(f'hx_{cat}_avail') else 'NA')
        if all((c.strip() == 'NA' for c in row.split('&'))):
            continue
        L(f'         {cat}病史, $n$~(\\%) & {row} \\\\')
    L('         \\addlinespace[2pt]')
    L('         \\multicolumn{7}{l}{\\textbf{术前检验}} \\\\')
    for grp in LAB_GROUPS:
        key = f'abn_{grp}'
        row = row6_custom(lambda ds, src: fmt_pct(src[ds].get(key, 0), src[ds]['_n']) if src[ds].get(f'{key}_avail') else 'NA')
        L(f'         {grp}异常, $n$~(\\%) & {row} \\\\')
    L('         \\addlinespace[2pt]')
    L('         \\multicolumn{7}{l}{\\textbf{拟行手术}} \\\\')
    L(f"         择期手术, $n$~(\\%) & {row6_custom(lambda ds, src: fmt_pct(src[ds].get('elective', 0), src[ds]['_n']) if src[ds].get('elective', 0) > 0 else 'NA')} \\\\")
    L(f"         急诊手术, $n$~(\\%) & {row6_custom(lambda ds, src: fmt_pct(src[ds].get('emergency', 0), src[ds]['_n']) if src[ds].get('emergency', 0) > 0 else 'NA')} \\\\")
    L(f"         住院手术, $n$~(\\%) & {row6_custom(lambda ds, src: fmt_pct(src[ds].get('inpatient', 0), src[ds]['_n']) if src[ds].get('inpatient', 0) > 0 else 'NA')} \\\\")
    L(f"         门诊/日间手术, $n$~(\\%) & {row6_custom(lambda ds, src: fmt_pct(src[ds].get('outpatient', 0), src[ds]['_n']) if src[ds].get('outpatient', 0) > 0 else 'NA')} \\\\")
    L(f"         术前诊断可用, $n$~(\\%) & {row6('dx_avail', 'pct')} \\\\")
    L(f"         拟行手术名称可用, $n$~(\\%) & {row6('proc_avail', 'pct')} \\\\")
    L(f"         外科科室可用, $n$~(\\%) & {row6_custom(lambda ds, src: fmt_pct(src[ds].get('dept_avail', 0), src[ds]['_n']) if src[ds].get('dept_has_col') else 'NA')} \\\\")
    L('         \\addlinespace[2pt]')
    L('         \\multicolumn{7}{l}{\\textbf{拟行麻醉}} \\\\')
    for key, label in [('anes_GA', '拟行全身麻醉'), ('anes_neuraxial', '拟行椎管内麻醉'), ('anes_regional', '拟行区域/阻滞麻醉'), ('anes_MAC', '拟行MAC/镇静'), ('anes_other', '其他拟行麻醉')]:
        L(f"         {label}, $n$~(\\%) & {row6(key, 'pct')} \\\\")
    L('         \\addlinespace[2pt]')
    L('         \\multicolumn{7}{l}{\\textbf{术前用药}} \\\\')
    L(f"         用药记录可用, $n$~(\\%) & {row6_custom(lambda ds, src: fmt_pct(src[ds].get('med_avail', 0), src[ds]['_n']) if src[ds].get('med_has_col') else 'NA')} \\\\")
    L('         \\addlinespace[2pt]')
    L('         \\multicolumn{7}{l}{\\textbf{任务目标 (阳性样本 / 总样本)}} \\\\')
    task_order = ['asa', 'ioh', 'ioht', 'aki', 'icu_admit', 'icu_los', 'icu_los_prolonged', 'hospital_los', 'hospital_los_prolonged', 'inhospital_mortality']
    task_labels = {'asa': 'ASA', 'ioh': 'IOH', 'ioht': 'IOHT', 'aki': 'AKI', 'icu_admit': 'ICU-Adm', 'icu_los': 'ICU-LOS', 'icu_los_prolonged': 'ICU-LOS-Prolonged', 'hospital_los': 'Hosp-LOS', 'hospital_los_prolonged': 'Hosp-LOS-Prolonged', 'inhospital_mortality': 'Mort-InHosp'}
    for task in task_order:
        label = task_labels.get(task, task)
        cells = []
        for ds in ds_order:
            for task_src in [full_task, bench_task]:
                n_pos, n_tot = task_src.get(task, {}).get(ds, (0, 0))
                if n_tot == 0:
                    cells.append('NA')
                elif task in ['hospital_los', 'icu_los', 'asa']:
                    cells.append(f'{n_tot:,}')
                else:
                    cells.append(f'{n_pos:,}/{n_tot:,}')
        L(f"         {label} & {' & '.join(cells)} \\\\")
    L(' \\bottomrule')
    L(' \\end{tabular}%')
    L(' }')
    L(' \\end{table*}')
    return '\n'.join(lines)

def main():
    print('=' * 60)
    print(' 全队列 vs 基准队列 统计')
    print(' 全队列 = 质量过滤后（六字段 + 病史/检验纳排）')
    print(' 基准队列 = 全队列经患者级采样(seed=42)的 ASA 任务样本')
    print('=' * 60)
    full_stats = {}
    bench_stats = {}
    for ds in DATASETS:
        df_full = load_quality_filtered_cohort(ds)
        df_bench = load_benchmark_cohort(ds)
        print(f'\n{ds.upper()}: 全队列(纳排后) {len(df_full)} 手术, 基准队列(ASA采样) {len(df_bench)} 手术')
        full_stats[ds] = compute_cohort_stats(df_full)
        bench_stats[ds] = compute_cohort_stats(df_bench)
    with open(OUT_DIR / 'cohort_stats.json', 'w') as f:
        json.dump({'full': full_stats, 'bench': bench_stats}, f, indent=2, ensure_ascii=False, default=str)
    full_task_stats = load_full_task_stats()
    bench_task_stats = load_bench_task_stats()
    for task in sorted(set(list(full_task_stats.keys()) + list(bench_task_stats.keys()))):
        parts = []
        for ds in DATASETS:
            f = full_task_stats.get(task, {}).get(ds, (0, 0))
            b = bench_task_stats.get(task, {}).get(ds, (0, 0))
            parts.append(f'{ds}:f={f[0]}/{f[1]} b={b[0]}/{b[1]}')
        if parts:
            print(f"  {task}: {' '.join(parts)}")
    tex = gen_latex(full_stats, bench_stats, full_task_stats, bench_task_stats)
    tex_path = OUT_DIR / 'cohort_characteristics.tex'
    tex_path.write_text(tex)
    print(f"\n{'=' * 60}")
    print(f'LaTeX: {tex_path}')
    print(f'JSON:  {OUT_DIR}/cohort_stats.json')
    print(f"{'=' * 60}")
    print(tex)
if __name__ == '__main__':
    main()

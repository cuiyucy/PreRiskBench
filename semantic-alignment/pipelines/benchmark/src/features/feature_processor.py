"""
统一特征处理器

根据用户指定的清洗规则处理所有特征文件，包括:
- 删除冗余列（summary、detail、_cn、_all等）
- 去重跨文件的重复字段（surg_admission_chars）
- 文本列解析和独热编码（desc_ens、med_rec、手术名等）
- 枚举列独热编码（sex、race、麻醉类型等）
- 数值/布尔列保留

支持两种模式:
- unaligned: 使用 prerisk_bench_feat（非对齐原始数据）
- aligned:   使用 alignment_feat（SNOMED对齐后数据）
"""
import os as _release_os
from release_runtime import expand_path as _release_expand, expand_config as _release_expand_config
import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict, Counter
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureProcessor:
    """
    统一特征处理器

    工作流: fit() → transform() 或 fit_transform_all()
    fit阶段在多个数据集上学习编码词汇表，确保跨数据集列空间一致。
    """
    FEATURE_TYPES = ['feat_pat_demo', 'feat_anes_vars', 'feat_surg_vars', 'feat_test_vars', 'feat_past_his', 'feat_med_rec']
    DROP_COLS = {'patient_id', 'mrn', 'patient_class_nm', 'surgery_urgency'}
    DROP_SUMMARY_COLS = {'pat_demo_summary', 'anes_vars_summary', 'surg_vars_summary', 'past_his_summary', 'test_vars_summary', 'other_tests_summary'}
    DROP_COMPLEX_COLS = {'ecg_detail', 'pft_detail'}
    DETAIL_PATTERNS = ['_detail', '_details']
    DROP_SUFFIXES = ['_cn', '_all', '_all_cn']
    SURG_ADMISSION_SOURCE = 'feat_pat_demo'
    DATASET_EXCLUDE_COLS = {'mover': {'surgery_urgency'}}
    DROP_DESC_PATTERNS = ['_desc_ens', '_desc_cns']
    ONEHOT_COLUMNS = {'feat_pat_demo': ['sex', 'race'], 'feat_anes_vars': ['scheduled_anesthesia_type'], 'feat_surg_vars': ['surgical_department', 'scheduled_operative_type', 'scheduled_surgical_approach', 'scheduled_surgical_position', 'preop_diagnosis', 'scheduled_procedure_name'], 'feat_test_vars': [], 'feat_past_his': [], 'feat_med_rec': []}

    def __init__(self, base_dir: str='dataset', mode: str='unaligned', max_onehot_categories: int=500, encode_text: bool=True):
        """
        Args:
            base_dir: 数据集根目录
            mode: 'unaligned' 或 'aligned'
            max_onehot_categories: 单个列最大独热编码类别数，超出则只取topN
            encode_text: 是否编码文本列。Exp1设为False只保留数值/布尔列
        """
        self.base_dir = Path(base_dir)
        self.mode = mode
        self.max_onehot_categories = max_onehot_categories
        self.encode_text = encode_text
        self.onehot_vocabularies: Dict[str, List[str]] = {}
        self.fitted = False
        self.final_column_order: List[str] = []

    def _get_feat_dir(self, dataset: str) -> Path:
        """获取特征目录"""
        subdir = 'prerisk_bench_feat' if self.mode == 'unaligned' else 'alignment_feat'
        return self.base_dir / dataset / subdir

    def load_features(self, dataset: str) -> Dict[str, pd.DataFrame]:
        """加载数据集的所有特征文件"""
        feat_dir = self._get_feat_dir(dataset)
        suffix = '_aligned' if self.mode == 'aligned' else ''
        features = {}
        for ft in self.FEATURE_TYPES:
            path = feat_dir / f'{ft}{suffix}.csv'
            if path.exists():
                features[ft] = pd.read_csv(path, low_memory=False)
            else:
                logger.debug(f'  文件不存在: {path}')
        return features

    def _should_drop_detail(self, col_name: str) -> bool:
        """判断是否detail类列（past_his的hx_*_detail保留用于解析编码）"""
        if col_name.startswith('hx_') and (col_name.endswith('_detail') or col_name.endswith('_details')):
            return False
        for p in self.DETAIL_PATTERNS:
            if col_name.endswith(p):
                return True
        return False

    def _should_drop_suffix(self, col_name: str) -> bool:
        """判断是否应丢弃的后缀列（_cn, _all, _all_cn）"""
        for s in self.DROP_SUFFIXES:
            if col_name.endswith(s):
                return True
        return False

    def clean_features(self, features: Dict[str, pd.DataFrame], dataset: str='combined') -> Dict[str, pd.DataFrame]:
        """
        对加载的特征字典应用清洗规则

        Returns:
            {feat_type: cleaned_df} 清洗后的特征字典
        """
        cleaned = {}
        surg_adm_df = None
        if self.SURG_ADMISSION_SOURCE in features:
            df_src = features[self.SURG_ADMISSION_SOURCE]
            if 'surgery_id' in df_src.columns and 'surg_admission_chars' in df_src.columns:
                surg_adm_df = df_src[['surgery_id', 'surg_admission_chars']].copy()
        for ft, df in features.items():
            df_clean = df.copy()
            if dataset == 'vitaldb' and ft == 'feat_past_his':
                rename_map = {'hx_circ_details': 'hx_circ_detail', 'hx_endo_details': 'hx_endo_detail'}
                df_clean = df_clean.rename(columns=rename_map)
            cols_to_drop = set()
            for col in df_clean.columns:
                if col == 'surgery_id':
                    continue
                if col in self.DROP_COLS:
                    cols_to_drop.add(col)
                elif col in self.DROP_SUMMARY_COLS:
                    cols_to_drop.add(col)
                elif col in self.DROP_COMPLEX_COLS:
                    cols_to_drop.add(col)
                elif self._should_drop_detail(col):
                    cols_to_drop.add(col)
                elif self._should_drop_suffix(col):
                    cols_to_drop.add(col)
                elif any((col.endswith(p) for p in self.DROP_DESC_PATTERNS)):
                    cols_to_drop.add(col)
                elif col.endswith('_flag'):
                    cols_to_drop.add(col)
            if ft != self.SURG_ADMISSION_SOURCE and 'surg_admission_chars' in df_clean.columns:
                cols_to_drop.add('surg_admission_chars')
            df_clean = df_clean.drop(columns=list(cols_to_drop), errors='ignore')
            cleaned[ft] = df_clean
        if surg_adm_df is not None:
            cleaned['surg_admission_chars'] = surg_adm_df
        return cleaned

    @staticmethod
    @staticmethod
    def parse_med_rec_drugs(series: pd.Series) -> Dict[str, List[str]]:
        """
        解析 med_rec_summary 提取药名列表

        格式: "时间:药物(途径),药物(途径).时间:药物(途径)."
        示例:
          "Recent Use (<1wk): ACETAMINOPHEN 500 MG OR TABS(Oral), SODIUM CHLORIDE 0.9 % IV SOLN(IV)."
          "Past History: Product containing precisely paracetamol 500 milligram/1 each conventional release oral tablet(Oral route)."
          "Recent Use (<1wk): famotidine(iv), flomoxef(iv)."

        处理规则:
          1. 按 . 分割时间段（仅当.后跟大写字母，避免误切0.9%等）
          2. 取 : 后内容作为药物部分
          3. 按逗号分割各药物条目，但排除数字中的逗号（如 200,000）
          4. 去掉 (途径) 括号部分，但保留 SNOMED 标签中的括号（如 (product)）
          5. 保留完整药名原样

        Returns:
            {surgery_id: [drug_name, drug_name, ...]}
        """
        drug_sets = {}
        for idx, val in series.items():
            if pd.isna(val) or not isinstance(val, str):
                drug_sets[idx] = []
                continue
            if 'no preoperative' in val.lower() and 'found' in val.lower():
                drug_sets[idx] = []
                continue
            drugs = set()
            import re
            segments = re.split('\\.\\s*(?=[A-Z])', val)
            for seg in segments:
                seg = seg.strip()
                if not seg:
                    continue
                if ':' in seg:
                    drug_part = seg.split(':', 1)[1].strip()
                else:
                    drug_part = seg
                drug_entries = re.split(',\\s*(?![0-9])', drug_part)
                for drug_entry in drug_entries:
                    drug_entry = drug_entry.strip()
                    if not drug_entry or len(drug_entry) < 2:
                        continue
                    if drug_entry.endswith('.'):
                        drug_entry = drug_entry[:-1].strip()
                    route_keywords = ['oral', 'iv', 'intravenous', 'subcutaneous', 'sc', 'im', 'intramuscular', 'topical', 'nasal', 'rectal', 'route', 'inhalation']
                    if '(' in drug_entry and drug_entry.endswith(')'):
                        match = re.search('\\(([^)]+)\\)$', drug_entry)
                        if match:
                            content = match.group(1).lower()
                            if any((kw in content for kw in route_keywords)):
                                drug_name = drug_entry[:drug_entry.rfind('(')].strip()
                            else:
                                drug_name = drug_entry.strip()
                        else:
                            drug_name = drug_entry.strip()
                    else:
                        drug_name = drug_entry.strip()
                    if drug_name and len(drug_name) >= 3 and (not drug_name.isdigit()):
                        drugs.add(drug_name)
            drug_sets[idx] = sorted(drugs)
        return drug_sets

    @staticmethod
    def extract_unique_drugs(drug_dict: Dict[str, List[str]]) -> List[str]:
        """从药物字典中提取所有唯一药名并排序"""
        all_drugs = set()
        for drug_list in drug_dict.values():
            all_drugs.update((d for d in drug_list if d and len(d) >= 3))
        return sorted(all_drugs)

    @staticmethod
    def parse_past_his_details(series: pd.Series, dataset: str='combined') -> Dict[str, List[str]]:
        """
        解析 past_his _detail 列，提取病史内容

        支持格式:
          - inspire: "4 days ago diagnosed as Other and unspecified disorders of nose; 3 days ago underwent Bypass..."
            用户要求: 对于 inspire，只取 as/underwent 后的第一个病史（逗号分隔前的内容）
          - mover:  "急性呼吸衰竭伴缺氧 (CMS-HCC); 慢性阻塞性肺疾病急性加重"
          - vitaldb: "Hypertension"

        提取规则:
          - "diagnosed as xxx" → 提取 xxx
          - "underwent xxx"    → 提取 xxx
          - 分号分隔的多条病史分别提取
          - 纯文本（vitaldb/mover格式）按分号分割
          - 对于 inspire: 每个 segment 提取的内容如果包含逗号，只保留逗号前的内容

        Returns:
            {surgery_id: [history_item, history_item, ...]}
        """
        import re
        result = {}
        for idx, val in series.items():
            if pd.isna(val) or not isinstance(val, str) or (not val.strip()):
                result[idx] = []
                continue
            items = set()
            val = val.strip()
            segments = re.split(';\\s*', val)
            for seg in segments:
                seg = seg.strip()
                if not seg or len(seg) < 2:
                    continue
                diagnosed_match = re.search('diagnosed\\s+as\\s+(.+?)(?:;\\s*|$)', seg, re.IGNORECASE)
                underwent_match = re.search('underwent\\s+(.+?)(?:;\\s*|$)', seg, re.IGNORECASE)
                item = None
                if diagnosed_match:
                    item = diagnosed_match.group(1).strip()
                elif underwent_match:
                    item = underwent_match.group(1).strip()
                elif not any((kw in seg.lower() for kw in ['days ago', 'day ago', 'months ago', 'year ago', 'years ago'])):
                    if len(seg) >= 2 and (not seg.isdigit()):
                        item = seg
                if item:
                    if dataset == 'inspire':
                        depth = 0
                        split_pos = -1
                        for i, ch in enumerate(item):
                            if ch in '(<':
                                depth += 1
                            elif ch in ')>':
                                depth -= 1
                            elif ch == ',' and depth == 0:
                                split_pos = i
                                break
                        if split_pos > 0:
                            item = item[:split_pos].strip()
                    if item.startswith('(') and item.endswith(')'):
                        item = item[1:-1].strip()
                    if len(item) >= 2:
                        items.add(item)
            result[idx] = sorted(items)
        return result

    @staticmethod
    def extract_unique_history_items(history_dict: Dict[str, List[str]]) -> List[str]:
        """从病史字典中提取所有唯一条目并排序"""
        all_items = set()
        for item_list in history_dict.values():
            all_items.update((i for i in item_list if i and len(i) >= 2))
        return sorted(all_items)

    @staticmethod
    def parse_desc_ens(series: pd.Series, col_name: str) -> pd.Series:
        """
        清洗 desc_ens 列，提取SNOMED概念名用于后续独热编码

        desc_ens 格式示例:
          "4 days ago diagnosed as Other and unspecified disorders of nose"
          "306 days ago diagnosed as Essential (primary) hypertension; 301 days..."
          "292 days ago underwent Bypass of Coronary Artery..."

        策略: 去除时间前缀（"N days ago diagnosed as" / "N days ago underwent"），
              分号分隔多个概念时取第一个

        Returns:
            清洗后的Series（索引不变，值为概念名）
        """

        def clean_desc(val):
            if pd.isna(val) or not isinstance(val, str):
                return np.nan
            cleaned = re.sub('\\d+\\s*(day|month|year)s?\\s*ago\\s*(diagnosed\\s*as|underwent)\\s*', '', val, flags=re.IGNORECASE).strip()
            if ';' in cleaned:
                cleaned = cleaned.split(';')[0].strip()
            return cleaned if cleaned else np.nan
        return series.apply(clean_desc)

    @staticmethod
    def _classify_column(series: pd.Series) -> str:
        """识别列类型: 'numeric' | 'boolean' | 'categorical' | 'text'"""
        s = series.dropna()
        if len(s) == 0:
            return 'empty'
        n_unique = s.nunique()
        try:
            pd.to_numeric(s)
            return 'boolean' if n_unique <= 2 else 'numeric'
        except (ValueError, TypeError):
            pass
        return 'text'

    def fit(self, datasets: List[str]):
        """
        在多个数据集上学习独热编码词汇表

        Args:
            datasets: 数据集名称列表 (如 ['vitaldb', 'mover', 'inspire'])
        """
        logger.info(f'[FIT] 模式={self.mode}, 数据集={datasets}')
        vocab: Dict[str, Counter] = defaultdict(Counter)
        for ds in datasets:
            features = self.load_features(ds)
            if not features:
                continue
            features = self.clean_features(features, ds)
            for ft, df in features.items():
                if ft == 'feat_med_rec' and 'med_rec_summary' in df.columns:
                    drug_dict = self.parse_med_rec_drugs(df['med_rec_summary'])
                    for drugs in drug_dict.values():
                        vocab['med_rec_summary'].update(drugs)
                if ft == 'feat_past_his':
                    detail_cols = [c for c in df.columns if (c.endswith('_detail') or c.endswith('_details')) and c.startswith('hx_')]
                    for col in detail_cols:
                        history_dict = self.parse_past_his_details(df[col], ds)
                        for items in history_dict.values():
                            vocab[col].update(items)
                onehot_targets = self.ONEHOT_COLUMNS.get(ft, [])
                for col in onehot_targets:
                    if col in df.columns:
                        vocab[col].update(df[col].dropna().astype(str))
                if self.encode_text:
                    text_cols = ['preop_diagnosis', 'scheduled_procedure_name']
                    hx_cols = [c for c in df.columns if c.startswith('hx_') and (c.endswith('_detail') or c.endswith('_details'))]
                    text_cols.extend(hx_cols)
                    for col in text_cols:
                        if col in df.columns:
                            vocab[col].update(df[col].dropna().astype(str))
            if 'surg_admission_chars' in features:
                src = features['surg_admission_chars']
                if 'surg_admission_chars' in src.columns:
                    vocab['surg_admission_chars'].update(src['surg_admission_chars'].dropna().astype(str))
        self.onehot_vocabularies = {}
        for col, counts in vocab.items():
            most_common = counts.most_common(self.max_onehot_categories)
            sorted_vals = sorted([val for val, count in most_common])
            if len(counts) > self.max_onehot_categories:
                logger.warning(f'  {col}: 总共 {len(counts)} 个唯一值 > max({self.max_onehot_categories}), 已保留频率最高的 {self.max_onehot_categories} 个')
            self.onehot_vocabularies[col] = sorted_vals
            logger.debug(f'  {col}: {len(sorted_vals)} 类')
        self.fitted = True
        total_categories = sum((len(v) for v in self.onehot_vocabularies.values()))
        logger.info(f'[FIT] 完成: {len(self.onehot_vocabularies)} 个编码列, 共 {total_categories} 个类别')

    def transform(self, dataset: str) -> pd.DataFrame:
        """
        处理单个数据集: 加载 → 清洗 → 编码 → 返回数值特征矩阵

        Returns:
            DataFrame: surgery_id 为索引的特征矩阵
        """
        if not self.fitted:
            raise RuntimeError('必须先调用 fit() 学习词汇表')
        features = self.load_features(dataset)
        if not features:
            raise ValueError(f'数据集 {dataset} 无特征文件')
        features = self.clean_features(features, dataset)
        exclude = self.DATASET_EXCLUDE_COLS.get(dataset, set())
        if exclude:
            for ft in list(features.keys()):
                drop_here = [c for c in exclude if c in features[ft].columns]
                if drop_here:
                    logger.info(f'  [DROP] {dataset}/{ft}: {drop_here} (数据集特定)')
                    features[ft] = features[ft].drop(columns=drop_here)
        result_dfs = []
        for ft, df in features.items():
            if ft == 'surg_admission_chars':
                df_proc = self._encode_feature_df(df, ft, dataset)
                if df_proc is not None:
                    result_dfs.append(df_proc)
                continue
            if df.empty or 'surgery_id' not in df.columns:
                continue
            df_proc = self._encode_feature_df(df, ft, dataset)
            if df_proc is not None:
                result_dfs.append(df_proc)
        if not result_dfs:
            raise ValueError(f'数据集 {dataset} 处理后无可用特征')
        merged = result_dfs[0]
        for df in result_dfs[1:]:
            merged = merged.join(df, how='outer')
        merged = merged.fillna(0)
        logger.info(f'[TRANSFORM] {dataset} ({self.mode}): {merged.shape[0]} 样本, {merged.shape[1]} 特征')
        return merged

    @staticmethod
    def _onehot_encode_column(series: pd.Series, col_prefix: str, vocab: List[str]) -> Optional[pd.DataFrame]:
        """
        高效独热编码单列
        """
        if series.empty:
            return None
        vocab_map = {v: i for i, v in enumerate(vocab)}
        vocab_cols = [f'{col_prefix}_{v}' for v in vocab]
        idx = series.index
        data = np.zeros((len(idx), len(vocab)), dtype=np.float32)
        for i, val in enumerate(series.fillna('__missing__').values):
            j = vocab_map.get(str(val), -1)
            if j >= 0:
                data[i, j] = 1.0
        return pd.DataFrame(data, index=idx, columns=vocab_cols, dtype=np.float32)

    def _encode_feature_df(self, df: pd.DataFrame, ft: str, dataset: str) -> Optional[pd.DataFrame]:
        """
        对单个特征DataFrame进行编码

        Args:
            df: 含 surgery_id 的原始DataFrame
            ft: 特征类型键
            dataset: 数据集名（用于日志）

        Returns:
            编码后的DataFrame（surgery_id为索引，其余为数值特征列）
            若无可编码特征则返回None
        """
        df = df.copy()
        df = df.set_index('surgery_id')
        if 'patient_id' in df.columns:
            df = df.drop(columns=['patient_id'])
        encoded_parts = []
        for col in df.columns:
            if col in self.onehot_vocabularies:
                continue
            col_type = self._classify_column(df[col])
            if col_type in ('numeric', 'boolean'):
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                encoded_parts.append(df[[col]].astype(np.float32))
        if ft == 'feat_med_rec' and 'med_rec_summary' in df.columns:
            if 'med_rec_summary' in self.onehot_vocabularies:
                drug_vocab = self.onehot_vocabularies['med_rec_summary']
                drug_matrix = pd.DataFrame(0, index=df.index, columns=[f'med_rec_summary_{d}' for d in drug_vocab], dtype=np.float32)
                for idx, val in df['med_rec_summary'].items():
                    if pd.isna(val) or not isinstance(val, str) or (not val.strip()):
                        continue
                    if ';' in val or (val and len(val) > 0):
                        drugs = [d.strip() for d in val.split(';') if d.strip()]
                    else:
                        drugs = self.parse_med_rec_drugs(pd.Series({idx: val}))[idx]
                    for d in drugs:
                        if d in drug_vocab:
                            drug_matrix.loc[idx, f'med_rec_summary_{d}'] = 1.0
                encoded_parts.append(drug_matrix)
        if ft == 'feat_past_his':
            detail_cols = [c for c in df.columns if (c.endswith('_detail') or c.endswith('_details')) and c.startswith('hx_')]
            for col in detail_cols:
                if col in self.onehot_vocabularies:
                    hist_vocab = self.onehot_vocabularies[col]
                    hist_matrix = pd.DataFrame(0, index=df.index, columns=[f'{col}_{item}' for item in hist_vocab], dtype=np.float32)
                    for idx, val in df[col].items():
                        if pd.isna(val) or not isinstance(val, str) or (not val.strip()):
                            continue
                        if ';' in val or (val and len(val) > 0):
                            items = [i.strip() for i in val.split(';') if i.strip()]
                        else:
                            items = self.parse_past_his_details(pd.Series({idx: val}), dataset)[idx]
                        for item in items:
                            col_name = f'{col}_{item}'
                            if col_name in hist_matrix.columns:
                                hist_matrix.loc[idx, col_name] = 1.0
                    encoded_parts.append(hist_matrix)
        onehot_targets = self.ONEHOT_COLUMNS.get(ft, [])
        for col in onehot_targets:
            if col in df.columns and col in self.onehot_vocabularies:
                dummies = self._onehot_encode_column(df[col], col, self.onehot_vocabularies[col])
                if dummies is not None:
                    encoded_parts.append(dummies)
        if ft == 'surg_admission_chars' and 'surg_admission_chars' in df.columns:
            col = 'surg_admission_chars'
            if col in self.onehot_vocabularies:
                dummies = self._onehot_encode_column(df[col], col, self.onehot_vocabularies[col])
                if dummies is not None:
                    encoded_parts.append(dummies)
        if not encoded_parts:
            return None
        result = pd.concat(encoded_parts, axis=1)
        return result

    def fit_transform_all(self, datasets: List[str]) -> Dict[str, pd.DataFrame]:
        """
        fit + transform 所有数据集

        Returns:
            {dataset_name: feature_matrix_df}
        """
        self.fit(datasets)
        results = {}
        for ds in datasets:
            try:
                results[ds] = self.transform(ds)
            except Exception as e:
                logger.warning(f'  {ds} 处理失败: {e}')
        return results

    def clean_and_merge(self, dataset: str) -> Optional[pd.DataFrame]:
        """
        清洗并合并所有特征，保留原始数据类型（不编码）

        用于benchmark构建：只做清洗，后期按需one-hot/BERT编码
        解析 med_rec 和 hx_*_detail 字段并存储解析后的文本（分号分隔）

        Returns:
            DataFrame: surgery_id为索引的原始特征矩阵
            - 数值列: 保持原值
            - 枚举列: 保持字符串
            - 文本列: 保持解析后/清洗后的文本
        """
        features = self.load_features(dataset)
        if not features:
            logger.warning(f'  {dataset}: 无特征文件')
            return None
        cleaned = self.clean_features(features, dataset)
        if not cleaned:
            return None
        merged_parts = []
        for ft, df in cleaned.items():
            df_part = df.copy()
            if 'surgery_id' in df_part.columns:
                df_part = df_part.set_index('surgery_id')
            if 'patient_id' in df_part.columns:
                df_part = df_part.drop(columns=['patient_id'])
            if ft == 'feat_med_rec' and 'med_rec_summary' in df_part.columns:
                drug_dict = self.parse_med_rec_drugs(df_part['med_rec_summary'])
                df_part['med_rec_summary'] = pd.Series({sid: '; '.join(drugs) if drugs else '' for sid, drugs in drug_dict.items()})
            if ft == 'feat_past_his':
                detail_cols = [c for c in df_part.columns if (c.endswith('_detail') or c.endswith('_details')) and c.startswith('hx_')]
                for col in detail_cols:
                    history_dict = self.parse_past_his_details(df_part[col], dataset)
                    df_part[col] = pd.Series({sid: '; '.join(items) if items else '' for sid, items in history_dict.items()})
            merged_parts.append(df_part)
        if not merged_parts:
            return None
        result = pd.concat(merged_parts, axis=1)
        result = result.loc[:, ~result.columns.duplicated()]
        logger.info(f'  {dataset}: clean_and_merge 完成 {result.shape[0]}行 x {result.shape[1]}列')
        return result

    def get_feature_dim(self) -> int:
        """返回编码后的特征维度（需要先fit）"""
        if not self.fitted:
            return 0
        total = sum((len(v) for v in self.onehot_vocabularies.values()))
        return total

    def print_summary(self):
        """打印编码词汇表摘要"""
        if not self.fitted:
            print('未fit，无词汇表')
            return
        print(f"\n{'=' * 60}")
        print(f' 编码词汇表 [{self.mode}]')
        print(f"{'=' * 60}")
        for col, vals in sorted(self.onehot_vocabularies.items()):
            preview = ', '.join(vals[:5])
            if len(vals) > 5:
                preview += f', ... (+{len(vals) - 5})'
            print(f'  {col} ({len(vals)}): {preview}')

    def find_common_columns(self, datasets: List[str]) -> List[str]:
        """
        找出所有数据集共有的特征列

        Returns:
            共有的列名列表（字母序）
        """
        all_columns = []
        for ds in datasets:
            try:
                X = self.transform(ds)
                all_columns.append(set(X.columns))
            except Exception as e:
                logger.warning(f'  {ds} 变换失败: {e}')
                return []
        common = sorted(set.intersection(*all_columns)) if all_columns else []
        logger.info(f"  共有列: {len(common)} / {', '.join((f'{ds}={len(c)}' for ds, c in zip(datasets, [set.intersection(*all_columns)] * len(datasets))))}")
        return common

    def filter_to_common(self, feature_dict: Dict[str, pd.DataFrame], common_columns: List[str]) -> Dict[str, pd.DataFrame]:
        """将多数据集特征矩阵裁剪到共有列空间"""
        filtered = {}
        for ds, df in feature_dict.items():
            available = [c for c in common_columns if c in df.columns]
            filtered[ds] = df[available]
            logger.info(f'  {ds}: {len(available)}/{len(common_columns)} 共有列')
        return filtered

def create_processor(mode: str='unaligned', max_onehot_categories: int=500) -> FeatureProcessor:
    return FeatureProcessor(mode=mode, max_onehot_categories=max_onehot_categories)
if __name__ == '__main__':
    print('=' * 60)
    print(' 测试: 未对齐特征处理 (unaligned)')
    print('=' * 60)
    proc = create_processor('unaligned')
    proc.fit(['vitaldb', 'mover', 'inspire'])
    proc.print_summary()
    for ds in ['vitaldb', 'mover', 'inspire']:
        try:
            X = proc.transform(ds)
            print(f'\n  {ds}: {X.shape}')
            print(f'  前10列: {list(X.columns[:10])}')
        except Exception as e:
            print(f'  {ds}: ERROR - {e}')
    print('\n' + '=' * 60)
    print(' 测试: 已对齐特征处理 (aligned)')
    print('=' * 60)
    proc2 = create_processor('aligned')
    proc2.fit(['vitaldb', 'mover', 'inspire'])
    proc2.print_summary()
    for ds in ['vitaldb', 'mover', 'inspire']:
        try:
            X = proc2.transform(ds)
            print(f'\n  {ds}: {X.shape}')
            print(f'  前10列: {list(X.columns[:10])}')
        except Exception as e:
            print(f'  {ds}: ERROR - {e}')

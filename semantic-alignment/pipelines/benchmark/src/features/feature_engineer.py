import os as _release_os
from release_runtime import expand_path as _release_expand, expand_config as _release_expand_config
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SNOMEDAlignmentAnalyzer:
    """
    SNOMED CT对齐分析器

    用于分析prerisk_bench（未对齐）和alignment（已对齐）数据之间的差异。

    **重要理解：**
    - SNOMED对齐的不是字段名，而是**字段值/描述性文本**
    - 字段名在两个版本中完全相同
    - 对齐将不同数据集的描述统一到SNOMED CT标准术语

    **被对齐的内容类型：**
    1. 诊断名称 (preop_diagnosis) → SNOMED疾病概念
       例: "Rectal cancer" → "(Fallopian tube Ca) or"
           "Unknown" → "(Atrial fibrillation) or (atrial flutter)"

    2. 病史详情 (hx_*_detail/details) → SNOMED临床发现
       例: "Hypertension" → "Hypertensive disorder, systemic arterial"
           "306 days ago diagnosed as (ICD-10-CM) I10" →
           "8 years ago diagnosed as (Acute myocardial infarction)"

    3. 手术/操作名称 (scheduled_procedure_name) → SNOMED操作过程
       例: "Low anterior resection" → "Colostomy route"
           "GI EGD DILATION" → "Cardiac ablation for atrial fibrillation"
           "0YQ50" (ICD-PCS编码) → "Injury of tympanic membrane"

    4. 麻醉类型 (scheduled_anesthesia_type) → SNOMED操作
       例: "General" → "General anesthesia"
           "General" (MOVER) → "Epidural route" (可能修正)

    5. 药品记录 (med_rec_summary) → SNOMED药物物质
       例: "ACETAMINOPHEN 500 MG OR TABS(Oral)" →
           "Product containing precisely paracetamol 500 milligram/1 each..."

    6. 入院类型 (surg_admission_chars) → 统一术语
       例: "Elective Surgery" → "Optional surgery"
           "Hospital Outpatient Surgery" → "Inpatient surgery care"

    7. 科室/部门 (surgical_department) → 标准化
       例: "General Surgery" → "Ophthalmology" (可能基于实际科室映射)

    **不被对齐的内容：**
    - 数值型特征 (age, bmi, *_val等)
    - 标志位 (*_flag: 0/1)
    - 基本人口统计学 (sex, race)
    """
    ALIGNMENT_TARGET_COLUMNS = ['preop_diagnosis', 'preop_diagnosis_cn', 'scheduled_procedure_name', 'scheduled_procedure_name_cn', 'scheduled_anesthesia_type', 'scheduled_surgical_approach', 'scheduled_operative_type', 'scheduled_surgical_position', 'surg_admission_chars', 'surgical_department', 'hx_circ_detail', 'hx_circ_details', 'hx_circ_detail_cn', 'hx_endo_detail', 'hx_endo_details', 'hx_dige_detail', 'hx_dige_details', 'hx_immu_detail', 'hx_neuro_detail', 'hx_urine_detail', 'hx_spine_detail', 'med_rec_summary', 'anes_vars_summary', 'surg_vars_summary', 'past_his_summary']

    def __init__(self):
        self.alignment_examples = {}

    def compare_values(self, df_unaligned: pd.DataFrame, df_aligned: pd.DataFrame, column: str, n_samples: int=5) -> Dict:
        """
        对比同一列在未对齐和已对齐数据中的值

        Args:
            df_unaligned: 未对齐DataFrame
            df_aligned: 已对齐DataFrame
            column: 要对比的列名
            n_samples: 显示的样本数

        Returns:
            包含对比结果的字典
        """
        if column not in df_unaligned.columns or column not in df_aligned.columns:
            return {'column': column, 'error': '列不存在'}
        vals_un = df_unaligned[column].dropna().head(n_samples).tolist()
        vals_al = df_aligned[column].dropna().head(n_samples).tolist()
        differences = []
        for i in range(min(len(vals_un), len(vals_al))):
            if str(vals_un[i]) != str(vals_al[i]):
                differences.append({'index': i, 'unaligned': str(vals_un[i])[:100], 'aligned': str(vals_al[i])[:100]})
        result = {'column': column, 'unaligned_samples': [str(v)[:80] for v in vals_un[:n_samples]], 'aligned_samples': [str(v)[:80] for v in vals_al[:n_samples]], 'differences_count': len(differences), 'is_aligned': len(differences) > 0, 'examples': differences[:3]}
        if result['is_aligned']:
            self.alignment_examples[column] = result
            logger.info(f'[SNOMED] 发现对齐: {column} ({len(differences)} 处差异)')
        return result

    def identify_aligned_columns(self, dataset_name: str, data_loader_unaligned, data_loader_aligned) -> Dict:
        """
        识别特定数据集中哪些列被SNOMED对齐了

        Args:
            dataset_name: 数据集名称
            data_loader_unaligned: 未对齐数据的DataLoader
            data_loader_aligned: 已对齐数据的DataLoader

        Returns:
            对齐分析结果字典
        """
        feat_types = ['feat_pat_demo', 'feat_anes_vars', 'feat_surg_vars', 'feat_test_vars', 'feat_past_his', 'feat_med_rec']
        aligned_columns = []
        unchanged_columns = []
        only_in_unaligned = []
        only_in_aligned = []
        for feat_type in feat_types:
            try:
                df_un = data_loader_unaligned.load_features(feat_type)
                df_al = data_loader_aligned.load_features(feat_type)
                if df_un.empty or df_al.empty:
                    continue
                common_cols = set(df_un.columns) & set(df_al.columns)
                non_id_cols = [c for c in common_cols if c not in ['surgery_id', 'patient_id'] and '_summary' not in c.lower()]
                for col in non_id_cols[:10]:
                    comparison = self.compare_values(df_un, df_al, col, n_samples=3)
                    if comparison['is_aligned']:
                        aligned_columns.append(col)
                    else:
                        unchanged_columns.append(col)
            except Exception as e:
                logger.warning(f'分析 {feat_type} 失败: {e}')
                continue
        result = {'dataset': dataset_name, 'total_checked': len(aligned_columns) + len(unchanged_columns), 'aligned_columns': aligned_columns, 'unchanged_columns': unchanged_columns, 'alignment_rate': len(aligned_columns) / max(1, len(aligned_columns) + len(unchanged_columns))}
        logger.info(f'\n[SNOMED] {dataset_name} 对齐分析:')
        logger.info(f"  总检查列数: {result['total_checked']}")
        logger.info(f"  已对齐列数: {len(result['aligned_columns'])} ({result['alignment_rate'] * 100:.1f}%)")
        logger.info(f"  未变化列数: {len(result['unchanged_columns'])}")
        logger.info(f"  主要对齐列: {result['aligned_columns'][:10]}")
        return result

class FeatureExtractor:
    """基础特征提取器"""

    def __init__(self):
        self.feature_names = []

    def extract_features(self, features_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        合并所有特征文件

        Args:
            features_dict: 特征类型到DataFrame的映射

        Returns:
            合并后的特征DataFrame
        """
        merged = None
        for feat_name, df in features_dict.items():
            if merged is None:
                merged = df.copy()
            else:
                on_cols = ['surgery_id', 'patient_id']
                available_on_cols = [col for col in on_cols if col in merged.columns and col in df.columns]
                if available_on_cols:
                    merged = merged.merge(df, on=available_on_cols, how='inner', suffixes=('', f'_{feat_name}'))
        id_cols = ['surgery_id', 'patient_id']
        summary_cols = [col for col in merged.columns if '_summary' in col.lower()]
        cols_to_drop = [col for col in id_cols + summary_cols if col in merged.columns]
        feature_df = merged.drop(columns=cols_to_drop, errors='ignore')
        self.feature_names = feature_df.columns.tolist()
        logger.info(f'特征提取完成: {len(self.feature_names)} 个特征')
        return feature_df

class LiteralMatchFeatureExtractor(FeatureExtractor):
    """
    字面匹配特征提取器

    用于未对齐数据 (prerisk_bench) 的联合训练，
    仅保留多个数据集间字段名完全相同的特征
    """

    def __init__(self):
        super().__init__()
        self.common_fields: Set[str] = set()

    def find_common_fields(self, datasets_features: Dict[str, Dict[str, pd.DataFrame]]) -> List[str]:
        """
        找出所有数据集共有的字段名

        Args:
            datasets_features: 数据集名 -> (特征类型->DataFrame) 的嵌套字典

        Returns:
            共有字段名列表
        """
        all_field_sets = []
        for dataset_name, features_dict in datasets_features.items():
            dataset_fields = set()
            temp_merged = None
            for feat_type, df in features_dict.items():
                if temp_merged is None:
                    temp_merged = df.copy()
                else:
                    on_cols = ['surgery_id', 'patient_id']
                    available_on_cols = [col for col in on_cols if col in temp_merged.columns and col in df.columns]
                    if available_on_cols:
                        temp_merged = temp_merged.merge(df, on=available_on_cols, how='inner')
            if temp_merged is not None:
                exclude_fields = {'surgery_id', 'patient_id'}
                dataset_fields = set(temp_merged.columns) - exclude_fields
                dataset_fields = {col for col in dataset_fields if '_summary' not in col.lower()}
            all_field_sets.append(dataset_fields)
            logger.info(f'{dataset_name} 特征数: {len(dataset_fields)}')
        if all_field_sets:
            self.common_fields = set.intersection(*all_field_sets)
        else:
            self.common_fields = set()
        logger.info(f'字面匹配 - 共有字段数: {len(self.common_fields)}')
        logger.info(f'共有字段示例: {list(self.common_fields)[:10]}...')
        return list(self.common_fields)

    def find_common_numeric_fields(self, datasets_features: Dict[str, Dict[str, pd.DataFrame]]) -> List[str]:
        """
        找出所有数据集共有的数值型/布尔型字段名

        用于Exp1(未对齐联合训练): 仅保留可跨数据集直接比较的数值和布尔特征，
        排除文本字段(诊断名称、手术描述等)，避免语义不匹配问题

        Args:
            datasets_features: 数据集名 -> (特征类型->DataFrame) 的嵌套字典

        Returns:
            共有数值/布尔字段名列表
        """
        all_field_sets = []
        for dataset_name, features_dict in datasets_features.items():
            dataset_fields = set()
            temp_merged = None
            for feat_type, df in features_dict.items():
                if temp_merged is None:
                    temp_merged = df.copy()
                else:
                    on_cols = ['surgery_id', 'patient_id']
                    available_on_cols = [col for col in on_cols if col in temp_merged.columns and col in df.columns]
                    if available_on_cols:
                        temp_merged = temp_merged.merge(df, on=available_on_cols, how='inner')
            if temp_merged is not None:
                exclude_fields = {'surgery_id', 'patient_id'}
                all_cols = set(temp_merged.columns) - exclude_fields
                all_cols = {col for col in all_cols if '_summary' not in col.lower()}
                numeric_bool_fields = set()
                for col in all_cols:
                    dtype = temp_merged[col].dtype
                    if pd.api.types.is_numeric_dtype(dtype):
                        unique_vals = temp_merged[col].dropna().unique()
                        if len(unique_vals) <= 2:
                            numeric_bool_fields.add(col)
                        else:
                            numeric_bool_fields.add(col)
                dataset_fields = numeric_bool_fields
            all_field_sets.append(dataset_fields)
            n_numeric = len(dataset_fields)
            n_total_all = len(set(temp_merged.columns) - {'surgery_id', 'patient_id'}) if temp_merged is not None else 0
            logger.info(f'{dataset_name} 数值/布尔特征: {n_numeric}/{n_total_all}')
        if all_field_sets:
            common_numeric = set.intersection(*all_field_sets)
        else:
            common_numeric = set()
        self.common_fields = common_numeric
        logger.info(f'[EXP1-数值模式] 共有数值/布尔字段数: {len(common_numeric)}')
        logger.info(f'共有数值字段: {sorted(list(common_numeric))[:20]}...')
        return list(common_numeric)

    def extract_features(self, features_dict: Dict[str, pd.DataFrame], common_fields: Optional[List[str]]=None) -> pd.DataFrame:
        """
        提取仅包含共有字段的特征

        Args:
            features_dict: 特征字典
            common_fields: 共有字段列表（如果为None则使用之前找到的）

        Returns:
            仅包含共有字段的DataFrame
        """
        merged = super().extract_features(features_dict)
        if common_fields is None:
            common_fields = list(self.common_fields)
        available_fields = [f for f in common_fields if f in merged.columns]
        if available_fields:
            feature_df = merged[available_fields]
        else:
            logger.warning('没有可用的共有字段，返回全部特征')
            feature_df = merged
        self.feature_names = feature_df.columns.tolist()
        logger.info(f'字面匹配提取: {len(self.feature_names)} 个特征')
        return feature_df

class OneHotEncoder:
    """
    独热编码器 (支持两种模式)

    mode='standard': 常规独热编码，基于原始文本值
    mode='snomed': SNOMED概念ID编码，将文本值映射为concept_id后再编码

    SNOMED模式工作流:
    1. 加载 snomed_fsn_to_conceptid.csv 映射表
    2. 将列中的文本值(如"Blepharitis...")替换为对应的concept_id
    3. 对concept_id进行独热编码
    4. 无法匹配的值归入"unknown"类别
    """

    def __init__(self, max_categories: int=20, drop_first: bool=True, mode: str='standard', snomed_mapping_path: Optional[str]=None):
        self.max_categories = max_categories
        self.drop_first = drop_first
        self.mode = mode
        self.snomed_mapping_path = snomed_mapping_path
        self.encoders = {}
        self.categorical_columns = []
        self.snomed_map: Dict[str, str] = {}
        self._snomed_loaded = False

    def _load_snomed_mapping(self):
        """加载SNOMED FSN -> ConceptID 映射表"""
        if self._snomed_loaded:
            return
        if self.snomed_mapping_path is None:
            default_paths = ['dataset/snomed_mapping/snomed_fsn_to_conceptid.csv', '../dataset/snomed_mapping/snomed_fsn_to_conceptid.csv']
            for p in default_paths:
                if Path(p).exists():
                    self.snomed_mapping_path = p
                    break
        if self.snomed_mapping_path and Path(self.snomed_mapping_path).exists():
            df = pd.read_csv(self.snomed_mapping_path)
            df['fsn_clean'] = df['fsn'].astype(str).str.strip()
            df['conceptId_str'] = df['conceptId'].astype(str).str.strip()
            self.snomed_map = dict(zip(df['fsn_clean'], df['conceptId_str']))
            self._snomed_loaded = True
            logger.info(f'加载SNOMED映射: {len(self.snomed_map)} 条记录')
        else:
            logger.warning(f'SNOMED映射文件未找到: {self.snomed_mapping_path}, 回退到标准模式')
            self.mode = 'standard'
            self._snomed_loaded = True

    def _map_to_snomed_id(self, value) -> str:
        """将文本值映射为SNOMED concept ID"""
        if pd.isna(value):
            return '__missing__'
        value_str = str(value).strip()
        if value_str in self.snomed_map:
            return f'sid_{self.snomed_map[value_str]}'
        for fsn_key, cid in self.snomed_map.items():
            if value_str in fsn_key or fsn_key in value_str:
                return f'sid_{cid}'
        return '__unknown__'

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        对分类变量进行编码

        Args:
            X: 输入特征DataFrame

        Returns:
            编码后的DataFrame
        """
        if self.mode == 'snomed':
            self._load_snomed_mapping()
        X_encoded = X.copy()
        self.categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
        for col in self.categorical_columns:
            if col not in X_encoded.columns:
                continue
            if self.mode == 'snomed':
                X_encoded[f'{col}_snomid'] = X_encoded[col].apply(self._map_to_snomed_id)
                encode_col = f'{col}_snomid'
                prefix = f'{col}_sid'
            else:
                encode_col = col
                prefix = col
            n_unique = X_encoded[encode_col].nunique(dropna=True)
            if n_unique <= self.max_categories:
                dummies = pd.get_dummies(X_encoded[encode_col], prefix=prefix, drop_first=self.drop_first, dummy_na=False)
                X_encoded = pd.concat([X_encoded.drop([encode_col, col], axis=1, errors='ignore'), dummies], axis=1)
                self.encoders[col] = {'mode': self.mode, 'n_categories': n_unique, 'dummy_columns': dummies.columns.tolist(), 'encode_column': encode_col}
                mode_tag = '[SNOMED]' if self.mode == 'snomed' else ''
                logger.info(f'独热编码{mode_tag}: {col} ({n_unique} 类) -> {dummies.shape[1]} 列')
            else:
                X_encoded = X_encoded.drop([encode_col, col], axis=1, errors='ignore')
                mode_tag = '[SNOMED]' if self.mode == 'snomed' else ''
                logger.warning(f'{mode_tag}删除高基数列: {col} (唯一值={n_unique})')
        X_encoded = X_encoded.astype(float)
        logger.info(f'编码完成({self.mode}): {X.shape[1]} -> {X_encoded.shape[1]} 维度')
        return X_encoded

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        使用已拟合的编码器转换新数据

        Args:
            X: 输入特征DataFrame

        Returns:
            编码后的DataFrame
        """
        if self.mode == 'snomed':
            self._load_snomed_mapping()
        X_encoded = X.copy()
        for col in self.categorical_columns:
            if col not in X_encoded.columns:
                continue
            if col not in self.encoders:
                X_encoded = X_encoded.drop(col, axis=1, errors='ignore')
                continue
            encoder_info = self.encoders[col]
            encode_col = encoder_info.get('encode_column', col)
            if encoder_info.get('mode') == 'snomed' and encode_col not in X_encoded.columns:
                X_encoded[encode_col] = X_encoded[col].apply(self._map_to_snomed_id)
            actual_col = encode_col if encode_col in X_encoded.columns else col
            if actual_col not in X_encoded.columns:
                continue
            dummies = pd.get_dummies(X_encoded[actual_col], prefix=encoder_info.get('dummy_columns')[0].rsplit('_', 1)[0] if '_' in encoder_info.get('dummy_columns', [''])[0] else actual_col, drop_first=self.drop_first)
            expected_cols = encoder_info['dummy_columns']
            for expected_col in expected_cols:
                if expected_col not in dummies.columns:
                    dummies[expected_col] = 0
            dummies = dummies[expected_cols]
            X_encoded = pd.concat([X_encoded.drop([actual_col, col], axis=1, errors='ignore'), dummies], axis=1)
        X_encoded = X_encoded.astype(float)
        return X_encoded

class BERTFeatureEmbedder:
    """
    ClinicalBERT特征嵌入器

    将特征名称通过ClinicalBERT转换为语义向量，
    用于隐式对齐或增强语义表示
    """

    def __init__(self, model_name: str='emilyalsentzer/Bio_ClinicalBERT'):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.embedding_cache = {}
        self.device = None
        self._initialized = False

    def initialize_model(self):
        """初始化ClinicalBERT模型"""
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
            logger.info(f'加载ClinicalBERT模型: {self.model_name}')
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            self.model.eval()
            self._initialized = True
            logger.info(f'ClinicalBERT加载成功, 设备: {self.device}')
        except Exception as e:
            logger.error(f'ClinicalBERT模型加载失败: {e}')
            raise

    def get_embedding(self, text: str) -> np.ndarray:
        """
        获取文本的BERT嵌入向量

        Args:
            text: 输入文本

        Returns:
            嵌入向量 (768维)
        """
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        if not self._initialized:
            self.initialize_model()
        import torch
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
        self.embedding_cache[text] = embedding
        return embedding

    def embed_feature_names(self, feature_names: List[str]) -> np.ndarray:
        """
        将特征名列表转换为BERT嵌入矩阵

        Args:
            feature_names: 特征名列表

        Returns:
            嵌入矩阵 (n_features, 768)
        """
        if not self._initialized:
            self.initialize_model()
        embeddings = []
        for feat_name in feature_names:
            text = feat_name.replace('_', ' ')
            try:
                emb = self.get_embedding(text)
                embeddings.append(emb)
            except Exception as e:
                logger.warning(f'嵌入生成失败: {feat_name}, 错误: {e}')
                embeddings.append(np.zeros(768))
        embedding_matrix = np.array(embeddings)
        logger.info(f'特征名嵌入矩阵形状: {embedding_matrix.shape}')
        return embedding_matrix

def create_feature_extractor(method: str) -> FeatureExtractor:
    """
    工厂函数：创建特征提取器

    Args:
        method: 提取方法 ('raw' 或 'literal_match')

    Returns:
        特征提取器实例
    """
    extractors = {'raw': FeatureExtractor(), 'literal_match': LiteralMatchFeatureExtractor()}
    if method not in extractors:
        raise ValueError(f'不支持的特征提取方法: {method}. 支持: {list(extractors.keys())}')
    return extractors[method]
if __name__ == '__main__':
    from data.dataset import MultiDatasetLoader
    print('\n' + '=' * 70)
    print(' [TEST] Feature Engineering Test')
    print('=' * 70)
    loader = MultiDatasetLoader('dataset')
    print('\n--- 测试原始特征提取 ---')
    vitaldb_features_unaligned = loader.get_loader('vitaldb', 'unaligned').load_all_features()
    raw_extractor = FeatureExtractor()
    raw_features = raw_extractor.extract_features(vitaldb_features_unaligned)
    print(f'VitalDB (未对齐) 原始特征数: {raw_features.shape[1]}')
    print(f'示例特征: {list(raw_features.columns[:5])}')
    print('\n--- 测试字面匹配 ---')
    mover_features_unaligned = loader.get_loader('mover', 'unaligned').load_all_features()
    inspire_features_unaligned = loader.get_loader('inspire', 'unaligned').load_all_features()
    datasets_features = {'vitaldb': vitaldb_features_unaligned, 'mover': mover_features_unaligned, 'inspire': inspire_features_unaligned}
    literal_extractor = LiteralMatchFeatureExtractor()
    common_fields = literal_extractor.find_common_fields(datasets_features)
    print(f'\n三个数据集的共有字段数: {len(common_fields)}')
    print(f'共有字段: {common_fields[:15]}...')
    print('\n--- 测试独热编码 ---')
    vitaldb_aligned_features = loader.get_loader('vitaldb', 'aligned').load_all_features()
    aligned_extractor = FeatureExtractor()
    aligned_features = aligned_extractor.extract_features(vitaldb_aligned_features)
    encoder = OneHotEncoder(max_categories=20)
    encoded_features = encoder.fit_transform(aligned_features)
    print(f'\nVitalDB (已对齐) 独热编码:')
    print(f'  编码前: {aligned_features.shape[1]} 维')
    print(f'  编码后: {encoded_features.shape[1]} 维')

import os as _release_os
from release_runtime import expand_path as _release_expand, expand_config as _release_expand_config
import os
import pandas as pd
from pathlib import Path
import json
import re

def load_snomed_data(description_file: str, output_dir: str=None):
    """
    加载SNOMED CT数据并设置输出目录

    参数:
        description_file: SNOMED CT Description文件路径
        output_dir: 输出目录

    返回:
        tuple: (desc_df, filtered_df, output_dir)
    """
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(root_dir), 'data', 'processed', 'snomed')
    os.makedirs(output_dir, exist_ok=True)
    print(f'输入文件: {description_file}')
    print(f'输出目录: {output_dir}')
    print('\n正在加载Description数据...')
    try:
        desc_df = pd.read_csv(description_file, sep='\t', dtype=str, low_memory=False)
        print(f'成功加载数据，总行数: {len(desc_df):,}')
    except Exception as e:
        print(f'加载文件失败: {e}')
        return (None, None, None)
    print('\n过滤活跃的英语FSN概念...')
    filtered_df = desc_df[(desc_df['active'] == '1') & (desc_df['languageCode'] == 'en') & (desc_df['typeId'] == '900000000000003001')].copy()
    print(f'活跃英语FSN概念数量: {len(filtered_df):,}')
    return (desc_df, filtered_df, output_dir)

def extract_fsn_suffixes(description_file: str, output_dir: str=None):
    """
    提取FSN类型术语后缀（括号内的内容）的唯一值

    参数:
        description_file: SNOMED CT Description文件路径
        output_dir: 输出目录，默认为当前目录下的output文件夹
    """
    print('SNOMED CT FSN术语后缀提取工具')
    print('=' * 50)
    desc_df, filtered_df, output_dir = load_snomed_data(description_file, output_dir)
    if desc_df is None:
        return
    print('\n提取FSN术语后缀...')
    suffixes = []
    for term in filtered_df['term']:
        matches = re.findall('\\((.*?)\\)', term)
        suffixes.extend(matches)
    unique_suffixes = sorted(set(suffixes))
    print(f'找到的唯一后缀数量: {len(unique_suffixes):,}')
    if len(unique_suffixes) == 0:
        print('未找到包含括号的FSN术语')
        return
    result_df = pd.DataFrame({'suffix': unique_suffixes, 'count': [suffixes.count(suffix) for suffix in unique_suffixes]})
    result_df = result_df.sort_values('count', ascending=False)
    csv_file = os.path.join(output_dir, 'snomed_fsn_suffixes.csv')
    result_df.to_csv(csv_file, index=False, encoding='utf-8')
    print(f'CSV文件已保存: {csv_file}')
    print('\n' + '=' * 50)
    print('统计信息:')
    print(f'总唯一后缀数量: {len(result_df):,}')
    print(f"总后缀出现次数: {sum(result_df['count']):,}")
    print(f"平均每个后缀出现次数: {sum(result_df['count']) / len(result_df):.1f}")
    print('\n前10个最常出现的后缀:')
    for i, (_, row) in enumerate(result_df.head(10).iterrows()):
        print(f"{i + 1:2d}. {row['suffix']} (出现次数: {row['count']:,})")
    print('\n' + '=' * 50)
    print('FSN术语后缀提取完成!')

def extract_attribute_concepts(description_file: str, output_dir: str=None):
    """
    提取以(attribute)结尾的关系概念

    参数:
        description_file: SNOMED CT Description文件路径
        output_dir: 输出目录，默认为当前目录下的output文件夹
    """
    print('SNOMED CT关系概念提取工具')
    print('=' * 50)
    desc_df, filtered_df, output_dir = load_snomed_data(description_file, output_dir)
    if desc_df is None:
        return
    print('\n提取以(attribute)结尾的关系概念...')
    attribute_df = filtered_df[filtered_df['term'].str.endswith('(attribute)', na=False)].copy()
    print(f'找到的关系概念数量: {len(attribute_df):,}')
    if len(attribute_df) == 0:
        print('未找到以(attribute)结尾的概念')
        return
    attribute_df['term_clean'] = attribute_df['term'].str.replace('\\s*\\(attribute\\)\\s*$', '', regex=True)
    result_df = attribute_df[['conceptId', 'term', 'term_clean']].copy()
    result_df = result_df.sort_values('conceptId')
    csv_file = os.path.join(output_dir, 'snomed_attribute_concepts.csv')
    result_df.to_csv(csv_file, index=False, encoding='utf-8')
    print(f'CSV文件已保存: {csv_file}')
    print('\n' + '=' * 50)
    print('统计信息:')
    print(f'总关系概念数量: {len(result_df):,}')
    print(f"概念ID范围: {result_df['conceptId'].min()} - {result_df['conceptId'].max()}")
    print('\n前10个关系概念示例:')
    for i, (_, row) in enumerate(result_df.head(10).iterrows()):
        print(f"{i + 1:2d}. {row['conceptId']}: {row['term_clean']}")
    print('\n' + '=' * 50)
    print('关系概念提取完成!')

def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description='SNOMED CT数据处理工具')
    parser.add_argument('--mode', choices=['attribute', 'suffix', 'all'], default='all', help='处理模式: attribute(仅提取关系概念), suffix(仅提取后缀), all(全部处理)')
    parser.add_argument('--description-file', help='SNOMED CT Description文件路径')
    parser.add_argument('--output-dir', help='输出目录')
    args = parser.parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    if args.description_file:
        description_file = args.description_file
    else:
        description_file = os.path.join(project_root, 'data', 'raw', 'snomed_ct', 'sct2_Description_Snapshot-en_INT_20260201.txt')
    if not os.path.exists(description_file):
        print(f'错误: Description文件不存在: {description_file}')
        print('请确保SNOMED CT数据文件已正确放置')
        return
    if args.mode in ['attribute', 'all']:
        print('\n' + '=' * 60)
        print('开始提取关系概念...')
        extract_attribute_concepts(description_file, args.output_dir)
    if args.mode in ['suffix', 'all']:
        print('\n' + '=' * 60)
        print('开始提取FSN术语后缀...')
        extract_fsn_suffixes(description_file, args.output_dir)
    print('\n' + '=' * 60)
    print('所有处理完成!')
if __name__ == '__main__':
    main()

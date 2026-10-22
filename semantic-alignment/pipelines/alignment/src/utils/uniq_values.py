import os as _release_os
from release_runtime import expand_path as _release_expand, expand_config as _release_expand_config
import os
import yaml
import psycopg2
import pandas as pd
import json
from typing import Dict, List, Any
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
config_path = os.path.join(project_root, 'config', 'config.yaml')
with open(config_path, 'r', encoding='utf-8') as f:
    config = _release_expand_config(yaml.safe_load(f))
databases_config = config.get('databases', {})
TARGET_TABLES = ['feat_anes_vars', 'feat_med_rec', 'feat_past_his', 'feat_pat_demo', 'feat_surg_vars', 'feat_test_vars']
EXCLUDED_FIELDS = ['surgery_id', 'patient_id']

def get_db_connection(db_config: Dict[str, Any]) -> psycopg2.extensions.connection:
    """建立数据库连接"""
    try:
        conn = psycopg2.connect(host=db_config['host'], port=db_config['port'], database=db_config['database'], user=db_config['username'], password=db_config.get('password', ''))
        return conn
    except Exception as e:
        print(f"连接数据库 {db_config['name']} 失败: {e}")
        return None

def get_table_columns(conn: psycopg2.extensions.connection, schema: str, table: str) -> List[str]:
    """获取表的列名"""
    try:
        cursor = conn.cursor()
        cursor.execute('\n            SELECT column_name \n            FROM information_schema.columns \n            WHERE table_schema = %s AND table_name = %s\n            ORDER BY ordinal_position\n        ', (schema, table))
        columns = [row[0] for row in cursor.fetchall()]
        cursor.close()
        return columns
    except Exception as e:
        print(f'获取表 {schema}.{table} 的列名失败: {e}')
        return []

def get_unique_values(conn: psycopg2.extensions.connection, schema: str, table: str, column: str) -> List[str]:
    """获取指定字段的唯一值"""
    try:
        cursor = conn.cursor()
        cursor.execute(f'\n            SELECT DISTINCT {column} \n            FROM {schema}.{table} \n            WHERE {column} IS NOT NULL\n            ORDER BY {column}\n        ')
        values = [str(row[0]) if row[0] is not None else '' for row in cursor.fetchall()]
        cursor.close()
        return values
    except Exception as e:
        print(f'获取字段 {column} 的唯一值失败: {e}')
        return []

def process_database(database_name: str, db_config: Dict[str, Any]):
    """处理单个数据库的所有目标表"""
    print(f'正在处理数据库: {database_name}')
    conn = get_db_connection(db_config)
    if not conn:
        return
    schema = 'prerisk_bench'
    for table in TARGET_TABLES:
        print(f'  处理表: {table}')
        columns = get_table_columns(conn, schema, table)
        if not columns:
            print(f'    表 {schema}.{table} 不存在或无法访问')
            continue
        target_columns = [col for col in columns if col not in EXCLUDED_FIELDS]
        if not target_columns:
            print(f'    表 {schema}.{table} 没有需要处理的字段')
            continue
        unique_values_dict = {}
        for column in target_columns:
            print(f'    处理字段: {column}')
            values = get_unique_values(conn, schema, table, column)
            unique_values_dict[column] = values
        if unique_values_dict:
            output_filename = f'{database_name}_{table}.json'
            save_to_json(database_name, table, unique_values_dict, output_filename)
            print(f'    已保存: {output_filename}')
    conn.close()

def save_to_json(database_name: str, table_name: str, unique_values_dict: Dict[str, List[str]], filename: str):
    """保存数据到JSON文件"""
    if not unique_values_dict:
        return
    output_dir = os.path.join(project_root, 'data', 'raw', 'unique_values')
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    json_data = {'dataset': database_name, 'table_name': table_name, 'unique_values': unique_values_dict}
    with open(filepath, 'w', encoding='utf-8') as jsonfile:
        json.dump(json_data, jsonfile, ensure_ascii=False, indent=2)
    print(f'    文件保存位置: {filepath}')

def main():
    """主函数"""
    print('开始处理数据库唯一值...')
    for db_name, db_config in databases_config.items():
        process_database(db_name, db_config)
    print('处理完成！')
if __name__ == '__main__':
    main()

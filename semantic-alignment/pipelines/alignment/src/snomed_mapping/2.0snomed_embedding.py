import os as _release_os
from release_runtime import expand_path as _release_expand, expand_config as _release_expand_config
import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import yaml
import argparse

def load_model_config():
    """从config.yaml加载模型配置"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    config_path = os.path.join(project_root, 'config', 'config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = _release_expand_config(yaml.safe_load(f))
    return config['models']['clinical_bert']

def generate_rag_embeddings(processed_dir: str, embedding_dir: str, rag_filename: str='snomed_rag_texts.json', index_filename: str='snomed_optimized_index.json'):
    """生成SNOMED CT RAG文本嵌入"""
    print('开始生成SNOMED CT RAG文本嵌入...')
    rag_texts_path = os.path.join(processed_dir, rag_filename)
    with open(rag_texts_path, 'r', encoding='utf-8') as f:
        rag_data = json.load(f)
    optimized_index_path = os.path.join(processed_dir, index_filename)
    with open(optimized_index_path, 'r', encoding='utf-8') as f:
        optimized_index = json.load(f)
    concept_to_fsn = optimized_index.get('concept_id_to_fsn', {})
    all_texts = []
    all_concept_ids = []
    all_term_types = []
    all_concept_types = []
    for item in rag_data:
        text = item.get('text', '')
        concept_id = item.get('conceptId', '')
        term_type = item.get('term_type', '')
        concept_type = item.get('concept_type', '')
        if text and concept_id:
            all_texts.append(text)
            all_concept_ids.append(concept_id)
            all_term_types.append(term_type)
            all_concept_types.append(concept_type)
    print(f'从RAG文本数据生成了 {len(all_texts)} 个嵌入项')
    print(f'覆盖 {len(set(all_concept_ids))} 个SNOMED概念')
    print(f'FSN映射: {len(concept_to_fsn)} 个概念')
    model_config = load_model_config()
    model_path = model_config['path']
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    full_model_path = os.path.join(project_root, model_path)
    print(f"正在加载模型: {model_config['name']}")
    print(f'模型路径: {full_model_path}')
    import torch
    if torch.cuda.is_available():
        device = 'cuda'
        print(f'✓ 检测到GPU: {torch.cuda.device_count()}个GPU可用')
        for i in range(torch.cuda.device_count()):
            print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
    else:
        device = 'cpu'
        print('⚠ 未检测到GPU，使用CPU模式')
    model = SentenceTransformer(full_model_path, device=device)
    print(f'✓ 模型加载成功 (设备: {device})')
    print(f'开始生成 {len(all_texts)} 个RAG文本的嵌入向量...')
    embeddings = model.encode(all_texts, batch_size=32, show_progress_bar=True)
    print(f'嵌入向量生成完成，形状: {embeddings.shape}')
    print('正在归一化嵌入向量...')
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    print(f'向量归一化完成')
    os.makedirs(embedding_dir, exist_ok=True)
    embeddings_path = os.path.join(embedding_dir, 'snomed_rag_embeddings.npy')
    np.save(embeddings_path, embeddings)
    print('正在构建FAISS索引...')
    dimension = embeddings.shape[1]
    nlist = min(100, len(embeddings) // 100)
    quantizer = faiss.IndexFlatIP(dimension)
    index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
    index.train(embeddings)
    index.add(embeddings)
    print(f'✓ 索引构建完成，包含 {index.ntotal} 个向量')
    index_path = os.path.join(embedding_dir, 'snomed_rag_faiss_index.index')
    faiss.write_index(index, index_path)
    print('✓ FAISS 索引已保存')
    mapping_path = os.path.join(embedding_dir, 'snomed_rag_embedding_mapping.pkl')
    with open(mapping_path, 'wb') as f:
        pickle.dump({'texts': all_texts, 'concept_ids': all_concept_ids, 'term_types': all_term_types, 'concept_types': all_concept_types, 'concept_to_fsn': concept_to_fsn}, f)
    info_path = os.path.join(embedding_dir, 'rag_embedding_info.json')
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump({'num_texts': len(all_texts), 'num_concepts': len(set(all_concept_ids)), 'embedding_dim': embeddings.shape[1], 'index_size': index.ntotal, 'model_name': model_config['name'], 'avg_text_length': sum((len(text) for text in all_texts)) / len(all_texts) if all_texts else 0, 'term_type_distribution': {'fsn': all_term_types.count('fsn'), 'synonym': all_term_types.count('synonym')}, 'concept_type_distribution': {t: all_concept_types.count(t) for t in set(all_concept_types)}}, f, indent=2)
    print(f'\nRAG嵌入数据已保存到: {embedding_dir}')
    print(f'  - 嵌入维度: {embeddings.shape[1]}')
    print(f'  - 索引大小: {index.ntotal} 个向量')
    print(f'  - 平均文本长度: {sum((len(text) for text in all_texts)) / len(all_texts):.0f} 字符')
    print(f'  - 文本数量: {len(all_texts)}')
    print(f'  - 覆盖概念数量: {len(set(all_concept_ids))}')
    print(f"  - FSN术语: {all_term_types.count('fsn')}")
    print(f"  - 同义词: {all_term_types.count('synonym')}")

def main():
    print('SNOMED CT RAG文本嵌入流程')
    print('=' * 60)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    parser = argparse.ArgumentParser(description='SNOMED CT RAG文本嵌入')
    parser.add_argument('--dataset', choices=['snomed', 'allconcepts'], default='snomed', help='选择数据集: snomed(筛选后的) 或 allconcepts(全概念)')
    args = parser.parse_args()
    if args.dataset == 'allconcepts':
        processed_dir = os.path.join(project_root, 'data', 'processed', 'snomed_allconcepts')
        embedding_dir = os.path.join(project_root, 'data', 'embeddings', 'snomed_allconcepts')
        rag_filename = 'snomed_allconcepts_rag_texts.json'
        index_filename = 'snomed_allconcepts_optimized_index.json'
        print(f'数据集: 全概念 (allconcepts)')
    else:
        processed_dir = os.path.join(project_root, 'data', 'processed', 'snomed')
        embedding_dir = os.path.join(project_root, 'data', 'embeddings', 'snomed')
        rag_filename = 'snomed_rag_texts.json'
        index_filename = 'snomed_optimized_index.json'
        print(f'数据集: 筛选概念 (snomed)')
    generate_rag_embeddings(processed_dir, embedding_dir, rag_filename=rag_filename, index_filename=index_filename)
    print('\n' + '=' * 60)
    print('SNOMED CT RAG文本嵌入流程完成!')
    print('=' * 60)
    print(f'RAG文本数据目录: {processed_dir}')
    print(f'嵌入数据目录: {embedding_dir}')
if __name__ == '__main__':
    main()

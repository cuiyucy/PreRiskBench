"""
图构建工具

基于患者特征相似度构建k-NN图，用于图神经网络训练。

构建策略:
- 使用余弦相似度或欧氏距离度量患者间相似度
- 为每个患者连接k个最相似邻居
- 返回PyTorch稀疏格式的邻接矩阵
"""
import os as _release_os
from release_runtime import expand_path as _release_expand, expand_config as _release_expand_config
import numpy as np
import pandas as pd
import torch
import logging
from typing import Optional, Tuple
from sklearn.neighbors import NearestNeighbors
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GraphBuilder:
    """
    从特征矩阵构建k-NN患者相似度图

    Args:
        k: 每个节点的邻居数
        metric: 距离度量 ('cosine' 或 'euclidean')
        self_loop: 是否添加自环
    """

    def __init__(self, k: int=15, metric: str='cosine', self_loop: bool=True):
        self.k = k
        self.metric = metric
        self.self_loop = self_loop

    def build_graph(self, X: np.ndarray, node_ids: Optional[np.ndarray]=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        构建k-NN图

        Args:
            X: 特征矩阵 (n_nodes, n_features)
            node_ids: 节点ID数组 (用于调试/日志)

        Returns:
            edge_index: (2, E) 的边索引张量
            edge_weight: (E,) 的边权重张量
        """
        n = X.shape[0]
        k_actual = min(self.k, n - 1)
        if k_actual < 1:
            logger.warning('样本数过少，无法构建k-NN图，至少需要2个样本')
            empty_edge = torch.zeros((2, 0), dtype=torch.long)
            return (empty_edge, torch.zeros(0))
        logger.info(f'  构建k-NN图: {n}个节点, k={k_actual}, metric={self.metric}')
        nn = NearestNeighbors(n_neighbors=k_actual + 1, metric=self.metric, algorithm='brute' if n < 10000 else 'auto', n_jobs=-1)
        nn.fit(X)
        distances, indices = nn.kneighbors(X)
        rows, cols, weights = ([], [], [])
        for i in range(n):
            for j_idx in range(1, k_actual + 1):
                j = indices[i, j_idx]
                if i < j:
                    rows.append(i)
                    cols.append(j)
                    if self.metric == 'cosine':
                        w = 1.0 - distances[i, j_idx]
                    else:
                        w = 1.0 / (1.0 + distances[i, j_idx])
                    weights.append(w)
        if self.self_loop:
            for i in range(n):
                rows.append(i)
                cols.append(i)
                weights.append(1.0)
        edge_index = torch.tensor([rows, cols], dtype=torch.long)
        edge_weight = torch.tensor(weights, dtype=torch.float)
        logger.info(f'  图构建完成: {edge_index.size(1)}条边 (含自环)')
        return (edge_index, edge_weight)

    def build_graph_from_df(self, X_df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """从DataFrame构建图"""
        X_np = X_df.values.astype(np.float32)
        node_ids = X_df.index.values if hasattr(X_df, 'index') else None
        return self.build_graph(X_np, node_ids)

class BatchedGraphBuilder:
    """
    批处理图构建器（用于大规模数据集分块构建）

    将大矩阵分块构建k-NN，减少内存占用。
    """

    def __init__(self, k: int=15, metric: str='cosine', batch_size: int=5000):
        self.k = k
        self.metric = metric
        self.batch_size = batch_size

    def build_graph(self, X: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        n = X.shape[0]
        k_actual = min(self.k, n - 1)
        logger.info(f'  批量构建k-NN图: {n}个节点, k={k_actual}, batch={self.batch_size}')
        rows, cols, weights = ([], [], [])
        nn = NearestNeighbors(n_neighbors=k_actual + 1, metric=self.metric, algorithm='brute' if n < 10000 else 'auto', n_jobs=-1)
        nn.fit(X)
        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            batch_X = X[start:end]
            distances, indices = nn.kneighbors(batch_X)
            for i_local, i_global in enumerate(range(start, end)):
                for j_idx in range(1, k_actual + 1):
                    j = indices[i_local, j_idx]
                    if i_global < j:
                        rows.append(i_global)
                        cols.append(j)
                        w = 1.0 - distances[i_local, j_idx] if self.metric == 'cosine' else 1.0 / (1.0 + distances[i_local, j_idx])
                        weights.append(w)
        for i in range(n):
            rows.append(i)
            cols.append(i)
            weights.append(1.0)
        edge_index = torch.tensor([rows, cols], dtype=torch.long)
        edge_weight = torch.tensor(weights, dtype=torch.float)
        logger.info(f'  图构建完成: {edge_index.size(1)}条边')
        return (edge_index, edge_weight)

"""
神经网络基线模型

包含:
- MLPModel: 多层感知机分类器
- GNNModel: 图卷积神经网络分类器（基于纯PyTorch实现）
- TabTransformerModel: 自注意力分类器（多头注意力，适合稀疏One-Hot特征）
- CNNModel: 1D卷积分类器（多尺度卷积核，适合高维表格特征）
"""
import os as _release_os
from release_runtime import expand_path as _release_expand, expand_config as _release_expand_config
import math
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import joblib
from typing import Dict, Optional, Tuple, Union
from pathlib import Path
from src.models.base_models import BaseModel
warnings.filterwarnings('ignore', message='Sparse invariant checks are implicitly disabled')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MLPNetwork(nn.Module):
    """多层感知机网络"""

    def __init__(self, input_dim: int, hidden_dims: list, num_classes: int, dropout: float=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, h_dim), nn.BatchNorm1d(h_dim), nn.ReLU(), nn.Dropout(dropout)])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class GCNLayer(nn.Module):
    """
    图卷积层（纯PyTorch实现）

    H' = σ(A_hat @ H @ W)
    其中 A_hat = D^{-1/2} A D^{-1/2} （对称归一化邻接矩阵）
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float=0.3):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_dim, out_dim))
        self.bias = nn.Parameter(torch.FloatTensor(out_dim))
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, adj_norm: torch.sparse.FloatTensor) -> torch.Tensor:
        x = self.dropout(x)
        support = torch.mm(x, self.weight)
        out = torch.sparse.mm(adj_norm, support)
        out = out + self.bias
        return F.relu(out)

class GNNNetwork(nn.Module):
    """
    简单图卷积网络（2层GCN）

    架构: GCN(ReLU) → Dropout → GCN → LogSoftmax
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout: float=0.3):
        super().__init__()
        self.conv1 = GCNLayer(input_dim, hidden_dim, dropout)
        self.conv2 = GCNLayer(hidden_dim, num_classes, dropout)

    def forward(self, x: torch.Tensor, adj_norm: torch.sparse.FloatTensor) -> torch.Tensor:
        x = self.conv1(x, adj_norm)
        x = self.conv2(x, adj_norm)
        return F.log_softmax(x, dim=1)

def normalize_adjacency(edge_index: torch.Tensor, n_nodes: int, edge_weight: Optional[torch.Tensor]=None) -> torch.sparse.FloatTensor:
    """
    对称归一化邻接矩阵: A_hat = D^{-1/2} A D^{-1/2}

    Args:
        edge_index: (2, E) 边索引
        n_nodes: 节点数
        edge_weight: (E,) 边权重, 默认为全1

    Returns:
        稀疏归一化邻接矩阵
    """
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1))
    adj = torch.sparse_coo_tensor(edge_index, edge_weight, (n_nodes, n_nodes)).coalesce()
    deg = torch.sparse.sum(adj, dim=1).to_dense()
    deg_inv_sqrt = torch.pow(deg.clamp(min=1e-08), -0.5)
    values = adj.values()
    rows = adj.indices()[0]
    cols = adj.indices()[1]
    norm_values = values * deg_inv_sqrt[rows] * deg_inv_sqrt[cols]
    adj_norm = torch.sparse_coo_tensor(adj.indices(), norm_values, (n_nodes, n_nodes)).coalesce()
    return adj_norm

class EarlyStopping:
    """早停机制"""

    def __init__(self, patience: int=20, min_delta: float=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.best_state = None

    def step(self, loss: float, model: nn.Module) -> bool:
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore(self, model: nn.Module):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)

def _prepare_val_loader(model_obj, X_val, y_val, batch_size):
    """构建验证集 DataLoader 用于早停

    过滤训练中未见过的标签（无法映射的样本不参与验证集 loss）。
    X_val/y_val 为 None 或全部标签不可映射时返回 None。
    """
    if X_val is None or y_val is None:
        return None
    X_val_np, y_val_np = model_obj._to_numpy(X_val, y_val)
    valid = np.array([v in model_obj._label_map for v in y_val_np])
    if not valid.any():
        return None
    Xv = torch.FloatTensor(X_val_np[valid])
    yv = torch.LongTensor(np.array([model_obj._label_map[v] for v in y_val_np[valid]]))
    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Xv, yv), batch_size=batch_size, shuffle=False, drop_last=False)

def _compute_val_loss(net, criterion, val_loader):
    """计算验证集平均 loss（网络先置 eval 后恢复 train）"""
    net.eval()
    total_loss, total_n = (0.0, 0)
    with torch.no_grad():
        for bX, by in val_loader:
            bX, by = (bX.to(DEVICE), by.to(DEVICE))
            logits = net(bX)
            total_loss += criterion(logits, by).item() * len(bX)
            total_n += len(bX)
    net.train()
    return total_loss / max(total_n, 1)

class MLPModel(BaseModel):
    """
    多层感知机分类器

    用法:
        model = MLPModel({'hidden_dims': [128, 64], 'lr': 0.001})
        model.train(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)
    """

    def __init__(self, params: Optional[Dict]=None):
        super().__init__('MLP')
        default_params = {'hidden_dims': [256, 128, 64], 'dropout': 0.3, 'lr': 0.001, 'weight_decay': 0.0001, 'batch_size': 128, 'max_epochs': 200, 'patience': 30, 'random_state': 42}
        self.params = {**default_params, **(params or {})}
        self._net = None
        self._num_classes = None
        self._input_dim = None
        self._label_map = None

    def _set_random_seed(self):
        torch.manual_seed(self.params['random_state'])
        np.random.seed(self.params['random_state'])

    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs) -> 'MLPModel':
        self._set_random_seed()
        X, y = self._to_numpy(X_train, y_train)
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)
        self._input_dim = X.shape[1]
        self._num_classes = len(np.unique(y))
        self._label_map = {v: i for i, v in enumerate(np.unique(y))}
        y_mapped = np.array([self._label_map[v] for v in y])
        y_tensor = torch.LongTensor(y_mapped)
        self._net = MLPNetwork(input_dim=self._input_dim, hidden_dims=self.params['hidden_dims'], num_classes=self._num_classes, dropout=self.params['dropout']).to(DEVICE)
        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.params['lr'], weight_decay=self.params['weight_decay'])
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        stopper = EarlyStopping(patience=self.params['patience'])
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.params['batch_size'], shuffle=True, drop_last=True)
        val_loader = _prepare_val_loader(self, X_val, y_val, self.params['batch_size'])
        for epoch in range(self.params['max_epochs']):
            self._net.train()
            epoch_loss = 0.0
            for batch_X, batch_y in loader:
                batch_X, batch_y = (batch_X.to(DEVICE), batch_y.to(DEVICE))
                optimizer.zero_grad()
                logits = self._net(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._net.parameters(), 5.0)
                optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(loader)
            monitor_loss = _compute_val_loss(self._net, criterion, val_loader) if val_loader is not None else avg_loss
            scheduler.step(monitor_loss)
            if (epoch + 1) % 50 == 0:
                logger.info(f"  [MLP] Epoch {epoch + 1}/{self.params['max_epochs']}, Loss: {avg_loss:.4f}, ValLoss: {monitor_loss:.4f}")
            if stopper.step(monitor_loss, self._net):
                logger.info(f'  [MLP] 早停于 epoch {epoch + 1}')
                break
        stopper.restore(self._net)
        self.is_fitted = True
        logger.info(f'[MLP] 训练完成: {self._input_dim}维 → {self._num_classes}类')
        return self

    def predict(self, X) -> np.ndarray:
        if not self.is_fitted or self._net is None:
            raise ValueError('MLP模型尚未训练')
        X_np = self._to_numpy(X, None)[0]
        self._net.eval()
        with torch.no_grad():
            logits = self._net(torch.FloatTensor(X_np).to(DEVICE))
            preds = logits.argmax(dim=1).cpu().numpy()
        inv_map = {v: k for k, v in self._label_map.items()}
        return np.array([inv_map[p] for p in preds])

    def predict_proba(self, X) -> np.ndarray:
        if not self.is_fitted or self._net is None:
            raise ValueError('MLP模型尚未训练')
        X_np = self._to_numpy(X, None)[0]
        self._net.eval()
        with torch.no_grad():
            logits = self._net(torch.FloatTensor(X_np).to(DEVICE))
            probs = F.softmax(logits, dim=1).cpu().numpy()
        return probs

    def _to_numpy(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            X = X.values.astype(np.float32)
        elif isinstance(X, pd.Series):
            X = X.values.astype(np.float32)
        elif isinstance(X, (list,)):
            X = np.array(X, dtype=np.float32)
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        if y is not None:
            if isinstance(y, (pd.Series, pd.DataFrame)):
                y = y.values.ravel()
            elif isinstance(y, (list,)):
                y = np.array(y)
            elif isinstance(y, torch.Tensor):
                y = y.cpu().numpy()
            return (X, y)
        return (X,)

    def save_model(self, path: str) -> None:
        data = {'model_state': self._net.state_dict() if self._net else None, 'params': self.params, 'num_classes': self._num_classes, 'input_dim': self._input_dim, 'label_map': self._label_map, 'model_name': self.model_name, 'is_fitted': self.is_fitted}
        joblib.dump(data, path)
        logger.info(f'MLP模型已保存: {path}')

    def load_model(self, path: str) -> 'MLPModel':
        data = joblib.load(path)
        self.params = data['params']
        self._num_classes = data['num_classes']
        self._input_dim = data['input_dim']
        self._label_map = data['label_map']
        self.model_name = data['model_name']
        self.is_fitted = data['is_fitted']
        if data['model_state'] and self.is_fitted:
            self._net = MLPNetwork(input_dim=self._input_dim, hidden_dims=self.params['hidden_dims'], num_classes=self._num_classes, dropout=self.params['dropout']).to(DEVICE)
            self._net.load_state_dict(data['model_state'])
        logger.info(f'MLP模型已加载: {path}')
        return self

class GNNModel(BaseModel):
    """
    图卷积神经网络分类器

    GNN在BaseModel接口基础上增加图结构需求。

    train() 会从特征矩阵构建k-NN图：
        model.train(X_train, y_train)

    或外部传入预构建图：
        model.train(X_train, y_train, edge_index=edge_index, edge_weight=edge_weight)
    """

    def __init__(self, params: Optional[Dict]=None):
        super().__init__('GNN')
        default_params = {'hidden_dim': 128, 'dropout': 0.3, 'lr': 0.005, 'weight_decay': 0.0005, 'max_epochs': 300, 'patience': 40, 'k_neighbors': 15, 'graph_metric': 'cosine', 'random_state': 42}
        self.params = {**default_params, **(params or {})}
        self._net = None
        self._num_classes = None
        self._input_dim = None
        self._label_map = None
        self._adj_norm = None

    def _set_random_seed(self):
        torch.manual_seed(self.params['random_state'])
        np.random.seed(self.params['random_state'])

    def _build_graph(self, X: np.ndarray) -> torch.sparse.FloatTensor:
        """从特征矩阵构建归一化邻接矩阵"""
        from src.models.graph_builder import GraphBuilder
        n = X.shape[0]
        builder = GraphBuilder(k=self.params['k_neighbors'], metric=self.params['graph_metric'], self_loop=True)
        edge_index, edge_weight = builder.build_graph(X)
        adj_norm = normalize_adjacency(edge_index, n, edge_weight).to(DEVICE)
        return adj_norm

    def train(self, X_train, y_train, edge_index: Optional[torch.Tensor]=None, edge_weight: Optional[torch.Tensor]=None, **kwargs) -> 'GNNModel':
        self._set_random_seed()
        X, y = self._to_numpy(X_train, y_train)
        X_tensor = torch.FloatTensor(X).to(DEVICE)
        self._input_dim = X.shape[1]
        unique_classes = np.unique(y)
        self._num_classes = len(unique_classes)
        self._label_map = {v: i for i, v in enumerate(unique_classes)}
        y_mapped = np.array([self._label_map[v] for v in y])
        y_tensor = torch.LongTensor(y_mapped).to(DEVICE)
        if edge_index is not None:
            n = X.shape[0]
            self._adj_norm = normalize_adjacency(edge_index, n, edge_weight).to(DEVICE)
            logger.info(f'  [GNN] 使用外部图结构: {edge_index.size(1)}条边')
        else:
            logger.info('  [GNN] 从特征构建k-NN图...')
            self._adj_norm = self._build_graph(X)
            logger.info(f'  [GNN] 图构建完成')
        self._net = GNNNetwork(input_dim=self._input_dim, hidden_dim=self.params['hidden_dim'], num_classes=self._num_classes, dropout=self.params['dropout']).to(DEVICE)
        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.params['lr'], weight_decay=self.params['weight_decay'])
        criterion = nn.NLLLoss()
        stopper = EarlyStopping(patience=self.params['patience'])
        for epoch in range(self.params['max_epochs']):
            self._net.train()
            optimizer.zero_grad()
            log_probs = self._net(X_tensor, self._adj_norm)
            loss = criterion(log_probs, y_tensor)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._net.parameters(), 5.0)
            optimizer.step()
            if (epoch + 1) % 50 == 0:
                acc = (log_probs.argmax(dim=1) == y_tensor).float().mean()
                logger.info(f"  [GNN] Epoch {epoch + 1}/{self.params['max_epochs']}, Loss: {loss.item():.4f}, Acc: {acc.item():.4f}")
            if stopper.step(loss.item(), self._net):
                logger.info(f'  [GNN] 早停于 epoch {epoch + 1}')
                break
        stopper.restore(self._net)
        self.is_fitted = True
        logger.info(f'[GNN] 训练完成: {self._input_dim}维 → {self._num_classes}类')
        return self

    def predict(self, X) -> np.ndarray:
        if not self.is_fitted or self._net is None:
            raise ValueError('GNN模型尚未训练')
        X_np = self._to_numpy(X, None)[0]
        self._net.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_np).to(DEVICE)
            from src.models.graph_builder import GraphBuilder
            builder = GraphBuilder(k=self.params['k_neighbors'], metric=self.params['graph_metric'], self_loop=True)
            edge_index, edge_weight = builder.build_graph(X_np)
            n = X_np.shape[0]
            adj_norm = normalize_adjacency(edge_index, n, edge_weight).to(DEVICE)
            log_probs = self._net(X_tensor, adj_norm)
            preds = log_probs.argmax(dim=1).cpu().numpy()
        inv_map = {v: k for k, v in self._label_map.items()}
        return np.array([inv_map[p] for p in preds])

    def predict_proba(self, X) -> np.ndarray:
        if not self.is_fitted or self._net is None:
            raise ValueError('GNN模型尚未训练')
        X_np = self._to_numpy(X, None)[0]
        self._net.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_np).to(DEVICE)
            from src.models.graph_builder import GraphBuilder
            builder = GraphBuilder(k=self.params['k_neighbors'], metric=self.params['graph_metric'], self_loop=True)
            edge_index, edge_weight = builder.build_graph(X_np)
            n = X_np.shape[0]
            adj_norm = normalize_adjacency(edge_index, n, edge_weight).to(DEVICE)
            log_probs = self._net(X_tensor, adj_norm)
            probs = torch.exp(log_probs).cpu().numpy()
        return probs

    def _to_numpy(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            X = X.values.astype(np.float32)
        elif isinstance(X, pd.Series):
            X = X.values.astype(np.float32)
        elif isinstance(X, (list,)):
            X = np.array(X, dtype=np.float32)
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        if y is not None:
            if isinstance(y, (pd.Series, pd.DataFrame)):
                y = y.values.ravel()
            elif isinstance(y, (list,)):
                y = np.array(y)
            elif isinstance(y, torch.Tensor):
                y = y.cpu().numpy()
            return (X, y)
        return (X,)

    def save_model(self, path: str) -> None:
        data = {'model_state': self._net.state_dict() if self._net else None, 'params': self.params, 'num_classes': self._num_classes, 'input_dim': self._input_dim, 'label_map': self._label_map, 'model_name': self.model_name, 'is_fitted': self.is_fitted}
        joblib.dump(data, path)
        logger.info(f'GNN模型已保存: {path}')

    def load_model(self, path: str) -> 'GNNModel':
        data = joblib.load(path)
        self.params = data['params']
        self._num_classes = data['num_classes']
        self._input_dim = data['input_dim']
        self._label_map = data['label_map']
        self.model_name = data['model_name']
        self.is_fitted = data['is_fitted']
        if data['model_state'] and self.is_fitted:
            self._net = GNNNetwork(input_dim=self._input_dim, hidden_dim=self.params['hidden_dim'], num_classes=self._num_classes, dropout=self.params['dropout']).to(DEVICE)
            self._net.load_state_dict(data['model_state'])
        logger.info(f'GNN模型已加载: {path}')
        return self

class ResidualBlock(nn.Module):
    """残差块: Linear → BN → ReLU → Dropout → Linear → BN, + skip connection"""

    def __init__(self, dim: int, dropout: float=0.3):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.dropout(out)
        out = self.bn2(self.fc2(out))
        out = F.relu(out + residual)
        return out

class TabResNetNetwork(nn.Module):
    """Tabular ResNet: 输入投影 → N个残差块 → 分类头"""

    def __init__(self, input_dim: int, hidden_dim: int, num_blocks: int, num_classes: int, dropout: float=0.3):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.res_blocks = nn.ModuleList([ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)])
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for block in self.res_blocks:
            x = block(x)
        return self.classifier(x)

class TabResNetModel(BaseModel):
    """
    Tabular ResNet 分类器

    用法:
        model = TabResNetModel({'hidden_dim': 256, 'num_blocks': 5, 'lr': 0.001})
        model.train(X_train, y_train)
        preds = model.predict(X_test)
    """

    def __init__(self, params: Optional[Dict]=None):
        super().__init__('TabResNet')
        default_params = {'hidden_dim': 256, 'num_blocks': 5, 'dropout': 0.3, 'lr': 0.001, 'weight_decay': 0.0001, 'batch_size': 256, 'max_epochs': 200, 'patience': 30, 'random_state': 42}
        self.params = {**default_params, **(params or {})}
        self._net = None
        self._num_classes = None
        self._input_dim = None
        self._label_map = None

    def _set_random_seed(self):
        torch.manual_seed(self.params['random_state'])
        np.random.seed(self.params['random_state'])

    def train(self, X_train, y_train, **kwargs) -> 'TabResNetModel':
        self._set_random_seed()
        X, y = self._to_numpy(X_train, y_train)
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)
        self._input_dim = X.shape[1]
        self._num_classes = len(np.unique(y))
        self._label_map = {v: i for i, v in enumerate(np.unique(y))}
        y_mapped = np.array([self._label_map[v] for v in y])
        y_tensor = torch.LongTensor(y_mapped)
        self._net = TabResNetNetwork(input_dim=self._input_dim, hidden_dim=self.params['hidden_dim'], num_blocks=self.params['num_blocks'], num_classes=self._num_classes, dropout=self.params['dropout']).to(DEVICE)
        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.params['lr'], weight_decay=self.params['weight_decay'])
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        stopper = EarlyStopping(patience=self.params['patience'])
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.params['batch_size'], shuffle=True, drop_last=True)
        for epoch in range(self.params['max_epochs']):
            self._net.train()
            epoch_loss = 0.0
            for batch_X, batch_y in loader:
                batch_X, batch_y = (batch_X.to(DEVICE), batch_y.to(DEVICE))
                optimizer.zero_grad()
                logits = self._net(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._net.parameters(), 5.0)
                optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(loader)
            scheduler.step(avg_loss)
            if (epoch + 1) % 50 == 0:
                logger.info(f"  [TabResNet] Epoch {epoch + 1}/{self.params['max_epochs']}, Loss: {avg_loss:.4f}")
            if stopper.step(avg_loss, self._net):
                logger.info(f'  [TabResNet] 早停于 epoch {epoch + 1}')
                break
        stopper.restore(self._net)
        self.is_fitted = True
        logger.info(f'[TabResNet] 训练完成: {self._input_dim}维 → {self._num_classes}类')
        return self

    def predict(self, X) -> np.ndarray:
        if not self.is_fitted or self._net is None:
            raise ValueError('TabResNet模型尚未训练')
        X_np = self._to_numpy(X, None)[0]
        self._net.eval()
        with torch.no_grad():
            logits = self._net(torch.FloatTensor(X_np).to(DEVICE))
            preds = logits.argmax(dim=1).cpu().numpy()
        inv_map = {v: k for k, v in self._label_map.items()}
        return np.array([inv_map[p] for p in preds])

    def predict_proba(self, X) -> np.ndarray:
        if not self.is_fitted or self._net is None:
            raise ValueError('TabResNet模型尚未训练')
        X_np = self._to_numpy(X, None)[0]
        self._net.eval()
        with torch.no_grad():
            logits = self._net(torch.FloatTensor(X_np).to(DEVICE))
            probs = F.softmax(logits, dim=1).cpu().numpy()
        return probs

    def _to_numpy(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            X = X.values.astype(np.float32)
        elif isinstance(X, pd.Series):
            X = X.values.astype(np.float32)
        elif isinstance(X, (list,)):
            X = np.array(X, dtype=np.float32)
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        if y is not None:
            if isinstance(y, (pd.Series, pd.DataFrame)):
                y = y.values.ravel()
            elif isinstance(y, (list,)):
                y = np.array(y)
            elif isinstance(y, torch.Tensor):
                y = y.cpu().numpy()
            return (X, y)
        return (X,)

    def save_model(self, path: str) -> None:
        data = {'model_state': self._net.state_dict() if self._net else None, 'params': self.params, 'num_classes': self._num_classes, 'input_dim': self._input_dim, 'label_map': self._label_map, 'model_name': self.model_name, 'is_fitted': self.is_fitted}
        joblib.dump(data, path)

    def load_model(self, path: str) -> 'TabResNetModel':
        data = joblib.load(path)
        self.params = data['params']
        self._num_classes = data['num_classes']
        self._input_dim = data['input_dim']
        self._label_map = data['label_map']
        self.model_name = data['model_name']
        self.is_fitted = data['is_fitted']
        if data['model_state'] and self.is_fitted:
            self._net = TabResNetNetwork(input_dim=self._input_dim, hidden_dim=self.params['hidden_dim'], num_blocks=self.params['num_blocks'], num_classes=self._num_classes, dropout=self.params['dropout']).to(DEVICE)
            self._net.load_state_dict(data['model_state'])
        return self

class TabTransformerNetwork(nn.Module):
    """
    简化版 TabTransformer：
    输入投影 → 位置编码 → N个TransformerEncoder层 → CLS token → 分类头
    省去embedding lookup（输入已是稠密One-Hot/连续值），直接用线性投影。
    """

    def __init__(self, input_dim: int, d_model: int, n_heads: int, n_layers: int, num_classes: int, dropout: float=0.2):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(input_dim, d_model), nn.LayerNorm(d_model), nn.Dropout(dropout))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4, dropout=dropout, activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.classifier = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x).unsqueeze(1)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.transformer(x)
        return self.classifier(x[:, 0])

class TabTransformerModel(BaseModel):
    """
    TabTransformer 分类器

    用法:
        model = TabTransformerModel({'d_model': 256, 'n_heads': 8, 'n_layers': 3})
        model.train(X_train, y_train)
        preds = model.predict(X_test)
    """

    def __init__(self, params: Optional[Dict]=None):
        super().__init__('TabTransformer')
        default_params = {'d_model': 128, 'n_heads': 4, 'n_layers': 3, 'dropout': 0.2, 'lr': 0.001, 'weight_decay': 1e-05, 'batch_size': 256, 'max_epochs': 200, 'patience': 30, 'random_state': 42}
        self.params = {**default_params, **(params or {})}
        self._net = None
        self._num_classes = None
        self._input_dim = None
        self._label_map = None

    def _set_random_seed(self):
        torch.manual_seed(self.params['random_state'])
        np.random.seed(self.params['random_state'])

    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs) -> 'TabTransformerModel':
        self._set_random_seed()
        X, y = self._to_numpy(X_train, y_train)
        X_tensor = torch.FloatTensor(X)
        self._input_dim = X.shape[1]
        self._num_classes = len(np.unique(y))
        self._label_map = {v: i for i, v in enumerate(np.unique(y))}
        y_mapped = np.array([self._label_map[v] for v in y])
        y_tensor = torch.LongTensor(y_mapped)
        self._net = TabTransformerNetwork(input_dim=self._input_dim, d_model=self.params['d_model'], n_heads=self.params['n_heads'], n_layers=self.params['n_layers'], num_classes=self._num_classes, dropout=self.params['dropout']).to(DEVICE)
        optimizer = torch.optim.AdamW(self._net.parameters(), lr=self.params['lr'], weight_decay=self.params['weight_decay'])
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
        stopper = EarlyStopping(patience=self.params['patience'])
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.params['batch_size'], shuffle=True, drop_last=True)
        val_loader = _prepare_val_loader(self, X_val, y_val, self.params['batch_size'])
        for epoch in range(self.params['max_epochs']):
            self._net.train()
            epoch_loss = 0.0
            for batch_X, batch_y in loader:
                batch_X, batch_y = (batch_X.to(DEVICE), batch_y.to(DEVICE))
                optimizer.zero_grad()
                logits = self._net(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._net.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(loader)
            monitor_loss = _compute_val_loss(self._net, criterion, val_loader) if val_loader is not None else avg_loss
            scheduler.step()
            if (epoch + 1) % 50 == 0:
                logger.info(f"  [TabTransformer] Epoch {epoch + 1}/{self.params['max_epochs']}, Loss: {avg_loss:.4f}, ValLoss: {monitor_loss:.4f}")
            if stopper.step(monitor_loss, self._net):
                logger.info(f'  [TabTransformer] 早停于 epoch {epoch + 1}')
                break
        stopper.restore(self._net)
        self.is_fitted = True
        logger.info(f'[TabTransformer] 训练完成: {self._input_dim}维 → {self._num_classes}类')
        return self

    def predict(self, X) -> np.ndarray:
        if not self.is_fitted or self._net is None:
            raise ValueError('TabTransformer模型尚未训练')
        X_np = self._to_numpy(X, None)[0]
        self._net.eval()
        with torch.no_grad():
            logits = self._net(torch.FloatTensor(X_np).to(DEVICE))
            preds = logits.argmax(dim=1).cpu().numpy()
        inv_map = {v: k for k, v in self._label_map.items()}
        return np.array([inv_map[p] for p in preds])

    def predict_proba(self, X) -> np.ndarray:
        if not self.is_fitted or self._net is None:
            raise ValueError('TabTransformer模型尚未训练')
        X_np = self._to_numpy(X, None)[0]
        self._net.eval()
        with torch.no_grad():
            logits = self._net(torch.FloatTensor(X_np).to(DEVICE))
            probs = F.softmax(logits, dim=1).cpu().numpy()
        return probs

    def _to_numpy(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            X = X.values.astype(np.float32)
        elif isinstance(X, pd.Series):
            X = X.values.astype(np.float32)
        elif isinstance(X, (list,)):
            X = np.array(X, dtype=np.float32)
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        if y is not None:
            if isinstance(y, (pd.Series, pd.DataFrame)):
                y = y.values.ravel()
            elif isinstance(y, (list,)):
                y = np.array(y)
            elif isinstance(y, torch.Tensor):
                y = y.cpu().numpy()
            return (X, y)
        return (X,)

    def save_model(self, path: str) -> None:
        data = {'model_state': self._net.state_dict() if self._net else None, 'params': self.params, 'num_classes': self._num_classes, 'input_dim': self._input_dim, 'label_map': self._label_map, 'model_name': self.model_name, 'is_fitted': self.is_fitted}
        joblib.dump(data, path)

    def load_model(self, path: str) -> 'TabTransformerModel':
        data = joblib.load(path)
        self.params = data['params']
        self._num_classes = data['num_classes']
        self._input_dim = data['input_dim']
        self._label_map = data['label_map']
        self.model_name = data['model_name']
        self.is_fitted = data['is_fitted']
        if data['model_state'] and self.is_fitted:
            self._net = TabTransformerNetwork(input_dim=self._input_dim, d_model=self.params['d_model'], n_heads=self.params['n_heads'], n_layers=self.params['n_layers'], num_classes=self._num_classes, dropout=self.params['dropout']).to(DEVICE)
            self._net.load_state_dict(data['model_state'])
        return self

class CNNNetwork(nn.Module):
    """
    多尺度 1D CNN: 将特征 reshape 为 (batch, 1, D) → 多个不同 kernel_size
    的 Conv1d → 拼接 → 全连接分类头。适合捕捉 One-Hot 特征的局部模式。
    """

    def __init__(self, input_dim: int, num_classes: int, dropout: float=0.3):
        super().__init__()
        kernel_sizes = [3, 5, 7]
        out_channels = 64
        self.convs = nn.ModuleList([nn.Sequential(nn.Conv1d(1, out_channels, ks, padding=ks // 2), nn.BatchNorm1d(out_channels), nn.ReLU()) for ks in kernel_sizes])
        conv_out_dim = out_channels * len(kernel_sizes)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(conv_out_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        pooled = []
        for conv in self.convs:
            out = conv(x)
            out = F.adaptive_avg_pool1d(out, 1).squeeze(-1)
            pooled.append(out)
        x = torch.cat(pooled, dim=1)
        return self.classifier(x)

class CNNModel(BaseModel):
    """1D-CNN 表格分类器"""

    def __init__(self, params: Optional[Dict]=None):
        super().__init__('CNN')
        default_params = {'dropout': 0.3, 'lr': 0.001, 'weight_decay': 0.0001, 'batch_size': 256, 'max_epochs': 200, 'patience': 30, 'random_state': 42}
        self.params = {**default_params, **(params or {})}
        self._net = None
        self._num_classes = None
        self._input_dim = None
        self._label_map = None

    def _set_random_seed(self):
        torch.manual_seed(self.params['random_state'])
        np.random.seed(self.params['random_state'])

    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs) -> 'CNNModel':
        self._set_random_seed()
        X, y = self._to_numpy(X_train, y_train)
        X_tensor = torch.FloatTensor(X)
        self._input_dim = X.shape[1]
        self._num_classes = len(np.unique(y))
        self._label_map = {v: i for i, v in enumerate(np.unique(y))}
        y_mapped = np.array([self._label_map[v] for v in y])
        y_tensor = torch.LongTensor(y_mapped)
        self._net = CNNNetwork(input_dim=self._input_dim, num_classes=self._num_classes, dropout=self.params['dropout']).to(DEVICE)
        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.params['lr'], weight_decay=self.params['weight_decay'])
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        stopper = EarlyStopping(patience=self.params['patience'])
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.params['batch_size'], shuffle=True, drop_last=True)
        val_loader = _prepare_val_loader(self, X_val, y_val, self.params['batch_size'])
        for epoch in range(self.params['max_epochs']):
            self._net.train()
            epoch_loss = 0.0
            for batch_X, batch_y in loader:
                batch_X, batch_y = (batch_X.to(DEVICE), batch_y.to(DEVICE))
                optimizer.zero_grad()
                logits = self._net(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._net.parameters(), 5.0)
                optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(loader)
            monitor_loss = _compute_val_loss(self._net, criterion, val_loader) if val_loader is not None else avg_loss
            scheduler.step(monitor_loss)
            if (epoch + 1) % 50 == 0:
                logger.info(f"  [CNN] Epoch {epoch + 1}/{self.params['max_epochs']}, Loss: {avg_loss:.4f}, ValLoss: {monitor_loss:.4f}")
            if stopper.step(monitor_loss, self._net):
                logger.info(f'  [CNN] 早停于 epoch {epoch + 1}')
                break
        stopper.restore(self._net)
        self.is_fitted = True
        logger.info(f'[CNN] 训练完成: {self._input_dim}维 → {self._num_classes}类')
        return self

    def predict(self, X) -> np.ndarray:
        if not self.is_fitted or self._net is None:
            raise ValueError('CNN模型尚未训练')
        X_np = self._to_numpy(X, None)[0]
        self._net.eval()
        with torch.no_grad():
            logits = self._net(torch.FloatTensor(X_np).to(DEVICE))
            preds = logits.argmax(dim=1).cpu().numpy()
        inv_map = {v: k for k, v in self._label_map.items()}
        return np.array([inv_map[p] for p in preds])

    def predict_proba(self, X) -> np.ndarray:
        if not self.is_fitted or self._net is None:
            raise ValueError('CNN模型尚未训练')
        X_np = self._to_numpy(X, None)[0]
        self._net.eval()
        with torch.no_grad():
            logits = self._net(torch.FloatTensor(X_np).to(DEVICE))
            probs = F.softmax(logits, dim=1).cpu().numpy()
        return probs

    def _to_numpy(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            X = X.values.astype(np.float32)
        elif isinstance(X, pd.Series):
            X = X.values.astype(np.float32)
        elif isinstance(X, (list,)):
            X = np.array(X, dtype=np.float32)
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        if y is not None:
            if isinstance(y, (pd.Series, pd.DataFrame)):
                y = y.values.ravel()
            elif isinstance(y, (list,)):
                y = np.array(y)
            elif isinstance(y, torch.Tensor):
                y = y.cpu().numpy()
            return (X, y)
        return (X,)

    def save_model(self, path: str) -> None:
        data = {'model_state': self._net.state_dict() if self._net else None, 'params': self.params, 'num_classes': self._num_classes, 'input_dim': self._input_dim, 'label_map': self._label_map, 'model_name': self.model_name, 'is_fitted': self.is_fitted}
        joblib.dump(data, path)

    def load_model(self, path: str) -> 'CNNModel':
        data = joblib.load(path)
        self.params = data['params']
        self._num_classes = data['num_classes']
        self._input_dim = data['input_dim']
        self._label_map = data['label_map']
        self.model_name = data['model_name']
        self.is_fitted = data['is_fitted']
        if data['model_state'] and self.is_fitted:
            self._net = CNNNetwork(input_dim=self._input_dim, num_classes=self._num_classes, dropout=self.params['dropout']).to(DEVICE)
            self._net.load_state_dict(data['model_state'])
        return self



import logging
import random
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import scipy.sparse as sp
import os
from copy import deepcopy

from torch_geometric.nn import SAGEConv
from sklearn.metrics import (roc_auc_score, f1_score,
                              precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 只过滤已知无害的deprecation warning，不全局屏蔽
warnings.filterwarnings(
    'ignore', category=UserWarning,
    message=".*NeighborSampler.*deprecated.*")
warnings.filterwarnings(
    'ignore', category=FutureWarning,
    message=".*torch.load.*")

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s')
logger = logging.getLogger(__name__)


def set_all_seeds(seed: int) -> None:
    """Reset all RNGs so a method is invariant to execution order."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 常量（魔法数字集中管理）
# ============================================================
TRAIN_VAL_SPLIT       = 0.7    # 与蒸馏保持一致（EllipticData/YelpData内部70:15:15）
VAL_SPLIT             = 0.15   # 验证集比例
TEST_SPLIT_RATIO      = 0.15   # 各机构取15%作为公共测试集（Yelp）
FRAUD_OVERSAMPLE_RATIO = 0.3   # 合成图中欺诈节点占比
MIN_SYN_NODES         = 10     # 合成节点最小数量
GNN_TRAIN_EPOCHS      = 300    # GNN训练轮数
GNN_EVAL_INTERVAL     = 50     # GNN验证间隔
HIDDEN_DIM            = 256    # GNN隐藏层维度
LEARNING_RATE         = 0.01   # 学习率
WEIGHT_DECAY          = 5e-4   # 权重衰减
FEDAVG_ROUNDS         = 10     # 联邦学习轮数
FEDAVG_LOCAL_EPOCHS   = 30     # 联邦学习本地训练轮数
KNN_NEIGHBORS         = 5      # KNN图的k值
MAX_GNFS_SAMPLES      = 500    # GNFS计算时最大样本数
SEED_DEFAULT          = 15     # 默认随机种子

# Elliptic时间步划分（无数据泄露）
ELLIPTIC_INST_SPLITS = {
    'A': (1,  17),
    'B': (17, 33),
    'C': (33, 40),  # 不含测试集t=40-49
}
ELLIPTIC_TEST_START = 40
ELLIPTIC_TEST_END   = 50


DISTILL_TAG = 'uw1_ltfocal_ov0.0'


MODEL_VAL_SPLIT = 0.15
CALIBRATION_SPLIT = 0.15


# ============================================================
# 实验配置 DataClass
# ============================================================
@dataclass
class ExperimentConfig:
    """所有实验超参数集中管理，避免散落的魔法数字"""
    # 数据集
    dataset:        str   = 'elliptic'
    reduction_rate: float = 0.01
    seed:           int   = SEED_DEFAULT
    gpu_id:         int   = 0

    # GNN训练
    gnn_epochs:     int   = GNN_TRAIN_EPOCHS
    hidden_dim:     int   = HIDDEN_DIM
    lr:             float = LEARNING_RATE
    weight_decay:   float = WEIGHT_DECAY

    # 联邦学习
    fed_rounds:        int = FEDAVG_ROUNDS
    fed_local_epochs:  int = FEDAVG_LOCAL_EPOCHS

    distill_use_weight: int   = 1
    distill_loss_type:  str   = 'focal'
    distill_oversample: float = 0.0      # 最优配置不过采样



    fraud_oversample_ratio: float = 0.0 

    # 数据划分
    train_val_split: float = TRAIN_VAL_SPLIT
    model_val_split: float = MODEL_VAL_SPLIT
    calibration_split: float = CALIBRATION_SPLIT
    test_split_ratio: float = TEST_SPLIT_RATIO

    @property
    def distill_tag(self) -> str:
        return (f'uw{self.distill_use_weight}_'
                f'lt{self.distill_loss_type}_'
                f'ov{self.distill_oversample}')


def _validate_canonical_split(name: str, split: Dict[str, np.ndarray],
                              pool_idx: np.ndarray,
                              labels: np.ndarray) -> None:
    """Fail fast if a train/model-val/calibration partition is not isolated."""
    keys = ('train', 'model_val', 'calibration')
    arrays = {key: np.asarray(split[key], dtype=np.int64) for key in keys}
    for key, idx in arrays.items():
        if idx.size == 0:
            raise ValueError(f"Institution {name}: {key} split is empty")
        if np.unique(idx).size != idx.size:
            raise ValueError(f"Institution {name}: duplicate nodes in {key}")

    for i, left in enumerate(keys):
        for right in keys[i + 1:]:
            if np.intersect1d(arrays[left], arrays[right]).size:
                raise ValueError(
                    f"Institution {name}: {left} overlaps {right}")

    union = np.sort(np.concatenate([arrays[key] for key in keys]))
    if not np.array_equal(union, np.sort(np.asarray(pool_idx, dtype=np.int64))):
        raise ValueError(
            f"Institution {name}: canonical split does not cover its history pool")

    for key in ('model_val', 'calibration'):
        if np.unique(labels[arrays[key]]).size < 2:
            raise ValueError(
                f"Institution {name}: {key} must contain both classes")


def _canonical_institution_split(inst: Dict, labels: np.ndarray,
                                 cfg: ExperimentConfig,
                                 name: str) -> Dict[str, np.ndarray]:
    """Create the split shared by distillation and downstream evaluation."""
    idx = np.asarray(inst['idx'], dtype=np.int64)
    timesteps = inst.get('timestep')

    if timesteps is not None:
        timesteps = np.asarray(timesteps)
        if timesteps.shape[0] != idx.shape[0]:
            raise ValueError(
                f"Institution {name}: timestep/index length mismatch")
        unique_ts = np.sort(np.unique(timesteps))
        if unique_ts.size < 3:
            raise ValueError(
                f"Institution {name}: at least three timesteps are required")
        n_train_ts = max(1, int(unique_ts.size * cfg.train_val_split))
        n_model_val_ts = max(1, int(unique_ts.size * cfg.model_val_split))
        if n_train_ts + n_model_val_ts >= unique_ts.size:
            n_train_ts = max(1, unique_ts.size - 2)
            n_model_val_ts = 1
        train_ts = unique_ts[:n_train_ts]
        model_val_ts = unique_ts[n_train_ts:n_train_ts + n_model_val_ts]
        calibration_ts = unique_ts[n_train_ts + n_model_val_ts:]
        split = {
            'train': idx[np.isin(timesteps, train_ts)],
            'model_val': idx[np.isin(timesteps, model_val_ts)],
            'calibration': idx[np.isin(timesteps, calibration_ts)],
        }
    else:
        holdout = cfg.model_val_split + cfg.calibration_split
        if not np.isclose(cfg.train_val_split + holdout, 1.0):
            raise ValueError(
                "train/model_val/calibration ratios must sum to one")
        train_idx, holdout_idx = train_test_split(
            idx,
            train_size=cfg.train_val_split,
            random_state=cfg.seed,
            stratify=labels[idx],
        )
        model_share = cfg.model_val_split / holdout
        model_val_idx, calibration_idx = train_test_split(
            holdout_idx,
            train_size=model_share,
            random_state=cfg.seed,
            stratify=labels[holdout_idx],
        )
        split = {
            'train': np.sort(train_idx),
            'model_val': np.sort(model_val_idx),
            'calibration': np.sort(calibration_idx),
        }

    _validate_canonical_split(name, split, idx, labels)
    return split


def attach_canonical_splits(institutions: Dict, labels: np.ndarray,
                            cfg: ExperimentConfig) -> None:
    """Attach one immutable canonical split to each institution."""
    for name, inst in institutions.items():
        split = _canonical_institution_split(inst, labels, cfg, name)
        inst['split'] = split
        logger.info(
            f"  Institution {name} split: train={len(split['train']):,}, "
            f"model-val={len(split['model_val']):,}, "
            f"calibration={len(split['calibration']):,}; "
            f"train fraud={(labels[split['train']] == 1).sum():,}")


def validate_protocol_isolation(institutions: Dict,
                                global_data: Dict) -> None:
    """Verify that no historical protocol split contains a test node."""
    test_idx = np.asarray(global_data['test_idx'], dtype=np.int64)
    history_idx = np.concatenate([
        np.asarray(inst['idx'], dtype=np.int64)
        for inst in institutions.values()
    ])
    if np.intersect1d(history_idx, test_idx).size:
        raise ValueError("Institution history overlaps the external test set")

    for name, inst in institutions.items():
        for split_name in ('train', 'model_val', 'calibration'):
            split_idx = np.asarray(inst['split'][split_name], dtype=np.int64)
            if np.intersect1d(split_idx, test_idx).size:
                raise ValueError(
                    f"Institution {name}: {split_name} overlaps the test set")


# ============================================================
# 1. 数据加载
# ============================================================
def load_and_split(
        processed_path: str = 'data/elliptic_bitcoin_dataset/processed.pt',
        cfg: ExperimentConfig = None):
    """加载Elliptic数据集并按时间步划分三个机构"""
    if cfg is None:
        cfg = ExperimentConfig()

    logger.info("Loading Elliptic and splitting into 3 institutions...")

    ck       = torch.load(processed_path, map_location='cpu')
    pyg_data = ck['data']
    timestep = pyg_data.timestep.numpy()
    labels   = pyg_data.y.numpy()
    features = pyg_data.x.numpy()
    src_all  = pyg_data.edge_index[0].numpy()
    dst_all  = pyg_data.edge_index[1].numpy()

    # 只保留有标注节点（排除unknown=-1）
    labeled  = (labels != -1)
    lab_idx  = np.where(labeled)[0]
    old2new  = np.full(len(labels), -1, dtype=np.int64)
    old2new[lab_idx] = np.arange(len(lab_idx))

    feat_lab  = features[lab_idx].astype(np.float32)
    label_lab = labels[lab_idx]
    time_lab  = timestep[lab_idx]

    valid    = labeled[src_all] & labeled[dst_all]
    new_edge = np.stack(
        [old2new[src_all[valid]], old2new[dst_all[valid]]], axis=0)

    institutions = _build_institutions(
        feat_lab, label_lab, time_lab,
        ELLIPTIC_INST_SPLITS, split_by='timestep')

    attach_canonical_splits(institutions, label_lab, cfg)
    all_train_idx = np.concatenate([
        inst['split']['train'] for inst in institutions.values()])
    # processed.pt is already in a common coordinate system. Refit the final
    # downstream scaler on canonical training nodes only, and apply the same
    # affine map to legacy synthetic features when they are loaded.
    eval_scaler = StandardScaler().fit(feat_lab[all_train_idx])
    feat_lab = eval_scaler.transform(feat_lab).astype(np.float32)
    for inst in institutions.values():
        inst['feat'] = feat_lab[inst['idx']]
        inst['syn_source_mean'] = np.zeros(feat_lab.shape[1], dtype=np.float64)
        inst['syn_source_scale'] = np.ones(feat_lab.shape[1], dtype=np.float64)
        inst['eval_target_mean'] = eval_scaler.mean_.copy()
        inst['eval_target_scale'] = eval_scaler.scale_.copy()

    test_mask = ((time_lab >= ELLIPTIC_TEST_START) &
                 (time_lab < ELLIPTIC_TEST_END))
    test_idx  = np.where(test_mask)[0]
    n_f_test  = (label_lab[test_idx] == 1).sum()
    logger.info(
        f"  Shared test (t={ELLIPTIC_TEST_START}-{ELLIPTIC_TEST_END-1}): "
        f"{len(test_idx):,} nodes "
        f"(fraud={n_f_test}, {n_f_test/len(test_idx)*100:.1f}%)")

    return institutions, {
        'feat':       feat_lab,
        'labels':     label_lab,
        'edge_index': new_edge,
        'n':          len(lab_idx),
        'test_idx':   test_idx,
        'history_idx': np.concatenate([
            inst['idx'] for inst in institutions.values()]),
        'dataset':    'elliptic',
    }


def load_and_split_yelp(
        data_dir: str = 'data/Yelpchi',
        cfg: ExperimentConfig = None):
    """
    加载YelpChi, 用Louvain社区检测划分为三个机构。
    Uses yelp_community_split() — identical to YelpData，保证
    蒸馏和评估使用同一 test set。

    社区划分保留图结构（vs 随机节点切分丢失67%边）。
    不同社区自然具有不同的 fraud rate。
    """
    import scipy.io
    from sklearn.preprocessing import StandardScaler
    from load_yelp import yelp_community_split, _load_yelp_adj

    if cfg is None:
        cfg = ExperimentConfig()

    logger.info("Loading YelpChi (Louvain community split)...")

    mat = scipy.io.loadmat(f'{data_dir}/YelpChi.mat')
    features = mat['features']
    if sp.issparse(features):
        features = features.toarray()
    features = features.astype(np.float32)
    labels = mat['label'].flatten().astype(np.int64)
    n_total = len(labels)

    adj_full, adj_key = _load_yelp_adj(mat)
    adj_coo = adj_full.tocoo()
    edge_all = np.stack([adj_coo.row, adj_coo.col], axis=0)
    logger.info(f"  Adj: {adj_key} ({adj_full.nnz:,} edges)")

    # Community-based split (与 YelpData 完全一致)
    trainval_per_inst, shared_test, partition = yelp_community_split(
        adj_full, labels, seed=cfg.seed)

    # Report edge retention
    total_edges = adj_full.nnz
    retained = sum(
        adj_full[np.ix_(p, p)].nnz for p in partition)
    logger.info(f"  Edge retention: {retained:,}/{total_edges:,} "
                f"({retained/total_edges:.1%})")

    institutions = {}
    for name in ['A', 'B', 'C']:
        tv_idx = trainval_per_inst[name]
        n_fraud = (labels[tv_idx] == 1).sum()
        fraud_rate = n_fraud / len(tv_idx) * 100
        logger.info(
            f"  Institution {name}: {len(tv_idx):,} trainval nodes, "
            f"fraud={n_fraud} ({fraud_rate:.1f}%)")
        institutions[name] = {
            'idx':    tv_idx,
            'feat':   features[tv_idx],
            'labels': labels[tv_idx],
        }

    attach_canonical_splits(institutions, labels, cfg)
    all_train_idx = np.concatenate([
        inst['split']['train'] for inst in institutions.values()])
    eval_scaler = StandardScaler().fit(features[all_train_idx])
    features = eval_scaler.transform(features).astype(np.float32)
    for inst in institutions.values():
        inst['feat'] = features[inst['idx']]
        inst['syn_source_mean'] = np.zeros(features.shape[1], dtype=np.float64)
        inst['syn_source_scale'] = np.ones(features.shape[1], dtype=np.float64)
        inst['eval_target_mean'] = eval_scaler.mean_.copy()
        inst['eval_target_scale'] = eval_scaler.scale_.copy()

    n_f_test = (labels[shared_test] == 1).sum()
    logger.info(
        f"  Shared test: {len(shared_test):,} nodes "
        f"(fraud={n_f_test})")

    return institutions, {
        'feat':       features,
        'labels':     labels,
        'edge_index': edge_all,
        'n':          n_total,
        'test_idx':   shared_test,
        'history_idx': np.concatenate([
            inst['idx'] for inst in institutions.values()]),
        'dataset':    'yelp',
    }


def load_and_split_amazon(
        data_dir: str = 'data/Amazon',
        cfg: ExperimentConfig = None):
    """
    加载Amazon-Fraud并按图关系划分为三个机构

    机构划分：same nodes, different relation views
      A: net_upu（共同购买）  B: net_usu（评分）  C: net_uvu（品类）

    Split: uses amazon_split() — identical to AmazonData，保证
    蒸馏和评估使用同一 test set。
    """
    import scipy.io
    from sklearn.preprocessing import StandardScaler
    from load_amazon import (INST_TO_RELATION, _load_adj,
                             _build_global_adj, AMAZON_ADJ_FALLBACK,
                             amazon_split)

    if cfg is None:
        cfg = ExperimentConfig()

    logger.info("Loading Amazon-Fraud (relation-based institution split)...")

    mat = scipy.io.loadmat(f'{data_dir}/Amazon.mat')
    features = mat['features']
    if sp.issparse(features):
        features = features.toarray()
    features = features.astype(np.float32)
    labels = mat['label'].flatten().astype(np.int64)
    n_total = len(labels)

    # 与 AmazonData 完全一致的 70:15:15 划分
    idx_train, idx_val, idx_test = amazon_split(n_total, labels, cfg.seed)
    trainval_idx = np.sort(np.concatenate([idx_train, idx_val]))

    # Scaler: fit on train only（修复原代码对全数据 fit 的泄露）
    scaler = StandardScaler()
    scaler.fit(features[idx_train])
    features = scaler.transform(features).astype(np.float32)

    n_f_test = (labels[idx_test] == 1).sum()
    logger.info(f"  Split: train={len(idx_train):,} "
                f"val={len(idx_val):,} test={len(idx_test):,} "
                f"(test fraud={n_f_test})")

    institutions = {}
    for inst_name, rel_key in INST_TO_RELATION.items():
        adj_mat, used_key = _load_adj(mat, rel_key, AMAZON_ADJ_FALLBACK)
        adj_coo = adj_mat.tocoo()
        edge_all = np.stack([adj_coo.row, adj_coo.col], axis=0)

        n_fraud = (labels[trainval_idx] == 1).sum()
        logger.info(
            f"  Institution {inst_name} ({used_key}): "
            f"{len(trainval_idx):,} trainval nodes, "
            f"fraud={n_fraud}")

        institutions[inst_name] = {
            'idx':        trainval_idx,
            'feat':       features[trainval_idx],
            'labels':     labels[trainval_idx],
            'edge_index': edge_all,
            'adj_full':   adj_mat,
        }

    model_val_idx, calibration_idx = train_test_split(
        idx_val,
        train_size=0.5,
        random_state=cfg.seed,
        stratify=labels[idx_val],
    )
    for name, inst in institutions.items():
        split = {
            'train': np.sort(idx_train),
            'model_val': np.sort(model_val_idx),
            'calibration': np.sort(calibration_idx),
        }
        _validate_canonical_split(name, split, trainval_idx, labels)
        inst['split'] = split
        inst['syn_source_mean'] = np.zeros(features.shape[1], dtype=np.float64)
        inst['syn_source_scale'] = np.ones(features.shape[1], dtype=np.float64)
        inst['eval_target_mean'] = np.zeros(features.shape[1], dtype=np.float64)
        inst['eval_target_scale'] = np.ones(features.shape[1], dtype=np.float64)

    # Global graph: union of all relations（比单一 net_upu 更全面）
    adj_global, used_key = _build_global_adj(mat)
    adj_coo_g = adj_global.tocoo()
    edge_global = np.stack([adj_coo_g.row, adj_coo_g.col], axis=0)
    logger.info(f"  Global graph: {used_key} ({adj_global.nnz:,} edges)")

    return institutions, {
        'feat':       features,
        'labels':     labels,
        'edge_index': edge_global,
        'n':          n_total,
        'test_idx':   idx_test,
        'history_idx': trainval_idx,
        'dataset':    'amazon',
    }


def load_and_split_elliptic_pp(
        data_dir: str = 'data/Elliptic++ Dataset',
        cfg: ExperimentConfig = None):
    """
    加载 Elliptic++ 交易视角数据集，按时间步划分机构。
    机构边界与 Elliptic 相同: A(1-16), B(17-32), C(33-39), test(40-49)。
    """
    from sklearn.preprocessing import StandardScaler
    if cfg is None:
        cfg = ExperimentConfig()

    processed = os.path.join(data_dir, 'processed.pt')
    if not os.path.exists(processed):
        raise FileNotFoundError(
            f"Run: python load_elliptic_pp.py --preprocess "
            f"--data_dir \"{data_dir}\"")

    logger.info("Loading Elliptic++ (transaction view)...")

    ck = torch.load(processed, map_location='cpu', weights_only=False)
    features = ck['features'].numpy()
    labels = ck['labels'].numpy()
    edge_index = ck['edge_index'].numpy()
    N = int(ck['num_nodes'])
    time_steps = ck.get('time_steps')
    if time_steps is not None:
        time_steps = time_steps.numpy()
    else:
        raise ValueError("Elliptic++ processed.pt missing time_steps")

    # Filter to labeled nodes only
    labeled = (labels != -1)
    lab_idx = np.where(labeled)[0]
    old2new = np.full(N, -1, dtype=np.int64)
    old2new[lab_idx] = np.arange(len(lab_idx))

    feat_lab = features[lab_idx].astype(np.float32)
    label_lab = labels[lab_idx]
    time_lab = time_steps[lab_idx]

    # Remap edges to labeled subgraph
    src, dst = edge_index[0], edge_index[1]
    valid = labeled[src] & labeled[dst]
    new_edge = np.stack(
        [old2new[src[valid]], old2new[dst[valid]]], axis=0)

    # Build institutions (same boundaries as Elliptic)
    institutions = _build_institutions(
        feat_lab, label_lab, time_lab,
        ELLIPTIC_INST_SPLITS, split_by='timestep')

    attach_canonical_splits(institutions, label_lab, cfg)
    all_train_idx = np.concatenate([
        inst['split']['train'] for inst in institutions.values()])
    eval_scaler = StandardScaler().fit(feat_lab[all_train_idx])

    # Existing Elliptic++ distilled artifacts were generated after a local
    # train-only scaler at each institution. Store both source and target
    # affine maps so loading can move them into the common evaluation space.
    for inst in institutions.values():
        source_scaler = StandardScaler().fit(
            feat_lab[inst['split']['train']])
        inst['syn_source_mean'] = source_scaler.mean_.copy()
        inst['syn_source_scale'] = source_scaler.scale_.copy()
        inst['eval_target_mean'] = eval_scaler.mean_.copy()
        inst['eval_target_scale'] = eval_scaler.scale_.copy()

    feat_lab = eval_scaler.transform(feat_lab).astype(np.float32)
    for inst in institutions.values():
        inst['feat'] = feat_lab[inst['idx']]

    # Shared test set
    test_mask = (time_lab >= ELLIPTIC_TEST_START) & \
                (time_lab < ELLIPTIC_TEST_END)
    test_idx = np.where(test_mask)[0]
    n_f_test = (label_lab[test_idx] == 1).sum()
    logger.info(
        f"  Shared test (t={ELLIPTIC_TEST_START}-{ELLIPTIC_TEST_END-1}): "
        f"{len(test_idx):,} nodes "
        f"(fraud={n_f_test}, {n_f_test/len(test_idx)*100:.1f}%)")
    logger.info(f"  Feature dim: {feat_lab.shape[1]}")

    return institutions, {
        'feat':       feat_lab,
        'labels':     label_lab,
        'edge_index': new_edge,
        'n':          len(lab_idx),
        'test_idx':   test_idx,
        'history_idx': np.concatenate([
            inst['idx'] for inst in institutions.values()]),
        'dataset':    'elliptic_pp',
    }


def _build_institutions(feat, labels, time_or_idx, splits, split_by):
    """按时间步或索引范围构建机构字典（内部工具函数）"""
    institutions = {}
    for name, (t0, t1) in splits.items():
        if split_by == 'timestep':
            mask = (time_or_idx >= t0) & (time_or_idx < t1)
        else:
            mask = (time_or_idx >= t0) & (time_or_idx < t1)
        idx  = np.where(mask)[0]
        n_f  = (labels[idx] == 1).sum()
        logger.info(
            f"  Institution {name} ({t0}-{t1-1}): "
            f"{len(idx):,} nodes, fraud={n_f} "
            f"({n_f/len(idx)*100:.1f}%)")
        institutions[name] = {
            'idx':    idx,
            'feat':   feat[idx],
            'labels': labels[idx],
            'timestep': np.asarray(time_or_idx)[idx]
                        if split_by == 'timestep' else None,
        }
    return institutions


# ============================================================
# 2. GNN模型和训练工具
# ============================================================
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_dim: int, hidden: int = HIDDEN_DIM,
                 out_dim: int = 2, layers: int = 2,
                 dropout: float = 0.5):
        super().__init__()
        self.convs   = torch.nn.ModuleList()
        self.dropout = dropout
        self.convs.append(SAGEConv(in_dim, hidden))
        for _ in range(layers - 2):
            self.convs.append(SAGEConv(hidden, hidden))
        self.convs.append(SAGEConv(hidden, out_dim))

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)


def compute_class_weight(labels: np.ndarray,
                          device: torch.device) -> torch.Tensor:
    """按训练集不平衡比例计算类别权重"""
    n_legit = (labels == 0).sum()
    n_fraud = max((labels == 1).sum(), 1)
    return torch.tensor(
        [1.0, n_legit / n_fraud], dtype=torch.float).to(device)


def split_train_val(idx: np.ndarray,
                    train_ratio: float = TRAIN_VAL_SPLIT,
                    seed: int = SEED_DEFAULT
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    按比例划分训练/验证索引（固定seed保证可复现）
    train_ratio=0.7 → 70%训练，30%验证
    剩余30%全部作验证（不丢弃任何节点）
    与蒸馏的70%训练集完全对齐
    """
    rng    = np.random.RandomState(seed)
    perm   = rng.permutation(len(idx))
    n_train = int(len(idx) * train_ratio)
    return idx[perm[:n_train]], idx[perm[n_train:]]


def make_fixed_institution_splits(
        institutions: Dict,
        cfg: ExperimentConfig) -> Dict[str, Dict[str, np.ndarray]]:
    """Return the canonical split attached by the dataset loader."""
    del cfg  # the loader has already fixed the protocol for this run
    missing = [name for name, inst in institutions.items()
               if 'split' not in inst]
    if missing:
        raise ValueError(
            f"Canonical institution splits are missing for: {missing}")
    return {
        name: {key: np.asarray(value, dtype=np.int64).copy()
               for key, value in inst['split'].items()}
        for name, inst in institutions.items()
    }


def induced_edge_index(edge_index: np.ndarray,
                       allowed_idx: np.ndarray,
                       num_nodes: int) -> np.ndarray:
    """Return edges whose two endpoints are both in the allowed node set."""
    edge_index = np.asarray(edge_index, dtype=np.int64)
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, num_edges]")
    allowed_idx = np.asarray(allowed_idx, dtype=np.int64)
    if allowed_idx.size == 0:
        return np.empty((2, 0), dtype=np.int64)
    if allowed_idx.min() < 0 or allowed_idx.max() >= num_nodes:
        raise ValueError("allowed node index is outside the feature matrix")
    allowed = np.zeros(num_nodes, dtype=bool)
    allowed[allowed_idx] = True
    keep = allowed[edge_index[0]] & allowed[edge_index[1]]
    return edge_index[:, keep]


def _institution_edge_index(inst: Dict,
                            global_edge_index: np.ndarray) -> np.ndarray:
    """Use a relation-specific institutional graph when one is available."""
    return np.asarray(inst.get('edge_index', global_edge_index), dtype=np.int64)


def train_gnn(feat: np.ndarray, edge_index: np.ndarray,
              labels: np.ndarray, tr_idx: np.ndarray,
              va_idx: np.ndarray, device: torch.device,
              cfg: ExperimentConfig = None) -> GraphSAGE:
    """在给定图上训练GNN，返回验证集AUC最优的模型"""
    if cfg is None:
        cfg = ExperimentConfig()

    model = GraphSAGE(feat.shape[1], cfg.hidden_dim, 2).to(device)
    opt   = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    x     = torch.tensor(feat,       dtype=torch.float).to(device)
    y     = torch.tensor(labels,     dtype=torch.long).to(device)
    tr_idx = np.asarray(tr_idx, dtype=np.int64)
    va_idx = np.asarray(va_idx, dtype=np.int64)
    if np.intersect1d(tr_idx, va_idx).size:
        raise ValueError("Training and model-validation indices overlap")

    # Gradient updates see only the train-induced graph. Model selection gets
    # a second forward on train+model-validation, never on calibration/test.
    train_edge = induced_edge_index(edge_index, tr_idx, len(feat))
    selection_nodes = np.union1d(tr_idx, va_idx)
    selection_edge = induced_edge_index(
        edge_index, selection_nodes, len(feat))
    train_ei = torch.tensor(train_edge, dtype=torch.long).to(device)
    selection_ei = torch.tensor(
        selection_edge, dtype=torch.long).to(device)
    # Compute imbalance weights from training labels only.  Using all labels
    # would leak the class prevalence of validation/test nodes into training.
    cw    = compute_class_weight(labels[tr_idx], device)

    best_auc, best_state = 0.0, None

    for epoch in range(1, cfg.gnn_epochs + 1):
        model.train()
        opt.zero_grad()
        F.cross_entropy(model(x, train_ei)[tr_idx],
                        y[tr_idx], weight=cw).backward()
        opt.step()

        if epoch % GNN_EVAL_INTERVAL == 0:
            model.eval()
            with torch.no_grad():
                probs = F.softmax(
                    model(x, selection_ei), dim=1)[va_idx, 1].cpu().numpy()
                auc = roc_auc_score(labels[va_idx], probs)
                if auc > best_auc:
                    best_auc   = auc
                    best_state = deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def select_validation_f1_threshold(labels_val: np.ndarray,
                                   probs_val: np.ndarray) -> float:
    """Select the fraud-F1 operating point using calibration data only."""
    from sklearn.metrics import precision_recall_curve

    labels_val = np.asarray(labels_val)
    probs_val = np.asarray(probs_val)
    if labels_val.shape != probs_val.shape or labels_val.size == 0:
        raise ValueError(
            "Calibration labels and probabilities must be non-empty and aligned")
    if np.unique(labels_val).size < 2:
        raise ValueError("Calibration data must contain both classes")

    precisions, recalls, thresholds = precision_recall_curve(
        labels_val, probs_val)
    if thresholds.size == 0:
        raise RuntimeError("Calibration produced no candidate threshold")
    f1s = 2.0 * precisions[:-1] * recalls[:-1] / (
        precisions[:-1] + recalls[:-1] + 1e-12)
    # Thresholds are ascending, so this deterministically resolves exact ties
    # in favor of the lower operating point (and therefore higher recall).
    return float(thresholds[int(np.nanargmax(f1s))])


@torch.no_grad()
def eval_gnn(model: GraphSAGE, feat: np.ndarray,
             edge_index: np.ndarray, labels: np.ndarray,
             test_idx: np.ndarray, device: torch.device,
             threshold: float = 0.5,
             val_idx: np.ndarray = None,
             threshold_mode: str = 'validation_f1',
             context_idx: np.ndarray = None) -> Dict:
    """
    Evaluate on the test set.

    threshold_mode='fixed':
        Use the supplied threshold. This mode is reserved for the appendix
        threshold-sensitivity sweep.
    threshold_mode='validation_f1':
        Select the threshold maximizing fraud-class F1 on validation data only,
        freeze it, and then apply it once to the test set. This is the
        main-table mode.

    Test labels are never used to select a threshold.
    """
    from sklearn.metrics import average_precision_score

    if threshold_mode not in {'fixed', 'validation_f1'}:
        raise ValueError(
            f"threshold_mode must be 'fixed' or 'validation_f1', "
            f"got {threshold_mode!r}")

    model.eval()
    x = torch.tensor(feat, dtype=torch.float).to(device)
    test_idx = np.asarray(test_idx, dtype=np.int64)
    if context_idx is None:
        context_idx = np.setdiff1d(
            np.arange(len(feat), dtype=np.int64), test_idx)
    context_idx = np.asarray(context_idx, dtype=np.int64)
    if np.intersect1d(context_idx, test_idx).size:
        raise ValueError("Evaluation context must exclude test nodes")

    selected_threshold = float(threshold)
    if threshold_mode == 'validation_f1':
        if val_idx is None or len(val_idx) == 0:
            raise ValueError(
                "val_idx is required when threshold_mode='validation_f1'")
        val_idx = np.asarray(val_idx, dtype=np.int64)
        if not np.all(np.isin(val_idx, context_idx)):
            raise ValueError(
                "Threshold-calibration nodes must be inside the non-test context")
        calibration_edge = induced_edge_index(
            edge_index, context_idx, len(feat))
        calibration_ei = torch.tensor(
            calibration_edge, dtype=torch.long).to(device)
        probs_val = F.softmax(
            model(x, calibration_ei), dim=1)[val_idx, 1].cpu().numpy()
        labels_val = labels[val_idx]
        selected_threshold = select_validation_f1_threshold(
            labels_val, probs_val)

    # Only after the operating threshold is frozen do test nodes enter a
    # forward pass. Their labels are used solely for the metrics below.
    deployment_nodes = np.union1d(context_idx, test_idx)
    deployment_edge = induced_edge_index(
        edge_index, deployment_nodes, len(feat))
    deployment_ei = torch.tensor(
        deployment_edge, dtype=torch.long).to(device)
    probs = F.softmax(
        model(x, deployment_ei), dim=1)[test_idx, 1].cpu().numpy()
    preds = (probs >= selected_threshold).astype(int)
    y = labels[test_idx]

    return {
        'auc': round(float(roc_auc_score(y, probs)), 4),
        'pr_auc': round(float(average_precision_score(y, probs)), 4),
        'f1_fraud': round(float(f1_score(
            y, preds, pos_label=1, average='binary', zero_division=0)), 4),
        'f1_macro': round(float(f1_score(
            y, preds, average='macro', zero_division=0)), 4),
        'precision': round(float(precision_score(
            y, preds, pos_label=1, zero_division=0)), 4),
        'recall': round(float(recall_score(
            y, preds, pos_label=1, zero_division=0)), 4),
        'threshold': round(selected_threshold, 4),
        'threshold_mode': threshold_mode,
    }


# ============================================================
# 3. 合成图工具函数
# ============================================================
def build_knn_edge(feat: np.ndarray,
                   k: int = KNN_NEIGHBORS) -> np.ndarray:
    """构建KNN图edge_index"""
    from sklearn.neighbors import kneighbors_graph
    k   = min(k, len(feat) - 1)
    knn = kneighbors_graph(
        feat, k, mode='connectivity', include_self=False).tocoo()
    return np.stack([knn.row, knn.col], axis=0)


def build_disjoint_knn_edge(feat_blocks: List[np.ndarray],
                            k: int = KNN_NEIGHBORS) -> np.ndarray:
    """Build one KNN graph per institution and return their block union."""
    edges = []
    offset = 0
    for feat in feat_blocks:
        feat = np.asarray(feat)
        if len(feat) > 1:
            edges.append(build_knn_edge(feat, k) + offset)
        offset += len(feat)
    return (np.hstack(edges) if edges
            else np.empty((2, 0), dtype=np.int64))


def sample_fraud_aware(feat: np.ndarray, labels: np.ndarray,
                        n_total: int,
                        fraud_ratio: float = FRAUD_OVERSAMPLE_RATIO
                        ) -> Tuple[np.ndarray, np.ndarray]:

    n_fraud = max(int(round(n_total * fraud_ratio)), 1)
    n_legit = n_total - n_fraud

    fi = np.where(labels == 1)[0]
    li = np.where(labels == 0)[0]

    if len(fi) == 0 or len(li) == 0:
        raise ValueError(
            "Fraud-aware sampling requires both classes in the train split")
    fs = np.random.choice(fi, n_fraud, replace=len(fi) < n_fraud)
    ls = np.random.choice(li, n_legit, replace=len(li) < n_legit)

    syn_feat   = np.vstack([feat[fs], feat[ls]]).astype(np.float32)
    syn_labels = np.array([1] * len(fs) + [0] * len(ls))
    return syn_feat, syn_labels


def load_synthetic_graph(
        inst_name: str, dataset: str,
        reduction_rate: float, seed: int,
        tag: str,
        artifact_kind: str = 'fiad') -> Tuple[Optional[np.ndarray], Optional[str]]:

    base = (f'saved_ours/feat_{dataset}_inst{inst_name}_'
            f'{reduction_rate}_{seed}_{tag}')

    # 优先兼容 FIAD 和 GCond 两种保存后缀
    if artifact_kind == 'fiad':
        candidates = [f'{base}_fiad.pt']
    elif artifact_kind == 'gcond':
        candidates = [f'{base}_gcond.pt']
    elif artifact_kind == 'sfgc':
        candidates = [f'{base}_fiad_sfgc.pt',
                      f'{base}_gcond_sfgc.pt',
                      f'{base}_sfgc.pt']
    else:
        raise ValueError(f"Unknown synthetic artifact kind: {artifact_kind}")

    for candidate in candidates:
        if os.path.exists(candidate):
            tensor = torch.load(candidate, map_location='cpu')
            return tensor.detach().cpu().numpy(), candidate
    return None, None


def make_syn_labels(n_syn: int,
                    train_labels: np.ndarray,
                    reduction_rate: float,
                    fraud_ratio: float = 0.0) -> np.ndarray:

    if fraud_ratio > 0.0:
        n_fraud = int(n_syn * fraud_ratio)
    else:
        n_fraud_real = int((np.asarray(train_labels) == 1).sum())
        n_fraud = int(n_fraud_real * reduction_rate)
    n_fraud = max(n_fraud, 1)
    n_fraud = min(n_fraud, n_syn - 1)
    return np.array([1] * n_fraud + [0] * (n_syn - n_fraud))


def _load_synthetic(institutions: Dict, reduction_rate: float,
                    in_dim: int, dataset: str = 'elliptic',
                    tag: str = DISTILL_TAG,
                    artifact_kind: str = 'fiad',
                    cfg: ExperimentConfig = None,
                    strict: bool = True) -> Tuple[Optional[List], Optional[List]]:

    if cfg is None:
        cfg = ExperimentConfig()

    all_feat, all_labels = [], []

    for name, inst in institutions.items():
        syn_feat, feat_path = load_synthetic_graph(
            name, dataset, reduction_rate,
            cfg.seed, tag, artifact_kind)

        if syn_feat is None:
            message = (
                f"Institution {name}: {artifact_kind} artifact missing "
                f"(tag={tag}, seed={cfg.seed}, rr={reduction_rate})")
            if strict:
                logger.warning(message + "; skipping this scenario.")
                return None, None
            raise FileNotFoundError(message)

        split = inst.get('split')
        if split is None:
            raise ValueError(
                f"Institution {name}: canonical split missing before artifact load")
        train_idx = np.asarray(split['train'], dtype=np.int64)
        # inst['labels'] is aligned with inst['idx']; map canonical global IDs
        # explicitly so no broader label array can accidentally affect counts.
        pos = {int(node): i for i, node in enumerate(inst['idx'])}
        local_train = np.array([pos[int(node)] for node in train_idx])
        train_labels = np.asarray(inst['labels'])[local_train]

        expected_n = int(len(train_idx) * reduction_rate)
        if len(syn_feat) != expected_n:
            raise ValueError(
                f"Institution {name}: artifact has {len(syn_feat)} nodes, "
                f"but canonical train split implies {expected_n}. "
                "Regenerate the artifact with the canonical split.")
        if syn_feat.ndim != 2 or syn_feat.shape[1] != in_dim:
            raise ValueError(
                f"Institution {name}: artifact feature shape {syn_feat.shape} "
                f"does not match expected (*, {in_dim})")

        # Map a locally standardized synthetic artifact into the common
        # train-only evaluation coordinate system.
        if all(key in inst for key in (
                'syn_source_mean', 'syn_source_scale',
                'eval_target_mean', 'eval_target_scale')):
            source_scale = np.asarray(inst['syn_source_scale'])
            target_scale = np.asarray(inst['eval_target_scale'])
            source_scale = np.where(source_scale == 0, 1.0, source_scale)
            target_scale = np.where(target_scale == 0, 1.0, target_scale)
            base_feat = (np.asarray(syn_feat) * source_scale
                         + np.asarray(inst['syn_source_mean']))
            syn_feat = ((base_feat - np.asarray(inst['eval_target_mean']))
                        / target_scale).astype(np.float32)

        expected_labels = make_syn_labels(
            len(syn_feat), train_labels, reduction_rate,
            fraud_ratio=cfg.fraud_oversample_ratio)
        label_path = feat_path.replace('feat_', 'label_', 1)
        if os.path.exists(label_path):
            syn_labels = torch.load(
                label_path, map_location='cpu').detach().cpu().numpy()
            label_source = 'saved'
        else:
            syn_labels = expected_labels
            label_source = 'canonical-train reconstruction'

        syn_labels = np.asarray(syn_labels, dtype=np.int64)
        if len(syn_labels) != len(syn_feat):
            raise ValueError(
                f"Institution {name}: feature/label artifact length mismatch")
        if not set(np.unique(syn_labels)).issubset({0, 1}):
            raise ValueError(
                f"Institution {name}: synthetic labels are not binary")
        if not np.array_equal(syn_labels, expected_labels):
            raise ValueError(
                f"Institution {name}: saved synthetic labels do not match "
                "the canonical training split and generator class-count rule")
        logger.info(
            f"  Institution {name}: loaded {artifact_kind} "
            f"({len(syn_feat)} nodes, fraud={(syn_labels == 1).sum()}, "
            f"labels={label_source})")

        all_feat.append(syn_feat)
        all_labels.append(syn_labels)

    return all_feat, all_labels


def _run_compression_scenario(
        name: str, institutions: Dict, global_data: Dict,
        device: torch.device,
        syn_feats: List[np.ndarray],
        syn_labels_list: List[np.ndarray],
        cfg: ExperimentConfig = None) -> Dict:
 
    if cfg is None:
        cfg = ExperimentConfig()

    feat_all  = global_data['feat']
    label_all = global_data['labels']
    edge_all  = global_data['edge_index']
    test_idx  = global_data['test_idx']
    n_real    = len(feat_all)

    # 合成图总体统计
    combined_feat   = np.vstack(syn_feats).astype(np.float32)
    combined_labels = np.concatenate(syn_labels_list)
    n_syn           = len(combined_feat)
    n_f = (combined_labels == 1).sum()
    logger.info(
        f"  Synthetic: {n_syn} nodes total "
        f"(fraud={n_f}, legit={n_syn-n_f})")

    # 合成节点追加到真实图后面（KNN边偏移）
    joint_feat   = np.vstack([feat_all, combined_feat]).astype(np.float32)
    joint_labels = np.concatenate([label_all, combined_labels])
    syn_edge_shifted = (
        build_disjoint_knn_edge(syn_feats, KNN_NEIGHBORS) + n_real)

    # 每个机构：本地真实训练集 + 其他机构合成节点
    inst_names  = list(institutions.keys())
    n_inst      = len(inst_names)
    inst_results = {}

    for i, (inst_name, inst) in enumerate(institutions.items()):
        # 本机构真实训练集
        split = inst['split']
        tr_real = split['train']
        model_val_real = split['model_val']
        calibration_real = split['calibration']

        # 其他机构的合成节点（排除本机构自己的合成图）
        other_syn_feats  = [syn_feats[j]       for j in range(n_inst) if j != i]
        other_syn_labels = [syn_labels_list[j] for j in range(n_inst) if j != i]

        if other_syn_feats:
            other_syn_idx = np.concatenate([
                np.arange(n_real + sum(len(syn_feats[j]) for j in range(j2)),
                          n_real + sum(len(syn_feats[j]) for j in range(j2+1)))
                for j2 in range(n_inst) if j2 != i
            ])
        else:
            other_syn_idx = np.array([], dtype=np.int64)

        # 训练集 = 本机构真实 + 其他机构合成
        all_tr = np.concatenate([tr_real, other_syn_idx])
        allowed_history = np.concatenate([inst['idx'], other_syn_idx])
        local_real_edge = _institution_edge_index(inst, edge_all)
        joint_edge = np.hstack([local_real_edge, syn_edge_shifted])

        logger.info(f"  Institution {inst_name}: "
                    f"real_train={len(tr_real)}, "
                    f"other_syn={len(other_syn_idx)}, "
                    f"total={len(all_tr)}")

        model = train_gnn(joint_feat, joint_edge, joint_labels,
                          all_tr, model_val_real, device, cfg)
        m = eval_gnn(model, joint_feat, joint_edge,
                     joint_labels, test_idx, device,
                     val_idx=calibration_real,
                     threshold_mode='validation_f1',
                     context_idx=allowed_history)
        inst_results[inst_name] = m
        logger.info(f"  {inst_name}: AUC={m['auc']:.4f}  "
                    f"F1={m['f1_fraud']:.4f}  "
                    f"Recall={m['recall']:.4f}")

    # 取各机构结果的均值作为S4-V1最终结果
    avg_keys = ['auc', 'pr_auc', 'f1_fraud', 'f1_macro', 'precision', 'recall']
    avg = {k: round(float(np.mean([r[k] for r in inst_results.values()])), 4)
           for k in avg_keys}
    avg['threshold'] = round(float(np.mean(
        [r['threshold'] for r in inst_results.values()])), 4)
    avg['threshold_std'] = round(float(np.std(
        [r['threshold'] for r in inst_results.values()])), 4)
    avg['threshold_mode'] = 'validation_selected_per_institution'
    logger.info(f"\n  {name} Average: AUC={avg['auc']:.4f}  "
                f"F1={avg['f1_fraud']:.4f}")
    return avg


# ============================================================
# 4. S1: 孤立训练
# ============================================================
def scenario_isolated(institutions: Dict, global_data: Dict,
                       device: torch.device,
                       cfg: ExperimentConfig = None
                       ) -> Tuple[Dict, Dict]:
    if cfg is None:
        cfg = ExperimentConfig()

    logger.info("\n" + "="*60)
    logger.info("S1: Isolated Training (No Collaboration)")
    logger.info("="*60)

    feat_all  = global_data['feat']
    label_all = global_data['labels']
    edge_all  = global_data['edge_index']
    test_idx  = global_data['test_idx']

    results = {}
    for name, inst in institutions.items():
        idx          = inst['idx']
        split = inst['split']
        tr_idx = split['train']
        model_val_idx = split['model_val']
        calibration_idx = split['calibration']
        local_edge = _institution_edge_index(inst, edge_all)

        logger.info(f"\n  Training Institution {name} ({len(idx):,} nodes)...")
        model = train_gnn(feat_all, local_edge, label_all,
                          tr_idx, model_val_idx, device, cfg)
        m     = eval_gnn(model, feat_all, local_edge,
                         label_all, test_idx, device,
                         val_idx=calibration_idx,
                         threshold_mode='validation_f1',
                         context_idx=idx)
        results[name] = m
        logger.info(
            f"  {name}: AUC={m['auc']:.4f}  "
            f"F1-fraud={m['f1_fraud']:.4f}  "
            f"Recall={m['recall']:.4f}")

    avg = {k: round(float(np.mean([r[k] for r in results.values()])), 4)
           for k in ['auc', 'pr_auc', 'f1_fraud', 'f1_macro',
                     'precision', 'recall']}
    avg['threshold'] = round(float(np.mean(
        [r['threshold'] for r in results.values()])), 4)
    avg['threshold_std'] = round(float(np.std(
        [r['threshold'] for r in results.values()])), 4)
    avg['threshold_mode'] = 'validation_selected_per_institution'
    logger.info(
        f"\n  Average: AUC={avg['auc']:.4f}  "
        f"F1-fraud={avg['f1_fraud']:.4f}")
    return results, avg


# ============================================================
# 5. S2: 共享原始数据（上界）
# ============================================================
def scenario_shared_data(institutions: Dict, global_data: Dict,
                          device: torch.device,
                          cfg: ExperimentConfig = None) -> Dict:
    if cfg is None:
        cfg = ExperimentConfig()

    logger.info("\n" + "="*60)
    logger.info("S2: Shared Raw Data (Upper Bound, Not Privacy-Safe)")
    logger.info("="*60)

    feat_all  = global_data['feat']
    label_all = global_data['labels']
    edge_all  = global_data['edge_index']
    test_idx  = global_data['test_idx']

    all_idx = np.concatenate([i['idx'] for i in institutions.values()])
    tr_idx = np.concatenate([
        inst['split']['train'] for inst in institutions.values()])
    model_val_idx = np.concatenate([
        inst['split']['model_val'] for inst in institutions.values()])
    calibration_idx = np.concatenate([
        inst['split']['calibration'] for inst in institutions.values()])

    logger.info(f"  Combined training: {len(tr_idx):,} nodes")
    model = train_gnn(feat_all, edge_all, label_all,
                      tr_idx, model_val_idx, device, cfg)
    m     = eval_gnn(model, feat_all, edge_all,
                     label_all, test_idx, device,
                     val_idx=calibration_idx,
                     threshold_mode='validation_f1',
                     context_idx=all_idx)
    logger.info(
        f"  AUC={m['auc']:.4f}  F1-fraud={m['f1_fraud']:.4f}  "
        f"Recall={m['recall']:.4f}")
    return m


# ============================================================
# 6. S3: 联邦学习（FedAvg）
# ============================================================
def scenario_federated(institutions: Dict, global_data: Dict,
                        device: torch.device,
                        cfg: ExperimentConfig = None) -> Dict:
    if cfg is None:
        cfg = ExperimentConfig()

    logger.info("\n" + "="*60)
    logger.info("S3: Federated Learning (FedAvg)")
    logger.info("="*60)

    feat_all  = global_data['feat']
    label_all = global_data['labels']
    edge_all  = global_data['edge_index']
    test_idx  = global_data['test_idx']

    x  = torch.tensor(feat_all,  dtype=torch.float).to(device)
    y  = torch.tensor(label_all, dtype=torch.long).to(device)
    fixed_splits = make_fixed_institution_splits(institutions, cfg)
    all_train_idx = np.concatenate(
        [fixed_splits[name]['train'] for name in institutions])
    all_calibration_idx = np.concatenate(
        [fixed_splits[name]['calibration'] for name in institutions])
    history_idx = np.concatenate([
        inst['idx'] for inst in institutions.values()])
    local_train_ei = {}
    local_cw = {}
    for name, inst in institutions.items():
        tr_idx = fixed_splits[name]['train']
        local_edge = _institution_edge_index(inst, edge_all)
        local_train_ei[name] = torch.tensor(
            induced_edge_index(local_edge, tr_idx, len(feat_all)),
            dtype=torch.long, device=device)
        local_cw[name] = compute_class_weight(label_all[tr_idx], device)

    global_model = GraphSAGE(feat_all.shape[1], cfg.hidden_dim, 2).to(device)

    for rnd in range(1, cfg.fed_rounds + 1):
        local_params, local_sizes = [], []

        for name, inst in institutions.items():
            tr_idx = fixed_splits[name]['train']
            lm  = deepcopy(global_model)
            opt = torch.optim.Adam(lm.parameters(),
                                    lr=cfg.lr,
                                    weight_decay=cfg.weight_decay)
            lm.train()
            for _ in range(cfg.fed_local_epochs):
                opt.zero_grad()
                F.cross_entropy(lm(x, local_train_ei[name])[tr_idx],
                                y[tr_idx], weight=local_cw[name]).backward()
                opt.step()
            local_params.append(
                {k: v.clone() for k, v in lm.state_dict().items()})
            local_sizes.append(len(tr_idx))

        # FedAvg聚合
        total = sum(local_sizes)
        agg   = {
            k: sum(local_params[i][k] * (local_sizes[i] / total)
                   for i in range(len(local_params)))
            for k in global_model.state_dict()
        }
        global_model.load_state_dict(agg)

        if rnd % 5 == 0 or rnd == 1:
            logger.info(f"  Round {rnd:2d}: training completed")

    m = eval_gnn(global_model, feat_all, edge_all,
                 label_all, test_idx, device,
                 val_idx=all_calibration_idx,
                 threshold_mode='validation_f1',
                 context_idx=history_idx)
    logger.info(
        f"\n  Final: AUC={m['auc']:.4f}  "
        f"F1-fraud={m['f1_fraud']:.4f}  "
        f"Recall={m['recall']:.4f}")
    return m


# ============================================================
# 6b. S3b: Federated Learning (FedProx)
# ============================================================
FEDPROX_MU = 0.01  # proximal term weight

def scenario_fedprox(institutions: Dict, global_data: Dict,
                     device: torch.device,
                     cfg: ExperimentConfig = None) -> Dict:
    if cfg is None:
        cfg = ExperimentConfig()

    logger.info("\n" + "="*60)
    logger.info("S3b: Federated Learning (FedProx)")
    logger.info("="*60)

    feat_all  = global_data['feat']
    label_all = global_data['labels']
    edge_all  = global_data['edge_index']
    test_idx  = global_data['test_idx']

    x  = torch.tensor(feat_all,  dtype=torch.float).to(device)
    y  = torch.tensor(label_all, dtype=torch.long).to(device)
    fixed_splits = make_fixed_institution_splits(institutions, cfg)
    all_train_idx = np.concatenate(
        [fixed_splits[name]['train'] for name in institutions])
    all_calibration_idx = np.concatenate(
        [fixed_splits[name]['calibration'] for name in institutions])
    history_idx = np.concatenate([
        inst['idx'] for inst in institutions.values()])
    local_train_ei = {}
    local_cw = {}
    for name, inst in institutions.items():
        tr_idx = fixed_splits[name]['train']
        local_edge = _institution_edge_index(inst, edge_all)
        local_train_ei[name] = torch.tensor(
            induced_edge_index(local_edge, tr_idx, len(feat_all)),
            dtype=torch.long, device=device)
        local_cw[name] = compute_class_weight(label_all[tr_idx], device)

    global_model = GraphSAGE(feat_all.shape[1], cfg.hidden_dim, 2).to(device)

    for rnd in range(1, cfg.fed_rounds + 1):
        local_params, local_sizes = [], []

        for name, inst in institutions.items():
            tr_idx = fixed_splits[name]['train']
            lm  = deepcopy(global_model)
            opt = torch.optim.Adam(lm.parameters(),
                                    lr=cfg.lr,
                                    weight_decay=cfg.weight_decay)

            # 保存全局参数用于 proximal term
            global_params = {k: v.clone()
                             for k, v in global_model.state_dict().items()}

            lm.train()
            for _ in range(cfg.fed_local_epochs):
                opt.zero_grad()
                loss = F.cross_entropy(
                    lm(x, local_train_ei[name])[tr_idx],
                    y[tr_idx], weight=local_cw[name])
                # FedProx proximal term: mu/2 * ||w - w_global||^2
                prox = 0.0
                for k, param in lm.named_parameters():
                    prox += ((param - global_params[k]) ** 2).sum()
                loss = loss + (FEDPROX_MU / 2.0) * prox
                loss.backward()
                opt.step()

            local_params.append(
                {k: v.clone() for k, v in lm.state_dict().items()})
            local_sizes.append(len(tr_idx))

        # FedAvg聚合（FedProx的聚合方式与FedAvg相同）
        total = sum(local_sizes)
        agg   = {
            k: sum(local_params[i][k] * (local_sizes[i] / total)
                   for i in range(len(local_params)))
            for k in global_model.state_dict()
        }
        global_model.load_state_dict(agg)

        if rnd % 5 == 0 or rnd == 1:
            logger.info(f"  Round {rnd:2d}: training completed")

    m = eval_gnn(global_model, feat_all, edge_all,
                 label_all, test_idx, device,
                 val_idx=all_calibration_idx,
                 threshold_mode='validation_f1',
                 context_idx=history_idx)
    logger.info(
        f"\n  Final: AUC={m['auc']:.4f}  "
        f"F1-fraud={m['f1_fraud']:.4f}  "
        f"Recall={m['recall']:.4f}")
    return m


# ============================================================
# 6c. S3c: DP-FedAvg (Federated Learning with Differential Privacy)
# ============================================================
DP_CLIP_NORM      = 1.0   # gradient clipping bound C
DP_NOISE_LOW      = 0.5   # noise multiplier σ (relaxed privacy)
DP_NOISE_HIGH     = 1.0   # noise multiplier σ (strict privacy)


def scenario_dp_fedavg(institutions: Dict, global_data: Dict,
                       device: torch.device,
                       noise_multiplier: float = DP_NOISE_LOW,
                       cfg: ExperimentConfig = None) -> Dict:
    """
    DP-FedAvg: FedAvg + DP-SGD (gradient clipping + Gaussian noise)
    用于验证：给FL加DP后性能下降多少？
    如果下降显著 → 蒸馏是更好的 privacy-utility tradeoff
    """
    if cfg is None:
        cfg = ExperimentConfig()

    label = f"σ={noise_multiplier}"
    logger.info("\n" + "="*60)
    logger.info(f"S3c: DP-FedAvg ({label}, clip={DP_CLIP_NORM})")
    logger.info("="*60)

    feat_all  = global_data['feat']
    label_all = global_data['labels']
    edge_all  = global_data['edge_index']
    test_idx  = global_data['test_idx']

    x  = torch.tensor(feat_all,  dtype=torch.float).to(device)
    y  = torch.tensor(label_all, dtype=torch.long).to(device)
    fixed_splits = make_fixed_institution_splits(institutions, cfg)
    all_train_idx = np.concatenate(
        [fixed_splits[name]['train'] for name in institutions])
    all_calibration_idx = np.concatenate(
        [fixed_splits[name]['calibration'] for name in institutions])
    history_idx = np.concatenate([
        inst['idx'] for inst in institutions.values()])
    local_train_ei = {}
    local_cw = {}
    for name, inst in institutions.items():
        tr_idx = fixed_splits[name]['train']
        local_edge = _institution_edge_index(inst, edge_all)
        local_train_ei[name] = torch.tensor(
            induced_edge_index(local_edge, tr_idx, len(feat_all)),
            dtype=torch.long, device=device)
        local_cw[name] = compute_class_weight(label_all[tr_idx], device)

    global_model = GraphSAGE(feat_all.shape[1], cfg.hidden_dim, 2).to(device)

    for rnd in range(1, cfg.fed_rounds + 1):
        local_params, local_sizes = [], []

        for name, inst in institutions.items():
            tr_idx = fixed_splits[name]['train']
            lm  = deepcopy(global_model)
            opt = torch.optim.Adam(lm.parameters(),
                                    lr=cfg.lr,
                                    weight_decay=cfg.weight_decay)
            lm.train()
            for _ in range(cfg.fed_local_epochs):
                opt.zero_grad()
                F.cross_entropy(lm(x, local_train_ei[name])[tr_idx],
                                y[tr_idx], weight=local_cw[name]).backward()
                # DP-SGD: clip gradients + add Gaussian noise
                torch.nn.utils.clip_grad_norm_(
                    lm.parameters(), max_norm=DP_CLIP_NORM)
                with torch.no_grad():
                    for param in lm.parameters():
                        if param.grad is not None:
                            param.grad += torch.randn_like(param.grad) * \
                                          (noise_multiplier * DP_CLIP_NORM)
                opt.step()

            local_params.append(
                {k: v.clone() for k, v in lm.state_dict().items()})
            local_sizes.append(len(tr_idx))

        total = sum(local_sizes)
        agg   = {
            k: sum(local_params[i][k] * (local_sizes[i] / total)
                   for i in range(len(local_params)))
            for k in global_model.state_dict()
        }
        global_model.load_state_dict(agg)

        if rnd % 5 == 0 or rnd == 1:
            logger.info(f"  Round {rnd:2d}: training completed")

    m = eval_gnn(global_model, feat_all, edge_all,
                 label_all, test_idx, device,
                 val_idx=all_calibration_idx,
                 threshold_mode='validation_f1',
                 context_idx=history_idx)
    logger.info(
        f"\n  Final ({label}): AUC={m['auc']:.4f}  "
        f"F1-fraud={m['f1_fraud']:.4f}  "
        f"Recall={m['recall']:.4f}")
    return m


# ============================================================
# 7. S4-V1: FIAD Joint Training
# ============================================================
def scenario_distill_v1_joint(institutions: Dict, global_data: Dict,
                               device: torch.device,
                               cfg: ExperimentConfig = None) -> Dict:
    if cfg is None:
        cfg = ExperimentConfig()

    logger.info("\n" + "="*60)
    logger.info("S4-V1: FIAD Joint Training (Synthetic + Real)")
    logger.info("="*60)

    in_dim   = global_data['feat'].shape[1]
    dataset  = global_data.get('dataset', 'elliptic')

    all_syn_feat, all_syn_labels = _load_synthetic(
        institutions, cfg.reduction_rate, in_dim,
        dataset=dataset, tag=cfg.distill_tag,
        artifact_kind='fiad', cfg=cfg,
        strict=True)

    if all_syn_feat is None:
        logger.warning("  S4-V1 skipped: FIAD distilled files not found.")
        return None

    return _run_compression_scenario(
        "S4-V1", institutions, global_data, device,
        all_syn_feat, all_syn_labels, cfg)


# ============================================================
# 8. S4-V2: FIAD Pretrain + FedAvg
# ============================================================
def scenario_distill_v2_pretrain_fed(
        institutions: Dict, global_data: Dict,
        device: torch.device,
        cfg: ExperimentConfig = None) -> Dict:
    if cfg is None:
        cfg = ExperimentConfig()

    logger.info("\n" + "="*60)
    logger.info("S4-V2: FIAD Pretrain + FedAvg")
    logger.info("="*60)

    feat_all  = global_data['feat']
    label_all = global_data['labels']
    edge_all  = global_data['edge_index']
    test_idx  = global_data['test_idx']
    in_dim    = feat_all.shape[1]
    dataset   = global_data.get('dataset', 'elliptic')

    # Step1: 合成图预训练
    all_syn_feat, all_syn_labels = _load_synthetic(
        institutions, cfg.reduction_rate, in_dim,
        dataset=dataset, tag=cfg.distill_tag,
        artifact_kind='fiad', cfg=cfg,
        strict=True)

    if all_syn_feat is None:
        logger.warning("  S4-V2 skipped: FIAD distilled files not found.")
        return None

    combined_feat   = np.vstack(all_syn_feat).astype(np.float32)
    combined_labels = np.concatenate(all_syn_labels)
    n_syn           = len(combined_feat)

    # 合成图KNN边
    syn_edge = build_disjoint_knn_edge(
        all_syn_feat, KNN_NEIGHBORS)

    # One strict real-data holdout is used for all fine-tuning rounds and for
    # selecting the deployment threshold.
    fixed_splits = make_fixed_institution_splits(institutions, cfg)
    all_train_idx = np.concatenate(
        [fixed_splits[name]['train'] for name in institutions])
    all_calibration_idx = np.concatenate(
        [fixed_splits[name]['calibration'] for name in institutions])
    history_idx = np.concatenate([
        inst['idx'] for inst in institutions.values()])

    # Step1: 在合成图上预训练（只用合成数据，验证集也用合成图内部）
    syn_tr_idx, syn_va_idx = split_train_val(
        np.arange(n_syn), cfg.train_val_split, seed=cfg.seed)

    logger.info("  Step1: Pretraining on synthetic graph only...")
    pretrained = train_gnn(combined_feat, syn_edge, combined_labels,
                           syn_tr_idx, syn_va_idx, device,
                           ExperimentConfig(gnn_epochs=200))

    # Step2+3: 各机构fine-tune + FedAvg聚合
    logger.info("  Step2: Local fine-tuning + FedAvg aggregation...")
    x_real  = torch.tensor(feat_all,  dtype=torch.float).to(device)
    y_real  = torch.tensor(label_all, dtype=torch.long).to(device)
    local_train_ei = {}
    local_cw = {}
    for name, inst in institutions.items():
        tr_idx = fixed_splits[name]['train']
        local_edge = _institution_edge_index(inst, edge_all)
        local_train_ei[name] = torch.tensor(
            induced_edge_index(local_edge, tr_idx, len(feat_all)),
            dtype=torch.long, device=device)
        local_cw[name] = compute_class_weight(label_all[tr_idx], device)

    global_model = deepcopy(pretrained)
    fed_rounds   = 5  # 预训练后只需少量轮次

    for rnd in range(1, fed_rounds + 1):
        local_params, local_sizes = [], []
        for name, inst in institutions.items():
            tr_idx = fixed_splits[name]['train']
            lm  = deepcopy(global_model)
            opt = torch.optim.Adam(lm.parameters(),
                                    lr=cfg.lr * 0.5,
                                    weight_decay=cfg.weight_decay)
            lm.train()
            for _ in range(cfg.fed_local_epochs):
                opt.zero_grad()
                F.cross_entropy(
                    lm(x_real, local_train_ei[name])[tr_idx],
                    y_real[tr_idx], weight=local_cw[name]).backward()
                opt.step()
            local_params.append(
                {k: v.clone() for k, v in lm.state_dict().items()})
            local_sizes.append(len(tr_idx))

        total = sum(local_sizes)
        agg   = {
            k: sum(local_params[i][k] * (local_sizes[i] / total)
                   for i in range(len(local_params)))
            for k in global_model.state_dict()
        }
        global_model.load_state_dict(agg)
        logger.info(f"  Fine-tune round {rnd}: training completed")

    m = eval_gnn(global_model, feat_all, edge_all,
                 label_all, test_idx, device,
                 val_idx=all_calibration_idx,
                 threshold_mode='validation_f1',
                 context_idx=history_idx)
    logger.info(
        f"\n  V2 Result: AUC={m['auc']:.4f}  "
        f"F1-fraud={m['f1_fraud']:.4f}  "
        f"Recall={m['recall']:.4f}")
    return m


# ============================================================
# 8b. Vanilla GCond S4-V2 (ablation baseline)
# ============================================================
GCOND_VANILLA_TAG = 'uw0_ltnll_ov0.0'

def scenario_distill_v2_gcond_vanilla(
        institutions: Dict, global_data: Dict,
        device: torch.device,
        cfg: ExperimentConfig = None) -> Dict:
    """
    与S4-V2相同的pipeline，但使用vanilla GCond蒸馏图（原始coeff + NLL + 无align）
    用于ablation：证明FIAD修复（coeff fix + focal loss + centroid init）的贡献
    """
    if cfg is None:
        cfg = ExperimentConfig()

    logger.info("\n" + "="*60)
    logger.info("S4-V2-GCond: Vanilla GCond Pretrain + FedAvg")
    logger.info("="*60)

    feat_all  = global_data['feat']
    label_all = global_data['labels']
    edge_all  = global_data['edge_index']
    test_idx  = global_data['test_idx']
    in_dim    = feat_all.shape[1]
    dataset   = global_data.get('dataset', 'elliptic')

    # 加载vanilla GCond蒸馏图（strict=True: 文件不存在则跳过，不fallback）
    all_syn_feat, all_syn_labels = _load_synthetic(
        institutions, cfg.reduction_rate, in_dim,
        dataset=dataset, tag=GCOND_VANILLA_TAG,
        artifact_kind='gcond', cfg=cfg,
        strict=True)

    if all_syn_feat is None:
        logger.info("  Skipped: GCond distilled files not found for this config.")
        return None

    combined_feat   = np.vstack(all_syn_feat).astype(np.float32)
    combined_labels = np.concatenate(all_syn_labels)
    n_syn           = len(combined_feat)

    syn_edge = build_disjoint_knn_edge(
        all_syn_feat, KNN_NEIGHBORS)

    fixed_splits = make_fixed_institution_splits(institutions, cfg)
    all_train_idx = np.concatenate(
        [fixed_splits[name]['train'] for name in institutions])
    all_calibration_idx = np.concatenate(
        [fixed_splits[name]['calibration'] for name in institutions])
    history_idx = np.concatenate([
        inst['idx'] for inst in institutions.values()])

    # Pretrain on synthetic
    syn_tr_idx, syn_va_idx = split_train_val(
        np.arange(n_syn), cfg.train_val_split, seed=cfg.seed)
    pretrained = train_gnn(combined_feat, syn_edge, combined_labels,
                           syn_tr_idx, syn_va_idx, device,
                           ExperimentConfig(gnn_epochs=200))

    # Fine-tune + FedAvg（与S4-V2完全相同）
    x_real  = torch.tensor(feat_all,  dtype=torch.float).to(device)
    y_real  = torch.tensor(label_all, dtype=torch.long).to(device)
    local_train_ei = {}
    local_cw = {}
    for name, inst in institutions.items():
        tr_idx = fixed_splits[name]['train']
        local_edge = _institution_edge_index(inst, edge_all)
        local_train_ei[name] = torch.tensor(
            induced_edge_index(local_edge, tr_idx, len(feat_all)),
            dtype=torch.long, device=device)
        local_cw[name] = compute_class_weight(label_all[tr_idx], device)

    global_model = deepcopy(pretrained)

    for rnd in range(1, 6):
        local_params, local_sizes = [], []
        for name, inst in institutions.items():
            tr_idx = fixed_splits[name]['train']
            lm  = deepcopy(global_model)
            opt = torch.optim.Adam(lm.parameters(),
                                    lr=cfg.lr * 0.5,
                                    weight_decay=cfg.weight_decay)
            lm.train()
            for _ in range(cfg.fed_local_epochs):
                opt.zero_grad()
                F.cross_entropy(
                    lm(x_real, local_train_ei[name])[tr_idx],
                    y_real[tr_idx], weight=local_cw[name]).backward()
                opt.step()
            local_params.append(
                {k: v.clone() for k, v in lm.state_dict().items()})
            local_sizes.append(len(tr_idx))

        total = sum(local_sizes)
        agg   = {
            k: sum(local_params[i][k] * (local_sizes[i] / total)
                   for i in range(len(local_params)))
            for k in global_model.state_dict()
        }
        global_model.load_state_dict(agg)

    m = eval_gnn(global_model, feat_all, edge_all,
                 label_all, test_idx, device,
                 val_idx=all_calibration_idx,
                 threshold_mode='validation_f1',
                 context_idx=history_idx)
    logger.info(
        f"\n  GCond-V2 Result: AUC={m['auc']:.4f}  "
        f"F1-fraud={m['f1_fraud']:.4f}  "
        f"Recall={m['recall']:.4f}")
    return m


# ============================================================
# 8c. Communication Cost Analysis
# ============================================================

def communication_cost_analysis(institutions: Dict, global_data: Dict,
                                cfg: ExperimentConfig = None):
    """
    计算各协作方式的通信成本（单位：bytes）
    返回dict并打印对比表
    """
    if cfg is None:
        cfg = ExperimentConfig()

    feat_all = global_data['feat']
    n_total  = feat_all.shape[0]
    d        = feat_all.shape[1]  # feature dimension

    # GraphSAGE model size: 2-layer SAGEConv
    # SAGEConv(in, out) has lin_l + lin_r = 2 * (in * out + out)
    # conv1: SAGEConv(d, hidden) = 2 * hidden * (d + 1)
    # conv2: SAGEConv(hidden, 2) = 2 * 2 * (hidden + 1)
    hidden = cfg.hidden_dim
    model_params = sum(
        p.numel() for p in GraphSAGE(d, hidden, 2).parameters())
    bytes_per_param = 4  # float32
    bytes_per_label = 1  # binary label serialized as uint8
    bytes_per_edge_endpoint = 8  # int64 edge_index

    costs = {}

    # S2: Share raw data — each institution sends all training features + labels
    total_train = sum(
        len(inst['split']['train']) for inst in institutions.values())
    raw_edge_count = 0
    for inst in institutions.values():
        train_idx = inst['split']['train']
        local_edge = _institution_edge_index(
            inst, global_data['edge_index'])
        raw_edge_count += induced_edge_index(
            local_edge, train_idx, len(feat_all)).shape[1]
    s2_bytes = (
        total_train * (d * bytes_per_param + bytes_per_label)
        + raw_edge_count * 2 * bytes_per_edge_endpoint)
    costs['S2: Raw data'] = s2_bytes

    # S3: FedAvg — each round, K institutions upload model, server sends back 1 model
    # Total = rounds * (K upload + 1 broadcast) * model_size
    K = len(institutions)
    s3_bytes = cfg.fed_rounds * (K + 1) * model_params * bytes_per_param
    costs['S3: FedAvg'] = s3_bytes

    # S3b: FedProx — same communication as FedAvg
    costs['S3b: FedProx'] = s3_bytes

    # S3c: DP-FedAvg — same communication as FedAvg (noise is local)
    costs['S3c: DP-FedAvg'] = s3_bytes

    # C1-C3 / S4-V2: Share compressed/synthetic graphs
    total_syn = sum(
        max(int(len(inst['split']['train']) * cfg.reduction_rate), 2)
        for inst in institutions.values())
    # Synthetic adjacency is reconstructed locally via per-institution KNN.
    c_bytes = total_syn * (d * bytes_per_param + bytes_per_label)
    costs['C1-C3: Samples'] = c_bytes
    costs['S4-V2: Mechanisms'] = c_bytes  # same size, different content

    # S4-V2 total (synthetic + FedAvg fine-tune rounds)
    fed_finetune_rounds = 5
    s4v2_total = c_bytes + fed_finetune_rounds * (K + 1) * model_params * bytes_per_param
    costs['S4-V2: Total (syn+FT)'] = s4v2_total

    logger.info("\n" + "="*60)
    logger.info("Communication Cost Analysis")
    logger.info("="*60)
    logger.info(f"  Feature dim: {d}, Model params: {model_params:,}")
    logger.info(f"  Institutions: {K}, Fed rounds: {cfg.fed_rounds}")
    logger.info(f"  Total train nodes: {total_train:,}, "
                f"Total syn nodes: {total_syn}")
    logger.info(f"\n  {'Method':<30} {'Bytes':>12} {'KB':>10} {'Relative':>10}")
    logger.info("  " + "-"*65)

    ref = costs['S2: Raw data']
    for name, nbytes in costs.items():
        logger.info(
            f"  {name:<30} {nbytes:>12,} {nbytes/1024:>10.1f} "
            f"{nbytes/ref:>10.2%}")

    # 保存到 CSV
    os.makedirs('res', exist_ok=True)
    comm_rows = []
    for name, nbytes in costs.items():
        comm_rows.append({
            'dataset':        cfg.dataset if cfg else 'elliptic',
            'reduction_rate': cfg.reduction_rate if cfg else 0.01,
            'method':         name,
            'bytes':          nbytes,
            'kb':             round(nbytes / 1024, 1),
            'relative':       round(nbytes / ref, 4),
        })
    comm_path = 'res/communication_cost.csv'
    pd.DataFrame(comm_rows).to_csv(
        comm_path, mode='a',
        header=not os.path.exists(comm_path),
        index=False)
    logger.info(f"\n  Saved to: {comm_path}")

    return costs


# ============================================================
# 9. 压缩方法对比（Table 2）
# ============================================================
def scenario_random(institutions: Dict, global_data: Dict,
                    device: torch.device,
                    cfg: ExperimentConfig = None) -> Dict:
    """C1: Random Sampling - 随机采样baseline"""
    if cfg is None:
        cfg = ExperimentConfig()

    logger.info("\n" + "="*60)
    logger.info("C1: Random Sampling")
    logger.info("="*60)

    feat_all = global_data['feat']
    label_all = global_data['labels']
    syn_feats, syn_labels_list = [], []
    for name, inst in institutions.items():
        train_idx = inst['split']['train']
        train_feat = feat_all[train_idx]
        train_labels = label_all[train_idx]
        n_total = max(int(len(train_idx) * cfg.reduction_rate), 2)
        target_labels = make_syn_labels(
            n_total, train_labels, cfg.reduction_rate,
            fraud_ratio=cfg.fraud_oversample_ratio)
        target_fraud_rate = float((target_labels == 1).mean())
        syn_feat, syn_labels = sample_fraud_aware(
            train_feat, train_labels, n_total,
            target_fraud_rate)
        if (len(syn_labels) != len(target_labels)
                or int((syn_labels == 1).sum())
                != int((target_labels == 1).sum())):
            raise RuntimeError(
                f"Institution {name}: random baseline class-count mismatch")
        syn_feats.append(syn_feat)
        syn_labels_list.append(syn_labels)
        logger.info(
            f"  Institution {name}: {len(syn_feat)} nodes "
            f"(fraud={(syn_labels==1).sum()}, "
            f"legit={(syn_labels==0).sum()})")

    return _run_compression_scenario(
        "Random", institutions, global_data, device,
        syn_feats, syn_labels_list, cfg)


def herding_select(feats: np.ndarray, n_select: int) -> np.ndarray:
    """
    Herding Coreset Selection (Welling 2009)

    目标：选出n_select个样本使其均值最接近全集均值μ
    算法：每步t找离目标向量target_t = (t+1)*μ - current_sum最近的点

    数学推导：
      最小化 ||μ - (1/(t+1)) Σ_{i=0}^{t} x_i||²
      等价于：最小化 ||(t+1)*μ - current_sum - x_{t+1}||²
      即找离 target_t = (t+1)*μ - current_sum 最近的点

    注意：μ只计算一次，不随selected集合变化
    """
    if len(feats) <= n_select:
        return np.arange(len(feats))

    mu          = feats.mean(axis=0)   # 全集均值，固定不变
    current_sum = np.zeros_like(mu)
    selected    = []
    selected_set = set()

    for t in range(n_select):
        # 目标向量：如果当前均值要等于μ，下一个点应该在哪里
        target = (t + 1) * mu - current_sum
        # 找离target最近且未被选过的点
        dists  = np.linalg.norm(feats - target[None, :], axis=1)
        # 已选的点距离设为inf，避免重复选
        for s in selected_set:
            dists[s] = np.inf
        best = int(np.argmin(dists))
        selected.append(best)
        selected_set.add(best)
        current_sum += feats[best]

    return np.array(selected)


def scenario_herding(institutions: Dict, global_data: Dict,
                     device: torch.device,
                     cfg: ExperimentConfig = None) -> Dict:
    """C2: Herding Coreset Selection"""
    if cfg is None:
        cfg = ExperimentConfig()

    logger.info("\n" + "="*60)
    logger.info("C2: Herding Coreset Selection")
    logger.info("="*60)

    feat_all = global_data['feat']
    label_all = global_data['labels']
    syn_feats, syn_labels_list = [], []
    for name, inst in institutions.items():
        train_idx = inst['split']['train']
        feat = feat_all[train_idx]
        labels = label_all[train_idx]
        n_total = max(int(len(train_idx) * cfg.reduction_rate), 2)
        target_labels = make_syn_labels(
            n_total, labels, cfg.reduction_rate,
            fraud_ratio=cfg.fraud_oversample_ratio)
        n_fraud = int((target_labels == 1).sum())
        n_legit = int((target_labels == 0).sum())

        fi = np.where(labels == 1)[0]
        li = np.where(labels == 0)[0]

        # 对欺诈类和合法类分别做herding
        fs_local = herding_select(feat[fi], n_fraud)
        ls_local = herding_select(feat[li], n_legit)
        fs = fi[fs_local]
        ls = li[ls_local]

        if len(fs) != n_fraud or len(ls) != n_legit:
            raise ValueError(
                f"Institution {name}: requested herding class counts exceed "
                "the available canonical training examples")

        syn_feat   = np.vstack([feat[fs], feat[ls]]).astype(np.float32)
        syn_labels = np.array([1] * len(fs) + [0] * len(ls))
        syn_feats.append(syn_feat)
        syn_labels_list.append(syn_labels)
        logger.info(
            f"  Institution {name}: {len(syn_feat)} nodes "
            f"(fraud={len(fs)}, legit={len(ls)})")

    return _run_compression_scenario(
        "Herding", institutions, global_data, device,
        syn_feats, syn_labels_list, cfg)


def scenario_sfgc(institutions: Dict, global_data: Dict,
                  device: torch.device,
                  cfg: ExperimentConfig = None) -> Dict:
    """C3: SFGC-Lite - Structure-Free Graph Condensation"""
    if cfg is None:
        cfg = ExperimentConfig()

    logger.info("\n" + "="*60)
    logger.info("C3: SFGC-Lite")
    logger.info("="*60)

    in_dim  = global_data['feat'].shape[1]
    dataset = global_data.get('dataset', 'elliptic')

    all_syn_feat, all_syn_labels = _load_synthetic(
        institutions, cfg.reduction_rate, in_dim,
        dataset=dataset, tag=cfg.distill_tag,
        artifact_kind='sfgc', cfg=cfg, strict=True)

    if all_syn_feat is None:
        logger.warning("  SFGC skipped: method-specific artifacts not found.")
        return None

    return _run_compression_scenario(
        "SFGC", institutions, global_data, device,
        all_syn_feat, all_syn_labels, cfg)


# ============================================================
# 10. 主函数
# ============================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(
        description=(
            'Validation-selected-threshold main-table rerun with strict '
            'train/model-validation/calibration/test isolation'))
    parser.add_argument('--gpu_id',         type=int,   default=0)
    parser.add_argument('--reduction_rate', type=float, default=0.05)
    parser.add_argument('--seed',           type=int,   default=SEED_DEFAULT)
    parser.add_argument('--fed_rounds',     type=int,   default=50)
    parser.add_argument('--output_dir',     type=str,
                        default='res/validation_selected')
    parser.add_argument('--dataset',        type=str,   default='elliptic',
                        choices=['elliptic', 'elliptic_pp'])
    args = parser.parse_args()

    set_all_seeds(args.seed)

    device = torch.device(
        f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    logger.info(f"Dataset: {args.dataset}")

    cfg = ExperimentConfig(
        dataset=args.dataset,
        reduction_rate=args.reduction_rate,
        seed=args.seed,
        gpu_id=args.gpu_id,
        fed_rounds=args.fed_rounds,
    )

    if args.dataset == 'elliptic':
      institutions, global_data = load_and_split(cfg=cfg)
    else:
      institutions, global_data = load_and_split_elliptic_pp(cfg=cfg)

    validate_protocol_isolation(institutions, global_data)

    # Reset before every method so selecting/reordering methods elsewhere does
    # not change a model's initialization or stochastic DP noise stream.
    set_all_seeds(args.seed)
    s1_res, s1_avg = scenario_isolated(institutions, global_data, device, cfg)
    set_all_seeds(args.seed)
    s2_res         = scenario_shared_data(institutions, global_data, device, cfg)
    set_all_seeds(args.seed)
    s3_res         = scenario_federated(institutions, global_data, device, cfg)
    set_all_seeds(args.seed)
    s3b_res        = scenario_fedprox(institutions, global_data, device, cfg)
    set_all_seeds(args.seed)
    dp_low_res     = scenario_dp_fedavg(institutions, global_data, device,
                                        noise_multiplier=DP_NOISE_LOW, cfg=cfg)
    set_all_seeds(args.seed)
    dp_high_res    = scenario_dp_fedavg(institutions, global_data, device,
                                        noise_multiplier=DP_NOISE_HIGH, cfg=cfg)
    set_all_seeds(args.seed)
    s4v1_res       = scenario_distill_v1_joint(institutions, global_data, device, cfg)
    set_all_seeds(args.seed)
    s4v2_res       = scenario_distill_v2_pretrain_fed(institutions, global_data, device, cfg)
    set_all_seeds(args.seed)
    gcond_res = scenario_distill_v2_gcond_vanilla(institutions, global_data, device, cfg)
    set_all_seeds(args.seed)
    rand_res       = scenario_random(institutions, global_data, device, cfg)
    set_all_seeds(args.seed)
    herd_res       = scenario_herding(institutions, global_data, device, cfg)
    set_all_seeds(args.seed)
    sfgc_res       = scenario_sfgc(institutions, global_data, device, cfg)
    comm_costs     = communication_cost_analysis(institutions, global_data, cfg)

    # 汇总
    logger.info("\n" + "="*70)
    logger.info(f"Results Summary - {args.dataset.upper()} "
                f"(reduction_rate={args.reduction_rate})")
    logger.info("="*70)
    logger.info(
        f"{'Scenario':<45} {'AUC':>7} {'PR-AUC':>8} {'F1-fraud':>9} "
        f"{'Precision':>10} {'Recall':>8}")
    logger.info("-"*85)

    scenario_results = [
        ("S1: Isolated (avg, no collaboration)",     s1_avg),
        ("S2: Shared Raw Data (upper bound)",         s2_res),
        ("S3: Federated Learning (FedAvg)",           s3_res),
        ("S3b: Federated Learning (FedProx)",         s3b_res),
        (f"S3c: DP-FedAvg (σ={DP_NOISE_LOW})",       dp_low_res),
        (f"S3c: DP-FedAvg (σ={DP_NOISE_HIGH})",      dp_high_res),
        ("C1: Random Sampling",                       rand_res),
        ("C2: Herding Coreset",                       herd_res),
    ]
    # 蒸馏场景：文件不存在时返回None，自动跳过
    for name, res in [
        ("C3: SFGC-Lite",                         sfgc_res),
        ("S4-V1: FIAD Joint Training",            s4v1_res),
        ("S4-V2: FIAD Pretrain+FedAvg",           s4v2_res),
        ("S4-V2-GCond: Vanilla GCond+FedAvg",     gcond_res),
    ]:
        if res is not None:
            scenario_results.append((name, res))

    for name, result in scenario_results:
        threshold_mode = str(result.get('threshold_mode', ''))
        if not threshold_mode.startswith('validation'):
            raise RuntimeError(
                f"{name} returned threshold_mode={threshold_mode!r}; "
                "the primary rerun requires a validation-selected threshold.")

    rows = []
    for name, r in scenario_results:
        reported_threshold = float(r['threshold'])
        logger.info(
            f"{name:<45} {r['auc']:>7.4f} {r.get('pr_auc',0):>8.4f} "
            f"{r['f1_fraud']:>9.4f} "
            f"{r['precision']:>10.4f} {r['recall']:>8.4f}")
        rows.append({
            'dataset':            cfg.dataset,
            'reduction_rate':     cfg.reduction_rate,
            'seed':               cfg.seed,
            'scenario':           name,
            'distill_tag':        cfg.distill_tag,
            'auc':                r['auc'],
            'pr_auc':             r.get('pr_auc', None),
            'f1_fraud':           r['f1_fraud'],
            'f1_macro':           r.get('f1_macro', None),
            'precision':          r['precision'],
            'recall':             r['recall'],
            'threshold':          reported_threshold,
            'threshold_mode':     r['threshold_mode'],
        })

    # 关键对比输出（FIAD主线是S4-V2）
    if s4v2_res is not None:
        logger.info(f"\n  === FIAD (S4-V2) vs Baselines ===")
        logger.info(f"  S4-V2 vs S1: {(s4v2_res['auc']-s1_avg['auc'])*100:+.1f}% AUC  "
                    f"{(s4v2_res['f1_fraud']-s1_avg['f1_fraud'])*100:+.1f}% F1")
        logger.info(f"  S4-V2 vs S3: {(s4v2_res['auc']-s3_res['auc'])*100:+.1f}% AUC  "
                    f"(gap to federated learning)")
        logger.info(f"  S4-V2 vs S2: {(s4v2_res['auc']-s2_res['auc'])*100:+.1f}% AUC  "
                    f"(gap to upper bound)")
    else:
        logger.warning("\n  S4-V2 results unavailable (distilled files not found).")

    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)

    # 汇总文件（追加，不覆盖）
    summary_path = os.path.join(
        args.output_dir, 'cross_institution_validation_selected_all.csv')
    pd.DataFrame(rows).to_csv(
        summary_path, mode='a',
        header=not os.path.exists(summary_path),
        index=False)
    logger.info(f"\n  Appended to: {summary_path}")

    # 单次实验文件（按dataset+rate+seed命名，避免覆盖）
    save_path = os.path.join(
        args.output_dir,
        f'cross_institution_validation_selected_{cfg.dataset}_'
        f'r{cfg.reduction_rate}_fl{cfg.fed_rounds}_s{cfg.seed}.csv')
    pd.DataFrame(rows).to_csv(save_path, index=False)
    logger.info(f"  Saved to: {save_path}")


if __name__ == '__main__':
    main()

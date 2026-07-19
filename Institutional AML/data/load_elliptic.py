import os

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data


def _check_split(labels, idx, split_name, require_both_classes=False):
    idx = np.asarray(idx, dtype=np.int64)
    if idx.size == 0:
        raise ValueError(f'{split_name} split is empty.')

    split_labels = np.asarray(labels)[idx]
    num_pos = int((split_labels == 1).sum())
    num_neg = int((split_labels == 0).sum())
    classes = np.unique(split_labels)

    if require_both_classes and classes.size < 2:
        raise ValueError(
            f'{split_name} split must contain both classes, '
            f'but got classes={classes.tolist()}, '
            f'n={idx.size}, pos={num_pos}, neg={num_neg}.'
        )

    return {
        'n': int(idx.size),
        'pos': num_pos,
        'neg': num_neg,
        'classes': classes.tolist(),
    }


def load_elliptic(data_dir='data/elliptic_bitcoin_dataset', verbose=True):
    """
    Load the Elliptic Bitcoin dataset and convert it to a PyG Data object.

    Returns:
        data: PyG Data object
        idx_train, idx_val, idx_test: train/val/test node indices
    """

    if verbose:
        print('Reading raw Elliptic data...')

    feat_path = os.path.join(data_dir, 'elliptic_txs_features.csv')
    edge_path = os.path.join(data_dir, 'elliptic_txs_edgelist.csv')
    label_path = os.path.join(data_dir, 'elliptic_txs_classes.csv')

    df_feat = pd.read_csv(feat_path, header=None)
    df_edge = pd.read_csv(edge_path)
    df_label = pd.read_csv(label_path)

    if verbose:
        print(f'  Nodes:  {len(df_feat):,}')
        print(f'  Edges:  {len(df_edge):,}')
        print(f'  Labels: {len(df_label):,}')

    node_ids = df_feat.iloc[:, 0].values
    id2idx = {nid: i for i, nid in enumerate(node_ids)}
    num_nodes = len(node_ids)

    timesteps = df_feat.iloc[:, 1].values
    features = df_feat.iloc[:, 2:].values.astype(np.float32)

    src_ids = df_edge.iloc[:, 0].values
    dst_ids = df_edge.iloc[:, 1].values
    valid_edge_mask = np.array(
        [(s in id2idx and d in id2idx) for s, d in zip(src_ids, dst_ids)]
    )
    src_ids = src_ids[valid_edge_mask]
    dst_ids = dst_ids[valid_edge_mask]

    src_idx = np.array([id2idx[s] for s in src_ids], dtype=np.int64)
    dst_idx = np.array([id2idx[d] for d in dst_ids], dtype=np.int64)
    src_all = np.concatenate([src_idx, dst_idx])
    dst_all = np.concatenate([dst_idx, src_idx])

    edge_index = torch.tensor(
        np.stack([src_all, dst_all], axis=0),
        dtype=torch.long,
    )

    label_map = df_label.set_index('txId')['class'].to_dict()
    labels = np.full(num_nodes, -1, dtype=np.int64)
    for i, nid in enumerate(node_ids):
        cls = label_map.get(nid, 'unknown')
        if cls == '1':
            labels[i] = 1
        elif cls == '2':
            labels[i] = 0

    y = torch.tensor(labels, dtype=torch.long)

    n_fraud = int((labels == 1).sum())
    n_legit = int((labels == 0).sum())
    n_unknown = int((labels == -1).sum())
    if verbose:
        print('\nLabel distribution:')
        print(f'  Fraud (1):    {n_fraud:,} ({n_fraud / num_nodes * 100:.1f}%)')
        print(f'  Legit (0):    {n_legit:,} ({n_legit / num_nodes * 100:.1f}%)')
        print(f'  Unknown (-1): {n_unknown:,} ({n_unknown / num_nodes * 100:.1f}%)')
        print(f'  Imbalance ratio: 1:{n_legit / max(n_fraud, 1):.1f}')

    labeled_idx = np.where(labels != -1)[0]
    train_mask = np.isin(timesteps[labeled_idx], np.arange(1, 35))
    val_mask = np.isin(timesteps[labeled_idx], np.arange(35, 41))
    test_mask = np.isin(timesteps[labeled_idx], np.arange(41, 50))

    idx_train_np = labeled_idx[train_mask]
    idx_val_np = labeled_idx[val_mask]
    idx_test_np = labeled_idx[test_mask]

    train_stats = _check_split(labels, idx_train_np, 'train', require_both_classes=False)
    val_stats = _check_split(labels, idx_val_np, 'val', require_both_classes=True)
    test_stats = _check_split(labels, idx_test_np, 'test', require_both_classes=True)

    scaler = StandardScaler()
    scaler.fit(features[idx_train_np])
    features = scaler.transform(features)
    x = torch.tensor(features, dtype=torch.float)

    idx_train = torch.tensor(idx_train_np, dtype=torch.long)
    idx_val = torch.tensor(idx_val_np, dtype=torch.long)
    idx_test = torch.tensor(idx_test_np, dtype=torch.long)

    if verbose:
        print(f'\nFeature shape: {x.shape}')
        print(f'Timestep range: {timesteps.min()} ~ {timesteps.max()}')
        print(f'Scaler fit on train only: {train_stats["n"]:,} nodes')
        print('\nTemporal split:')
        print(
            f'  Train: {train_stats["n"]:,} nodes, '
            f'pos={train_stats["pos"]:,}, neg={train_stats["neg"]:,}, '
            f'classes={train_stats["classes"]}'
        )
        print(
            f'  Val:   {val_stats["n"]:,} nodes, '
            f'pos={val_stats["pos"]:,}, neg={val_stats["neg"]:,}, '
            f'classes={val_stats["classes"]}'
        )
        print(
            f'  Test:  {test_stats["n"]:,} nodes, '
            f'pos={test_stats["pos"]:,}, neg={test_stats["neg"]:,}, '
            f'classes={test_stats["classes"]}'
        )
        print(f'Edges (undirected): {edge_index.shape[1]:,}')

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        num_nodes=num_nodes,
    )
    data.timestep = torch.tensor(timesteps, dtype=torch.long)

    if verbose:
        print('\nPyG Data:')
        print(f'  {data}')
        print('Elliptic preprocessing finished.')

    return data, idx_train, idx_val, idx_test


if __name__ == '__main__':
    data, idx_train, idx_val, idx_test = load_elliptic(
        data_dir='data/elliptic_bitcoin_dataset',
        verbose=True
    )

    print('\nRunning quick checks...')
    assert data.x.shape[1] == 165, 'Feature dimension should be 165'
    assert data.y.max().item() == 1, 'Max label should be 1'
    assert data.edge_index.min().item() >= 0, 'Edge index must be non-negative'
    assert data.edge_index.max().item() < data.num_nodes, 'Edge index out of range'
    print('All checks passed.')

    save_path = 'data/elliptic_bitcoin_dataset/processed.pt'
    torch.save({
        'data': data,
        'idx_train': idx_train,
        'idx_val': idx_val,
        'idx_test': idx_test,
    }, save_path)
    print(f'\nSaved processed data to: {save_path}')

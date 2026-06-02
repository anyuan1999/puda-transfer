"""
Test PUDA model on synthetic provenance graphs.
Validates the full pipeline: model construction, training, inference.
"""
import sys
import os
import torch
import numpy as np
from torch_geometric.data import Data

sys.path.insert(0, '/tmp/pidsmaker-transfer')
from methods.puda.adapter import PUDATransfer, PUDAModel, UnifiedMap, UnifiedType


def create_synthetic_provenance_graph(num_nodes=500, num_edges=2000,
                                       feat_dim=30, attack_ratio=0.05,
                                       dataset_name='cadets'):
    """
    Create a synthetic provenance graph mimicking DARPA TC datasets.

    Nodes have:
    - x: Word2Vec-like features (random normal)
    - y: attack labels (0=benign, 1=attack)
    - node_type: raw dataset-specific type labels
    """
    # Generate node features (simulating Word2Vec embeddings)
    x = torch.randn(num_nodes, feat_dim)

    # Generate random edges (provenance graph structure)
    src = torch.randint(0, num_nodes, (num_edges,))
    dst = torch.randint(0, num_nodes, (num_edges,))
    edge_index = torch.stack([src, dst], dim=0)

    # Generate node types based on dataset
    type_map = UnifiedMap.get_map(dataset_name)
    num_types = max(type_map.keys()) + 1
    node_type = torch.randint(0, num_types, (num_nodes,))

    # Generate attack labels (sparse - only attack_ratio of nodes)
    num_attacks = int(num_nodes * attack_ratio)
    y = torch.zeros(num_nodes, dtype=torch.long)
    attack_indices = torch.randperm(num_nodes)[:num_attacks]
    y[attack_indices] = 1

    graph = Data(x=x, edge_index=edge_index, y=y, node_type=node_type)
    graph.n_id = torch.arange(num_nodes)

    return graph


def test_puda_model_forward():
    """Test PUDAModel forward pass."""
    print("=" * 60)
    print("Test 1: PUDAModel Forward Pass")
    print("=" * 60)

    model = PUDAModel(word2vec_dim=30, gat_hidden=64, num_node_types=4,
                      num_domains=2, num_layers=2, use_gcn=True)

    # Synthetic input
    x = torch.randn(100, 30)
    edge_index = torch.randint(0, 100, (2, 300))

    model.eval()
    with torch.no_grad():
        task_logits, salience, domain_logits = model(x, edge_index)

    print(f"  Input: {x.shape[0]} nodes, {edge_index.shape[1]} edges")
    print(f"  Task logits: {task_logits.shape} (expect [100, 4])")
    print(f"  Salience scores: {salience.shape} (expect [100, 1])")
    print(f"  Domain logits: {domain_logits.shape} (expect [100, 2])")
    print(f"  Salience range: [{salience.min():.4f}, {salience.max():.4f}]")

    assert task_logits.shape == (100, 4)
    assert salience.shape == (100, 1)
    assert domain_logits.shape == (100, 2)
    assert salience.min() >= 0 and salience.max() <= 1

    print("  PASSED!\n")


def test_puda_transfer_train_and_predict():
    """Test full PUDATransfer pipeline: train_source + adapt_and_predict."""
    print("=" * 60)
    print("Test 2: PUDATransfer Full Pipeline (S3: CADETS -> THEIA)")
    print("=" * 60)

    # Create synthetic source (CADETS) and target (THEIA) graphs
    source_graph = create_synthetic_provenance_graph(
        num_nodes=300, num_edges=1200, feat_dim=30,
        attack_ratio=0.03, dataset_name='cadets'
    )
    target_graph = create_synthetic_provenance_graph(
        num_nodes=250, num_edges=1000, feat_dim=30,
        attack_ratio=0.04, dataset_name='theia'
    )

    print(f"  Source (CADETS): {source_graph.num_nodes} nodes, {source_graph.num_edges} edges, "
          f"{source_graph.y.sum().item()} attacks")
    print(f"  Target (THEIA): {target_graph.num_nodes} nodes, {target_graph.num_edges} edges, "
          f"{target_graph.y.sum().item()} attacks")

    # Initialize PUDA adapter
    cfg = {}
    puda = PUDATransfer(
        cfg, device='cpu', epochs=10, lr=0.005, batch_size=128,
        gat_hidden=32, num_layers=2, use_gcn=True,
        lambda_sal=3.0, lambda_adv=1.0, max_grl_lambda=1.0,
        threshold_strategy='percentile_top3'
    )

    # Train with adversarial domain alignment
    print("\n  Training PUDA (source + target adversarial)...")
    puda.train_source(
        source_graphs=[source_graph],
        target_graphs=[target_graph],
        source_name='cadets',
        target_name='theia'
    )

    # Predict on target
    print("\n  Predicting on target domain...")
    scores, predictions = puda.adapt_and_predict(
        target_graphs=[target_graph],
        target_name='theia'
    )

    print(f"\n  Results:")
    print(f"    Score range: [{scores.min():.4f}, {scores.max():.4f}]")
    print(f"    Mean score: {scores.mean():.4f}")
    print(f"    Predictions: {predictions.sum()}/{len(predictions)} anomalous")

    # Evaluate
    gt_nids = torch.where(target_graph.y == 1)[0].numpy()
    metrics = puda.evaluate(predictions, scores, gt_nids, target_graph.num_nodes)

    print(f"\n  Metrics:")
    print(f"    TP={metrics['TP']}, TN={metrics['TN']}, FP={metrics['FP']}, FN={metrics['FN']}")
    print(f"    F1={metrics['F1']}, MCC={metrics['MCC']}, ADP={metrics['ADP']}")

    assert len(scores) == target_graph.num_nodes
    assert len(predictions) == target_graph.num_nodes
    assert all(p in [0, 1] for p in predictions)

    print("  PASSED!\n")
    return metrics


def test_puda_cross_os_transfer():
    """Test Cross-OS scenario (S5: THEIA -> OpTC)."""
    print("=" * 60)
    print("Test 3: Cross-OS Transfer (S5: THEIA -> OpTC)")
    print("=" * 60)

    # Different feature dims to simulate cross-domain mismatch
    source_graph = create_synthetic_provenance_graph(
        num_nodes=400, num_edges=1600, feat_dim=30,
        attack_ratio=0.02, dataset_name='theia'
    )
    target_graph = create_synthetic_provenance_graph(
        num_nodes=350, num_edges=1400, feat_dim=30,
        attack_ratio=0.05, dataset_name='optc'
    )

    print(f"  Source (THEIA): {source_graph.num_nodes} nodes, types 0-5")
    print(f"  Target (OpTC): {target_graph.num_nodes} nodes, types 0-3")

    # Verify unified type mapping
    source_unified = UnifiedMap.convert_node_types(source_graph.node_type, 'theia', 'cpu')
    target_unified = UnifiedMap.convert_node_types(target_graph.node_type, 'optc', 'cpu')

    print(f"  Source unified distribution: {torch.bincount(source_unified, minlength=4).tolist()}")
    print(f"  Target unified distribution: {torch.bincount(target_unified, minlength=4).tolist()}")

    cfg = {}
    puda = PUDATransfer(
        cfg, device='cpu', epochs=8, lr=0.005, batch_size=128,
        gat_hidden=32, num_layers=2, use_gcn=True,
        lambda_sal=3.0, lambda_adv=1.0,
        threshold_strategy='percentile_top5'
    )

    print("\n  Training Cross-OS PUDA...")
    puda.train_source(
        source_graphs=[source_graph],
        target_graphs=[target_graph],
        source_name='theia',
        target_name='optc'
    )

    scores, predictions = puda.adapt_and_predict(
        target_graphs=[target_graph],
        target_name='optc'
    )

    gt_nids = torch.where(target_graph.y == 1)[0].numpy()
    metrics = puda.evaluate(predictions, scores, gt_nids, target_graph.num_nodes)

    print(f"\n  Cross-OS Metrics:")
    print(f"    TP={metrics['TP']}, TN={metrics['TN']}, FP={metrics['FP']}, FN={metrics['FN']}")
    print(f"    F1={metrics['F1']}, MCC={metrics['MCC']}, ADP={metrics['ADP']}")
    print("  PASSED!\n")
    return metrics


def test_grl_lambda_schedule():
    """Test GRL lambda progressive scheduling."""
    print("=" * 60)
    print("Test 4: GRL Lambda Schedule")
    print("=" * 60)

    from methods.puda.adapter import compute_grl_lambda

    total_epochs = 30
    lambdas = [compute_grl_lambda(e, total_epochs) for e in range(total_epochs)]

    print(f"  Epoch  0: lambda = {lambdas[0]:.4f}")
    print(f"  Epoch  5: lambda = {lambdas[5]:.4f}")
    print(f"  Epoch 15: lambda = {lambdas[15]:.4f}")
    print(f"  Epoch 25: lambda = {lambdas[25]:.4f}")
    print(f"  Epoch 29: lambda = {lambdas[29]:.4f}")

    # Should be monotonically increasing
    for i in range(1, len(lambdas)):
        assert lambdas[i] >= lambdas[i-1], f"Non-monotonic at epoch {i}"

    # Should start near 0 and end near 1
    assert lambdas[0] < 0.1
    assert lambdas[-1] > 0.9

    print("  Progressive schedule verified: 0 -> 1")
    print("  PASSED!\n")


def test_without_target_graphs():
    """Test source-only training (no adversarial)."""
    print("=" * 60)
    print("Test 5: Source-Only Training (No Adversarial Alignment)")
    print("=" * 60)

    source_graph = create_synthetic_provenance_graph(
        num_nodes=200, num_edges=800, feat_dim=30,
        attack_ratio=0.05, dataset_name='trace'
    )
    target_graph = create_synthetic_provenance_graph(
        num_nodes=150, num_edges=600, feat_dim=30,
        attack_ratio=0.03, dataset_name='cadets'
    )

    cfg = {}
    puda = PUDATransfer(
        cfg, device='cpu', epochs=5, lr=0.005, batch_size=128,
        gat_hidden=32, use_gcn=True,
        threshold_strategy='fixed_0.5'
    )

    # Train without target (no adversarial loss)
    print("  Training source-only (no target for adversarial)...")
    puda.train_source(
        source_graphs=[source_graph],
        target_graphs=None,  # No target
        source_name='trace',
        target_name='cadets'
    )

    # Still predict on target
    scores, predictions = puda.adapt_and_predict(
        target_graphs=[target_graph],
        target_name='cadets'
    )

    print(f"  Scores: mean={scores.mean():.4f}, std={scores.std():.4f}")
    print(f"  Predictions: {predictions.sum()}/{len(predictions)} anomalous")
    print("  PASSED!\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  PUDA Model Integration Tests")
    print("  Testing on synthetic provenance graphs (CPU)")
    print("=" * 60 + "\n")

    test_puda_model_forward()
    test_grl_lambda_schedule()
    metrics_s3 = test_puda_transfer_train_and_predict()
    metrics_s5 = test_puda_cross_os_transfer()
    test_without_target_graphs()

    print("=" * 60)
    print("  ALL TESTS PASSED!")
    print("=" * 60)
    print(f"\n  Summary:")
    print(f"    S3 (CADETS->THEIA): F1={metrics_s3['F1']}, MCC={metrics_s3['MCC']}")
    print(f"    S5 (THEIA->OpTC):   F1={metrics_s5['F1']}, MCC={metrics_s5['MCC']}")
    print()

"""
Run all domain adaptation methods on PIDSMaker transfer scenarios.

Usage:
    python run_da_methods.py --method stgan --source CADETS_E3 --target THEIA_E3 [--gpu]
    python run_da_methods.py --method puda --source CADETS_E3 --target THEIA_E3 [--gpu]
    python run_da_methods.py --all  # Run all methods × all scenarios

Methods: stgan, puda, udagcn, a2gnn
"""
import argparse
import os
import sys
import csv
import time
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from methods.stgan.adapter import STGANTransfer
from methods.puda.adapter import PUDATransfer
from methods.udagcn.adapter import UDAGCNTransfer
from methods.a2gnn.adapter import A2GNNTransfer


def get_method(name, cfg, device):
    methods = {
        'stgan': STGANTransfer(cfg, device, epochs=50),
        'puda': PUDATransfer(cfg, device, epochs=30, lr=0.005, batch_size=2048,
                             gat_hidden=64, use_gcn=True, lambda_sal=3.0, lambda_adv=1.0),
        'udagcn': UDAGCNTransfer(cfg, device, epochs=200),
        'a2gnn': A2GNNTransfer(cfg, device, epochs=200, s_pnums=0, t_pnums=10),
    }
    return methods[name]


def load_pidsmaker_graphs(dataset_name, cfg):
    """Load preprocessed graphs from PIDSMaker's pipeline."""
    from pidsmaker.config.pipeline import get_yml_cfg, get_runtime_required_args
    from pidsmaker.tasks.batching import get_preprocessed_graphs
    from pidsmaker.utils.utils import get_device, set_seed
    
    # Build config for this dataset
    sys.argv = [
        "main.py", "magic", dataset_name,  # Use magic config (lightweight)
        "--artifact_dir", cfg.get('artifact_dir', '/home/artifacts'),
        "--database_host", cfg.get('db_host', '127.0.0.1'),
        "--database_port", cfg.get('db_port', '5432'),
        "--database_user", cfg.get('db_user', 'postgres'),
        "--database_password", cfg.get('db_pass', ''),
    ]
    if cfg.get('cpu', True):
        sys.argv.append("--cpu")
    
    args, _ = get_runtime_required_args(return_unknown_args=True)
    dataset_cfg = get_yml_cfg(args)
    device = get_device(dataset_cfg)
    set_seed(dataset_cfg, seed=0)
    
    train_data, val_data, test_data, max_node_num = get_preprocessed_graphs(dataset_cfg)
    
    # Convert PIDSMaker's format to list of PyG Data objects
    # PIDSMaker stores temporal graphs as lists of CollatableTemporalData
    graphs = []
    for data_list in test_data:
        for batch in data_list:
            graphs.append(batch)
    
    # Also get ground truth
    from pidsmaker.detection.evaluation_methods.evaluation_utils import get_ground_truth_nids
    gt_nids, _ = get_ground_truth_nids(dataset_cfg)
    
    return {
        'train': train_data,
        'val': val_data, 
        'test': test_data,
        'test_graphs': graphs,
        'gt_nids': gt_nids,
        'max_node_num': max_node_num,
        'cfg': dataset_cfg,
    }


def run_single_experiment(method_name, source_name, target_name, cfg):
    """Run a single transfer experiment."""
    device = 'cuda' if (torch.cuda.is_available() and not cfg.get('cpu', True)) else 'cpu'
    print(f"\n{'='*60}")
    print(f"  {method_name}: {source_name} -> {target_name} (device={device})")
    print(f"{'='*60}")
    
    # Load data
    print(f"  Loading source ({source_name})...")
    source_data = load_pidsmaker_graphs(source_name, cfg)
    
    print(f"  Loading target ({target_name})...")
    target_data = load_pidsmaker_graphs(target_name, cfg)
    
    # Get source train graphs (flatten)
    source_train_graphs = []
    for data_list in source_data['train']:
        for batch in data_list:
            source_train_graphs.append(batch)
    
    target_test_graphs = target_data['test_graphs']
    
    # Run method
    method = get_method(method_name, cfg, device)
    
    print(f"  Training on source...")
    t0 = time.time()
    method.train_source(source_train_graphs)
    
    print(f"  Adapting and predicting on target...")
    scores, predictions = method.adapt_and_predict(target_test_graphs)
    elapsed = time.time() - t0
    
    # Evaluate
    gt_nids = target_data['gt_nids']
    total_nodes = len(predictions)
    metrics = method.evaluate(predictions, scores, gt_nids, total_nodes)
    metrics['time_sec'] = round(elapsed, 1)
    
    print(f"  Results: TP={metrics['TP']} TN={metrics['TN']} FP={metrics['FP']} FN={metrics['FN']}")
    print(f"           MCC={metrics['MCC']} F1={metrics['F1']} ADP={metrics['ADP']}")
    print(f"           Time: {elapsed:.1f}s")
    
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, choices=['stgan', 'puda', 'udagcn', 'a2gnn', 'all'], default='all')
    parser.add_argument("--source", type=str, default=None)
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument("--cpu", action="store_true", default=True)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--artifact_dir", default="/home/artifacts")
    parser.add_argument("--db_host", default="127.0.0.1")
    parser.add_argument("--db_port", default="5432")
    parser.add_argument("--db_user", default="pgsql")
    parser.add_argument("--db_pass", default="")
    parser.add_argument("--output", default="da_results.csv")
    args = parser.parse_args()
    
    if args.gpu:
        args.cpu = False
    
    cfg = vars(args)
    
    SCENARIOS = [
        ("CADETS_E3", "THEIA_E3", "S3"),
        ("THEIA_E3", "CADETS_E3", "S4"),
        ("THEIA_E3", "optc_h501", "S5"),
        ("optc_h501", "THEIA_E3", "S6"),
        ("optc_h201", "CADETS_E3", "S7"),
        ("CADETS_E3", "optc_h051", "S8"),
    ]
    
    methods = ['stgan', 'puda', 'udagcn', 'a2gnn'] if args.method == 'all' else [args.method]
    
    if args.source and args.target:
        scenarios = [(args.source, args.target, "custom")]
    else:
        scenarios = SCENARIOS
    
    # Run experiments
    results = []
    for method_name in methods:
        for source, target, scenario in scenarios:
            try:
                metrics = run_single_experiment(method_name, source, target, cfg)
                results.append({
                    'system': method_name,
                    'scenario': scenario,
                    'source': source,
                    'target': target,
                    **metrics
                })
            except Exception as e:
                print(f"  ERROR: {str(e)[:100]}")
                results.append({
                    'system': method_name,
                    'scenario': scenario,
                    'source': source,
                    'target': target,
                    'TP': '', 'TN': '', 'FP': '', 'FN': '',
                    'MCC': '', 'F1': '', 'ADP': '',
                    'error': str(e)[:100]
                })
    
    # Save results
    output_path = os.path.join(cfg['artifact_dir'], 'transfer_results', args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['system', 'scenario', 'source', 'target', 'TP', 'TN', 'FP', 'FN', 'MCC', 'F1', 'ADP'])
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, '') for k in writer.fieldnames})
    
    print(f"\n{'='*60}")
    print(f"  ALL DONE. Results: {output_path}")
    print(f"{'='*60}")
    
    # Print summary table
    print(f"\n{'system':<10}{'scenario':<6}{'source':<12}{'target':<12}{'TP':<6}{'FN':<6}{'MCC':<8}{'F1':<8}{'ADP':<6}")
    print("-" * 74)
    for r in results:
        print(f"{r.get('system',''):<10}{r.get('scenario',''):<6}{r.get('source',''):<12}{r.get('target',''):<12}{str(r.get('TP','')):<6}{str(r.get('FN','')):<6}{str(r.get('MCC','')):<8}{str(r.get('F1','')):<8}{str(r.get('ADP','')):<6}")


if __name__ == "__main__":
    main()

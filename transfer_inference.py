"""Cross-domain Transfer Inference for PIDSMaker."""
import argparse, glob, os, sys
import torch
import wandb
wandb.init(mode="disabled")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pidsmaker.config.pipeline import get_yml_cfg, get_runtime_required_args
from pidsmaker.factory import build_model
from pidsmaker.tasks.batching import get_preprocessed_graphs
from pidsmaker.utils.utils import get_device, log, set_seed
from pidsmaker.detection.training_methods import inference_loop
from pidsmaker.tasks import evaluation as evaluation_task


def find_source_model(target_training_path, source_dataset):
    """Find source model in the same config hash as target."""
    hash_dir = os.path.dirname(target_training_path)
    source_path = os.path.join(hash_dir, source_dataset, "gnn_models")
    
    if not os.path.isdir(source_path):
        training_root = os.path.dirname(hash_dir)
        for h in os.listdir(training_root):
            candidate = os.path.join(training_root, h, source_dataset, "gnn_models")
            if os.path.isdir(candidate):
                source_path = candidate
                break
    
    if not os.path.isdir(source_path):
        return None
    
    epochs = sorted(
        glob.glob(os.path.join(source_path, "model_epoch_*")),
        key=lambda p: int(p.rsplit("_", 1)[-1])
    )
    return epochs[-1] if epochs else None


def load_model_with_resize(model, model_path, device):
    """Load model state dict, resizing mismatched tensors (e.g. TGN memory)."""
    state_dict_path = os.path.join(model_path, "state_dict.pkl")
    source_state = torch.load(state_dict_path, map_location=device)
    model_state = model.state_dict()

    for key in list(source_state.keys()):
        if key in model_state and source_state[key].shape != model_state[key].shape:
            src_shape = source_state[key].shape
            tgt_shape = model_state[key].shape
            log(f"  Resizing {key}: {src_shape} -> {tgt_shape}")
            new_tensor = torch.zeros(tgt_shape, dtype=source_state[key].dtype)
            slices = tuple(slice(0, min(s, t)) for s, t in zip(src_shape, tgt_shape))
            new_tensor[slices] = source_state[key][slices]
            source_state[key] = new_tensor

    model.load_state_dict(source_state, strict=False)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("system", type=str)
    parser.add_argument("source", type=str)
    parser.add_argument("target", type=str)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--artifact_dir", default="/home/artifacts")
    parser.add_argument("--database_host", default="postgres")
    parser.add_argument("--database_port", type=str, default="5432")
    parser.add_argument("--database_user", default="postgres")
    parser.add_argument("--database_password", default="postgres")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  PUDA Transfer: {args.source} -> {args.target} [{args.system}]")
    print(f"{'='*60}\n")

    sys.argv = [
        "main.py", args.system, args.target,
        "--artifact_dir", args.artifact_dir,
        "--database_host", args.database_host,
        "--database_port", args.database_port,
        "--database_user", args.database_user,
        "--database_password", args.database_password,
    ]
    if args.cpu:
        sys.argv.append("--cpu")

    target_args, _ = get_runtime_required_args(return_unknown_args=True)
    target_cfg = get_yml_cfg(target_args)
    device = get_device(target_cfg)
    set_seed(target_cfg, seed=target_cfg.training.seed)

    log("Loading target preprocessed graphs...")
    train_data, val_data, test_data, max_node_num = get_preprocessed_graphs(target_cfg)
    log(f"Target loaded. max_node={max_node_num}")

    model_path = find_source_model(target_cfg.training._task_path, args.source)
    if model_path is None:
        print(f"[ERROR] No source model for {args.source} with same config")
        print(f"  Target training path: {target_cfg.training._task_path}")
        sys.exit(1)
    log(f"Source model: {model_path}")

    model = build_model(
        data_sample=train_data[0][0], device=device, cfg=target_cfg, max_node_num=max_node_num
    )
    model = load_model_with_resize(model, model_path, device)
    model.to_device(device)
    model.eval()
    log("Model loaded OK")

    transfer_dir = os.path.join(
        args.artifact_dir, "transfer_results",
        f"{args.system}_{args.source}_to_{args.target}"
    )
    edge_losses_dir = os.path.join(transfer_dir, "edge_losses")
    pr_dir = os.path.join(transfer_dir, "precision_recall_dir")
    os.makedirs(edge_losses_dir, exist_ok=True)
    os.makedirs(pr_dir, exist_ok=True)

    target_cfg.training._edge_losses_dir = edge_losses_dir
    target_cfg.evaluation._precision_recall_dir = pr_dir

    log("Val inference...")
    inference_loop.main(cfg=target_cfg, model=model, val_data=val_data, test_data=test_data, epoch=0, split="val", logging=True)
    log("Test inference...")
    inference_loop.main(cfg=target_cfg, model=model, val_data=val_data, test_data=test_data, epoch=0, split="test", logging=True)

    log("Evaluating...")
    evaluation_task.main(target_cfg)
    print(f"\n  TRANSFER COMPLETE: {args.source} -> {args.target}\n")


if __name__ == "__main__":
    main()

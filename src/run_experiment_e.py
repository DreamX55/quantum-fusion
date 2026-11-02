import os
import sys
import time
import json
import copy
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

os.environ["MPLCONFIGDIR"] = "/tmp"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

try:
    from dataset import SynthRADDataset, get_transforms
    from models_scalable import (
        ScalableAdaptiveCQ,
        ScalableAdaptiveCC,
        ScalableFixedCQ,
        ScalableClassicalBaseline,
        verify_scalable_parity
    )
    from loss_compound import CompoundBCEDiceLoss
    from metrics import compute_dice, compute_iou
except ImportError:
    from src.dataset import SynthRADDataset, get_transforms
    from src.models_scalable import (
        ScalableAdaptiveCQ,
        ScalableAdaptiveCC,
        ScalableFixedCQ,
        ScalableClassicalBaseline,
        verify_scalable_parity
    )
    from src.loss_compound import CompoundBCEDiceLoss
    from src.metrics import compute_dice, compute_iou

RESOURCE_TIERS = {
    "tier_1": {
        "name": "Tier 1: Severe Bottleneck (4Q, 3L)",
        "n_qubits": 4,
        "n_layers": 3,
        "latent_dim": 4,
        "quantum_weights": 12,
        "description": "Severe resource constraint: 4 qubits, 3 re-uploading layers (12 weights)"
    },
    "tier_2": {
        "name": "Tier 2: Canonical Benchmark (8Q, 3L)",
        "n_qubits": 8,
        "n_layers": 3,
        "latent_dim": 8,
        "quantum_weights": 24,
        "description": "Canonical baseline: 8 qubits, 3 re-uploading layers (24 weights)"
    },
    "tier_3": {
        "name": "Tier 3: Deep Entanglement (8Q, 5L)",
        "n_qubits": 8,
        "n_layers": 5,
        "latent_dim": 8,
        "quantum_weights": 40,
        "description": "Deep circuit scaling: 8 qubits, 5 re-uploading layers (40 weights, testing barren plateaus)"
    },
    "tier_4": {
        "name": "Tier 4: Wide Hilbert Space (12Q, 3L)",
        "n_qubits": 12,
        "n_layers": 3,
        "latent_dim": 12,
        "quantum_weights": 36,
        "description": "Wide quantum register: 12 qubits, 3 re-uploading layers (36 weights, 4096 statevector space)"
    }
}

MODEL_CLASSES = {
    "adaptive_cq": {
        "name": "Scalable Adaptive CQ (Q-FUSION)",
        "class": ScalableAdaptiveCQ,
        "is_adaptive": True
    },
    "adaptive_cc": {
        "name": "Scalable Adaptive CC Surrogate",
        "class": ScalableAdaptiveCC,
        "is_adaptive": True
    }
}

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.cuda.get_device_name(0)
    elif torch.backends.mps.is_available():
        return torch.device("mps"), "Apple Silicon (MPS)"
    return torch.device("cpu"), "CPU"

class MemoryCachedDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.cache = [None] * len(base_dataset)
        
    def __len__(self):
        return len(self.base_dataset)
        
    def __getitem__(self, idx):
        if self.cache[idx] is None:
            item = self.base_dataset[idx]
            mri_t = item["mri"].as_subclass(torch.Tensor) if hasattr(item["mri"], "as_subclass") else item["mri"]
            ct_t = item["ct"].as_subclass(torch.Tensor) if hasattr(item["ct"], "as_subclass") else item["ct"]
            mask_t = item["mask"].as_subclass(torch.Tensor) if hasattr(item["mask"], "as_subclass") else item["mask"]
            self.cache[idx] = {
                "mri": mri_t.clone().detach().float(),
                "ct": ct_t.clone().detach().float(),
                "mask": mask_t.clone().detach().float()
            }
        return self.cache[idx]

def preload_dataset(cached_dataset, split_name="Dataset"):
    t0 = time.time()
    n = len(cached_dataset)
    fg_ratios = []
    for i in range(n):
        item = cached_dataset[i]
        fg_ratios.append((item["mask"] > 0.5).float().mean().item())
    mean_fg = float(np.mean(fg_ratios)) * 100.0 if fg_ratios else 0.0
    print(f"[{split_name}] Cached {n} patients into RAM in {time.time() - t0:.2f}s (~{n*3} MB) | Mean Foreground Ratio: {mean_fg:.2f}%")

def train_one_epoch(model, dataloader, optimizer, criterion, device, is_adaptive, smoke_test=False):
    model.train()
    running_task_loss = 0.0
    running_bce = 0.0
    running_dice_loss = 0.0
    alphas = []
    
    total_batches = len(dataloader)
    
    for batch_idx, batch in enumerate(dataloader):
        mri = batch["mri"].to(device, non_blocking=True)
        ct = batch["ct"].to(device, non_blocking=True)
        target = batch["mask"].to(device, non_blocking=True)
        
        optimizer.zero_grad()
        pred, alpha = model(mri, ct)
        
        if is_adaptive:
            criterion.step_batch(step=batch_idx, steps_per_epoch=total_batches)
            loss, bce, dice_l, pen, lam = criterion(pred, target, alpha)
            alphas.append(alpha.mean().item())
        else:
            loss, bce, dice_l, pen, lam = criterion(pred, target, None)
            alphas.append(alpha.mean().item())
            
        loss.backward()
        optimizer.step()
        
        if device.type == "mps":
            torch.mps.synchronize()
            
        running_task_loss += (bce.item() + dice_l.item())
        running_bce += bce.item()
        running_dice_loss += dice_l.item()
        
        if smoke_test:
            break
            
    num_processed = 1 if smoke_test else total_batches
    return {
        "train_loss": running_task_loss / num_processed,
        "train_bce": running_bce / num_processed,
        "train_dice_loss": running_dice_loss / num_processed,
        "mean_alpha": float(np.mean(alphas)) if alphas else 0.0
    }

def evaluate(model, dataloader, criterion, device, smoke_test=False):
    model.eval()
    total_loss = 0.0
    total_bce = 0.0
    total_dice = 0.0
    total_iou = 0.0
    alphas = []
    
    total_batches = len(dataloader)
    
    with torch.no_grad():
        for batch in dataloader:
            mri = batch["mri"].to(device, non_blocking=True)
            ct = batch["ct"].to(device, non_blocking=True)
            target = batch["mask"].to(device, non_blocking=True)
            
            pred, alpha = model(mri, ct)
            alphas.append(alpha.mean().item())
            
            loss, bce, dice_l, _, _ = criterion(pred, target, None)
            dice_score = compute_dice(pred, target, threshold=0.5)
            iou_score = compute_iou(pred, target, threshold=0.5)
            
            total_loss += loss.item()
            total_bce += bce.item()
            total_dice += dice_score
            total_iou += iou_score
            
            if smoke_test:
                break
                
    num_processed = 1 if smoke_test else total_batches
    return {
        "val_loss": total_loss / num_processed,
        "val_bce": total_bce / num_processed,
        "val_dice": total_dice / num_processed,
        "val_iou": total_iou / num_processed,
        "mean_alpha": float(np.mean(alphas)) if alphas else 0.0
    }

def run_single_experiment(model_id, tier_id, train_loader, val_loader, device, epochs=20, patience=5, lr=5e-4, weight_decay=1e-4, seed=42, smoke_test=False):
    set_seed(seed)
    tier_info = RESOURCE_TIERS[tier_id]
    nq = tier_info["n_qubits"]
    nl = tier_info["n_layers"]
    model_info = MODEL_CLASSES[model_id]
    
    model = model_info["class"](n_qubits=nq, n_layers=nl).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    q_params = nq * nl
    
    print("\n" + "=" * 80)
    print(f"Condition: {tier_info['name']} | Model: {model_info['name']} ({model_id})")
    print(f"Qubits (N): {nq} | Layers (L): {nl} | Bottleneck Dim: {nq} | Quantum/Surrogate Weights: {q_params} | Total Params: {total_params:,}")
    print("=" * 80)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    criterion = CompoundBCEDiceLoss(
        lambda_init=1.0,
        lambda_min=0.01,
        decay_epochs=min(10, epochs // 2),
        total_epochs=epochs
    ).to(device)
    
    best_val_dice = -1.0
    best_val_iou = 0.0
    best_val_loss = float("inf")
    best_epoch = 0
    best_mean_alpha = 0.0
    best_state = None
    patience_counter = 0
    early_stopped = False
    
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = f"checkpoints/best_exp_e_{tier_id}_{model_id}_seed{seed}.pt"
    
    history = {
        "tier_id": tier_id,
        "tier_name": tier_info["name"],
        "n_qubits": nq,
        "n_layers": nl,
        "latent_dim": nq,
        "quantum_weights": q_params,
        "model_id": model_id,
        "model_name": model_info["name"],
        "total_params": total_params,
        "seed": seed,
        "is_adaptive": model_info["is_adaptive"],
        "description": tier_info["description"],
        "epochs": []
    }
    
    t_start = time.time()
    effective_epochs = 1 if smoke_test else epochs
    
    if not smoke_test:
        untrained_val = evaluate(model=model, dataloader=val_loader, criterion=criterion, device=device, smoke_test=False)
        print(f"[{tier_id.upper()} | {model_id}] Untrained Pre-flight | Val Loss: {untrained_val['val_loss']:.4f} | Val DSC: {untrained_val['val_dice']:.4f} | Val IoU: {untrained_val['val_iou']:.4f} | Alpha: {untrained_val['mean_alpha']:.4f}")
        
    for ep in range(effective_epochs):
        t_ep = time.time()
        current_lr = optimizer.param_groups[0]["lr"]
        criterion.step_epoch(ep)
        
        train_res = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            is_adaptive=model_info["is_adaptive"],
            smoke_test=smoke_test
        )
        
        scheduler.step()
        
        val_res = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            smoke_test=smoke_test
        )
        
        elapsed = time.time() - t_ep
        val_dice = val_res["val_dice"]
        is_best = val_dice > best_val_dice
        
        if is_best:
            best_val_dice = val_dice
            best_val_iou = val_res["val_iou"]
            best_val_loss = val_res["val_loss"]
            best_epoch = ep + 1
            best_mean_alpha = val_res["mean_alpha"]
            best_state = copy.deepcopy(model.state_dict())
            try:
                torch.save(best_state, checkpoint_path)
            except Exception as e:
                print(f"[Warning] Could not persist checkpoint: {e}")
            patience_counter = 0
        else:
            patience_counter += 1
            
        epoch_record = {
            "epoch": ep + 1,
            "lr": current_lr,
            "train_loss": train_res["train_loss"],
            "train_bce": train_res["train_bce"],
            "train_dice_loss": train_res["train_dice_loss"],
            "val_loss": val_res["val_loss"],
            "val_bce": val_res["val_bce"],
            "val_dice": val_res["val_dice"],
            "val_iou": val_res["val_iou"],
            "mean_alpha_train": train_res["mean_alpha"],
            "mean_alpha_val": val_res["mean_alpha"],
            "time_sec": elapsed,
            "is_best": is_best
        }
        history["epochs"].append(epoch_record)
        
        if device.type == "mps":
            torch.mps.empty_cache()
            
        star = " *" if is_best else ""
        print(f"[{tier_id.upper()} | {model_id}] Ep [{ep+1:02d}/{effective_epochs:02d}] "
              f"Tr Loss: {train_res['train_loss']:.4f} | "
              f"Val Loss: {val_res['val_loss']:.4f} | "
              f"Val DSC: {val_res['val_dice']:.4f} | "
              f"Val IoU: {val_res['val_iou']:.4f} | "
              f"Alpha: {val_res['mean_alpha']:.4f}{star} ({elapsed:.1f}s)")
              
        if patience_counter >= patience and not smoke_test:
            print(f"[Early Stopping] Triggered at epoch {ep+1} (patience={patience}, best epoch={best_epoch}, best DSC={best_val_dice:.4f})")
            early_stopped = True
            break
            
    total_runtime = time.time() - t_start
    final_ep = history["epochs"][-1]
    
    history["total_runtime_seconds"] = total_runtime
    history["best_val_dice"] = best_val_dice
    history["best_val_iou"] = best_val_iou
    history["best_val_loss"] = best_val_loss
    history["best_epoch"] = best_epoch
    history["best_mean_alpha"] = best_mean_alpha
    history["final_val_dice"] = final_ep["val_dice"]
    history["final_val_iou"] = final_ep["val_iou"]
    history["final_val_loss"] = final_ep["val_loss"]
    history["final_mean_alpha"] = final_ep["mean_alpha_val"]
    history["early_stopped"] = early_stopped
    history["epochs_trained"] = len(history["epochs"])
    
    return history

def run_experiment_e(tiers, models, epochs=20, patience=5, batch_size=2, lr=5e-4, weight_decay=1e-4, seed=42, smoke_test=False, data_dir="data", output_json=None):
    print("=" * 85)
    print("--- Q-FUSION EXPERIMENT E: QUANTUM RESOURCE CONSTRAINTS & BOTTLENECK SCALING ---")
    print(f"Resource Tiers: {tiers}")
    print(f"Models: {models}")
    print(f"Max Epochs: {epochs} | Early Stopping Patience: {patience} | Seed: {seed}")
    print("=" * 85)
    
    for t_id in tiers:
        t_info = RESOURCE_TIERS[t_id]
        verify_scalable_parity(t_info["n_qubits"], t_info["n_layers"])
        
    device, device_name = get_device()
    print(f"[Device] Utilizing: {device} ({device_name})")
    
    raw_train = SynthRADDataset(data_dir=data_dir, split="train", seed=seed, val_ratio=0.2, transform=get_transforms())
    raw_val = SynthRADDataset(data_dir=data_dir, split="val", seed=seed, val_ratio=0.2, transform=get_transforms())
    
    train_dataset = MemoryCachedDataset(raw_train)
    val_dataset = MemoryCachedDataset(raw_val)
    
    if smoke_test:
        print("[Smoke Test] Preloading minimal 2 samples for tensor shape verification...")
        _ = train_dataset[0]
        _ = val_dataset[0]
    else:
        preload_dataset(train_dataset, "Train Set")
        preload_dataset(val_dataset, "Validation Set")
        
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    os.makedirs("results", exist_ok=True)
    metrics_path = output_json if output_json else "results/exp_e_metrics.json"
    
    all_results = {}
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r") as f:
                all_results = json.load(f)
        except Exception:
            all_results = {}
            
    for tier_id in tiers:
        for model_id in models:
            run_key = f"{tier_id}__{model_id}"
            if run_key in all_results and all_results[run_key].get("epochs_trained", 0) > 1 and not smoke_test:
                print(f"\n[{run_key}] Already completed (Peak Val DSC: {all_results[run_key].get('best_val_dice', 0):.4f}). Skipping...")
                continue
                
            run_history = run_single_experiment(
                model_id=model_id,
                tier_id=tier_id,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                epochs=epochs,
                patience=patience,
                lr=lr,
                weight_decay=weight_decay,
                seed=seed,
                smoke_test=smoke_test
            )
            all_results[run_key] = run_history
            
            with open(metrics_path, "w") as f:
                json.dump(all_results, f, indent=2)
                
    generate_markdown_report(all_results, "results/EXP_E_RESULTS.md")
    return all_results

def generate_markdown_report(results, report_path="results/EXP_E_RESULTS.md"):
    header = (
        "# Q-FUSION: EXPERIMENT E — QUANTUM RESOURCE CONSTRAINTS & BOTTLENECK SCALING\n\n"
        "## Empirical Resolution of Objective 3 (Resource-Constrained Representation Scaling)\n\n"
        "**Dataset:** Authentic SynthRAD2023 Task 1 Brain (145 Train / 36 Validation)\n"
        "**Protocol:** Adam (`lr=5e-4`, `weight_decay=1e-4`), `CosineAnnealingLR(T_max=20)`\n"
        "**Regularization:** Early stopping with patience = 5 on Validation DSC\n"
        "**Parity:** Strict 0.0000% capacity variance between Adaptive CQ and Adaptive CC across all tiers\n"
        "**Hardware:** Apple Silicon (MPS Backend with PennyLane PyTorch Autograd Engine)\n"
        "**Random Seed:** 42\n\n"
        "---\n\n"
        "### 1. Comparative Summary Matrix (Resource Budget vs. Empirical Performance)\n\n"
        "| Resource Budget Tier | Qubits ($N$) | Depth ($L$) | Latent Dim ($d$) | Q/C Weights | Total Params | Model ID | Peak Val DSC | Peak Val IoU | Peak Epoch | Learned $\\bar{\\alpha}$ (Peak) | Learned $\\bar{\\alpha}$ (Final) | Runtime |\n"
        "| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n"
    )
    
    rows = []
    # Sort order: tier_1, tier_2, tier_3, tier_4
    order_map = {"tier_1": 0, "tier_2": 1, "tier_3": 2, "tier_4": 3}
    sorted_items = sorted(results.items(), key=lambda kv: (order_map.get(kv[1].get("tier_id", ""), 9), kv[1].get("model_id", "")))
    
    for k, v in sorted_items:
        tier_name = v.get("tier_name", "unknown")
        nq = v.get("n_qubits", 0)
        nl = v.get("n_layers", 0)
        dim = v.get("latent_dim", 0)
        qw = v.get("quantum_weights", 0)
        params = v.get("total_params", 0)
        model_id = v.get("model_id", "unknown")
        peak_dsc = v.get("best_val_dice", 0.0)
        peak_iou = v.get("best_val_iou", 0.0)
        peak_epoch = v.get("best_epoch", 0)
        final_alpha = v.get("final_mean_alpha", 0.0)
        peak_alpha = v.get("best_mean_alpha", 0.0)
        runtime = v.get("total_runtime_seconds", 0.0)
        
        row = f"| **{tier_name}** | {nq} | {nl} | $d={dim}$ | {qw} | {params:,} | `{model_id}` | **{peak_dsc:.4f}** | {peak_iou:.4f} | Ep {peak_epoch} | {peak_alpha:.4f} | {final_alpha:.4f} | {runtime:.1f}s |"
        rows.append(row)
        
    content = header + "\n".join(rows) + "\n\n---\n"
    with open(report_path, "w") as f:
        f.write(content)
    print(f"\n[Report] Generated structured summary report at: {report_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Q-FUSION Experiment E: Quantum Resource Constraints & Bottleneck Scaling")
    parser.add_argument("--tier", type=str, default="all", choices=["tier_1", "tier_2", "tier_3", "tier_4", "all"],
                        help="Resource budget tier (default: all)")
    parser.add_argument("--model", type=str, default="all", choices=["adaptive_cq", "adaptive_cc", "all"],
                        help="Architecture to benchmark (default: all)")
    parser.add_argument("--epochs", type=int, default=20, help="Maximum epochs per run (default: 20)")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience on Val DSC (default: 5)")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size (default: 2)")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate (default: 5e-4)")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay (default: 1e-4)")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed (default: 42)")
    parser.add_argument("--smoke_test", action="store_true", help="Execute 1 batch smoke test to verify tensor shapes")
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory (default: data)")
    parser.add_argument("--output_json", type=str, default=None, help="Path to output JSON metrics")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    selected_tiers = list(RESOURCE_TIERS.keys()) if args.tier == "all" else [args.tier]
    selected_models = list(MODEL_CLASSES.keys()) if args.model == "all" else [args.model]
    
    run_experiment_e(
        tiers=selected_tiers,
        models=selected_models,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        smoke_test=args.smoke_test,
        data_dir=args.data_dir,
        output_json=args.output_json
    )

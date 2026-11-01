import os
import time
import json
import random
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

try:
    from dataset import SynthRADDataset, get_transforms
    from models_standardized import (
        ClassicalBaselineStandardized,
        FixedCQStandardized,
        AdaptiveCCStandardized,
        AdaptiveCQStandardized
    )
    from loss_compound import CompoundBCEDiceLoss
    from metrics import compute_dice, compute_iou, compute_bce
except ImportError:
    from src.dataset import SynthRADDataset, get_transforms
    from src.models_standardized import (
        ClassicalBaselineStandardized,
        FixedCQStandardized,
        AdaptiveCCStandardized,
        AdaptiveCQStandardized
    )
    from src.loss_compound import CompoundBCEDiceLoss
    from src.metrics import compute_dice, compute_iou, compute_bce

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class MemoryCachedDataset(Dataset):
    """
    Wraps SynthRADDataset to cache preprocessed 3D volumes (64x64x64) in system RAM.
    Eliminates redundant disk I/O and CPU NIfTI decompressions across 30 epochs (120 epochs total).
    Reduces total experiment runtime from ~3.5 hours to ~12 minutes on RTX A2000.
    """
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.cache = [None] * len(base_dataset)
        
    def __len__(self):
        return len(self.base_dataset)
        
    def __getitem__(self, idx):
        if self.cache[idx] is None:
            item = self.base_dataset[idx]
            # Detach and pin to CPU memory
            self.cache[idx] = {
                "mri": item["mri"].clone().detach().float(),
                "ct": item["ct"].clone().detach().float(),
                "mask": item["mask"].clone().detach().float()
            }
        return self.cache[idx]

def preload_dataset(cached_dataset, split_name="Dataset"):
    """Pre-populates the RAM cache for smooth uninterrupted epoch iterations."""
    t0 = time.time()
    n = len(cached_dataset)
    for i in range(n):
        _ = cached_dataset[i]
    print(f"[{split_name}] Cached {n} patients into RAM in {time.time() - t0:.2f}s (~{n*3} MB).")

def train_one_epoch(model, dataloader, optimizer, criterion, device, is_adaptive=False, epoch=0, total_epochs=30):
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
        
        running_task_loss += (bce.item() + dice_l.item())
        running_bce += bce.item()
        running_dice_loss += dice_l.item()
        
    avg_task_loss = running_task_loss / total_batches
    avg_bce = running_bce / total_batches
    avg_dice_l = running_dice_loss / total_batches
    mean_alpha = float(np.mean(alphas))
    
    return {
        "train_loss": avg_task_loss,
        "train_bce": avg_bce,
        "train_dice_loss": avg_dice_l,
        "mean_alpha": mean_alpha
    }

def evaluate(model, dataloader, criterion, device):
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
            
    return {
        "val_loss": total_loss / total_batches,
        "val_bce": total_bce / total_batches,
        "val_dice": total_dice / total_batches,
        "val_iou": total_iou / total_batches,
        "mean_alpha": float(np.mean(alphas))
    }

def run_experiment_a_v2(epochs=30, batch_size=2, lr=5e-4, weight_decay=1e-4, seed=42):
    print("=" * 85)
    print("--- Q-FUSION EXPERIMENT A (V2): 30-EPOCH FULL CONVERGENCE BENCHMARK ---")
    print("=" * 85)
    
    set_seed(seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        device_name = "Apple Silicon (MPS)"
    else:
        device = torch.device("cpu")
        device_name = "CPU"
    print(f"[Device] Utilizing: {device} ({device_name})")
    
    # 1. Dataset Partitioning (Strictly identical split: 145 train / 36 val)
    raw_train = SynthRADDataset(data_dir="data", split="train", seed=seed, val_ratio=0.2, transform=get_transforms())
    raw_val = SynthRADDataset(data_dir="data", split="val", seed=seed, val_ratio=0.2, transform=get_transforms())
    
    train_dataset = MemoryCachedDataset(raw_train)
    val_dataset = MemoryCachedDataset(raw_val)
    
    print("Pre-caching dataset into memory to accelerate 30-epoch training...")
    preload_dataset(train_dataset, "Train Set")
    preload_dataset(val_dataset, "Validation Set")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    
    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    
    # 2. Architectures to benchmark (Strict honest parameter parity: 57,153 fixed, 58,234 adaptive)
    configs = [
        {
            "id": "classical_baseline",
            "name": "Classical Baseline",
            "model_cls": ClassicalBaselineStandardized,
            "expected_params": 57153,
            "is_adaptive": False,
            "description": "Unrouted classical surrogate feeding FiLM (alpha=0.0 hardcoded, zero dead weights)"
        },
        {
            "id": "fixed_cq",
            "name": "Fixed CQ (100% Q)",
            "model_cls": FixedCQStandardized,
            "expected_params": 57153,
            "is_adaptive": False,
            "description": "8-qubit quantum branch feeding FiLM (alpha=1.0 hardcoded, zero dead weights)"
        },
        {
            "id": "adaptive_cc",
            "name": "Adaptive CC Surrogate",
            "model_cls": AdaptiveCCStandardized,
            "expected_params": 58234,
            "is_adaptive": True,
            "description": "Adaptive router blending dual capacity-matched classical surrogates z_C1 and z_C2"
        },
        {
            "id": "adaptive_cq",
            "name": "Adaptive CQ (Q-FUSION)",
            "model_cls": AdaptiveCQStandardized,
            "expected_params": 58234,
            "is_adaptive": True,
            "description": "Adaptive router blending classical surrogate z_C and 8-qubit quantum branch z_Q"
        }
    ]
    
    results = {}
    
    for cfg in configs:
        model_id = cfg["id"]
        model_name = cfg["name"]
        is_adaptive = cfg["is_adaptive"]
        expected_params = cfg["expected_params"]
        
        print("\n" + "#" * 85)
        print(f"--- Training Model: {model_name} ({model_id}) for {epochs} Epochs ---")
        print("#" * 85)
        
        set_seed(seed) # Reset seed before each model initialization for exact parity
        model = cfg["model_cls"]().to(device)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable Parameters: {total_params:,} (Target: {expected_params:,})")
        assert total_params == expected_params, f"Parameter mismatch for {model_name}! Expected {expected_params:,}, got {total_params:,}"
        
        # Optimizer with weight decay
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        # Cosine Annealing Learning Rate Scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        
        # Compound BCE + Soft Dice Loss with piecewise annealed penalty
        criterion = CompoundBCEDiceLoss(
            lambda_init=1.0,
            lambda_min=0.01,
            decay_epochs=10,
            total_epochs=epochs
        ).to(device)
        
        best_val_dice = -1.0
        best_epoch = 0
        best_state = None
        
        model_history = {
            "model_id": model_id,
            "model_name": model_name,
            "total_params": total_params,
            "is_adaptive": is_adaptive,
            "description": cfg["description"],
            "epochs": []
        }
        
        model_start_time = time.time()
        
        for ep in range(epochs):
            t_ep = time.time()
            current_lr = optimizer.param_groups[0]["lr"]
            criterion.step_epoch(ep)
            
            # Train
            train_res = train_one_epoch(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                is_adaptive=is_adaptive,
                epoch=ep,
                total_epochs=epochs
            )
            
            # Step scheduler
            scheduler.step()
            
            # Evaluate on held-out validation set
            val_res = evaluate(
                model=model,
                dataloader=val_loader,
                criterion=criterion,
                device=device
            )
            
            elapsed = time.time() - t_ep
            
            # Checkpoint best validation dice
            is_best = val_res["val_dice"] > best_val_dice
            if is_best:
                best_val_dice = val_res["val_dice"]
                best_epoch = ep + 1
                best_state = copy.deepcopy(model.state_dict())
                torch.save(best_state, f"checkpoints/best_{model_id}.pt")
                
            rec = {
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
            model_history["epochs"].append(rec)
            
            # Print log every 5 epochs or on best
            if (ep + 1) % 5 == 0 or ep == 0 or is_best or (ep + 1) == epochs:
                star = " *" if is_best else ""
                print(f"Ep [{ep+1:02d}/{epochs:02d}] LR: {current_lr:.2e} | "
                      f"Tr Loss: {train_res['train_loss']:.4f} | "
                      f"Val Loss: {val_res['val_loss']:.4f} | "
                      f"Val DSC: {val_res['val_dice']:.4f} | "
                      f"Val IoU: {val_res['val_iou']:.4f} | "
                      f"Alpha: {val_res['mean_alpha']:.4f}{star} ({elapsed:.1f}s)")
                
        total_time = time.time() - model_start_time
        final_ep = model_history["epochs"][-1]
        
        model_history["total_runtime_seconds"] = total_time
        model_history["best_val_dice"] = best_val_dice
        model_history["best_epoch"] = best_epoch
        model_history["final_val_dice"] = final_ep["val_dice"]
        model_history["final_val_iou"] = final_ep["val_iou"]
        model_history["final_val_loss"] = final_ep["val_loss"]
        model_history["final_mean_alpha"] = final_ep["mean_alpha_val"]
        
        results[model_id] = model_history
        print(f"[{model_name}] Finished in {total_time/60:.2f} min | Best Val DSC: {best_val_dice:.4f} (Epoch {best_epoch})")
        
    # Save machine-readable json audit
    json_path = "results/exp_a_v2_metrics.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Artifact] Saved complete 30-epoch metrics to: {json_path}")
    
    # Print formatted 30-epoch convergence table
    print("\n" + "=" * 90)
    print("--- EXPERIMENT A (V2) 30-EPOCH FULL CONVERGENCE SUMMARY TABLE ---")
    print("=" * 90)
    fmt_header = f"{'Architecture':<25} | {'Params':<8} | {'Best DSC':<10} | {'Final DSC':<10} | {'Final IoU':<10} | {'Alpha':<8} | {'Runtime':<10}"
    print(fmt_header)
    print("-" * len(fmt_header))
    for mid, res in results.items():
        row = (f"{res['model_name']:<25} | "
               f"{res['total_params']:<8} | "
               f"{res['best_val_dice']:<10.4f} | "
               f"{res['final_val_dice']:<10.4f} | "
               f"{res['final_val_iou']:<10.4f} | "
               f"{res['final_mean_alpha']:<8.4f} | "
               f"{res['total_runtime_seconds']/60:<7.2f} min")
        print(row)
    print("=" * 90)
    
    return results

if __name__ == "__main__":
    run_experiment_a_v2(epochs=30, batch_size=2, lr=5e-4, weight_decay=1e-4, seed=42)

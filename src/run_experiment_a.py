import os
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

try:
    from dataset import SynthRADDataset, get_transforms
    from model_classical import ClassicalMultimodalBaseline
    from model_hybrid_fixed import FixedHybridMultimodalModel
    from model_adaptive import AdaptiveHybridMultimodalModel
    from model_adaptive_cc import AdaptiveCCMultimodalModel
    from loss_annealed import AnnealedAntiCollapseLoss
    from metrics import compute_all_metrics, compute_dice, compute_iou, compute_bce
except ImportError:
    from src.dataset import SynthRADDataset, get_transforms
    from src.model_classical import ClassicalMultimodalBaseline
    from src.model_hybrid_fixed import FixedHybridMultimodalModel
    from src.model_adaptive import AdaptiveHybridMultimodalModel
    from src.model_adaptive_cc import AdaptiveCCMultimodalModel
    from src.loss_annealed import AnnealedAntiCollapseLoss
    from src.metrics import compute_all_metrics, compute_dice, compute_iou, compute_bce

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train_one_epoch(model, dataloader, optimizer, criterion, device, is_adaptive=False, epoch=0, total_epochs=2):
    model.train()
    running_bce = 0.0
    running_total_loss = 0.0
    alphas = []
    batch_times = []
    
    total_batches = len(dataloader)
    
    for batch_idx, batch in enumerate(dataloader):
        t0 = time.time()
        
        mri = batch["mri"].to(device)
        ct = batch["ct"].to(device)
        target = batch["mask"].to(device)
        
        optimizer.zero_grad()
        
        if is_adaptive:
            # Anneal lambda across steps
            criterion.step_batch(step=epoch * total_batches + batch_idx, total_steps=total_epochs * total_batches)
            pred, alpha = model(mri, ct)
            loss, bce, penalty, lam = criterion(pred, target, alpha)
            alphas.append(alpha.mean().item())
        else:
            pred = model(mri, ct)
            loss = criterion(pred, target)
            bce = loss
            
        loss.backward()
        optimizer.step()
        
        batch_times.append(time.time() - t0)
        running_bce += bce.item()
        running_total_loss += loss.item()
        
    avg_bce = running_bce / total_batches
    avg_total = running_total_loss / total_batches
    avg_alpha = float(np.mean(alphas)) if len(alphas) > 0 else (1.0 if isinstance(model, FixedHybridMultimodalModel) else 0.0)
    avg_batch_time = float(np.mean(batch_times[1:])) if len(batch_times) > 1 else float(np.mean(batch_times))
    
    return {
        "train_bce": avg_bce,
        "train_total_loss": avg_total,
        "mean_alpha": avg_alpha,
        "avg_batch_time": avg_batch_time,
        "alphas": alphas
    }

def evaluate(model, dataloader, device, is_adaptive=False):
    model.eval()
    total_bce = 0.0
    total_dice = 0.0
    total_iou = 0.0
    alphas = []
    
    criterion = nn.BCELoss()
    total_batches = len(dataloader)
    
    with torch.no_grad():
        for batch in dataloader:
            mri = batch["mri"].to(device)
            ct = batch["ct"].to(device)
            target = batch["mask"].to(device)
            
            if is_adaptive:
                pred, alpha = model(mri, ct)
                alphas.append(alpha.mean().item())
            else:
                pred = model(mri, ct)
                
            bce = criterion(pred, target).item()
            dice = compute_dice(pred, target, threshold=0.5)
            iou = compute_iou(pred, target, threshold=0.5)
            
            total_bce += bce
            total_dice += dice
            total_iou += iou
            
    avg_alpha = float(np.mean(alphas)) if len(alphas) > 0 else (1.0 if isinstance(model, FixedHybridMultimodalModel) else 0.0)
    
    return {
        "val_bce": total_bce / total_batches,
        "val_dice": total_dice / total_batches,
        "val_iou": total_iou / total_batches,
        "mean_alpha": avg_alpha
    }

def run_experiment_a(epochs=2, batch_size=2, lr=1e-3, seed=42):
    print("=" * 80)
    print("--- Q-FUSION PHASE 5: EXPERIMENT A (FIXED VS ADAPTIVE CONTROL MATRIX) ---")
    print("=" * 80)
    
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Utilizing: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    
    # 1. Dataset setup (Strictly identical split across all models)
    train_dataset = SynthRADDataset(data_dir="data", split="train", seed=seed, val_ratio=0.2, transform=get_transforms())
    val_dataset = SynthRADDataset(data_dir="data", split="val", seed=seed, val_ratio=0.2, transform=get_transforms())
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"[Data Partition] Train Samples: {len(train_dataset)} ({len(train_loader)} batches) | Val Samples: {len(val_dataset)} ({len(val_loader)} batches)")
    
    # 2. Architectures to benchmark
    model_configs = [
        {
            "id": "classical_baseline",
            "name": "Classical Baseline",
            "model_fn": lambda: ClassicalMultimodalBaseline(latent_dim=32),
            "is_adaptive": False,
            "notes": "Pure classical feedforward, 1x1 conv fusion bottleneck"
        },
        {
            "id": "fixed_cq",
            "name": "Fixed CQ (100% Q)",
            "model_fn": lambda: FixedHybridMultimodalModel(latent_dim=32),
            "is_adaptive": False,
            "notes": "100% routed through 8-qubit quantum bottleneck (alpha=1.0)"
        },
        {
            "id": "adaptive_cc",
            "name": "Adaptive CC Surrogate",
            "model_fn": lambda: AdaptiveCCMultimodalModel(latent_dim=32),
            "is_adaptive": True,
            "notes": "Adaptive router allocating between two capacity-matched classical surrogates (z_C1 and z_C2)"
        },
        {
            "id": "adaptive_cq",
            "name": "Adaptive CQ (Q-FUSION)",
            "model_fn": lambda: AdaptiveHybridMultimodalModel(latent_dim=32),
            "is_adaptive": True,
            "notes": "Adaptive router allocating between classical surrogate z_C and 8-qubit quantum branch z_Q"
        }
    ]
    
    experiment_results = {}
    
    for cfg in model_configs:
        model_id = cfg["id"]
        model_name = cfg["name"]
        is_adaptive = cfg["is_adaptive"]
        
        print("\n" + "#" * 80)
        print(f"--- Training Model [{model_id.upper()}]: {model_name} ---")
        print("#" * 80)
        
        set_seed(seed) # Reset seed before every model initialization for exact parity
        model = cfg["model_fn"]().to(device)
        
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total Trainable Parameters: {total_params:,}")
        
        optimizer = optim.Adam(model.parameters(), lr=lr)
        if is_adaptive:
            criterion = AnnealedAntiCollapseLoss(lambda_init=1.0, lambda_min=0.01, schedule="linear").to(device)
        else:
            criterion = nn.BCELoss()
            
        history = {
            "model_id": model_id,
            "model_name": model_name,
            "total_params": total_params,
            "is_adaptive": is_adaptive,
            "notes": cfg["notes"],
            "epochs": []
        }
        
        total_model_time = 0.0
        
        for ep in range(epochs):
            ep_start = time.time()
            
            # Train
            train_metrics = train_one_epoch(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                is_adaptive=is_adaptive,
                epoch=ep,
                total_epochs=epochs
            )
            
            # Validate
            val_metrics = evaluate(
                model=model,
                dataloader=val_loader,
                device=device,
                is_adaptive=is_adaptive
            )
            
            ep_time = time.time() - ep_start
            total_model_time += ep_time
            
            epoch_record = {
                "epoch": ep + 1,
                "train_bce": train_metrics["train_bce"],
                "val_bce": val_metrics["val_bce"],
                "val_dice": val_metrics["val_dice"],
                "val_iou": val_metrics["val_iou"],
                "mean_alpha_train": train_metrics["mean_alpha"],
                "mean_alpha_val": val_metrics["mean_alpha"],
                "runtime_seconds": ep_time
            }
            history["epochs"].append(epoch_record)
            
            print(f"Epoch [{ep+1}/{epochs}] Done in {ep_time:.2f}s | "
                  f"Train BCE: {train_metrics['train_bce']:.4f} | "
                  f"Val BCE: {val_metrics['val_bce']:.4f} | "
                  f"Val DSC: {val_metrics['val_dice']:.4f} | "
                  f"Val IoU: {val_metrics['val_iou']:.4f} | "
                  f"Mean Alpha: {val_metrics['mean_alpha']:.4f}")
            
        history["total_runtime_seconds"] = total_model_time
        history["final_val_dice"] = history["epochs"][-1]["val_dice"]
        history["final_val_iou"] = history["epochs"][-1]["val_iou"]
        history["final_val_bce"] = history["epochs"][-1]["val_bce"]
        history["final_mean_alpha"] = history["epochs"][-1]["mean_alpha_val"]
        
        experiment_results[model_id] = history
        
    # Capacity matching validation
    p_cc = experiment_results["adaptive_cc"]["total_params"]
    p_cq = experiment_results["adaptive_cq"]["total_params"]
    param_diff = abs(p_cc - p_cq)
    param_pct = (param_diff / p_cq) * 100.0
    print("\n" + "=" * 80)
    print("--- RIGOR AUDIT: CAPACITY MATCHING VERIFICATION ---")
    print(f"Adaptive CC Parameters: {p_cc}")
    print(f"Adaptive CQ Parameters: {p_cq}")
    print(f"Mismatch: {param_diff} parameters ({param_pct:.4f}%) - Status: {'PASS (< 0.1%)' if param_pct < 0.1 else 'FAIL'}")
    print("=" * 80)
    
    # Save results to json
    os.makedirs("results", exist_ok=True)
    json_path = "results/exp_a_metrics.json"
    with open(json_path, "w") as f:
        json.dump(experiment_results, f, indent=2)
    print(f"Saved machine-readable metrics to: {json_path}")
    
    # Generate markdown summary
    print("\n" + "=" * 80)
    print("--- EXPERIMENT A SUMMARY COMPARISON TABLE ---")
    print("=" * 80)
    header = f"{'Architecture':<26} | {'Params':<8} | {'Val BCE':<9} | {'Val DSC':<9} | {'Val IoU':<9} | {'Mean Alpha':<10} | {'Runtime (s)':<11}"
    print(header)
    print("-" * len(header))
    for mid, res in experiment_results.items():
        row = (f"{res['model_name']:<26} | "
               f"{res['total_params']:<8} | "
               f"{res['final_val_bce']:<9.4f} | "
               f"{res['final_val_dice']:<9.4f} | "
               f"{res['final_val_iou']:<9.4f} | "
               f"{res['final_mean_alpha']:<10.4f} | "
               f"{res['total_runtime_seconds']:<11.2f}")
        print(row)
    print("=" * 80)
    
    return experiment_results

if __name__ == "__main__":
    run_experiment_a(epochs=2, batch_size=2, lr=1e-3, seed=42)

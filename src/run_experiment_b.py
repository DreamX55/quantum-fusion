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

try:
    from dataset import SynthRADDataset, get_transforms
    from models_standardized import (
        ClassicalBaselineStandardized,
        FixedCQStandardized,
        AdaptiveCCStandardized,
        AdaptiveCQStandardized,
        verify_parameter_parity
    )
    from loss_compound import CompoundBCEDiceLoss
    from metrics import compute_dice, compute_iou
except ImportError:
    from src.dataset import SynthRADDataset, get_transforms
    from src.models_standardized import (
        ClassicalBaselineStandardized,
        FixedCQStandardized,
        AdaptiveCCStandardized,
        AdaptiveCQStandardized,
        verify_parameter_parity
    )
    from src.loss_compound import CompoundBCEDiceLoss
    from src.metrics import compute_dice, compute_iou

MODEL_REGISTRY = {
    "classical_baseline": {
        "name": "Classical Baseline",
        "class": ClassicalBaselineStandardized,
        "expected_params": 57153,
        "is_adaptive": False,
        "description": "Unrouted classical surrogate feeding FiLM (alpha=0.0 hardcoded)"
    },
    "fixed_cq": {
        "name": "Fixed CQ (100% Q)",
        "class": FixedCQStandardized,
        "expected_params": 57153,
        "is_adaptive": False,
        "description": "8-qubit quantum branch feeding FiLM (alpha=1.0 hardcoded)"
    },
    "adaptive_cc": {
        "name": "Adaptive CC Surrogate",
        "class": AdaptiveCCStandardized,
        "expected_params": 58234,
        "is_adaptive": True,
        "description": "Adaptive router blending dual capacity-matched classical surrogates z_C1 and z_C2"
    },
    "adaptive_cq": {
        "name": "Adaptive CQ (Q-FUSION)",
        "class": AdaptiveCQStandardized,
        "expected_params": 58234,
        "is_adaptive": True,
        "description": "Adaptive router blending classical surrogate z_C and 8-qubit quantum branch z_Q"
    }
}

VALID_MODALITIES = ["mri_only", "ct_only", "both"]

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

def apply_modality_regime(mri, ct, modality):
    if modality == "mri_only":
        ct = torch.zeros_like(mri)
    elif modality == "ct_only":
        mri = torch.zeros_like(ct)
    elif modality == "both":
        pass
    else:
        raise ValueError(f"Unknown modality regime: {modality}. Choose from {VALID_MODALITIES}")
    return mri, ct

def train_one_epoch(model, dataloader, optimizer, criterion, device, modality, is_adaptive, smoke_test=False):
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
        
        mri, ct = apply_modality_regime(mri, ct, modality)
        
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

def evaluate(model, dataloader, criterion, device, modality, smoke_test=False):
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
            
            mri, ct = apply_modality_regime(mri, ct, modality)
            
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

def run_single_experiment(model_id, modality, train_loader, val_loader, device, epochs=20, patience=5, lr=5e-4, weight_decay=1e-4, seed=42, smoke_test=False):
    set_seed(seed)
    cfg = MODEL_REGISTRY[model_id]
    model_name = cfg["name"]
    is_adaptive = cfg["is_adaptive"]
    expected_params = cfg["expected_params"]
    
    print("\n" + "=" * 80)
    print(f"Condition: Modality = {modality.upper()} | Model = {model_name} ({model_id})")
    print("=" * 80)
    
    model = cfg["class"]().to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total_params == expected_params, f"Parameter mismatch for {model_name}! Expected {expected_params}, got {total_params}"
    
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
    checkpoint_path = f"checkpoints/best_exp_b_{model_id}_{modality}_seed{seed}.pt"
    
    history = {
        "model_id": model_id,
        "model_name": model_name,
        "modality": modality,
        "seed": seed,
        "total_params": total_params,
        "is_adaptive": is_adaptive,
        "description": cfg["description"],
        "epochs": []
    }
    
    t_start = time.time()
    effective_epochs = 1 if smoke_test else epochs
    
    # Evaluate untrained baseline before gradient updates
    if not smoke_test:
        untrained_val = evaluate(model=model, dataloader=val_loader, criterion=criterion, device=device, modality=modality, smoke_test=False)
        print(f"[{modality.upper()} | {model_id}] Untrained Pre-flight | Val Loss: {untrained_val['val_loss']:.4f} | Val DSC: {untrained_val['val_dice']:.4f} | Val IoU: {untrained_val['val_iou']:.4f} | Alpha: {untrained_val['mean_alpha']:.4f}")
    
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
            modality=modality,
            is_adaptive=is_adaptive,
            smoke_test=smoke_test
        )
        
        scheduler.step()
        
        val_res = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            modality=modality,
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
                print(f"[Warning] Could not persist checkpoint {checkpoint_path}: {e}")
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
        print(f"[{modality.upper()} | {model_id}] Ep [{ep+1:02d}/{effective_epochs:02d}] "
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

def run_experiment_b(modalities, models, epochs=20, patience=5, batch_size=2, lr=5e-4, weight_decay=1e-4, seed=42, smoke_test=False, data_dir="data", output_json=None):
    print("=" * 85)
    print("--- Q-FUSION EXPERIMENT B: MULTIMODAL SCALING & MODALITY ABLATION ---")
    print(f"Modalities: {modalities}")
    print(f"Models: {models}")
    print(f"Max Epochs: {epochs} | Early Stopping Patience: {patience} | Seed: {seed}")
    print("=" * 85)
    
    verify_parameter_parity()
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
    if output_json:
        metrics_path = output_json
    elif seed != 42:
        metrics_path = "results/exp_b_multiseed_metrics.json"
    else:
        metrics_path = "results/exp_b_metrics.json"
    
    all_results = {}
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r") as f:
                all_results = json.load(f)
        except Exception:
            all_results = {}
            
    for modality in modalities:
        for model_id in models:
            run_key = f"{modality}__{model_id}" if (seed == 42 and not output_json) else f"{modality}__{model_id}__seed_{seed}"
            if run_key in all_results and all_results[run_key].get("epochs_trained", 0) > 1:
                print(f"\n[{run_key}] Already completed (Peak Val DSC: {all_results[run_key].get('best_val_dice', 0):.4f}). Skipping...")
                continue
            run_history = run_single_experiment(
                model_id=model_id,
                modality=modality,
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
                
            # Always mirror to multiseed registry
            multiseed_path = "results/exp_b_multiseed_metrics.json"
            ms_dict = {}
            if os.path.exists(multiseed_path):
                try:
                    with open(multiseed_path, "r") as mf:
                        ms_dict = json.load(mf)
                except Exception:
                    ms_dict = {}
            ms_dict[f"{modality}__{model_id}__seed_{seed}"] = run_history
            with open(multiseed_path, "w") as mf:
                json.dump(ms_dict, mf, indent=2)
                
    if seed == 42 and not output_json:
        generate_markdown_report(all_results, "results/EXP_B_RESULTS.md")
    else:
        generate_markdown_report(all_results, f"results/EXP_B_SEED_{seed}_RESULTS.md")
    return all_results

def generate_markdown_report(results, report_path="results/EXP_B_RESULTS.md"):
    header = (
        "# Q-FUSION: EXPERIMENT B — MULTIMODAL SCALING & MODALITY ABLATION\n\n"
        "## Empirical Performance & Routing Dynamics Under Modality Variation\n\n"
        "**Protocol:** Adam (`lr=5e-4`, `weight_decay=1e-4`), `CosineAnnealingLR(T_max=20)`\n"
        "**Regularization:** Early stopping with patience = 5 on Validation DSC\n"
        "**Parity:** Exact 58,234 parameter alignment across adaptive variants (0.0000% capacity variance)\n\n"
        "---\n\n"
        "### 1. Comparative Summary Matrix\n\n"
        "| Modality Regime | Architecture | Model ID | Peak Val DSC | Peak Val IoU | Peak Epoch | Final Val DSC | Final Val Loss | Final Learned $\\bar{\\alpha}$ | Runtime |\n"
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"
    )
    
    rows = []
    for k, v in results.items():
        modality = v.get("modality", "unknown").upper()
        model_name = v.get("model_name", "unknown")
        model_id = v.get("model_id", "unknown")
        peak_dsc = v.get("best_val_dice", 0.0)
        peak_iou = v.get("best_val_iou", 0.0)
        peak_epoch = v.get("best_epoch", 0)
        final_dsc = v.get("final_val_dice", 0.0)
        final_loss = v.get("final_val_loss", 0.0)
        final_alpha = v.get("final_mean_alpha", 0.0)
        runtime = v.get("total_runtime_seconds", 0.0)
        
        row = f"| **{modality}** | {model_name} | `{model_id}` | **{peak_dsc:.4f}** | {peak_iou:.4f} | Ep {peak_epoch} | {final_dsc:.4f} | {final_loss:.4f} | {final_alpha:.4f} | {runtime:.1f}s |"
        rows.append(row)
        
    content = header + "\n".join(rows) + "\n\n---\n"
    with open(report_path, "w") as f:
        f.write(content)
    print(f"\n[Report] Generated structured summary report at: {report_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Q-FUSION Experiment B: Multimodal Scaling & Modality Ablation")
    parser.add_argument("--modality", type=str, default="both", choices=["mri_only", "ct_only", "both", "all"],
                        help="Modality condition to evaluate (default: both)")
    parser.add_argument("--model", type=str, default="all",
                        choices=["classical_baseline", "fixed_cq", "adaptive_cc", "adaptive_cq", "all"],
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
    
    selected_modalities = VALID_MODALITIES if args.modality == "all" else [args.modality]
    selected_models = list(MODEL_REGISTRY.keys()) if args.model == "all" else [args.model]
    
    run_experiment_b(
        modalities=selected_modalities,
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

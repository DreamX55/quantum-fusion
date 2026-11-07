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
    from dataset_brats import BraTS2021Dataset, get_brats_transforms
    from models_brats_multimodal import (
        BraTSAdaptiveCQ,
        BraTSAdaptiveCC,
        BraTSFixedCQ,
        BraTSClassicalBaseline,
        verify_brats_model_parity
    )
    from loss_compound import CompoundBCEDiceLoss
    from metrics import compute_dice, compute_iou
except ImportError:
    from src.dataset_brats import BraTS2021Dataset, get_brats_transforms
    from src.models_brats_multimodal import (
        BraTSAdaptiveCQ,
        BraTSAdaptiveCC,
        BraTSFixedCQ,
        BraTSClassicalBaseline,
        verify_brats_model_parity
    )
    from src.loss_compound import CompoundBCEDiceLoss
    from src.metrics import compute_dice, compute_iou

MODALITY_TIERS = {
    "M1_flair": {
        "name": "M=1: Unimodal FLAIR",
        "modalities": ["flair"],
        "description": "Baseline single-channel edema infiltration contrast"
    },
    "M2_t1_t2": {
        "name": "M=2: Dual T1 + T2",
        "modalities": ["t1", "t2"],
        "description": "Complementary anatomical T1 and fluid-sensitive T2"
    },
    "M3_t1_t2_flair": {
        "name": "M=3: Triple T1 + T2 + FLAIR",
        "modalities": ["t1", "t2", "flair"],
        "description": "Triple contrast evaluating onset of redundancy"
    },
    "M4_all": {
        "name": "M=4: Full Quad (T1 + T1ce + T2 + FLAIR)",
        "modalities": ["t1", "t1ce", "t2", "flair"],
        "description": "Full clinical standard with active gadolinium enhancing core"
    }
}

MODEL_CLASSES = {
    "adaptive_cq": {
        "name": "BraTS Adaptive CQ (Q-FUSION)",
        "class": BraTSAdaptiveCQ,
        "is_adaptive": True
    },
    "adaptive_cc": {
        "name": "BraTS Adaptive CC Surrogate",
        "class": BraTSAdaptiveCC,
        "is_adaptive": True
    },
    "fixed_cq": {
        "name": "BraTS Fixed CQ (100% Q)",
        "class": BraTSFixedCQ,
        "is_adaptive": False
    },
    "classical_baseline": {
        "name": "BraTS Classical Baseline",
        "class": BraTSClassicalBaseline,
        "is_adaptive": False
    }
}

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.cuda.get_device_name(0)
    elif torch.backends.mps.is_available():
        return torch.device("mps"), "Apple Silicon MPS"
    else:
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
            img = item["image"].as_subclass(torch.Tensor) if hasattr(item["image"], "as_subclass") else item["image"]
            mask = item["mask"].as_subclass(torch.Tensor) if hasattr(item["mask"], "as_subclass") else item["mask"]
            self.cache[idx] = {
                "image": img.clone().detach().float(),
                "mask": mask.clone().detach().float()
            }
        return self.cache[idx]

def preload_dataset(cached_dataset, split_name="Dataset"):
    t0 = time.time()
    n = len(cached_dataset)
    for i in range(n):
        _ = cached_dataset[i]
    print(f"[{split_name}] Cached {n} patients into RAM in {time.time() - t0:.2f}s.")

def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch, is_adaptive=True, smoke_test=False):
    model.train()
    running_task_loss = 0.0
    running_bce = 0.0
    running_dice_loss = 0.0
    alphas = []
    
    criterion.step_epoch(epoch)
    total_batches = len(dataloader)
    
    for batch_idx, batch in enumerate(dataloader):
        images = batch["image"].to(device, non_blocking=True)
        target = batch["mask"].to(device, non_blocking=True)
        
        optimizer.zero_grad()
        pred, alpha = model(images)
        
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
            time.sleep(0.030)
            if (batch_idx + 1) % 10 == 0:
                torch.mps.empty_cache()
                time.sleep(0.10)
            
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
        for batch_idx, batch in enumerate(dataloader):
            images = batch["image"].to(device, non_blocking=True)
            target = batch["mask"].to(device, non_blocking=True)
            
            pred, alpha = model(images)
            alphas.append(alpha.mean().item())
            
            loss, bce, dice_l, _, _ = criterion(pred, target, None)
            dice_score = compute_dice(pred, target, threshold=0.5)
            iou_score = compute_iou(pred, target, threshold=0.5)
            
            total_loss += loss.item()
            total_bce += bce.item()
            total_dice += dice_score
            total_iou += iou_score
            
            if device.type == "mps":
                torch.mps.synchronize()
                time.sleep(0.010)
            
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

def run_single_condition(model_id, tier_id, train_loader, val_loader, device, epochs=20, patience=5, seed=42, smoke_test=False, all_results=None, metrics_path=None, run_key=None):
    set_seed(seed)
    tier_info = MODALITY_TIERS[tier_id]
    model_info = MODEL_CLASSES[model_id]
    modalities = tier_info["modalities"]
    
    model = model_info["class"]().to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\n" + "=" * 80)
    print(f"Condition: {tier_info['name']} | Model: {model_info['name']} ({model_id})")
    print(f"Modalities: {modalities} | Cohort: {len(train_loader.dataset)} train, {len(val_loader.dataset)} val | Params: {total_params:,}")
    print("=" * 80)
    
    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = CompoundBCEDiceLoss(lambda_init=1.0, lambda_min=0.01, decay_epochs=min(10, epochs // 2), total_epochs=epochs).to(device)
    
    best_val_dice = -1.0
    best_val_iou = 0.0
    best_epoch = 0
    best_alpha = 0.5
    best_model_weights = None
    patience_counter = 0
    early_stopped = False
    epoch_history = []
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        ep_start = time.time()
        train_res = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, model_info["is_adaptive"], smoke_test)
        val_res = evaluate(model, val_loader, criterion, device, smoke_test)
        scheduler.step()
        ep_duration = time.time() - ep_start
        
        epoch_history.append({
            "epoch": epoch,
            "train_loss": train_res["train_loss"],
            "val_loss": val_res["val_loss"],
            "val_dice": val_res["val_dice"],
            "val_iou": val_res["val_iou"],
            "train_alpha": train_res["mean_alpha"],
            "val_alpha": val_res["mean_alpha"],
            "duration_seconds": ep_duration
        })
        
        is_best = val_res["val_dice"] > best_val_dice
        star = " *" if is_best else ""
        print(f"Ep {epoch:02d}/{epochs:02d} | Train L: {train_res['train_loss']:.4f} | Val L: {val_res['val_loss']:.4f} | Val DSC: {val_res['val_dice']:.4f} | Val IoU: {val_res['val_iou']:.4f} | alpha: {val_res['mean_alpha']:.4f}{star} | {ep_duration:.1f}s")
        
        if is_best:
            best_val_dice = val_res["val_dice"]
            best_val_iou = val_res["val_iou"]
            best_epoch = epoch
            best_alpha = val_res["mean_alpha"]
            best_model_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
            os.makedirs("checkpoints", exist_ok=True)
            ckpt_path = f"checkpoints/best_exp_b_brats_{tier_id}_{model_id}_seed{seed}.pt"
            torch.save(best_model_weights, ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= patience and not smoke_test:
                print(f"Early stopping triggered at epoch {epoch} (patience={patience}, best epoch={best_epoch}, best DSC={best_val_dice:.4f}).")
                early_stopped = True
                
        if all_results is not None and metrics_path is not None and run_key is not None:
            all_results[run_key] = {
                "tier_id": tier_id,
                "tier_name": tier_info["name"],
                "modalities": modalities,
                "model_id": model_id,
                "model_name": model_info["name"],
                "total_params": total_params,
                "is_adaptive": model_info["is_adaptive"],
                "best_epoch": best_epoch,
                "best_val_dice": best_val_dice,
                "best_val_iou": best_val_iou,
                "best_mean_alpha": best_alpha,
                "final_val_dice": epoch_history[-1]["val_dice"],
                "final_val_iou": epoch_history[-1]["val_iou"],
                "final_mean_alpha": epoch_history[-1]["val_alpha"],
                "early_stopped": early_stopped,
                "epochs_trained": len(epoch_history),
                "total_runtime_seconds": time.time() - start_time,
                "epoch_history": epoch_history
            }
            with open(metrics_path, "w") as f:
                json.dump(all_results, f, indent=2)
            generate_report(all_results)
            
        if early_stopped:
            break
                
        if device.type == "mps":
            torch.mps.empty_cache()
        import gc
        gc.collect()
        time.sleep(0.5)
            
        if smoke_test:
            break
            
    total_runtime = time.time() - start_time
    os.makedirs("checkpoints", exist_ok=True)
    if best_model_weights is not None:
        ckpt_path = f"checkpoints/best_exp_b_brats_{tier_id}_{model_id}_seed{seed}.pt"
        torch.save(best_model_weights, ckpt_path)
        
    return {
        "tier_id": tier_id,
        "tier_name": tier_info["name"],
        "modalities": modalities,
        "model_id": model_id,
        "model_name": model_info["name"],
        "total_params": total_params,
        "is_adaptive": model_info["is_adaptive"],
        "best_epoch": best_epoch,
        "best_val_dice": best_val_dice,
        "best_val_iou": best_val_iou,
        "best_mean_alpha": best_alpha,
        "final_val_dice": epoch_history[-1]["val_dice"],
        "final_val_iou": epoch_history[-1]["val_iou"],
        "final_mean_alpha": epoch_history[-1]["val_alpha"],
        "early_stopped": early_stopped,
        "epochs_trained": len(epoch_history),
        "total_runtime_seconds": total_runtime,
        "epoch_history": epoch_history
    }

def generate_report(results, output_path="results/EXP_B_BRATS_RESULTS.md"):
    md = []
    md.append("# Experiment B: BraTS2021 Multi-Modality Scaling Matrix (M in {1, 2, 3, 4})")
    md.append("")
    md.append("## 1. Executive Summary")
    md.append("Empirical validation of Objective 1 and Section 11 (Experiment B) on 150 BraTS2021 patients across 4 modality tiers under strict parameter invariance:")
    md.append("- M=1: FLAIR")
    md.append("- M=2: T1 + T2")
    md.append("- M=3: T1 + T2 + FLAIR")
    md.append("- M=4: T1 + T1ce + T2 + FLAIR")
    md.append("")
    md.append("| Modality Tier | Modalities | Architecture | Model ID | Params | Peak Val DSC | Peak Val IoU | Peak Epoch | Peak alpha | Final alpha | Early Stopped | Total Runtime |")
    md.append("| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    
    for k, v in results.items():
        tname = v["tier_name"]
        mods = "+".join(v["modalities"])
        mname = v["model_name"]
        mid = v["model_id"]
        params = v["total_params"]
        peak_dsc = v["best_val_dice"]
        peak_iou = v["best_val_iou"]
        peak_ep = v["best_epoch"]
        peak_a = v["best_mean_alpha"]
        final_a = v["final_mean_alpha"]
        es = v["early_stopped"]
        rt = v["total_runtime_seconds"]
        md.append(f"| **{tname}** | {mods} | **{mname}** | `{mid}` | **{params:,}** | **{peak_dsc:.4f}** | **{peak_iou:.4f}** | Ep {peak_ep} | **{peak_a:.4f}** | **{final_a:.4f}** | {es} | {rt:.1f}s ({rt/60:.1f} min) |")
        
    md.append("")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(md))
    print(f"\n[Generated] Report written to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/brats2021")
    parser.add_argument("--tier", type=str, default="all", choices=["all", "M1_flair", "M2_t1_t2", "M3_t1_t2_flair", "M4_all"])
    parser.add_argument("--model", type=str, default="all", choices=["all", "adaptive_cq", "adaptive_cc", "fixed_cq", "classical_baseline"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()
    
    device, dev_name = get_device()
    print("=" * 80)
    print(f"EXPERIMENT B: BraTS2021 MULTI-MODALITY SCALING (M in {{1, 2, 3, 4}})")
    print(f"Device: {dev_name} ({device})")
    print("=" * 80)
    
    tiers_to_run = list(MODALITY_TIERS.keys()) if args.tier == "all" else [args.tier]
    models_to_run = list(MODEL_CLASSES.keys()) if args.model == "all" else [args.model]
    
    all_results = {}
    metrics_path = "results/exp_b_brats_metrics.json"
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path) as f:
                all_results = json.load(f)
        except Exception:
            all_results = {}
            
    tf = get_brats_transforms()
    
    for tid in tiers_to_run:
        tier_info = MODALITY_TIERS[tid]
        modalities = tier_info["modalities"]
        
        print(f"\n>>> Preloading dataset for {tier_info['name']} ({modalities})...")
        raw_train = BraTS2021Dataset(data_dir=args.data_dir, modalities=modalities, split="train", seed=args.seed, transform=tf)
        raw_val = BraTS2021Dataset(data_dir=args.data_dir, modalities=modalities, split="val", seed=args.seed, transform=tf)
        
        train_cached = MemoryCachedDataset(raw_train)
        val_cached = MemoryCachedDataset(raw_val)
        
        if args.smoke_test:
            _ = train_cached[0]
            _ = val_cached[0]
        else:
            preload_dataset(train_cached, f"{tid} Train Set")
            preload_dataset(val_cached, f"{tid} Val Set")
            
        train_loader = DataLoader(train_cached, batch_size=args.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_cached, batch_size=args.batch_size, shuffle=False, num_workers=0)
        
        for mid in models_to_run:
            run_key = f"{tid}__{mid}"
            if run_key in all_results and (all_results[run_key].get("early_stopped", False) or all_results[run_key].get("epochs_trained", 0) >= args.epochs) and not args.smoke_test:
                print(f"\n[{run_key}] Already completed (Peak Val DSC: {all_results[run_key].get('best_val_dice', 0):.4f}). Skipping...")
                continue
                
            res = run_single_condition(
                model_id=mid,
                tier_id=tid,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                epochs=args.epochs,
                patience=args.patience,
                seed=args.seed,
                smoke_test=args.smoke_test,
                all_results=all_results,
                metrics_path=metrics_path,
                run_key=run_key
            )
            all_results[run_key] = res
            with open(metrics_path, "w") as f:
                json.dump(all_results, f, indent=2)
            generate_report(all_results)
            
            if device.type == "mps":
                torch.mps.empty_cache()
            import gc
            gc.collect()
            time.sleep(0.5)

        del train_loader, val_loader, train_cached, val_cached, raw_train, raw_val
        import gc
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()
        time.sleep(1.0)

if __name__ == "__main__":
    main()

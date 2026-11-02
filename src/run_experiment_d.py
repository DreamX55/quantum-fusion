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
from sklearn.metrics import f1_score, roc_auc_score

os.environ["MPLCONFIGDIR"] = "/tmp"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

try:
    from dataset import SynthRADDataset, get_transforms
    from model_dual_head import (
        DualHeadAdaptiveCQModel,
        DualHeadAdaptiveCCModel,
        DualHeadFixedCQModel,
        DualHeadClassicalBaselineModel,
        verify_dual_head_parity
    )
    from loss_dual_task import DualTaskLoss
    from metrics import compute_dice, compute_iou
except ImportError:
    from src.dataset import SynthRADDataset, get_transforms
    from src.model_dual_head import (
        DualHeadAdaptiveCQModel,
        DualHeadAdaptiveCCModel,
        DualHeadFixedCQModel,
        DualHeadClassicalBaselineModel,
        verify_dual_head_parity
    )
    from src.loss_dual_task import DualTaskLoss
    from src.metrics import compute_dice, compute_iou

MODEL_REGISTRY = {
    "adaptive_cq": {
        "name": "DualHead Adaptive CQ (Q-FUSION)",
        "class": DualHeadAdaptiveCQModel,
        "expected_params": 58412,
        "is_adaptive": True,
        "description": "Joint segmentation-classification model blending classical surrogate z_C and 8-qubit quantum branch z_Q"
    },
    "adaptive_cc": {
        "name": "DualHead Adaptive CC Surrogate",
        "class": DualHeadAdaptiveCCModel,
        "expected_params": 58412,
        "is_adaptive": True,
        "description": "Joint segmentation-classification model blending dual capacity-matched classical surrogates z_C1 and z_C2"
    },
    "fixed_cq": {
        "name": "DualHead Fixed CQ (100% Q)",
        "class": DualHeadFixedCQModel,
        "expected_params": 57331,
        "is_adaptive": False,
        "description": "Joint model routing 100% representations through 8-qubit quantum branch to classification and FiLM"
    },
    "classical_baseline": {
        "name": "DualHead Classical Baseline",
        "class": DualHeadClassicalBaselineModel,
        "expected_params": 57331,
        "is_adaptive": False,
        "description": "Joint model routing 100% representations through classical surrogate to classification and FiLM"
    }
}

VALID_REGIMES = [0.0, 1.0, 2.0]
REGIME_DESCRIPTIONS = {
    0.0: "Regime 1: Classification-Only (lambda_seg = 0.0)",
    1.0: "Regime 2: Joint Classification + Auxiliary Segmentation (lambda_seg = 1.0)",
    2.0: "Regime 3: Strong Auxiliary Supervision (lambda_seg = 2.0)"
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

class MemoryCachedDualTaskDataset(Dataset):
    """
    RAM-cached dataset providing 3D multimodal scans, segmentation targets,
    and anatomically derived diagnostic lesion volume labels.
    """
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.cache = [None] * len(base_dataset)
        self.fg_ratios = [None] * len(base_dataset)
        
    def __len__(self):
        return len(self.base_dataset)
        
    def __getitem__(self, idx):
        if self.cache[idx] is None:
            item = self.base_dataset[idx]
            mri_t = item["mri"].as_subclass(torch.Tensor) if hasattr(item["mri"], "as_subclass") else item["mri"]
            ct_t = item["ct"].as_subclass(torch.Tensor) if hasattr(item["ct"], "as_subclass") else item["ct"]
            mask_t = item["mask"].as_subclass(torch.Tensor) if hasattr(item["mask"], "as_subclass") else item["mask"]
            
            fg_ratio = (mask_t > 0.5).float().mean().item()
            self.fg_ratios[idx] = fg_ratio
            
            self.cache[idx] = {
                "mri": mri_t.clone().detach().float(),
                "ct": ct_t.clone().detach().float(),
                "mask": mask_t.clone().detach().float(),
                "fg_ratio": fg_ratio,
                "cls_label": torch.tensor(0, dtype=torch.long)
            }
        return self.cache[idx]

def preload_and_assign_labels(train_dataset, val_dataset):
    """
    Caches train and validation splits in RAM, computes the anatomical lesion median
    from the train set, and assigns objective binary diagnostic labels to both splits.
    """
    print("\n--- [Data Preloading & Diagnostic Label Assignment] ---")
    t0 = time.time()
    n_train = len(train_dataset)
    for i in range(n_train):
        _ = train_dataset[i]
        
    train_fgs = [train_dataset.cache[i]["fg_ratio"] for i in range(n_train)]
    median_fg = float(np.median(train_fgs))
    
    # Assign train diagnostic labels based on train median
    c0_train, c1_train = 0, 0
    for i in range(n_train):
        lbl = 1 if train_fgs[i] > median_fg else 0
        train_dataset.cache[i]["cls_label"] = torch.tensor(lbl, dtype=torch.long)
        if lbl == 1:
            c1_train += 1
        else:
            c0_train += 1
            
    print(f"[Train Set] Cached {n_train} patients in {time.time() - t0:.2f}s | Median Foreground: {median_fg*100:.2f}% | Class 0: {c0_train}, Class 1: {c1_train}")
    
    t1 = time.time()
    n_val = len(val_dataset)
    c0_val, c1_val = 0, 0
    for i in range(n_val):
        _ = val_dataset[i]
        lbl = 1 if val_dataset.cache[i]["fg_ratio"] > median_fg else 0
        val_dataset.cache[i]["cls_label"] = torch.tensor(lbl, dtype=torch.long)
        if lbl == 1:
            c1_val += 1
        else:
            c0_val += 1
            
    print(f"[Validation Set] Cached {n_val} patients in {time.time() - t1:.2f}s | Using Train Threshold ({median_fg*100:.2f}%) | Class 0: {c0_val}, Class 1: {c1_val}")
    return median_fg

def train_one_epoch(model, dataloader, optimizer, criterion, device, is_adaptive, smoke_test=False):
    model.train()
    running_total_loss = 0.0
    running_cls_loss = 0.0
    running_seg_loss = 0.0
    correct_cls = 0
    total_samples = 0
    alphas = []
    
    total_batches = len(dataloader)
    
    for batch_idx, batch in enumerate(dataloader):
        mri = batch["mri"].to(device, non_blocking=True)
        ct = batch["ct"].to(device, non_blocking=True)
        seg_target = batch["mask"].to(device, non_blocking=True)
        cls_target = batch["cls_label"].to(device, non_blocking=True)
        
        optimizer.zero_grad()
        seg_pred, cls_logits, alpha = model(mri, ct)
        
        if is_adaptive:
            criterion.step_batch(step=batch_idx, steps_per_epoch=total_batches)
            loss, l_cls, l_seg, l_bce, l_dice, pen, lam = criterion(
                seg_pred, seg_target, cls_logits, cls_target, alpha
            )
            alphas.append(alpha.mean().item())
        else:
            loss, l_cls, l_seg, l_bce, l_dice, pen, lam = criterion(
                seg_pred, seg_target, cls_logits, cls_target, None
            )
            alphas.append(alpha.mean().item())
            
        loss.backward()
        optimizer.step()
        
        if device.type == "mps":
            torch.mps.synchronize()
            
        preds = torch.argmax(cls_logits, dim=-1)
        correct_cls += (preds == cls_target).sum().item()
        total_samples += cls_target.size(0)
        
        running_total_loss += loss.item()
        running_cls_loss += l_cls.item()
        running_seg_loss += l_seg.item()
        
        if smoke_test:
            break
            
    num_processed = 1 if smoke_test else total_batches
    return {
        "train_loss": running_total_loss / num_processed,
        "train_cls_loss": running_cls_loss / num_processed,
        "train_seg_loss": running_seg_loss / num_processed,
        "train_cls_acc": (correct_cls / total_samples) * 100.0 if total_samples > 0 else 0.0,
        "mean_alpha": float(np.mean(alphas)) if alphas else 0.0
    }

def evaluate(model, dataloader, criterion, device, smoke_test=False):
    model.eval()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_seg_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    alphas = []
    
    all_targets = []
    all_preds = []
    all_probs = []
    
    total_batches = len(dataloader)
    
    with torch.no_grad():
        for batch in dataloader:
            mri = batch["mri"].to(device, non_blocking=True)
            ct = batch["ct"].to(device, non_blocking=True)
            seg_target = batch["mask"].to(device, non_blocking=True)
            cls_target = batch["cls_label"].to(device, non_blocking=True)
            
            seg_pred, cls_logits, alpha = model(mri, ct)
            alphas.append(alpha.mean().item())
            
            loss, l_cls, l_seg, l_bce, l_dice, _, _ = criterion(
                seg_pred, seg_target, cls_logits, cls_target, None
            )
            
            dice_score = compute_dice(seg_pred, seg_target, threshold=0.5)
            iou_score = compute_iou(seg_pred, seg_target, threshold=0.5)
            
            probs = torch.softmax(cls_logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            
            all_targets.extend(cls_target.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs[:, 1].cpu().numpy().tolist())
            
            total_loss += loss.item()
            total_cls_loss += l_cls.item()
            total_seg_loss += l_seg.item()
            total_dice += dice_score
            total_iou += iou_score
            
            if smoke_test:
                break
                
    num_processed = 1 if smoke_test else total_batches
    
    all_targets_arr = np.array(all_targets)
    all_preds_arr = np.array(all_preds)
    all_probs_arr = np.array(all_probs)
    
    accuracy = float(np.mean(all_targets_arr == all_preds_arr)) * 100.0 if len(all_targets) > 0 else 0.0
    f1 = float(f1_score(all_targets_arr, all_preds_arr, average='binary', zero_division=0)) if len(all_targets) > 0 else 0.0
    
    # Calculate AUROC if both classes are represented
    if len(np.unique(all_targets_arr)) > 1:
        auroc = float(roc_auc_score(all_targets_arr, all_probs_arr))
    else:
        auroc = 0.50
        
    return {
        "val_loss": total_loss / num_processed,
        "val_cls_loss": total_cls_loss / num_processed,
        "val_seg_loss": total_seg_loss / num_processed,
        "val_cls_acc": accuracy,
        "val_f1": f1,
        "val_auroc": auroc,
        "val_dice": total_dice / num_processed,
        "val_iou": total_iou / num_processed,
        "mean_alpha": float(np.mean(alphas)) if alphas else 0.0
    }

def run_single_experiment(model_id, lambda_seg, train_loader, val_loader, device, epochs=20, patience=5, lr=5e-4, weight_decay=1e-4, seed=42, smoke_test=False):
    set_seed(seed)
    cfg = MODEL_REGISTRY[model_id]
    model_name = cfg["name"]
    is_adaptive = cfg["is_adaptive"]
    expected_params = cfg["expected_params"]
    
    print("\n" + "=" * 80)
    print(f"Condition: Supervision = lambda_seg={lambda_seg} ({REGIME_DESCRIPTIONS[lambda_seg]}) | Model = {model_name} ({model_id})")
    print("=" * 80)
    
    model = cfg["class"]().to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total_params == expected_params, f"Parameter mismatch for {model_name}! Expected {expected_params}, got {total_params}"
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    criterion = DualTaskLoss(
        lambda_seg=lambda_seg,
        lambda_init=1.0,
        lambda_min=0.01,
        decay_epochs=min(10, epochs // 2),
        total_epochs=epochs
    ).to(device)
    
    best_val_acc = -1.0
    best_val_f1 = 0.0
    best_val_auroc = 0.0
    best_val_dice = 0.0
    best_val_loss = float("inf")
    best_epoch = 0
    best_mean_alpha = 0.0
    best_state = None
    patience_counter = 0
    early_stopped = False
    
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = f"checkpoints/best_exp_d_{model_id}_lambda_{lambda_seg}_seed{seed}.pt"
    
    history = {
        "model_id": model_id,
        "model_name": model_name,
        "lambda_seg": lambda_seg,
        "regime_description": REGIME_DESCRIPTIONS[lambda_seg],
        "seed": seed,
        "total_params": total_params,
        "is_adaptive": is_adaptive,
        "description": cfg["description"],
        "epochs": []
    }
    
    t_start = time.time()
    effective_epochs = 1 if smoke_test else epochs
    
    if not smoke_test:
        untrained_val = evaluate(model=model, dataloader=val_loader, criterion=criterion, device=device, smoke_test=False)
        print(f"[Untrained Pre-flight] Val Loss: {untrained_val['val_loss']:.4f} | Val Acc: {untrained_val['val_cls_acc']:.2f}% | Val AUROC: {untrained_val['val_auroc']:.4f} | Val DSC: {untrained_val['val_dice']:.4f} | Alpha: {untrained_val['mean_alpha']:.4f}")
        
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
            is_adaptive=is_adaptive,
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
        val_acc = val_res["val_cls_acc"]
        
        # Primary selection: Validation Classification Accuracy. Secondary: Validation Loss
        is_best = (val_acc > best_val_acc) or (val_acc == best_val_acc and val_res["val_loss"] < best_val_loss)
        
        if is_best:
            best_val_acc = val_acc
            best_val_f1 = val_res["val_f1"]
            best_val_auroc = val_res["val_auroc"]
            best_val_dice = val_res["val_dice"]
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
            "train_cls_loss": train_res["train_cls_loss"],
            "train_seg_loss": train_res["train_seg_loss"],
            "train_cls_acc": train_res["train_cls_acc"],
            "val_loss": val_res["val_loss"],
            "val_cls_loss": val_res["val_cls_loss"],
            "val_seg_loss": val_res["val_seg_loss"],
            "val_cls_acc": val_res["val_cls_acc"],
            "val_f1": val_res["val_f1"],
            "val_auroc": val_res["val_auroc"],
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
        print(f"[Ep {ep+1:02d}/{effective_epochs:02d}] "
              f"Tr Loss: {train_res['train_loss']:.4f} | "
              f"Val Loss: {val_res['val_loss']:.4f} | "
              f"Val Acc: {val_res['val_cls_acc']:.2f}% | "
              f"Val AUROC: {val_res['val_auroc']:.4f} | "
              f"Val DSC: {val_res['val_dice']:.4f} | "
              f"Alpha: {val_res['mean_alpha']:.4f}{star} ({elapsed:.1f}s)")
              
        if patience_counter >= patience and not smoke_test:
            print(f"[Early Stopping] Triggered at epoch {ep+1} (patience={patience}, best epoch={best_epoch}, best Cls Acc={best_val_acc:.2f}%)")
            early_stopped = True
            break
            
    total_runtime = time.time() - t_start
    final_ep = history["epochs"][-1]
    
    history["total_runtime_seconds"] = total_runtime
    history["best_val_cls_acc"] = best_val_acc
    history["best_val_f1"] = best_val_f1
    history["best_val_auroc"] = best_val_auroc
    history["best_val_dice"] = best_val_dice
    history["best_val_loss"] = best_val_loss
    history["best_epoch"] = best_epoch
    history["best_mean_alpha"] = best_mean_alpha
    history["final_val_cls_acc"] = final_ep["val_cls_acc"]
    history["final_val_f1"] = final_ep["val_f1"]
    history["final_val_auroc"] = final_ep["val_auroc"]
    history["final_val_dice"] = final_ep["val_dice"]
    history["final_val_loss"] = final_ep["val_loss"]
    history["final_mean_alpha"] = final_ep["mean_alpha_val"]
    history["early_stopped"] = early_stopped
    history["epochs_trained"] = len(history["epochs"])
    
    return history

def run_experiment_d(regimes, models, epochs=20, patience=5, batch_size=2, lr=5e-4, weight_decay=1e-4, seed=42, smoke_test=False, data_dir="data", output_json=None):
    print("=" * 85)
    print("--- Q-FUSION EXPERIMENT D: AUXILIARY SPATIAL SUPERVISION BENCHMARK ---")
    print(f"Supervision Regimes (lambda_seg): {regimes}")
    print(f"Models: {models}")
    print(f"Max Epochs: {epochs} | Early Stopping Patience: {patience} | Seed: {seed}")
    print("=" * 85)
    
    verify_dual_head_parity()
    device, device_name = get_device()
    print(f"[Device] Utilizing: {device} ({device_name})")
    
    raw_train = SynthRADDataset(data_dir=data_dir, split="train", seed=seed, val_ratio=0.2, transform=get_transforms())
    raw_val = SynthRADDataset(data_dir=data_dir, split="val", seed=seed, val_ratio=0.2, transform=get_transforms())
    
    train_dataset = MemoryCachedDualTaskDataset(raw_train)
    val_dataset = MemoryCachedDualTaskDataset(raw_val)
    
    if smoke_test:
        print("[Smoke Test] Preloading minimal 2 samples for tensor shape verification...")
        _ = train_dataset[0]
        _ = val_dataset[0]
        train_dataset.cache[0]["cls_label"] = torch.tensor(0, dtype=torch.long)
        val_dataset.cache[0]["cls_label"] = torch.tensor(1, dtype=torch.long)
    else:
        preload_and_assign_labels(train_dataset, val_dataset)
        
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    os.makedirs("results", exist_ok=True)
    metrics_path = output_json if output_json else "results/exp_d_metrics.json"
    
    all_results = {}
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r") as f:
                all_results = json.load(f)
        except Exception:
            all_results = {}
            
    for lam in regimes:
        for model_id in models:
            run_key = f"lambda_{lam}__{model_id}"
            if run_key in all_results and all_results[run_key].get("epochs_trained", 0) > 1 and not smoke_test:
                print(f"\n[{run_key}] Already completed (Peak Val Acc: {all_results[run_key].get('best_val_cls_acc', 0):.2f}%). Skipping...")
                continue
                
            run_history = run_single_experiment(
                model_id=model_id,
                lambda_seg=lam,
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
                
    generate_markdown_report(all_results, "results/EXP_D_RESULTS.md")
    return all_results

def generate_markdown_report(results, report_path="results/EXP_D_RESULTS.md"):
    header = (
        "# Q-FUSION: EXPERIMENT D — AUXILIARY SPATIAL SUPERVISION BENCHMARK\n\n"
        "## Empirical Resolution of Hypothesis 2 (Shared Representation Sharpening via Spatial FiLM Supervision)\n\n"
        "**Dataset:** Authentic SynthRAD2023 Task 1 Brain (145 Train / 36 Validation)\n"
        "**Target:** Objective Anatomical Lesion Volume Classification (Binary: Low-Volume vs. High-Volume)\n"
        "**Protocol:** Adam (`lr=5e-4`, `weight_decay=1e-4`), `CosineAnnealingLR(T_max=20)`\n"
        "**Regularization:** Early stopping with patience = 5 on Validation Classification Accuracy\n"
        "**Parity:** Exact 58,412 parameter alignment across adaptive variants (0.0000% capacity variance)\n\n"
        "---\n\n"
        "### 1. Comparative Summary Matrix (Diagnostic Classification vs. Spatial Supervision)\n\n"
        "| Supervision Regime | Architecture | Model ID | Peak Val Acc (%) | Peak Val AUROC | Peak Val F1 | Peak Val DSC | Peak Epoch | Learned $\\bar{\\alpha}$ (Peak) | Learned $\\bar{\\alpha}$ (Final) | Runtime |\n"
        "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"
    )
    
    rows = []
    # Sort order: lambda 0.0, 1.0, 2.0
    order_map = {0.0: 0, 1.0: 1, 2.0: 2}
    sorted_items = sorted(results.items(), key=lambda kv: (order_map.get(kv[1].get("lambda_seg", 0.0), 9), kv[1].get("model_id", "")))
    
    for k, v in sorted_items:
        lam = v.get("lambda_seg", 0.0)
        regime_title = f"lambda_seg = {lam:.1f}"
        if lam == 0.0:
            regime_title += " (CLS ONLY)"
        elif lam == 1.0:
            regime_title += " (JOINT 1.0)"
        elif lam == 2.0:
            regime_title += " (STRONG 2.0)"
            
        model_name = v.get("model_name", "unknown")
        model_id = v.get("model_id", "unknown")
        peak_acc = v.get("best_val_cls_acc", 0.0)
        peak_auroc = v.get("best_val_auroc", 0.0)
        peak_f1 = v.get("best_val_f1", 0.0)
        peak_dsc = v.get("best_val_dice", 0.0)
        peak_epoch = v.get("best_epoch", 0)
        final_alpha = v.get("final_mean_alpha", 0.0)
        peak_alpha = v.get("best_mean_alpha", 0.0)
        runtime = v.get("total_runtime_seconds", 0.0)
        
        row = f"| **{regime_title}** | {model_name} | `{model_id}` | **{peak_acc:.2f}%** | **{peak_auroc:.4f}** | {peak_f1:.4f} | {peak_dsc:.4f} | Ep {peak_epoch} | {peak_alpha:.4f} | {final_alpha:.4f} | {runtime:.1f}s |"
        rows.append(row)
        
    content = header + "\n".join(rows) + "\n\n---\n"
    with open(report_path, "w") as f:
        f.write(content)
    print(f"\n[Report] Generated structured summary report at: {report_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Q-FUSION Experiment D: Auxiliary Spatial Supervision Benchmark")
    parser.add_argument("--lambda_seg", type=str, default="all", choices=["0.0", "1.0", "2.0", "all"],
                        help="Segmentation loss weight (default: all)")
    parser.add_argument("--model", type=str, default="core",
                        choices=["adaptive_cq", "adaptive_cc", "fixed_cq", "classical_baseline", "core", "all"],
                        help="Architecture to benchmark (default: core = adaptive_cq + adaptive_cc)")
    parser.add_argument("--epochs", type=int, default=20, help="Maximum epochs per run (default: 20)")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience on Val Cls Acc (default: 5)")
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
    
    if args.lambda_seg == "all":
        selected_regimes = VALID_REGIMES
    else:
        selected_regimes = [float(args.lambda_seg)]
        
    if args.model == "core":
        selected_models = ["adaptive_cq", "adaptive_cc"]
    elif args.model == "all":
        selected_models = list(MODEL_REGISTRY.keys())
    else:
        selected_models = [args.model]
        
    run_experiment_d(
        regimes=selected_regimes,
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

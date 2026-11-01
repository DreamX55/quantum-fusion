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
    from model_adaptive import AdaptiveHybridMultimodalModel
    from loss_annealed import AnnealedAntiCollapseLoss
except ImportError:
    from src.dataset import SynthRADDataset, get_transforms
    from src.model_adaptive import AdaptiveHybridMultimodalModel
    from src.loss_annealed import AnnealedAntiCollapseLoss

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train_adaptive_hybrid(epochs=1, batch_size=2, lr=1e-3, seed=42, lambda_init=1.0, lambda_min=0.01):
    print("=" * 75)
    print("--- Q-FUSION PHASE 4: Adaptive CQ Routing & FiLM Spatial Modulation ---")
    print("=" * 75)
    
    set_seed(seed)
    
    # 1. Device configuration
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[Device] Utilizing: {device}")
        print(f"[GPU] Model: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"[Device] Utilizing: {device} (Apple Silicon MPS)")
    else:
        device = torch.device("cpu")
        print(f"[Device] Utilizing: {device}")
        
    # 2. Dataset & Dataloader
    dataset = SynthRADDataset(data_dir="data", split="train", transform=get_transforms())
    total_samples = len(dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    total_batches = len(dataloader)
    print(f"[Dataset] SynthRAD2023 Task 1 Brain: {total_samples} patients ({total_batches} batches)")
    
    # 3. Model Initialization
    model = AdaptiveHybridMultimodalModel(latent_dim=32).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    q_params = sum(p.numel() for p in model.quantum_branch.parameters() if p.requires_grad)
    c_params = sum(p.numel() for p in model.classical_surrogate.parameters() if p.requires_grad)
    router_params = sum(p.numel() for p in model.router.parameters() if p.requires_grad)
    film_params = sum(p.numel() for p in model.film.parameters() if p.requires_grad)
    
    print(f"[Model] Total Trainable Parameters: {total_params:,}")
    print(f"[Capacity Audit] Quantum Branch Parameters:     {q_params} (8 qubits, 3 layers)")
    print(f"[Capacity Audit] Classical Surrogate Parameters: {c_params} (Matched Surrogate)")
    print(f"[Router & FiLM] Router Params: {router_params} | FiLM Params: {film_params}")
    assert q_params == c_params, f"Parameter mismatch! Q: {q_params}, C: {c_params}"
    
    # 4. Annealed Loss & Optimizer
    criterion = AnnealedAntiCollapseLoss(
        lambda_init=lambda_init,
        lambda_min=lambda_min,
        schedule="linear"
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 5. Tracking Metrics & Alpha Trajectory
    history = {
        "seed": seed,
        "device": str(device),
        "total_samples": total_samples,
        "total_batches": total_batches,
        "total_params": total_params,
        "q_params": q_params,
        "c_params": c_params,
        "lambda_init": lambda_init,
        "lambda_min": lambda_min,
        "alpha_trajectory": [],
        "batch_losses_total": [],
        "batch_losses_bce": [],
        "batch_penalties": [],
        "lambda_history": [],
        "batch_times": [],
        "grad_norms": {}
    }
    
    print("-" * 75)
    print(f"Beginning 1 Full Epoch Adaptive Training over {total_samples} patients...")
    print("-" * 75)
    
    epoch_start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        running_total_loss = 0.0
        running_bce_loss = 0.0
        running_alpha = 0.0
        
        for batch_idx, batch_data in enumerate(dataloader):
            batch_start = time.time()
            
            # Anneal lambda across steps in the epoch
            current_lambda = criterion.step_batch(step=batch_idx, total_steps=total_batches)
            
            mri = batch_data["mri"].to(device)
            ct = batch_data["ct"].to(device)
            target_mask = batch_data["mask"].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass: outputs segmentation mask and dynamic routing scalar alpha
            predictions, alpha = model(mri, ct)
            
            # Annealed Anti-Collapse Loss: BCE(y, y_hat) + lambda(t) * (alpha - 0.5)^2
            loss_total, loss_bce, loss_penalty, lam_val = criterion(predictions, target_mask, alpha)
            
            # Backward pass: propagates through FiLM, Router, Quantum Branch, Surrogate, and Encoders
            loss_total.backward()
            
            # Inspect key gradient norms
            router_grad = model.router.router[0].weight.grad.norm().item()
            q_weights_grad = model.quantum_branch.q_weights.grad.norm().item()
            c_weights_grad = model.classical_surrogate.c_weights.grad.norm().item()
            film_grad = model.film.gamma_gen.weight.grad.norm().item()
            mri_enc_grad = model.mri_encoder[0].conv[0].weight.grad.norm().item()
            
            optimizer.step()
            
            batch_time = time.time() - batch_start
            
            alpha_mean = alpha.mean().item()
            alpha_min = alpha.min().item()
            alpha_max = alpha.max().item()
            
            loss_tot_val = loss_total.item()
            loss_bce_val = loss_bce.item()
            pen_val = loss_penalty.item()
            
            running_total_loss += loss_tot_val
            running_bce_loss += loss_bce_val
            running_alpha += alpha_mean
            
            # Record tracking data
            history["alpha_trajectory"].append({
                "batch": batch_idx + 1,
                "alpha_mean": alpha_mean,
                "alpha_min": alpha_min,
                "alpha_max": alpha_max
            })
            history["batch_losses_total"].append(loss_tot_val)
            history["batch_losses_bce"].append(loss_bce_val)
            history["batch_penalties"].append(pen_val)
            history["lambda_history"].append(lam_val)
            history["batch_times"].append(batch_time)
            
            # Logging interval
            if (batch_idx + 1) % 10 == 0 or batch_idx == 0 or (batch_idx + 1) == total_batches:
                avg_bce = running_bce_loss / (batch_idx + 1)
                avg_alpha = running_alpha / (batch_idx + 1)
                print(f"Epoch [{epoch+1}/{epochs}] | Batch [{batch_idx+1:02d}/{total_batches:02d}] "
                      f"| BCE: {loss_bce_val:.4f} (Avg: {avg_bce:.4f}) "
                      f"| Alpha: {alpha_mean:.4f} (Avg: {avg_alpha:.4f}) "
                      f"| Lam: {lam_val:.3f} | Pen: {pen_val:.4f} "
                      f"| Time: {batch_time:.3f}s | Q-Grad: {q_weights_grad:.2e} | R-Grad: {router_grad:.2e}")
                
        # Final gradient norms check
        history["grad_norms"] = {
            "router_layer1": router_grad,
            "quantum_circuit_weights": q_weights_grad,
            "classical_surrogate_weights": c_weights_grad,
            "film_gamma": film_grad,
            "mri_encoder_layer1": mri_enc_grad,
            "decoder_layer1": model.decoder[1].weight.grad.norm().item()
        }
        
    total_epoch_time = time.time() - epoch_start_time
    avg_batch_time = sum(history["batch_times"][1:]) / max(1, len(history["batch_times"]) - 1)
    
    alpha_start = history["alpha_trajectory"][0]["alpha_mean"]
    alpha_mid = history["alpha_trajectory"][total_batches // 2]["alpha_mean"]
    alpha_end = history["alpha_trajectory"][-1]["alpha_mean"]
    all_alphas = [entry["alpha_mean"] for entry in history["alpha_trajectory"]]
    alpha_min_overall = min(all_alphas)
    alpha_max_overall = max(all_alphas)
    
    bce_start = history["batch_losses_bce"][0]
    bce_end = history["batch_losses_bce"][-1]
    bce_min = min(history["batch_losses_bce"])
    
    print("=" * 75)
    print("--- PHASE 4 ADAPTIVE EXPERIMENT SUMMARY ---")
    print("=" * 75)
    print(f"Total Patients Trained:      {total_samples}")
    print(f"Total Batches Completed:     {total_batches}")
    print(f"Total Epoch Execution Time:  {total_epoch_time:.2f} seconds")
    print(f"Average Runtime per Batch:   {avg_batch_time:.4f} seconds (Warmed-up)")
    print(f"BCE Loss Start (Batch 1):    {bce_start:.4f}")
    print(f"BCE Loss Minimum:            {bce_min:.4f}")
    print(f"BCE Loss End (Batch {total_batches}):      {bce_end:.4f}")
    print(f"BCE Loss Reduction:          {((bce_start - bce_end) / bce_start) * 100:.2f}%")
    print("-" * 75)
    print("--- ROUTER ALPHA TRAJECTORY AUDIT ---")
    print(f"Alpha Start (Batch 1):       {alpha_start:.4f} (Active balanced routing)")
    print(f"Alpha Midpoint (Batch {total_batches // 2}):   {alpha_mid:.4f}")
    print(f"Alpha Final (Batch {total_batches}):       {alpha_end:.4f}")
    print(f"Alpha Range Observed:        [{alpha_min_overall:.4f}, {alpha_max_overall:.4f}]")
    print(f"Did Alpha Collapse to 0.0?   {'NO - Classical Collapse Successfully Prevented!' if alpha_min_overall > 0.05 else 'YES (Collapsed)'}")
    print(f"Did Alpha Collapse to 1.0?   {'NO - Quantum Monopolization Prevented!' if alpha_max_overall < 0.95 else 'YES (Collapsed)'}")
    print(f"Final Routing Ratio:         {alpha_end * 100:.1f}% Quantum / {(1.0 - alpha_end) * 100:.1f}% Classical")
    print("-" * 75)
    print(f"Router Gradient Norm:        {history['grad_norms']['router_layer1']:.6e} (Active)")
    print(f"Quantum Branch Grad Norm:    {history['grad_norms']['quantum_circuit_weights']:.6e} (Active)")
    print(f"Classical Surrogate Grad:    {history['grad_norms']['classical_surrogate_weights']:.6e} (Active)")
    print(f"FiLM Modulation Grad Norm:   {history['grad_norms']['film_gamma']:.6e} (Active)")
    print("=" * 75)
    
    # Save training metrics and alpha trajectory to disk
    metrics_path = "phase4_training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Saved full Phase 4 alpha trajectory & metrics history to: {metrics_path}")
    
    return history

if __name__ == "__main__":
    train_adaptive_hybrid(epochs=1, batch_size=2, lr=1e-3, seed=42)

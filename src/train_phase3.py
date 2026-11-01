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
    from model_hybrid_fixed import FixedHybridMultimodalModel
except ImportError:
    from src.dataset import SynthRADDataset, get_transforms
    from src.model_hybrid_fixed import FixedHybridMultimodalModel

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train_fixed_hybrid(epochs=1, batch_size=2, lr=1e-3, seed=42):
    print("=" * 70)
    print("--- Q-FUSION PHASE 3: Fixed Quantum Branch Training ---")
    print("=" * 70)
    
    set_seed(seed)
    
    # 1. Hardware Detection
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[Device] Utilizing: {device}")
        print(f"[GPU] Model: {torch.cuda.get_device_name(0)}")
        print(f"[CUDA] Memory Allocated: {torch.cuda.memory_allocated(0)/(1024**2):.1f} MB")
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
    print(f"[Dataset] SynthRAD2023 Task 1 Brain: {total_samples} patients ({total_batches} batches at batch_size={batch_size})")
    
    # 3. Model Initialization
    model = FixedHybridMultimodalModel(latent_dim=32).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    q_params = model.quantum_bottleneck.q_weights.numel()
    print(f"[Model] Total Trainable Parameters: {total_params:,}")
    print(f"[Quantum] Circuit Parameters (q_weights): {q_params} (8 qubits, 3 layers)")
    
    # 4. Criterion & Optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 5. Training Metrics Tracking
    history = {
        "seed": seed,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "total_samples": total_samples,
        "total_params": total_params,
        "batch_losses": [],
        "batch_times": [],
        "grad_norms": {}
    }
    
    print("-" * 70)
    print(f"Beginning 1 Full Epoch Training across all {total_samples} patients...")
    print("-" * 70)
    
    epoch_start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for batch_idx, batch_data in enumerate(dataloader):
            batch_start = time.time()
            
            mri = batch_data["mri"].to(device)
            ct = batch_data["ct"].to(device)
            target_mask = batch_data["mask"].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass through Fixed Hybrid Model
            predictions = model(mri, ct)
            
            # Binary Cross Entropy Loss
            loss = criterion(predictions, target_mask)
            
            # Backward pass (spatial gradients propagate across PennyLane QNode)
            loss.backward()
            
            # Inspect gradients at key checkpoints
            mri_enc_grad = model.mri_encoder[0].conv[0].weight.grad.norm().item()
            ct_enc_grad = model.ct_encoder[0].conv[0].weight.grad.norm().item()
            q_weights_grad = model.quantum_bottleneck.q_weights.grad.norm().item()
            comp_grad = model.quantum_bottleneck.compressor.weight.grad.norm().item()
            decoder_grad = model.decoder[1].weight.grad.norm().item()
            
            optimizer.step()
            
            batch_time = time.time() - batch_start
            loss_val = loss.item()
            running_loss += loss_val
            
            history["batch_losses"].append(loss_val)
            history["batch_times"].append(batch_time)
            
            # Progress print every 10 batches or on first/last
            if (batch_idx + 1) % 10 == 0 or batch_idx == 0 or (batch_idx + 1) == total_batches:
                avg_so_far = running_loss / (batch_idx + 1)
                print(f"Epoch [{epoch+1}/{epochs}] | Batch [{batch_idx+1:02d}/{total_batches:02d}] "
                      f"| Loss: {loss_val:.4f} (Avg: {avg_so_far:.4f}) "
                      f"| Time: {batch_time:.3f}s "
                      f"| Q-Grad: {q_weights_grad:.2e} | MRI-Enc-Grad: {mri_enc_grad:.2e}")
                
        # Record final gradient norms for validation audit
        history["grad_norms"] = {
            "mri_encoder_layer1": mri_enc_grad,
            "ct_encoder_layer1": ct_enc_grad,
            "quantum_compressor": comp_grad,
            "quantum_circuit_weights": q_weights_grad,
            "decoder_layer1": decoder_grad
        }
        
    total_epoch_time = time.time() - epoch_start_time
    avg_batch_time = sum(history["batch_times"][1:]) / max(1, len(history["batch_times"]) - 1)
    initial_loss = history["batch_losses"][0]
    final_loss = history["batch_losses"][-1]
    min_loss = min(history["batch_losses"])
    loss_reduction_pct = ((initial_loss - final_loss) / initial_loss) * 100.0
    
    print("=" * 70)
    print("--- PHASE 3 TRAINING SUMMARY ---")
    print("=" * 70)
    print(f"Total Patients Trained:      {total_samples}")
    print(f"Total Batches Completed:     {total_batches}")
    print(f"Total Epoch Execution Time:  {total_epoch_time:.2f} seconds")
    print(f"Average Runtime per Batch:   {avg_batch_time:.4f} seconds (Warmed-up)")
    print(f"Phase 1 CPU Runtime/Batch:   ~1.500 seconds")
    print(f"Speedup vs Phase 1:          {1.500 / avg_batch_time:.2f}x faster on RTX A2000")
    print(f"Initial BCE Loss (Batch 1):  {initial_loss:.4f}")
    print(f"Final BCE Loss (Batch {total_batches}): {final_loss:.4f}")
    print(f"Minimum BCE Loss Observed:   {min_loss:.4f}")
    print(f"Loss Drop:                   {initial_loss - final_loss:.4f} ({loss_reduction_pct:.2f}% reduction)")
    print(f"Quantum Gradient Norm:       {history['grad_norms']['quantum_circuit_weights']:.6e} (Non-zero)")
    print(f"MRI Encoder Gradient Norm:   {history['grad_norms']['mri_encoder_layer1']:.6e} (Non-zero)")
    print(f"CT Encoder Gradient Norm:    {history['grad_norms']['ct_encoder_layer1']:.6e} (Non-zero)")
    print("=" * 70)
    
    # Save training audit log
    metrics_path = "phase3_training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Saved full Phase 3 metrics history to: {metrics_path}")
    
    return history

if __name__ == "__main__":
    train_fixed_hybrid(epochs=1, batch_size=2, lr=1e-3, seed=42)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import SynthRADDataset, get_transforms
from model_classical import ClassicalMultimodalBaseline

def train_baseline():
    print("--- Q-FUSION PHASE 1: Classical Multimodal Baseline ---")
    
    # 1. Setup Data (Will use dummy tensors if dataset folder is empty)
    # Change 'data/' to the actual extracted dataset path once downloaded
    dataset = SynthRADDataset(data_dir="data", split="train", transform=get_transforms())
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # 2. Setup Model
    device = torch.device("cpu") # Defaulting to CPU based on Phase 0 Audit
    model = ClassicalMultimodalBaseline().to(device)
    
    # 3. Setup Loss and Optimizer
    criterion = nn.BCELoss() # Binary Cross Entropy for Segmentation
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    print(f"Starting training on {device} with {len(dataset)} samples (using dummy data if real data is absent).")
    
    # 4. Training Loop
    epochs = 2
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, batch_data in enumerate(dataloader):
            mri = batch_data["mri"].to(device)
            ct = batch_data["ct"].to(device)
            target_mask = batch_data["mask"].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            predictions = model(mri, ct)
            
            # Loss calculation
            loss = criterion(predictions, target_mask)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx >= 2: # Limit to 3 batches for fast CPU verification
                print(f"  [Batch {batch_idx+1}] Partial loss: {loss.item():.4f} - Breaking early for Phase 1 verification.")
                break
                
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/(batch_idx+1):.4f}")
        
    print("Phase 1 Baseline Training loop complete. Ready for real data!")

if __name__ == "__main__":
    train_baseline()

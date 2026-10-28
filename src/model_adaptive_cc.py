import torch
import torch.nn as nn

try:
    from model_adaptive import ConvBlock3D, ClassicalSurrogateBottleneck, AdaptiveCQRouter, FiLMModulation3D
except ImportError:
    from src.model_adaptive import ConvBlock3D, ClassicalSurrogateBottleneck, AdaptiveCQRouter, FiLMModulation3D

class AdaptiveCCMultimodalModel(nn.Module):
    """
    Experiment A Control Model: Adaptive Classical-Classical (CC) Surrogate Architecture.
    
    Directly answers Pitfall 3 (Classical Nonlinear Control):
    Replaces the 8-qubit quantum branch (z_Q) with a second capacity-matched
    classical surrogate (z_C2) possessing exactly 24 trainable parameters.
    
    Both branches (z_C1 and z_C2) have identical capacity to z_Q:
    - Input dimension: 24
    - Output dimension: 8
    - Trainable parameter count: EXACTLY 24 parameters each.
    - Total Model Parameter Count: EXACTLY identical to Adaptive CQ (58,234 params).
    
    If Adaptive CQ > Adaptive CC under identical training and evaluation conditions,
    we obtain empirical Level 2 evidence that quantum representation processing provides
    unique utility beyond dynamic classical routing alone.
    """
    def __init__(self, latent_dim=32, q_features=24):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Branch 1: MRI 3D CNN Encoder (Identical capacity to all models)
        self.mri_encoder = nn.Sequential(
            ConvBlock3D(1, 8),           # (B, 1, 64, 64, 64)  -> (B, 8, 32, 32, 32)
            ConvBlock3D(8, 16),          # (B, 8, 32, 32, 32)  -> (B, 16, 16, 16, 16)
            ConvBlock3D(16, latent_dim)  # (B, 16, 16, 16, 16) -> (B, 32, 8, 8, 8)
        )
        
        # Branch 2: CT 3D CNN Encoder (Identical capacity to all models)
        self.ct_encoder = nn.Sequential(
            ConvBlock3D(1, 8),           # (B, 1, 64, 64, 64)  -> (B, 8, 32, 32, 32)
            ConvBlock3D(8, 16),          # (B, 8, 32, 32, 32)  -> (B, 16, 16, 16, 16)
            ConvBlock3D(16, latent_dim)  # (B, 16, 16, 16, 16) -> (B, 32, 8, 8, 8)
        )
        
        # Classical 3D Spatial Feature Map Fusion (F)
        self.spatial_fusion = nn.Sequential(
            nn.Conv3d(latent_dim * 2, latent_dim, kernel_size=1),
            nn.BatchNorm3d(latent_dim),
            nn.ReLU(inplace=True)
        )
        
        # Global pooling for the compact multimodal routing representation
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Shared compression layer: 64 features -> 24 bottleneck inputs
        self.compressor = nn.Linear(latent_dim * 2, q_features)
        
        # Pathway C1: Capacity-Matched Classical Surrogate 1 (Exactly 24 trainable params)
        self.classical_surrogate_1 = ClassicalSurrogateBottleneck(n_layers=3, n_features=8)
        
        # Pathway C2: Capacity-Matched Classical Surrogate 2 (Exactly 24 trainable params)
        # Replaces Pathway Q to form the CC control
        self.classical_surrogate_2 = ClassicalSurrogateBottleneck(n_layers=3, n_features=8)
        
        # Adaptive Router: Gating network computing alpha in [0, 1]
        self.router = AdaptiveCQRouter(in_features=latent_dim * 2, hidden_dim=16)
        
        # Quantum-Conditioned Spatial Modulation (FiLM Layer)
        self.film = FiLMModulation3D(in_features=8, num_channels=latent_dim)
        
        # 3D Spatial Decoder for Segmentation (Identical capacity to all models)
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False), # (8, 8, 8)   -> (16, 16, 16)
            nn.Conv3d(latent_dim, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False), # (16, 16, 16) -> (32, 32, 32)
            nn.Conv3d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False), # (32, 32, 32) -> (64, 64, 64)
            nn.Conv3d(8, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, mri, ct, return_details=False):
        batch_size = mri.shape[0]
        
        # 1. Classical Encoding
        z_mri = self.mri_encoder(mri)  # (B, 32, 8, 8, 8)
        z_ct = self.ct_encoder(ct)    # (B, 32, 8, 8, 8)
        z_concat = torch.cat([z_mri, z_ct], dim=1)  # (B, 64, 8, 8, 8)
        
        # 2. Extract base 3D Spatial Feature Map F
        F = self.spatial_fusion(z_concat)  # (B, 32, 8, 8, 8)
        
        # 3. Compact Multimodal Representation
        z_pooled = self.pool(z_concat).view(batch_size, -1)  # (B, 64)
        z_comp = self.compressor(z_pooled)                   # (B, 24)
        
        # 4. Parallel Classical Pathways Evaluation (C1 and C2)
        z_c1 = self.classical_surrogate_1(z_comp)            # (B, 8)
        z_c2 = self.classical_surrogate_2(z_comp)            # (B, 8)
        
        # 5. Dynamic Routing Decision alpha in [0, 1]
        alpha = self.router(z_pooled)                        # (B, 1)
        
        # 6. Convex Combination: z' = (1 - alpha) * z_C1 + alpha * z_C2
        z_prime = (1.0 - alpha) * z_c1 + alpha * z_c2        # (B, 8)
        
        # 7. Spatial Modulation (FiLM): F' = gamma * F + beta
        F_prime = self.film(F, z_prime)                      # (B, 32, 8, 8, 8)
        
        # 8. Decode to 3D Binary Mask
        out_mask = self.decoder(F_prime)                     # (B, 1, 64, 64, 64)
        
        if return_details:
            return out_mask, alpha, z_prime, z_c1, z_c2
        return out_mask, alpha

# Convenience alias
ModelAdaptiveCC = AdaptiveCCMultimodalModel

if __name__ == "__main__":
    print("--- Testing AdaptiveCCMultimodalModel (Experiment A) ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = AdaptiveCCMultimodalModel().to(device)
    
    # Audit parameter parity between Surrogate 1 and Surrogate 2
    c1_params = sum(p.numel() for p in model.classical_surrogate_1.parameters() if p.requires_grad)
    c2_params = sum(p.numel() for p in model.classical_surrogate_2.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Capacity Audit] Classical Surrogate 1 parameters: {c1_params}")
    print(f"[Capacity Audit] Classical Surrogate 2 parameters: {c2_params}")
    print(f"[Capacity Audit] Total Model Parameters:          {total_params}")
    assert c1_params == 24, f"C1 parameters must be 24, got {c1_params}"
    assert c2_params == 24, f"C2 parameters must be 24, got {c2_params}"
    assert total_params == 58234, f"Total parameters must be 58,234, got {total_params}"
    print("[SUCCESS] Exact Parameter Matching Verified for Adaptive CC.")
    
    dummy_mri = torch.randn(2, 1, 64, 64, 64, device=device)
    dummy_ct = torch.randn(2, 1, 64, 64, 64, device=device)
    
    out_mask, alpha = model(dummy_mri, dummy_ct)
    print(f"Output mask shape: {out_mask.shape}")
    print(f"Alpha values: {alpha.detach().cpu().numpy().flatten()}")
    assert out_mask.shape == (2, 1, 64, 64, 64), f"Shape mismatch: {out_mask.shape}"
    print("[SUCCESS] Adaptive CC Model successfully verified.")

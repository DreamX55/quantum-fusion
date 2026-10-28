import torch
import torch.nn as nn

try:
    from quantum_bottleneck import QuantumReuploadingBottleneck, reuploading_quantum_circuit
except ImportError:
    from src.quantum_bottleneck import QuantumReuploadingBottleneck, reuploading_quantum_circuit

class ConvBlock3D(nn.Module):
    """
    Identical 3D Convolutional block from Phase 1 Baseline and Phase 3.
    Preserves exact classical encoder capacity.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2)
        )
    def forward(self, x):
        return self.conv(x)

class ClassicalSurrogateBottleneck(nn.Module):
    """
    Capacity-Matched Classical Surrogate (Solves Pitfall 3).
    
    Strictly matches the 8-qubit, 3-layer Quantum Re-uploading circuit:
    - Input dimension: 24 (matching the 3 sequential chunks of 8 features)
    - Output dimension: 8 (matching the 8 Pauli-Z expectation values)
    - Trainable parameter count: EXACTLY 24 parameters (3 layers x 8 weights = 24)
    - Value range: [-1, 1] via tanh activation, matching Pauli-Z observable bounds
    """
    def __init__(self, n_layers=3, n_features=8):
        super().__init__()
        # Exactly 24 trainable parameters matching q_weights shape (3, 8)
        self.c_weights = nn.Parameter(torch.randn(n_layers, n_features) * 0.1)
        
    def forward(self, x):
        # x shape: (B, 24)
        # Slices matching the 3 sequential embedding layers of the quantum circuit
        x0 = x[:, 0:8]
        x1 = x[:, 8:16]
        x2 = x[:, 16:24]
        
        # Parallel classical transformation with identical parameter count
        out = (
            self.c_weights[0] * x0 +
            self.c_weights[1] * x1 +
            self.c_weights[2] * x2
        )
        # Bounded in [-1, 1] to match quantum expectation values
        return torch.tanh(out)

class AdaptiveCQRouter(nn.Module):
    """
    Adaptive Classical-Quantum (CQ) Router.
    
    Takes pooled multimodal latent features and outputs a dynamic gating scalar
    alpha in [0, 1] through a continuous Sigmoid relaxation.
    """
    def __init__(self, in_features=64, hidden_dim=16):
        super().__init__()
        self.router = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x: (B, 64) -> alpha: (B, 1) in [0, 1]
        return self.router(x)

class FiLMModulation3D(nn.Module):
    """
    Quantum-Conditioned Spatial Modulation (Solves Pitfall 2: Spatial Reconstruction Paradox).
    
    Rather than projecting compact 8-dimensional bottleneck features into 16,384 unflattened
    spatial locations, this FiLM layer preserves the high-resolution 3D spatial feature maps (F)
    from the CNN encoders and modulates them using scale gamma and shift beta factors:
    F' = gamma * F + beta
    """
    def __init__(self, in_features=8, num_channels=32):
        super().__init__()
        self.gamma_gen = nn.Linear(in_features, num_channels)
        self.beta_gen = nn.Linear(in_features, num_channels)
        
        # Initialize gamma around 1.0 and beta around 0.0 with small variance for gradient flow
        nn.init.normal_(self.gamma_gen.weight, std=0.02)
        nn.init.ones_(self.gamma_gen.bias)
        nn.init.normal_(self.beta_gen.weight, std=0.02)
        nn.init.zeros_(self.beta_gen.bias)
        
    def forward(self, F, z_prime):
        # F: (B, 32, 8, 8, 8)
        # z_prime: (B, 8)
        b = F.shape[0]
        gamma = self.gamma_gen(z_prime).view(b, -1, 1, 1, 1)
        beta = self.beta_gen(z_prime).view(b, -1, 1, 1, 1)
        
        # F' = gamma * F + beta
        F_prime = gamma * F + beta
        return F_prime

class QuantumBranchBottleneck(nn.Module):
    """
    8-qubit, 3-layer sequential Data Re-uploading Quantum Branch (Pitfall 3 match).
    
    Operates on 24 classical inputs and produces 8 Pauli-Z expectation values.
    Trainable parameters: EXACTLY 24 (q_weights of shape 3 x 8 = 24).
    """
    def __init__(self):
        super().__init__()
        self.q_weights = nn.Parameter(torch.randn(3, 8))
        
    def forward(self, x):
        batch_size = x.shape[0]
        orig_device = x.device
        x_clean = x.as_subclass(torch.Tensor) if hasattr(x, "as_subclass") else x
        x_cpu = x_clean.cpu()
        w_cpu = self.q_weights.cpu()
        
        q_out_list = []
        for i in range(batch_size):
            q_out_list.append(torch.stack(reuploading_quantum_circuit(x_cpu[i], w_cpu)))
            
        q_out = torch.stack(q_out_list).to(device=orig_device, dtype=x.dtype)
        return q_out

class AdaptiveHybridMultimodalModel(nn.Module):
    """
    Q-FUSION Phase 4 Architecture: Adaptive Classical-Quantum Routing & Spatial Modulation.
    
    Integrates:
    1. Dual 3D CNN Encoders (MRI & CT)
    2. Spatial feature fusion (F)
    3. Multimodal pooling & classical compression (64 -> 24)
    4. Parallel Quantum Branch (8 qubits, 3 layers, 24 params)
    5. Parallel Classical Surrogate (Capacity-matched, 24 params)
    6. Adaptive CQ Router (Outputs alpha in [0, 1])
    7. Convex combination: z' = (1 - alpha) * z_C + alpha * z_Q
    8. FiLM Spatial Modulation: F' = gamma * F + beta
    9. 3D Segmentation Decoder
    """
    def __init__(self, latent_dim=32, q_features=24):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Branch 1: MRI 3D CNN Encoder (Identical capacity to Phase 1 & Phase 3)
        self.mri_encoder = nn.Sequential(
            ConvBlock3D(1, 8),           # (B, 1, 64, 64, 64)  -> (B, 8, 32, 32, 32)
            ConvBlock3D(8, 16),          # (B, 8, 32, 32, 32)  -> (B, 16, 16, 16, 16)
            ConvBlock3D(16, latent_dim)  # (B, 16, 16, 16, 16) -> (B, 32, 8, 8, 8)
        )
        
        # Branch 2: CT 3D CNN Encoder (Identical capacity to Phase 1 & Phase 3)
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
        
        # Pathway Q: 8-qubit, 3-layer Data Re-uploading Quantum Circuit (Exactly 24 params)
        self.quantum_branch = QuantumBranchBottleneck()
        
        # Pathway C: Capacity-Matched Classical Surrogate (Exactly 24 trainable params)
        self.classical_surrogate = ClassicalSurrogateBottleneck(n_layers=3, n_features=8)
        
        # Adaptive Router: Gating network computing alpha in [0, 1]
        self.router = AdaptiveCQRouter(in_features=latent_dim * 2, hidden_dim=16)
        
        # Quantum-Conditioned Spatial Modulation (FiLM Layer)
        self.film = FiLMModulation3D(in_features=8, num_channels=latent_dim)
        
        # 3D Spatial Decoder for Segmentation (Identical capacity to Phase 1 & Phase 3)
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
        
        # 4. Parallel Pathway Evaluation
        z_c = self.classical_surrogate(z_comp)               # (B, 8)
        z_q = self.quantum_branch(z_comp)                    # (B, 8)
        
        # 5. Dynamic Routing Decision alpha in [0, 1]
        alpha = self.router(z_pooled)                        # (B, 1)
        
        # 6. Convex Combination: z' = (1 - alpha) * z_C + alpha * z_Q
        z_prime = (1.0 - alpha) * z_c + alpha * z_q          # (B, 8)
        
        # 7. Quantum-Conditioned Spatial Modulation (FiLM): F' = gamma * F + beta
        F_prime = self.film(F, z_prime)                      # (B, 32, 8, 8, 8)
        
        # 8. Decode to 3D Binary Mask
        out_mask = self.decoder(F_prime)                     # (B, 1, 64, 64, 64)
        
        if return_details:
            return out_mask, alpha, z_prime, z_c, z_q
        return out_mask, alpha

# Convenience alias
ModelAdaptive = AdaptiveHybridMultimodalModel

if __name__ == "__main__":
    print("--- Testing AdaptiveHybridMultimodalModel (Phase 4) ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = AdaptiveHybridMultimodalModel().to(device)
    
    # Audit parameter parity between Quantum and Classical Surrogate
    q_params = sum(p.numel() for p in model.quantum_branch.parameters() if p.requires_grad)
    c_params = sum(p.numel() for p in model.classical_surrogate.parameters() if p.requires_grad)
    print(f"[Capacity Audit] Quantum branch parameters: {q_params}")
    print(f"[Capacity Audit] Classical surrogate parameters: {c_params}")
    assert q_params == c_params, f"Parameter mismatch! Q: {q_params}, C: {c_params}"
    print("[SUCCESS] Exact 0.00% Parameter Matching Verified.")
    
    dummy_mri = torch.randn(2, 1, 64, 64, 64, device=device)
    dummy_ct = torch.randn(2, 1, 64, 64, 64, device=device)
    
    out_mask, alpha = model(dummy_mri, dummy_ct)
    print(f"Input MRI shape: {dummy_mri.shape}")
    print(f"Output mask shape: {out_mask.shape}")
    print(f"Alpha shape: {alpha.shape} | Alpha values: {alpha.detach().cpu().numpy().flatten()}")
    
    assert out_mask.shape == (2, 1, 64, 64, 64), f"Output shape mismatch: {out_mask.shape}"
    assert alpha.shape == (2, 1), f"Alpha shape mismatch: {alpha.shape}"
    print("[SUCCESS] Adaptive Model compiled and verified.")

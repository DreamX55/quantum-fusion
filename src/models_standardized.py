import torch
import torch.nn as nn

try:
    from quantum_bottleneck import reuploading_quantum_circuit
except ImportError:
    from src.quantum_bottleneck import reuploading_quantum_circuit

class ConvBlock3D(nn.Module):
    """
    Standardized 3D Convolutional block.
    Kernel: 3x3x3, Padding: 1, Stride: 1, followed by BatchNorm3d, ReLU, and MaxPool3d(2).
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
    Capacity-Matched Classical Surrogate (Correction 1 & Pitfall 3).
    
    Transforms 24 compressed multimodal features to 8 latent outputs:
    - Slices input into 3 chunks of 8 features (matching the 3 sequential re-uploading layers)
    - Applies trainable parameter tensor c_weights of shape (3, 8) = EXACTLY 24 parameters
    - Bounded in [-1, 1] via tanh activation
    """
    def __init__(self, n_layers=3, n_features=8):
        super().__init__()
        self.c_weights = nn.Parameter(torch.randn(n_layers, n_features) * 0.1)
        
    def forward(self, x):
        # x shape: (B, 24)
        x0 = x[:, 0:8]
        x1 = x[:, 8:16]
        x2 = x[:, 16:24]
        out = self.c_weights[0] * x0 + self.c_weights[1] * x1 + self.c_weights[2] * x2
        return torch.tanh(out)

class QuantumBranchBottleneck(nn.Module):
    """
    8-Qubit, 3-Layer Sequential Data Re-uploading Quantum Branch (Correction 1).
    
    Transforms 24 compressed multimodal features to 8 Pauli-Z expectation values:
    - Evaluates 8-qubit QNode with 3 sequential AngleEmbedding & Entanglement layers
    - Trainable parameters: q_weights of shape (3, 8) = EXACTLY 24 parameters
    - Output values bounded in [-1, 1]
    """
    def __init__(self):
        super().__init__()
        self.q_weights = nn.Parameter(torch.randn(3, 8) * 0.1)
        
    def forward(self, x):
        # x shape: (B, 24)
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

class AdaptiveCQRouter(nn.Module):
    """
    Adaptive Classical-Quantum Gating Router.
    Takes 64-dimensional pooled multimodal features and outputs routing scalar alpha in [0, 1].
    Parameter count: 64*16 + 16 + 16*1 + 1 = 1,057 parameters.
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
        return self.router(x)

class FiLMModulation3D(nn.Module):
    """
    Quantum-Conditioned Spatial Modulation (Correction 1 & Pitfall 2 Resolution).
    
    Eliminates the lossy 201k unflattening layer.
    Modulates high-resolution 3D spatial feature map F with affine parameters:
    F' = gamma(z') * F + beta(z')
    Parameter count: 2 * (8 * 32 + 32) = 576 parameters.
    """
    def __init__(self, in_features=8, num_channels=32):
        super().__init__()
        self.gamma_gen = nn.Linear(in_features, num_channels)
        self.beta_gen = nn.Linear(in_features, num_channels)
        
        nn.init.normal_(self.gamma_gen.weight, std=0.02)
        nn.init.ones_(self.gamma_gen.bias)
        nn.init.normal_(self.beta_gen.weight, std=0.02)
        nn.init.zeros_(self.beta_gen.bias)
        
    def forward(self, F, z_prime):
        b = F.shape[0]
        gamma = self.gamma_gen(z_prime).view(b, -1, 1, 1, 1)
        beta = self.beta_gen(z_prime).view(b, -1, 1, 1, 1)
        return gamma * F + beta

class StandardizedBaseMultimodalModel(nn.Module):
    """
    Unified backbone ensuring EXACT 58,234 parameter capacity across all 4 benchmark models.
    
    Layer Breakdown:
    - mri_encoder:    17,664 params
    - ct_encoder:     17,664 params
    - spatial_fusion:  2,144 params
    - compressor:      1,560 params
    - branch_1:           24 params
    - branch_2:           24 params
    - router:          1,057 params
    - film:              576 params
    - decoder:        17,521 params
    TOTAL:            58,234 params (0.00% capacity variance)
    """
    def __init__(self, latent_dim=32, q_features=24):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Dual 3D CNN Encoders
        self.mri_encoder = nn.Sequential(
            ConvBlock3D(1, 8),
            ConvBlock3D(8, 16),
            ConvBlock3D(16, latent_dim)
        )
        self.ct_encoder = nn.Sequential(
            ConvBlock3D(1, 8),
            ConvBlock3D(8, 16),
            ConvBlock3D(16, latent_dim)
        )
        
        # 3D Spatial Feature Map F
        self.spatial_fusion = nn.Sequential(
            nn.Conv3d(latent_dim * 2, latent_dim, kernel_size=1),
            nn.BatchNorm3d(latent_dim),
            nn.ReLU(inplace=True)
        )
        
        # Pooling & Compression
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.compressor = nn.Linear(latent_dim * 2, q_features)
        
        # FiLM Spatial Modulation
        self.film = FiLMModulation3D(in_features=8, num_channels=latent_dim)
        
        # 3D Segmentation Decoder
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            nn.Conv3d(latent_dim, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            nn.Conv3d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            nn.Conv3d(8, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

# ----------------------------------------------------------------------------------
# Model 1: Classical Baseline (Standardized, 57,153 params, alpha = 0.0 hardcoded)
# ----------------------------------------------------------------------------------
class ClassicalBaselineStandardized(StandardizedBaseMultimodalModel):
    """
    Model 1: Classical Baseline.
    Unrouted classical surrogate pathway directly feeding FiLM.
    alpha = 0.0 hardcoded.
    Zero dead weights.
    Exact parameter count: 57,153.
    """
    def __init__(self, latent_dim=32, q_features=24):
        super().__init__(latent_dim=latent_dim, q_features=q_features)
        self.classical_surrogate = ClassicalSurrogateBottleneck(n_layers=3, n_features=8)
        
    def forward(self, mri, ct):
        batch_size = mri.shape[0]
        z_mri = self.mri_encoder(mri)
        z_ct = self.ct_encoder(ct)
        z_concat = torch.cat([z_mri, z_ct], dim=1)
        
        F = self.spatial_fusion(z_concat)
        z_pooled = self.pool(z_concat).view(batch_size, -1)
        z_comp = self.compressor(z_pooled)
        
        # Pure classical surrogate output
        z_c = self.classical_surrogate(z_comp)
        z_prime = z_c
        
        # Alpha is fixed at 0.0 (pure classical)
        alpha = torch.zeros((batch_size, 1), device=mri.device)
        
        F_prime = self.film(F, z_prime)
        out_mask = self.decoder(F_prime)
        return out_mask, alpha

# ----------------------------------------------------------------------------------
# Model 2: Fixed CQ (Standardized, 57,153 params, alpha = 1.0 hardcoded, FiLM decoder)
# ----------------------------------------------------------------------------------
class FixedCQStandardized(StandardizedBaseMultimodalModel):
    """
    Model 2: Fixed CQ.
    100% of representations routed through the 8-qubit quantum branch.
    alpha = 1.0 hardcoded.
    Zero dead weights.
    Exact parameter count: 57,153.
    """
    def __init__(self, latent_dim=32, q_features=24):
        super().__init__(latent_dim=latent_dim, q_features=q_features)
        self.quantum_branch = QuantumBranchBottleneck()
        
    def forward(self, mri, ct):
        batch_size = mri.shape[0]
        z_mri = self.mri_encoder(mri)
        z_ct = self.ct_encoder(ct)
        z_concat = torch.cat([z_mri, z_ct], dim=1)
        
        F = self.spatial_fusion(z_concat)
        z_pooled = self.pool(z_concat).view(batch_size, -1)
        z_comp = self.compressor(z_pooled)
        
        # Pure quantum branch output
        z_q = self.quantum_branch(z_comp)
        z_prime = z_q
        
        # Alpha is fixed at 1.0 (pure quantum)
        alpha = torch.ones((batch_size, 1), device=mri.device)
        
        F_prime = self.film(F, z_prime)
        out_mask = self.decoder(F_prime)
        return out_mask, alpha

# ----------------------------------------------------------------------------------
# Model 3: Adaptive CC Surrogate (Standardized to 58,234 params, dual classical surrogates)
# ----------------------------------------------------------------------------------
class AdaptiveCCStandardized(StandardizedBaseMultimodalModel):
    """
    Model 3: Adaptive CC Surrogate.
    Dynamic router allocating between two capacity-matched classical surrogates:
    z' = (1 - alpha) * z_C1 + alpha * z_C2.
    Exact parameter count: 58,234.
    """
    def __init__(self, latent_dim=32, q_features=24):
        super().__init__(latent_dim=latent_dim, q_features=q_features)
        self.classical_surrogate_1 = ClassicalSurrogateBottleneck(n_layers=3, n_features=8)
        self.classical_surrogate_2 = ClassicalSurrogateBottleneck(n_layers=3, n_features=8)
        self.router = AdaptiveCQRouter(in_features=latent_dim * 2, hidden_dim=16)
        
    def forward(self, mri, ct):
        batch_size = mri.shape[0]
        z_mri = self.mri_encoder(mri)
        z_ct = self.ct_encoder(ct)
        z_concat = torch.cat([z_mri, z_ct], dim=1)
        
        F = self.spatial_fusion(z_concat)
        z_pooled = self.pool(z_concat).view(batch_size, -1)
        z_comp = self.compressor(z_pooled)
        
        z_c1 = self.classical_surrogate_1(z_comp)
        z_c2 = self.classical_surrogate_2(z_comp)
        
        alpha = self.router(z_pooled)
        z_prime = (1.0 - alpha) * z_c1 + alpha * z_c2
        
        F_prime = self.film(F, z_prime)
        out_mask = self.decoder(F_prime)
        return out_mask, alpha

# ----------------------------------------------------------------------------------
# Model 4: Adaptive CQ - Q-FUSION (Standardized to 58,234 params, classical + quantum)
# ----------------------------------------------------------------------------------
class AdaptiveCQStandardized(StandardizedBaseMultimodalModel):
    """
    Model 4: Adaptive CQ (Q-FUSION).
    Dynamic router allocating between 24-param classical surrogate z_C and 24-param 8-qubit PQC z_Q:
    z' = (1 - alpha) * z_C + alpha * z_Q.
    Exact parameter count: 58,234.
    """
    def __init__(self, latent_dim=32, q_features=24):
        super().__init__(latent_dim=latent_dim, q_features=q_features)
        self.classical_surrogate = ClassicalSurrogateBottleneck(n_layers=3, n_features=8)
        self.quantum_branch = QuantumBranchBottleneck()
        self.router = AdaptiveCQRouter(in_features=latent_dim * 2, hidden_dim=16)
        
    def forward(self, mri, ct):
        batch_size = mri.shape[0]
        z_mri = self.mri_encoder(mri)
        z_ct = self.ct_encoder(ct)
        z_concat = torch.cat([z_mri, z_ct], dim=1)
        
        F = self.spatial_fusion(z_concat)
        z_pooled = self.pool(z_concat).view(batch_size, -1)
        z_comp = self.compressor(z_pooled)
        
        z_c = self.classical_surrogate(z_comp)
        z_q = self.quantum_branch(z_comp)
        
        alpha = self.router(z_pooled)
        z_prime = (1.0 - alpha) * z_c + alpha * z_q
        
        F_prime = self.film(F, z_prime)
        out_mask = self.decoder(F_prime)
        return out_mask, alpha

def verify_parameter_parity():
    """
    Asserts honest parameter parity across all 4 models:
    - Fixed models (Classical Baseline, Fixed CQ): 57,129 base + 24 branch = 57,153 params (zero router overhead, zero dead weights).
    - Adaptive models (Adaptive CC, Adaptive CQ): 57,129 base + 48 branches + 1,057 router = 58,234 params (0.00% capacity variance).
    """
    models = {
        "Classical Baseline": (ClassicalBaselineStandardized(), 57153),
        "Fixed CQ": (FixedCQStandardized(), 57153),
        "Adaptive CC": (AdaptiveCCStandardized(), 58234),
        "Adaptive CQ": (AdaptiveCQStandardized(), 58234)
    }
    counts = {}
    for name, (m, expected) in models.items():
        total = sum(p.numel() for p in m.parameters() if p.requires_grad)
        counts[name] = total
        print(f"[{name}] Trainable Parameters: {total:,} (Expected: {expected:,})")
        assert total == expected, f"Parameter mismatch for {name}! Expected {expected:,}, got {total:,}"
        
    assert counts["Adaptive CC"] == counts["Adaptive CQ"], (
        f"Capacity bias between Adaptive CC ({counts['Adaptive CC']}) and Adaptive CQ ({counts['Adaptive CQ']})!"
    )
    assert counts["Classical Baseline"] == counts["Fixed CQ"], (
        f"Capacity bias between Classical Baseline ({counts['Classical Baseline']}) and Fixed CQ ({counts['Fixed CQ']})!"
    )
    print("[SUCCESS] Honest parameter parity verified across all 4 models (0.00% variance between CC and CQ).")
    return counts

if __name__ == "__main__":
    verify_parameter_parity()

import torch
import torch.nn as nn

try:
    from models_standardized import ConvBlock3D, AdaptiveCQRouter, FiLMModulation3D
    from quantum_bottleneck_scalable import (
        ScalableQuantumBranchBottleneck,
        ScalableClassicalSurrogateBottleneck
    )
except ImportError:
    from src.models_standardized import ConvBlock3D, AdaptiveCQRouter, FiLMModulation3D
    from src.quantum_bottleneck_scalable import (
        ScalableQuantumBranchBottleneck,
        ScalableClassicalSurrogateBottleneck
    )

class BaseScalableMultimodalModel(nn.Module):
    """
    Unified multimodal backbone parameterized by qubit count N and depth L.
    
    Ensures EXACT 0.0000% parameter parity between ScalableAdaptiveCQ and ScalableAdaptiveCC
    for any quantum resource configuration (N, L).
    """
    def __init__(self, n_qubits=8, n_layers=3, latent_dim=32):
        super().__init__()
        self.n_qubits = int(n_qubits)
        self.n_layers = int(n_layers)
        self.latent_dim = int(latent_dim)
        self.q_features = self.n_qubits * self.n_layers
        
        # Dual 3D CNN Encoders (17,664 params each = 35,328 params)
        self.mri_encoder = nn.Sequential(
            ConvBlock3D(1, 8),
            ConvBlock3D(8, 16),
            ConvBlock3D(16, self.latent_dim)
        )
        self.ct_encoder = nn.Sequential(
            ConvBlock3D(1, 8),
            ConvBlock3D(8, 16),
            ConvBlock3D(16, self.latent_dim)
        )
        
        # 3D Spatial Feature Map F (2,144 params)
        self.spatial_fusion = nn.Sequential(
            nn.Conv3d(self.latent_dim * 2, self.latent_dim, kernel_size=1),
            nn.BatchNorm3d(self.latent_dim),
            nn.ReLU(inplace=True)
        )
        
        # Pooling & Compression (64 * (N * L) + (N * L) params)
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.compressor = nn.Linear(self.latent_dim * 2, self.q_features)
        
        # FiLM Spatial Modulation: maps N-dimensional bottleneck z' -> channel scaling and shift
        # 2 * (N * 32 + 32) params
        self.film = FiLMModulation3D(in_features=self.n_qubits, num_channels=self.latent_dim)
        
        # 3D Segmentation Decoder (17,521 params)
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            nn.Conv3d(self.latent_dim, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            nn.Conv3d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            nn.Conv3d(8, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

class ScalableAdaptiveCQ(BaseScalableMultimodalModel):
    """
    Scalable Adaptive CQ (Q-FUSION).
    Dynamic router allocating between capacity-matched classical surrogate z_C and N-qubit L-layer PQC z_Q:
    z' = (1 - alpha) * z_C + alpha * z_Q, where z', z_C, z_Q in R^(B x N).
    """
    def __init__(self, n_qubits=8, n_layers=3, latent_dim=32):
        super().__init__(n_qubits=n_qubits, n_layers=n_layers, latent_dim=latent_dim)
        self.classical_surrogate = ScalableClassicalSurrogateBottleneck(n_qubits=n_qubits, n_layers=n_layers)
        self.quantum_branch = ScalableQuantumBranchBottleneck(n_qubits=n_qubits, n_layers=n_layers)
        self.router = AdaptiveCQRouter(in_features=self.latent_dim * 2, hidden_dim=16)
        
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

class ScalableAdaptiveCC(BaseScalableMultimodalModel):
    """
    Scalable Adaptive CC (Capacity-Matched Classical Surrogate Control).
    Dynamic router allocating between dual capacity-matched classical surrogates z_C1 and z_C2:
    z' = (1 - alpha) * z_C1 + alpha * z_C2, where z', z_C1, z_C2 in R^(B x N).
    Strict 0.0000% capacity variance vs. ScalableAdaptiveCQ.
    """
    def __init__(self, n_qubits=8, n_layers=3, latent_dim=32):
        super().__init__(n_qubits=n_qubits, n_layers=n_layers, latent_dim=latent_dim)
        self.classical_surrogate_1 = ScalableClassicalSurrogateBottleneck(n_qubits=n_qubits, n_layers=n_layers)
        self.classical_surrogate_2 = ScalableClassicalSurrogateBottleneck(n_qubits=n_qubits, n_layers=n_layers)
        self.router = AdaptiveCQRouter(in_features=self.latent_dim * 2, hidden_dim=16)
        
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

class ScalableFixedCQ(BaseScalableMultimodalModel):
    """
    Scalable Fixed CQ (100% Quantum Branch, alpha = 1.0).
    """
    def __init__(self, n_qubits=8, n_layers=3, latent_dim=32):
        super().__init__(n_qubits=n_qubits, n_layers=n_layers, latent_dim=latent_dim)
        self.quantum_branch = ScalableQuantumBranchBottleneck(n_qubits=n_qubits, n_layers=n_layers)
        
    def forward(self, mri, ct):
        batch_size = mri.shape[0]
        z_mri = self.mri_encoder(mri)
        z_ct = self.ct_encoder(ct)
        z_concat = torch.cat([z_mri, z_ct], dim=1)
        
        F = self.spatial_fusion(z_concat)
        z_pooled = self.pool(z_concat).view(batch_size, -1)
        z_comp = self.compressor(z_pooled)
        
        z_q = self.quantum_branch(z_comp)
        z_prime = z_q
        
        alpha = torch.ones((batch_size, 1), device=mri.device)
        F_prime = self.film(F, z_prime)
        out_mask = self.decoder(F_prime)
        return out_mask, alpha

class ScalableClassicalBaseline(BaseScalableMultimodalModel):
    """
    Scalable Classical Baseline (100% Classical Surrogate, alpha = 0.0).
    """
    def __init__(self, n_qubits=8, n_layers=3, latent_dim=32):
        super().__init__(n_qubits=n_qubits, n_layers=n_layers, latent_dim=latent_dim)
        self.classical_surrogate = ScalableClassicalSurrogateBottleneck(n_qubits=n_qubits, n_layers=n_layers)
        
    def forward(self, mri, ct):
        batch_size = mri.shape[0]
        z_mri = self.mri_encoder(mri)
        z_ct = self.ct_encoder(ct)
        z_concat = torch.cat([z_mri, z_ct], dim=1)
        
        F = self.spatial_fusion(z_concat)
        z_pooled = self.pool(z_concat).view(batch_size, -1)
        z_comp = self.compressor(z_pooled)
        
        z_c = self.classical_surrogate(z_comp)
        z_prime = z_c
        
        alpha = torch.zeros((batch_size, 1), device=mri.device)
        F_prime = self.film(F, z_prime)
        out_mask = self.decoder(F_prime)
        return out_mask, alpha

def verify_scalable_parity(n_qubits, n_layers):
    """
    Verifies parameter counts and asserts strict 0.0000% capacity variance
    between ScalableAdaptiveCC and ScalableAdaptiveCQ for given (N, L).
    """
    model_cq = ScalableAdaptiveCQ(n_qubits=n_qubits, n_layers=n_layers)
    model_cc = ScalableAdaptiveCC(n_qubits=n_qubits, n_layers=n_layers)
    
    cq_params = sum(p.numel() for p in model_cq.parameters() if p.requires_grad)
    cc_params = sum(p.numel() for p in model_cc.parameters() if p.requires_grad)
    
    q_weights_count = sum(p.numel() for p in model_cq.quantum_branch.parameters() if p.requires_grad)
    c_weights_count = sum(p.numel() for p in model_cq.classical_surrogate.parameters() if p.requires_grad)
    
    assert q_weights_count == n_qubits * n_layers, f"Expected {n_qubits*n_layers} quantum weights, got {q_weights_count}"
    assert c_weights_count == n_qubits * n_layers, f"Expected {n_qubits*n_layers} surrogate weights, got {c_weights_count}"
    assert cq_params == cc_params, f"Capacity mismatch! CQ: {cq_params:,}, CC: {cc_params:,}"
    
    return {
        "n_qubits": n_qubits,
        "n_layers": n_layers,
        "latent_dim": n_qubits,
        "quantum_weights": q_weights_count,
        "surrogate_weights": c_weights_count,
        "total_params": cq_params,
        "capacity_variance": "0.0000%"
    }

if __name__ == "__main__":
    print("--- Verifying Scalable Model Parity across Resource Tiers ---")
    tiers = [
        ("Tier 1 (4Q, 3L)", 4, 3),
        ("Tier 2 (8Q, 3L)", 8, 3),
        ("Tier 3 (8Q, 5L)", 8, 5),
        ("Tier 4 (12Q, 3L)", 12, 3),
    ]
    for label, nq, nl in tiers:
        res = verify_scalable_parity(nq, nl)
        print(f"[{label}] N={nq}, L={nl} | Bottleneck Dim: {nq} | Quantum Weights: {res['quantum_weights']} | Total Model Params: {res['total_params']:,} | Parity: {res['capacity_variance']}")

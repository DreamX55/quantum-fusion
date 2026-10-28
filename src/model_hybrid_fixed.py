import torch
import torch.nn as nn

try:
    from quantum_bottleneck import QuantumReuploadingBottleneck
except ImportError:
    from src.quantum_bottleneck import QuantumReuploadingBottleneck

class ConvBlock3D(nn.Module):
    """
    Identical 3D Convolutional block from Phase 1 Baseline.
    Preserves exact classical capacity.
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

class FixedHybridMultimodalModel(nn.Module):
    """
    Phase 3: Fixed Quantum Branch Multimodal Architecture.
    
    Replaces the classical 1x1 convolution fusion layer from Phase 1
    with an 8-qubit, 3-layer sequential Data Re-uploading Quantum Bottleneck.
    All multimodal information is forced through the quantum branch (no bypass/skip connections).
    Preserves identical classical capacity for the 3D CNN Encoders and 3D Segmentation Decoder.
    """
    def __init__(self, latent_dim=32, q_features=24):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Branch 1: MRI 3D CNN Encoder (Identical to Phase 1)
        self.mri_encoder = nn.Sequential(
            ConvBlock3D(1, 8),           # (B, 1, 64, 64, 64)  -> (B, 8, 32, 32, 32)
            ConvBlock3D(8, 16),          # (B, 8, 32, 32, 32)  -> (B, 16, 16, 16, 16)
            ConvBlock3D(16, latent_dim)  # (B, 16, 16, 16, 16) -> (B, 32, 8, 8, 8)
        )
        
        # Branch 2: CT 3D CNN Encoder (Identical to Phase 1)
        self.ct_encoder = nn.Sequential(
            ConvBlock3D(1, 8),           # (B, 1, 64, 64, 64)  -> (B, 8, 32, 32, 32)
            ConvBlock3D(8, 16),          # (B, 8, 32, 32, 32)  -> (B, 16, 16, 16, 16)
            ConvBlock3D(16, latent_dim)  # (B, 16, 16, 16, 16) -> (B, 32, 8, 8, 8)
        )
        
        # Global spatial pooling to compress concatenated latent representations (64 channels)
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Fixed Quantum Branch: 8-qubit, 3-layer Data Re-uploading Bottleneck
        # Compresses 64 classical features -> 24 circuit features -> 8 qubits (Pauli-Z expectations)
        self.quantum_bottleneck = QuantumReuploadingBottleneck(
            classical_in_dim=latent_dim * 2,
            q_features=q_features
        )
        
        # Quantum-to-Spatial Projection: Projects 8 quantum observables to the 3D latent spatial tensor
        self.q_expand = nn.Sequential(
            nn.Linear(8, latent_dim * 8 * 8 * 8),
            nn.ReLU(inplace=True)
        )
        
        # 3D Spatial Decoder for Segmentation (Identical to Phase 1)
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
        
    def forward(self, mri, ct):
        batch_size = mri.shape[0]
        
        # 1. Encode MRI and CT independently
        z_mri = self.mri_encoder(mri)  # (B, 32, 8, 8, 8)
        z_ct = self.ct_encoder(ct)    # (B, 32, 8, 8, 8)
        
        # 2. Concatenate multimodal representations
        z_concat = torch.cat([z_mri, z_ct], dim=1)  # (B, 64, 8, 8, 8)
        
        # 3. Pool spatially into a compact representation
        z_flat = self.pool(z_concat).view(batch_size, -1)  # (B, 64)
        
        # 4. Mandatory Quantum Bottleneck: 8 qubits, 3 layers of sequential Data Re-uploading
        q_out = self.quantum_bottleneck(z_flat)  # (B, 8)
        
        # 5. Project quantum features into spatial bottleneck tensor
        z_fused = self.q_expand(q_out).view(batch_size, self.latent_dim, 8, 8, 8)
        
        # 6. Decode into 3D binary segmentation mask
        out_mask = self.decoder(z_fused)  # (B, 1, 64, 64, 64)
        
        return out_mask

if __name__ == "__main__":
    print("--- Testing Fixed Hybrid Multimodal Architecture (Phase 3) ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = FixedHybridMultimodalModel().to(device)
    
    dummy_mri = torch.randn(2, 1, 64, 64, 64, device=device)
    dummy_ct = torch.randn(2, 1, 64, 64, 64, device=device)
    
    output = model(dummy_mri, dummy_ct)
    print(f"Input MRI shape: {dummy_mri.shape}")
    print(f"Input CT shape:  {dummy_ct.shape}")
    print(f"Output mask shape: {output.shape}")
    
    assert output.shape == (2, 1, 64, 64, 64), f"Shape mismatch: {output.shape}"
    print("[SUCCESS] Fixed Hybrid Model successfully compiled and verified.")

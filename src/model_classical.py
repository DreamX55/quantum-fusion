import torch
import torch.nn as nn

class ConvBlock3D(nn.Module):
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

class ClassicalMultimodalBaseline(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        
        # Branch 1: MRI Encoder
        self.mri_encoder = nn.Sequential(
            ConvBlock3D(1, 8),  # 64 -> 32
            ConvBlock3D(8, 16), # 32 -> 16
            ConvBlock3D(16, latent_dim) # 16 -> 8
        )
        
        # Branch 2: CT Encoder
        self.ct_encoder = nn.Sequential(
            ConvBlock3D(1, 8),
            ConvBlock3D(8, 16),
            ConvBlock3D(16, latent_dim)
        )
        
        # Fusion layer (Simulating the 'classical routing' bottleneck)
        self.fusion = nn.Sequential(
            nn.Conv3d(latent_dim * 2, latent_dim, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        
        # Spatial Decoder for Segmentation (Objective 2: Spatial Representation Enhancement)
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
        
    def forward(self, mri, ct):
        # 1. Encode separately
        z_mri = self.mri_encoder(mri)
        z_ct = self.ct_encoder(ct)
        
        # 2. Concatenate and classically fuse
        z_concat = torch.cat([z_mri, z_ct], dim=1)
        z_fused = self.fusion(z_concat)
        
        # 3. Decode to spatial segmentation mask
        out_mask = self.decoder(z_fused)
        
        return out_mask

if __name__ == "__main__":
    # Test the model with dummy data
    model = ClassicalMultimodalBaseline()
    dummy_mri = torch.randn(1, 1, 64, 64, 64)
    dummy_ct = torch.randn(1, 1, 64, 64, 64)
    output = model(dummy_mri, dummy_ct)
    print(f"Input shape: {dummy_mri.shape}")
    print(f"Output shape: {output.shape}")
    print("Classical Baseline Model successfully initialized.")

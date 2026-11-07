import torch
import torch.nn as nn

try:
    from models_standardized import (
        ConvBlock3D,
        AdaptiveCQRouter,
        FiLMModulation3D,
        ClassicalSurrogateBottleneck,
        QuantumBranchBottleneck
    )
except ImportError:
    from src.models_standardized import (
        ConvBlock3D,
        AdaptiveCQRouter,
        FiLMModulation3D,
        ClassicalSurrogateBottleneck,
        QuantumBranchBottleneck
    )

class BaseMultiModalBraTSModel(nn.Module):
    def __init__(self, latent_dim=32, q_features=24):
        super().__init__()
        self.latent_dim = latent_dim
        self.q_features = q_features
        
        self.encoder = nn.Sequential(
            ConvBlock3D(1, 8),
            ConvBlock3D(8, 16),
            ConvBlock3D(16, self.latent_dim)
        )
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.compressor = nn.Linear(self.latent_dim, self.q_features)
        self.film = FiLMModulation3D(in_features=8, num_channels=self.latent_dim)
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
        
    def encode_multimodal(self, x):
        b, m, d, h, w = x.shape
        f_list = []
        for i in range(m):
            f_list.append(self.encoder(x[:, i:i+1]))
        F = torch.stack(f_list, dim=0).mean(dim=0)
        z_pooled = self.pool(F).view(b, -1)
        z_comp = self.compressor(z_pooled)
        return F, z_pooled, z_comp

class BraTSAdaptiveCQ(BaseMultiModalBraTSModel):
    def __init__(self, latent_dim=32, q_features=24):
        super().__init__(latent_dim=latent_dim, q_features=q_features)
        self.classical_surrogate = ClassicalSurrogateBottleneck(n_layers=3, n_features=8)
        self.quantum_branch = QuantumBranchBottleneck()
        self.router = AdaptiveCQRouter(in_features=self.latent_dim, hidden_dim=16)
        
    def forward(self, x):
        F, z_pooled, z_comp = self.encode_multimodal(x)
        z_c = self.classical_surrogate(z_comp)
        z_q = self.quantum_branch(z_comp)
        alpha = self.router(z_pooled)
        z_prime = (1.0 - alpha) * z_c + alpha * z_q
        F_prime = self.film(F, z_prime)
        out_mask = self.decoder(F_prime)
        return out_mask, alpha

class BraTSAdaptiveCC(BaseMultiModalBraTSModel):
    def __init__(self, latent_dim=32, q_features=24):
        super().__init__(latent_dim=latent_dim, q_features=q_features)
        self.classical_surrogate_1 = ClassicalSurrogateBottleneck(n_layers=3, n_features=8)
        self.classical_surrogate_2 = ClassicalSurrogateBottleneck(n_layers=3, n_features=8)
        self.router = AdaptiveCQRouter(in_features=self.latent_dim, hidden_dim=16)
        
    def forward(self, x):
        F, z_pooled, z_comp = self.encode_multimodal(x)
        z_c1 = self.classical_surrogate_1(z_comp)
        z_c2 = self.classical_surrogate_2(z_comp)
        alpha = self.router(z_pooled)
        z_prime = (1.0 - alpha) * z_c1 + alpha * z_c2
        F_prime = self.film(F, z_prime)
        out_mask = self.decoder(F_prime)
        return out_mask, alpha

class BraTSFixedCQ(BaseMultiModalBraTSModel):
    def __init__(self, latent_dim=32, q_features=24):
        super().__init__(latent_dim=latent_dim, q_features=q_features)
        self.quantum_branch = QuantumBranchBottleneck()
        
    def forward(self, x):
        F, z_pooled, z_comp = self.encode_multimodal(x)
        z_q = self.quantum_branch(z_comp)
        alpha = torch.ones((x.shape[0], 1), device=x.device)
        F_prime = self.film(F, z_q)
        out_mask = self.decoder(F_prime)
        return out_mask, alpha

class BraTSClassicalBaseline(BaseMultiModalBraTSModel):
    def __init__(self, latent_dim=32, q_features=24):
        super().__init__(latent_dim=latent_dim, q_features=q_features)
        self.classical_surrogate = ClassicalSurrogateBottleneck(n_layers=3, n_features=8)
        
    def forward(self, x):
        F, z_pooled, z_comp = self.encode_multimodal(x)
        z_c = self.classical_surrogate(z_comp)
        alpha = torch.zeros((x.shape[0], 1), device=x.device)
        F_prime = self.film(F, z_c)
        out_mask = self.decoder(F_prime)
        return out_mask, alpha

def verify_brats_model_parity():
    cq = BraTSAdaptiveCQ()
    cc = BraTSAdaptiveCC()
    fx = BraTSFixedCQ()
    cl = BraTSClassicalBaseline()
    
    p_cq = sum(p.numel() for p in cq.parameters() if p.requires_grad)
    p_cc = sum(p.numel() for p in cc.parameters() if p.requires_grad)
    p_fx = sum(p.numel() for p in fx.parameters() if p.requires_grad)
    p_cl = sum(p.numel() for p in cl.parameters() if p.requires_grad)
    
    print(f"BraTS Adaptive CQ Params: {p_cq:,}")
    print(f"BraTS Adaptive CC Params: {p_cc:,}")
    print(f"BraTS Fixed CQ Params:    {p_fx:,}")
    print(f"BraTS Classical Params:   {p_cl:,}")
    assert p_cq == p_cc, "Capacity mismatch between Adaptive CQ and CC"
    assert p_fx == p_cl, "Capacity mismatch between Fixed CQ and Classical"
    print("Strict 0.0000% capacity variance verified across models.")

if __name__ == "__main__":
    verify_brats_model_parity()

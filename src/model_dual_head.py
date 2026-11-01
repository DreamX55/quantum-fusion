import torch
import torch.nn as nn

try:
    from models_standardized import (
        ConvBlock3D,
        ClassicalSurrogateBottleneck,
        QuantumBranchBottleneck,
        AdaptiveCQRouter,
        FiLMModulation3D,
        StandardizedBaseMultimodalModel
    )
except ImportError:
    from src.models_standardized import (
        ConvBlock3D,
        ClassicalSurrogateBottleneck,
        QuantumBranchBottleneck,
        AdaptiveCQRouter,
        FiLMModulation3D,
        StandardizedBaseMultimodalModel
    )

class ClassificationHead(nn.Module):
    """
    Auxiliary 2-layer MLP classification head (Experiment D).
    Maps 8D hybrid bottleneck representation z' to 2 binary diagnostic classes.
    Parameters:
    - Linear(8, 16): 8*16 + 16 = 144
    - Linear(16, 2): 16*2 + 2 = 34
    Total: 178 parameters.
    """
    def __init__(self, in_features=8, hidden_dim=16, num_classes=2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, z_prime):
        return self.classifier(z_prime)

class StandardizedBaseDualHeadModel(StandardizedBaseMultimodalModel):
    """
    Base dual-head model inheriting from StandardizedBaseMultimodalModel
    and appending the 178-parameter auxiliary classification head.
    Base params: 58,234 (adaptive) or 57,153 (fixed) + 178 = 58,412 / 57,331.
    """
    def __init__(self, latent_dim=32, q_features=24, num_classes=2):
        super().__init__(latent_dim=latent_dim, q_features=q_features)
        self.cls_head = ClassificationHead(in_features=8, hidden_dim=16, num_classes=num_classes)

class DualHeadAdaptiveCQModel(StandardizedBaseDualHeadModel):
    """
    Dual-Head Adaptive CQ (Q-FUSION).
    Joint multi-task segmentation + classification network with 8-qubit quantum bottleneck.
    Total trainable parameters: 58,412.
    """
    def __init__(self, latent_dim=32, q_features=24, num_classes=2):
        super().__init__(latent_dim=latent_dim, q_features=q_features, num_classes=num_classes)
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
        seg_mask = self.decoder(F_prime)
        cls_logits = self.cls_head(z_prime)
        
        return seg_mask, cls_logits, alpha

class DualHeadAdaptiveCCModel(StandardizedBaseDualHeadModel):
    """
    Dual-Head Adaptive CC (Capacity-Matched Classical Surrogate Control).
    Joint multi-task segmentation + classification network with dual classical bottlenecks.
    Total trainable parameters: 58,412 (0.0000% capacity variance vs. Adaptive CQ).
    """
    def __init__(self, latent_dim=32, q_features=24, num_classes=2):
        super().__init__(latent_dim=latent_dim, q_features=q_features, num_classes=num_classes)
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
        seg_mask = self.decoder(F_prime)
        cls_logits = self.cls_head(z_prime)
        
        return seg_mask, cls_logits, alpha

class DualHeadFixedCQModel(StandardizedBaseDualHeadModel):
    """
    Dual-Head Fixed CQ (100% Quantum Branch, alpha = 1.0).
    Total trainable parameters: 57,331.
    """
    def __init__(self, latent_dim=32, q_features=24, num_classes=2):
        super().__init__(latent_dim=latent_dim, q_features=q_features, num_classes=num_classes)
        self.quantum_branch = QuantumBranchBottleneck()
        
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
        seg_mask = self.decoder(F_prime)
        cls_logits = self.cls_head(z_prime)
        
        return seg_mask, cls_logits, alpha

class DualHeadClassicalBaselineModel(StandardizedBaseDualHeadModel):
    """
    Dual-Head Classical Baseline (100% Classical Surrogate, alpha = 0.0).
    Total trainable parameters: 57,331.
    """
    def __init__(self, latent_dim=32, q_features=24, num_classes=2):
        super().__init__(latent_dim=latent_dim, q_features=q_features, num_classes=num_classes)
        self.classical_surrogate = ClassicalSurrogateBottleneck(n_layers=3, n_features=8)
        
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
        seg_mask = self.decoder(F_prime)
        cls_logits = self.cls_head(z_prime)
        
        return seg_mask, cls_logits, alpha

def verify_dual_head_parity():
    """
    Verifies parameter counts and capacity parity across all dual-head models.
    """
    models = {
        "DualHead Classical Baseline": (DualHeadClassicalBaselineModel(), 57331),
        "DualHead Fixed CQ": (DualHeadFixedCQModel(), 57331),
        "DualHead Adaptive CC": (DualHeadAdaptiveCCModel(), 58412),
        "DualHead Adaptive CQ": (DualHeadAdaptiveCQModel(), 58412)
    }
    counts = {}
    for name, (m, expected) in models.items():
        total = sum(p.numel() for p in m.parameters() if p.requires_grad)
        counts[name] = total
        print(f"[{name}] Trainable Parameters: {total:,} (Expected: {expected:,})")
        assert total == expected, f"Parameter mismatch for {name}! Expected {expected:,}, got {total:,}"
        
    assert counts["DualHead Adaptive CC"] == counts["DualHead Adaptive CQ"], (
        f"Capacity bias between DualHead Adaptive CC ({counts['DualHead Adaptive CC']}) and DualHead Adaptive CQ ({counts['DualHead Adaptive CQ']})!"
    )
    assert counts["DualHead Classical Baseline"] == counts["DualHead Fixed CQ"], (
        f"Capacity bias between DualHead Classical Baseline ({counts['DualHead Classical Baseline']}) and DualHead Fixed CQ ({counts['DualHead Fixed CQ']})!"
    )
    print("[SUCCESS] Parameter parity strictly verified across all dual-head models (0.0000% capacity variance between Adaptive CC and Adaptive CQ).")
    return counts

if __name__ == "__main__":
    verify_dual_head_parity()

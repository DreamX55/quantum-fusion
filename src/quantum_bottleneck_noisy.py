import torch
import torch.nn as nn
import pennylane as qml

try:
    from models_scalable import (
        BaseScalableMultimodalModel,
        ScalableClassicalSurrogateBottleneck
    )
    from models_standardized import AdaptiveCQRouter
except ImportError:
    from src.models_scalable import (
        BaseScalableMultimodalModel,
        ScalableClassicalSurrogateBottleneck
    )
    from src.models_standardized import AdaptiveCQRouter

class NoisyQuantumBranchBottleneck(nn.Module):
    """
    NISQ Physical Hardware Noise Bottleneck (Experiment F).
    
    Implements a parameterized quantum data re-uploading circuit with open-system
    depolarizing noise channels on all active qubits after each entangling layer:
    
    rho -> (1 - p) rho + (p / 3) * (X rho X + Y rho Y + Z rho Z)
    
    - Pure statevector backend (default.qubit) when noise_p == 0.0
    - Density matrix backend (default.mixed) with backpropagation when noise_p > 0.0
    - Trainable Parameters: Exactly N * L weights (24 for N=8, L=3)
    """
    def __init__(self, n_qubits=8, n_layers=3, noise_p=0.0):
        super().__init__()
        self.n_qubits = int(n_qubits)
        self.n_layers = int(n_layers)
        self.q_weights = nn.Parameter(torch.randn(self.n_layers, self.n_qubits) * 0.1)
        self.set_noise(noise_p)
        
    def set_noise(self, noise_p):
        self.noise_p = float(noise_p)
        if self.noise_p > 0.0:
            self.dev = qml.device("default.mixed", wires=self.n_qubits)
            p = self.noise_p
            @qml.qnode(self.dev, interface="torch", diff_method="backprop")
            def _noisy_circuit(inputs, weights):
                for l in range(self.n_layers):
                    chunk = inputs[l * self.n_qubits : (l + 1) * self.n_qubits]
                    qml.AngleEmbedding(chunk, wires=range(self.n_qubits), rotation='Y')
                    qml.BasicEntanglerLayers(weights[l:l+1], wires=range(self.n_qubits))
                    for w in range(self.n_qubits):
                        qml.DepolarizingChannel(p, wires=w)
                return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
            self.circuit = _noisy_circuit
        else:
            self.dev = qml.device("default.qubit", wires=self.n_qubits)
            @qml.qnode(self.dev, interface="torch", diff_method="backprop")
            def _clean_circuit(inputs, weights):
                for l in range(self.n_layers):
                    chunk = inputs[l * self.n_qubits : (l + 1) * self.n_qubits]
                    qml.AngleEmbedding(chunk, wires=range(self.n_qubits), rotation='Y')
                    qml.BasicEntanglerLayers(weights[l:l+1], wires=range(self.n_qubits))
                return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
            self.circuit = _clean_circuit

    def forward(self, x):
        batch_size = x.shape[0]
        orig_device = x.device
        x_clean = x.as_subclass(torch.Tensor) if hasattr(x, "as_subclass") else x
        x_cpu = x_clean.cpu()
        w_cpu = self.q_weights.cpu()
        
        q_out_list = []
        for i in range(batch_size):
            q_out_list.append(torch.stack(self.circuit(x_cpu[i], w_cpu)).to(dtype=x.dtype))
        return torch.stack(q_out_list).to(device=orig_device)

class NoisyAdaptiveCQ(BaseScalableMultimodalModel):
    """
    Model 4: Noisy Adaptive CQ (Q-FUSION under NISQ Hardware Noise).
    Dynamic router allocating between 24-param classical surrogate z_C and 24-param noisy PQC z_Q:
    z' = (1 - alpha) * z_C + alpha * z_Q.
    Exact parameter count: 58,234 (N=8, L=3).
    """
    def __init__(self, n_qubits=8, n_layers=3, latent_dim=32, noise_p=0.0):
        super().__init__(n_qubits=n_qubits, n_layers=n_layers, latent_dim=latent_dim)
        self.classical_surrogate = ScalableClassicalSurrogateBottleneck(n_qubits=n_qubits, n_layers=n_layers)
        self.quantum_branch = NoisyQuantumBranchBottleneck(n_qubits=n_qubits, n_layers=n_layers, noise_p=noise_p)
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

class NoisyFixedCQ(BaseScalableMultimodalModel):
    """
    Model 2: Noisy Fixed CQ (100% Noisy Quantum Branch, alpha = 1.0).
    Exact parameter count: 57,153 (N=8, L=3).
    """
    def __init__(self, n_qubits=8, n_layers=3, latent_dim=32, noise_p=0.0):
        super().__init__(n_qubits=n_qubits, n_layers=n_layers, latent_dim=latent_dim)
        self.quantum_branch = NoisyQuantumBranchBottleneck(n_qubits=n_qubits, n_layers=n_layers, noise_p=noise_p)
        
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

def verify_noisy_parity():
    from models_scalable import ScalableAdaptiveCC, ScalableClassicalBaseline
    ad_cq = NoisyAdaptiveCQ(8, 3, 32, 0.0)
    ad_cc = ScalableAdaptiveCC(8, 3, 32)
    fx_cq = NoisyFixedCQ(8, 3, 32, 0.0)
    cl_bs = ScalableClassicalBaseline(8, 3, 32)
    
    p_ad_cq = sum(p.numel() for p in ad_cq.parameters() if p.requires_grad)
    p_ad_cc = sum(p.numel() for p in ad_cc.parameters() if p.requires_grad)
    p_fx_cq = sum(p.numel() for p in fx_cq.parameters() if p.requires_grad)
    p_cl_bs = sum(p.numel() for p in cl_bs.parameters() if p.requires_grad)
    
    print(f"NoisyAdaptiveCQ Params: {p_ad_cq:,}")
    print(f"ScalableAdaptiveCC Params: {p_ad_cc:,}")
    print(f"NoisyFixedCQ Params: {p_fx_cq:,}")
    print(f"ScalableClassicalBaseline Params: {p_cl_bs:,}")
    
    assert p_ad_cq == p_ad_cc == 58234, f"Parity mismatch: {p_ad_cq} vs {p_ad_cc}"
    assert p_fx_cq == p_cl_bs == 57153, f"Parity mismatch: {p_fx_cq} vs {p_cl_bs}"
    print("Exact 0.0000% capacity variance verified across noisy models.")

if __name__ == "__main__":
    verify_noisy_parity()

import torch
import torch.nn as nn
import pennylane as qml

# Define the 8-qubit quantum simulator
n_qubits = 8
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def reuploading_quantum_circuit(inputs, weights):
    """
    inputs: Tensor of shape (24,) - The classical features
    weights: Tensor of shape (3, 8) - The trainable quantum parameters
    """
    # LAYER 1: First 8 features
    qml.AngleEmbedding(inputs[0:8], wires=range(n_qubits), rotation='Y')
    qml.BasicEntanglerLayers(weights[0:1], wires=range(n_qubits))
    
    # LAYER 2: Next 8 features (Data Re-uploading)
    qml.AngleEmbedding(inputs[8:16], wires=range(n_qubits), rotation='Y')
    qml.BasicEntanglerLayers(weights[1:2], wires=range(n_qubits))
    
    # LAYER 3: Final 8 features (Data Re-uploading)
    qml.AngleEmbedding(inputs[16:24], wires=range(n_qubits), rotation='Y')
    qml.BasicEntanglerLayers(weights[2:3], wires=range(n_qubits))
    
    # Measure the expected value (Z-basis) of all 8 qubits
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

class QuantumReuploadingBottleneck(nn.Module):
    def __init__(self, classical_in_dim=256, q_features=24):
        super().__init__()
        # Classical compression down to 24 features (instead of 8)
        self.compressor = nn.Linear(classical_in_dim, q_features)
        
        # Trainable quantum weights (3 layers, 8 qubits)
        self.q_weights = nn.Parameter(torch.randn(3, 8))
        
    def forward(self, x):
        # 1. Classically compress to 24 features
        x_compressed = self.compressor(x)
        
        # 2. Process through the Sequential 8-qubit QNode
        # PennyLane QNodes currently process unbatched inputs natively, so we loop over the batch
        batch_size = x.shape[0]
        orig_device = x.device
        x_compressed = x_compressed.as_subclass(torch.Tensor) if hasattr(x_compressed, "as_subclass") else x_compressed
        x_comp_cpu = x_compressed.cpu()
        w_cpu = self.q_weights.cpu()
        
        q_out_list = []
        for i in range(batch_size):
            q_out_list.append(torch.stack(reuploading_quantum_circuit(x_comp_cpu[i], w_cpu)))
            
        q_out = torch.stack(q_out_list).to(device=orig_device, dtype=x.dtype)
        return q_out

if __name__ == "__main__":
    # Test the user's Re-uploading architecture
    print("--- Q-FUSION Phase 2/3: Data Re-uploading Prototype ---")
    model = QuantumReuploadingBottleneck(classical_in_dim=128, q_features=24)
    
    # Dummy classical latent vector (e.g., from the CNN fusion layer)
    dummy_latent = torch.randn(2, 128) 
    
    # Forward pass
    output = model(dummy_latent)
    print(f"Input shape (from Classical CNN): {dummy_latent.shape}")
    print(f"Compressed shape (fed into Qubits): (2, 24)")
    print(f"Output shape (from 8 Qubits): {output.shape}")
    print("Sequential Quantum Processing successful.")

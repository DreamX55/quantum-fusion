import pennylane as qml
from pennylane import numpy as np
import torch

def run_sanity_test():
    print("--- Q-FUSION Minimal Quantum Sanity Test ---")
    
    # 1. Check frameworks
    print(f"PyTorch version: {torch.__version__}")
    print(f"PennyLane version: {qml.__version__}")
    
    # 2. Define a simple quantum circuit with PyTorch interface
    n_qubits = 2
    dev = qml.device("default.qubit", wires=n_qubits)
    
    @qml.qnode(dev, interface="torch")
    def quantum_circuit(inputs, weights):
        # Embed inputs
        qml.AngleEmbedding(inputs, wires=range(n_qubits))
        # Basic trainable transformation
        qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
        # Measure
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    
    # 3. Initialize parameters
    inputs = torch.tensor([0.1, 0.2], requires_grad=False)
    # BasicEntanglerLayers expects weights of shape (n_layers, n_qubits)
    weights = torch.randn((1, n_qubits), requires_grad=True)
    
    print("\nInitial weights:", weights.detach().numpy())
    
    # 4. Forward pass
    output = quantum_circuit(inputs, weights)
    # Output is a list of tensors from multiple expvals; stack them
    output_tensor = torch.stack(output)
    print("Circuit output:", output_tensor.detach().numpy())
    
    # 5. Backward pass (Gradient check)
    loss = output_tensor.sum()
    loss.backward()
    
    print("Weight gradients:", weights.grad.detach().numpy())
    print("\n[SUCCESS] Quantum circuit executed and gradients successfully computed via PyTorch autograd.")

if __name__ == "__main__":
    run_sanity_test()

import torch
import torch.nn as nn
import pennylane as qml

class ScalableQuantumBranchBottleneck(nn.Module):
    """
    Scalable Parameterized Quantum Re-uploading Circuit (Experiment E).
    
    In accordance with Section 10 & 11 of the Novelty Framework:
    - N qubits, L sequential data re-uploading layers
    - Inputs: z_comp in R^(B x (N * L))
    - Measurement: Pauli-Z expectation values on all N qubits -> z_Q in R^(B x N)
    - Trainable Parameters: Exactly N * L weights
    """
    def __init__(self, n_qubits=8, n_layers=3):
        super().__init__()
        self.n_qubits = int(n_qubits)
        self.n_layers = int(n_layers)
        
        self.q_weights = nn.Parameter(torch.randn(self.n_layers, self.n_qubits) * 0.1)
        
        # Initialize PennyLane state-vector simulator device
        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        
        # Define dynamic QNode
        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def _circuit(inputs, weights):
            for l in range(self.n_layers):
                chunk = inputs[l * self.n_qubits : (l + 1) * self.n_qubits]
                qml.AngleEmbedding(chunk, wires=range(self.n_qubits), rotation='Y')
                qml.BasicEntanglerLayers(weights[l:l+1], wires=range(self.n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
            
        self.circuit = _circuit
        
    def forward(self, x):
        """
        x: Tensor of shape (B, N * L)
        Returns: Tensor of shape (B, N) bounded in [-1, 1]
        """
        batch_size = x.shape[0]
        orig_device = x.device
        x_clean = x.as_subclass(torch.Tensor) if hasattr(x, "as_subclass") else x
        x_cpu = x_clean.cpu()
        w_cpu = self.q_weights.cpu()
        
        q_out_list = []
        for i in range(batch_size):
            q_out_list.append(torch.stack(self.circuit(x_cpu[i], w_cpu)).to(dtype=x.dtype))
            
        q_out = torch.stack(q_out_list).to(device=orig_device)
        return q_out

class ScalableClassicalSurrogateBottleneck(nn.Module):
    """
    Capacity-Matched Scalable Classical Surrogate Control (Experiment E).
    
    Maps z_comp in R^(B x (N * L)) to z_C in R^(B x N):
    - Reshapes input to (B, L, N)
    - Applies trainable parameter tensor c_weights of shape (L, N)
    - Computes elementwise product and reduction over L layers
    - Bounded in [-1, 1] via tanh activation
    - Exact Parameter Count: N * L (0.0000% capacity variance vs. Quantum Branch)
    """
    def __init__(self, n_qubits=8, n_layers=3):
        super().__init__()
        self.n_qubits = int(n_qubits)
        self.n_layers = int(n_layers)
        self.c_weights = nn.Parameter(torch.randn(self.n_layers, self.n_qubits) * 0.1)
        
    def forward(self, x):
        """
        x: Tensor of shape (B, N * L)
        Returns: Tensor of shape (B, N) bounded in [-1, 1]
        """
        batch_size = x.shape[0]
        # Reshape to (B, L, N)
        x_reshaped = x.view(batch_size, self.n_layers, self.n_qubits)
        # Elementwise scaling and layer reduction: z_c = tanh(sum_{l=0}^{L-1} c_weights[l] * x_reshaped[:, l, :])
        out = (x_reshaped * self.c_weights).sum(dim=1)
        return torch.tanh(out)

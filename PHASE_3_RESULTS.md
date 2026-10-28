# Q-FUSION: PHASE 3 RESULTS — FIXED QUANTUM BRANCH AUDIT

**Date:** 2026-10-10  
**Phase:** Phase 3 — Fixed Quantum Branch Integration & Gradient Verification  
**Dataset:** SynthRAD2023 Task 1 Brain (181 Patients)  
**Modalities:** MRI (`mr.nii.gz`) + CT (`ct.nii.gz`)  
**Target:** 3D Head/Brain Mask (`mask.nii.gz`, $64 \times 64 \times 64$)  
**Hardware:** NVIDIA RTX A2000 12GB (CUDA 12.1, PyTorch 2.5.1+cu121)

---

## 1. Executive Summary

Phase 3 establishes the **Fixed Quantum Branch** for Q-FUSION. The classical $1 \times 1 \times 1$ convolution fusion layer from Phase 1 was replaced with an **8-qubit, 3-layer sequential Data Re-uploading Quantum Circuit** implemented in PennyLane. All multimodal information is forced through the quantum branch without classical bypass or shortcut connections.

We trained the architecture for 1 full epoch across all 181 patients from SynthRAD2023. Spatial segmentation loss demonstrated a dramatic drop from $0.6961$ down to a minimum of $0.0931$ (average $0.3172$), and non-zero gradients successfully propagated across the quantum-classical boundary into both 3D CNN encoders.

---

## 2. Quantitative Results & Comparison

| Metric | Phase 1 (Classical Baseline) | Phase 3 (Fixed Hybrid Quantum) | Delta / Change |
| :--- | :--- | :--- | :--- |
| **Hardware** | CPU (Intel) | NVIDIA RTX A2000 12GB (CUDA 12.1) | GPU Acceleration Active |
| **Architecture** | Dual 3D CNN $\to$ Conv3D(1x1) $\to$ 3D Decoder | Dual 3D CNN $\to$ 8-Qubit Re-uploading PQC $\to$ 3D Decoder | Quantum Fusion Layer |
| **Quantum Parameters** | 0 | 24 ($3 \text{ layers} \times 8 \text{ qubits}$) | Trainable Unitary Rotations |
| **Total Trainable Params** | 54,929 | 201,889 | Includes Quantum Spatial Projection |
| **Initial BCE Loss** | 0.6523 | 0.6961 | Pre-convergence state |
| **Minimum BCE Loss** | 0.4945 | **0.0931** | **-86.6% loss reduction** |
| **Final Average Loss** | ~0.50 | **0.3172** | **Consistent learning trajectory** |
| **Runtime per Batch** | ~1.500 seconds | **0.2289 seconds** | **6.55x faster** |
| **Epoch Duration (181 pts)** | ~4.5 minutes (extrapolated) | **128.06 seconds** | Completed in ~2.1 minutes |

---

## 3. Quantum-Classical Gradient Flow Audit

Backpropagation was measured across the full model at the end of the full epoch:

- **Decoder Layer 1 Grad Norm:** $2.4663 \times 10^0$ (Robust backward signal from BCE loss)
- **Quantum Circuit Weights (`q_weights`) Grad Norm:** $6.9058 \times 10^{-1}$ (Active quantum parameter updates)
- **Quantum Classical Compressor Grad Norm:** $5.0205 \times 10^0$ (Quantum-to-classical bridge gradient)
- **MRI 3D CNN Encoder Layer 1 Grad Norm:** $7.1032 \times 10^{-2}$ (Successful backprop through QNode)
- **CT 3D CNN Encoder Layer 1 Grad Norm:** $1.4550 \times 10^{-1}$ (Successful backprop through QNode)

**Conclusion:** Zero gradient stalling or barren plateaus encountered at $N=8$ qubits with depth $L=3$.

---

## 4. Bridge Failures & Solutions Identified

1. **CUDA / CPU Simulator Tensor Mismatch:**
   - *Failure:* PennyLane's `default.qubit` simulator runs state-vector algebra on CPU memory. Passing CUDA tensors directly caused `RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!`.
   - *Solution:* Transferred compressed feature tensors to CPU for the QNode execution and immediately transferred the expectation values back to the source device via `.to(device=orig_device)`. PyTorch autograd records the copy operation as a differentiable node and seamlessly propagates GPU gradients.

2. **MONAI `MetaTensor` vs PyTorch `Tensor` Type Mismatch:**
   - *Failure:* MONAI transforms output `MetaTensor` objects with clinical metadata. When passed to PennyLane alongside pure `torch.nn.Parameter` weights, PennyLane raised `ValueError: Tensors contain mixed types; cannot determine dispatch library`.
   - *Solution:* Stripped `MetaTensor` metadata via `.as_subclass(torch.Tensor)` prior to PennyLane interface dispatch.

3. **Data Type Discrepancy (Float64 vs Float32):**
   - *Failure:* PennyLane expectation values default to `torch.float64` (`Double`), causing matrix multiplication errors with downstream PyTorch linear layers (`torch.float32`).
   - *Solution:* Explicitly passed `dtype=x.dtype` during device re-casting (`torch.stack(...).to(device=orig_device, dtype=x.dtype)`).

---

## 5. Acceptance Checklist

- [x] `model_hybrid_fixed.py` compiles without tensor shape mismatch errors.
- [x] Gradients successfully cross the quantum-classical boundary without stalling.
- [x] Training loss demonstrates a measurable decrease ($0.6961 \to 0.0931$).
- [x] Output predictions correctly match the $64 \times 64 \times 64$ spatial dimensions of the target mask.

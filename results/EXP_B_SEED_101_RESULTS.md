# Q-FUSION: EXPERIMENT B — MULTIMODAL SCALING & MODALITY ABLATION

## Empirical Performance & Routing Dynamics Under Modality Variation

**Protocol:** Adam (`lr=5e-4`, `weight_decay=1e-4`), `CosineAnnealingLR(T_max=20)`
**Regularization:** Early stopping with patience = 5 on Validation DSC
**Parity:** Exact 58,234 parameter alignment across adaptive variants (0.0000% capacity variance)

---

### 1. Comparative Summary Matrix

| Modality Regime | Architecture | Model ID | Peak Val DSC | Peak Val IoU | Peak Epoch | Final Val DSC | Final Val Loss | Final Learned $\bar{\alpha}$ | Runtime |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **BOTH** | Classical Baseline | `classical_baseline` | **0.8901** | 0.8043 | Ep 1 | 0.6344 | 2.3072 | 0.0000 | 132.7s |
| **BOTH** | Fixed CQ (100% Q) | `fixed_cq` | **0.8929** | 0.8081 | Ep 1 | 0.6429 | 2.2314 | 1.0000 | 147.0s |
| **BOTH** | Adaptive CC Surrogate | `adaptive_cc` | **0.9051** | 0.8272 | Ep 1 | 0.6420 | 1.7484 | 0.4888 | 134.6s |
| **BOTH** | Adaptive CQ (Q-FUSION) | `adaptive_cq` | **0.9130** | 0.8401 | Ep 1 | 0.6540 | 1.5583 | 0.4673 | 147.3s |
| **BOTH** | Classical Baseline | `classical_baseline` | **0.7448** | 0.5939 | Ep 12 | 0.6442 | 1.8760 | 0.0000 | 366.5s |
| **BOTH** | Fixed CQ (100% Q) | `fixed_cq` | **0.6992** | 0.5378 | Ep 2 | 0.6495 | 1.6503 | 1.0000 | 166.3s |
| **BOTH** | Adaptive CC Surrogate | `adaptive_cc` | **0.7452** | 0.5944 | Ep 2 | 0.6998 | 1.2705 | 0.4781 | 158.9s |
| **BOTH** | Adaptive CQ (Q-FUSION) | `adaptive_cq` | **0.7462** | 0.5957 | Ep 2 | 0.6538 | 1.5821 | 0.4758 | 175.4s |

---

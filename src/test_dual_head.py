import torch
import torch.nn as nn
from model_dual_head import (
    DualHeadAdaptiveCQModel,
    DualHeadAdaptiveCCModel,
    DualHeadFixedCQModel,
    DualHeadClassicalBaselineModel,
    verify_dual_head_parity
)
from loss_dual_task import DualTaskLoss

def test_dual_head_pipeline():
    print("=" * 70)
    print("--- EXPERIMENT D: DUAL-HEAD ARCHITECTURE & GRADIENT FLOW VERIFICATION ---")
    print("=" * 70)
    
    # 1. Parameter parity check
    print("\n[Step 1] Checking parameter counts & parity...")
    counts = verify_dual_head_parity()
    
    # 2. Forward pass verification
    print("\n[Step 2] Testing forward pass with synthetic batch (B=2, C=1, D=64, H=64, W=64)...")
    device = torch.device("cpu")
    model = DualHeadAdaptiveCQModel().to(device)
    
    mri = torch.randn(2, 1, 64, 64, 64, device=device)
    ct = torch.randn(2, 1, 64, 64, 64, device=device)
    seg_target = torch.randint(0, 2, (2, 1, 64, 64, 64), device=device).float()
    cls_target = torch.tensor([0, 1], device=device).long()
    
    seg_mask, cls_logits, alpha = model(mri, ct)
    
    print(f"  Segmentation Mask Shape: {tuple(seg_mask.shape)} (Expected: (2, 1, 64, 64, 64))")
    print(f"  Classification Logits Shape: {tuple(cls_logits.shape)} (Expected: (2, 2))")
    print(f"  Routing Alpha Shape: {tuple(alpha.shape)} (Expected: (2, 1))")
    print(f"  Alpha values: {alpha.squeeze().tolist()}")
    
    assert seg_mask.shape == (2, 1, 64, 64, 64), f"Seg shape mismatch: {seg_mask.shape}"
    assert cls_logits.shape == (2, 2), f"Cls shape mismatch: {cls_logits.shape}"
    assert alpha.shape == (2, 1), f"Alpha shape mismatch: {alpha.shape}"
    print("  [SUCCESS] Forward pass shapes verified.")
    
    # 3. Dual-task loss and simultaneous backward gradient flow
    print("\n[Step 3] Testing composite loss and simultaneous dual-head backward flow...")
    criterion = DualTaskLoss(lambda_seg=1.0, lambda_init=1.0)
    loss, l_cls, l_seg, l_bce, l_dice, pen, lam = criterion(seg_mask, seg_target, cls_logits, cls_target, alpha)
    
    print(f"  Composite Loss: {loss.item():.4f} (Cls: {l_cls.item():.4f}, Seg: {l_seg.item():.4f}, Pen: {pen.item():.4f}, Lam: {lam:.4f})")
    
    model.zero_grad()
    loss.backward()
    
    # Inspect critical layer gradients
    critical_params = {
        "mri_encoder": model.mri_encoder[0].conv[0].weight.grad,
        "ct_encoder": model.ct_encoder[0].conv[0].weight.grad,
        "spatial_fusion": model.spatial_fusion[0].weight.grad,
        "compressor": model.compressor.weight.grad,
        "router": model.router.router[0].weight.grad,
        "classical_surrogate": model.classical_surrogate.c_weights.grad,
        "quantum_branch": model.quantum_branch.q_weights.grad,
        "classification_head": model.cls_head.classifier[0].weight.grad,
        "film_generator": model.film.gamma_gen.weight.grad,
        "segmentation_decoder": model.decoder[1].weight.grad,
    }
    
    print("\n  Gradient Norm Check across Architectural Subsystems:")
    all_active = True
    for name, grad in critical_params.items():
        if grad is None:
            print(f"    [FAIL] {name}: grad is None!")
            all_active = False
        else:
            gnorm = grad.norm().item()
            print(f"    [PASS] {name:22s} | Grad Norm: {gnorm:.6f}")
            if gnorm == 0.0:
                print(f"    [FAIL] {name}: grad norm is exactly 0.0!")
                all_active = False
                
    assert all_active, "One or more critical subsystems failed to receive backward gradients!"
    print("\n[SUCCESS] Dual-Head architecture and simultaneous backpropagation verified across both heads!")

if __name__ == "__main__":
    test_dual_head_pipeline()

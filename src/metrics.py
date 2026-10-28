import torch
import torch.nn as nn

def compute_dice(y_pred, y_true, threshold=0.5, epsilon=1e-7):
    r"""
    Computes Dice Similarity Coefficient (DSC) on binarized predictions.
    DSC = (2 * |X \cap Y|) / (|X| + |Y|)
    """
    with torch.no_grad():
        pred_bin = (y_pred > threshold).float()
        true_bin = (y_true > threshold).float()
        
        intersection = torch.sum(pred_bin * true_bin)
        cardinality = torch.sum(pred_bin) + torch.sum(true_bin)
        
        dice = (2.0 * intersection + epsilon) / (cardinality + epsilon)
        return dice.item()

def compute_iou(y_pred, y_true, threshold=0.5, epsilon=1e-7):
    r"""
    Computes Intersection-over-Union (IoU / Jaccard Index) on binarized predictions.
    IoU = |X \cap Y| / |X \cup Y|
    """
    with torch.no_grad():
        pred_bin = (y_pred > threshold).float()
        true_bin = (y_true > threshold).float()
        
        intersection = torch.sum(pred_bin * true_bin)
        union = torch.sum(pred_bin) + torch.sum(true_bin) - intersection
        
        iou = (intersection + epsilon) / (union + epsilon)
        return iou.item()

def compute_bce(y_pred, y_true):
    """
    Computes Binary Cross-Entropy loss between continuous prediction and true mask.
    """
    criterion = nn.BCELoss()
    return criterion(y_pred, y_true).item()

def compute_all_metrics(y_pred, y_true, threshold=0.5):
    """
    Convenience function returning dictionary with BCE, DSC, and IoU.
    """
    bce = compute_bce(y_pred, y_true)
    dice = compute_dice(y_pred, y_true, threshold=threshold)
    iou = compute_iou(y_pred, y_true, threshold=threshold)
    return {
        "bce": bce,
        "dice": dice,
        "iou": iou
    }

if __name__ == "__main__":
    print("--- Testing Metrics (DSC, IoU, BCE) ---")
    pred = torch.tensor([0.9, 0.8, 0.2, 0.1])
    target = torch.tensor([1.0, 1.0, 0.0, 0.0])
    
    m = compute_all_metrics(pred, target)
    print(f"BCE:  {m['bce']:.4f}")
    print(f"DSC:  {m['dice']:.4f}")
    print(f"IoU:  {m['iou']:.4f}")
    assert m['dice'] > 0.99, "Perfect match should have DSC near 1.0"
    print("[SUCCESS] Metrics verified.")

import torch
import torch.nn as nn

try:
    from loss_compound import SoftDiceLoss
except ImportError:
    from src.loss_compound import SoftDiceLoss

class DualTaskLoss(nn.Module):
    """
    Experiment D: Multi-Task Composite Loss Function.
    
    In accordance with Section 6 & 11 of the Novelty Framework:
    L_total = L_cls + lambda_seg * L_seg + lambda(t) * (alpha - 0.5)^2
    
    where:
    - L_cls: Cross-Entropy loss on binary diagnostic classification from hybrid bottleneck z'
    - L_seg: Compound BCE + Soft Dice loss on spatial FiLM-modulated segmentation
    - lambda(t) * (alpha - 0.5)^2: Annealed anti-collapse routing penalty preventing premature collapse
    """
    def __init__(
        self,
        lambda_seg=1.0,
        lambda_init=1.0,
        lambda_min=0.01,
        decay_epochs=10,
        total_epochs=20,
        epsilon=1e-5
    ):
        super().__init__()
        self.lambda_seg = float(lambda_seg)
        self.cls_criterion = nn.CrossEntropyLoss()
        self.bce = nn.BCELoss()
        self.soft_dice = SoftDiceLoss(epsilon=epsilon)
        
        self.lambda_init = float(lambda_init)
        self.lambda_min = float(lambda_min)
        self.decay_epochs = int(decay_epochs)
        self.total_epochs = int(total_epochs)
        self.current_epoch = 0
        self.lambda_t = self.lambda_init
        
    def get_lambda(self, epoch=None, step=None, steps_per_epoch=None):
        if step is not None and steps_per_epoch is not None and steps_per_epoch > 0:
            current_progress_epochs = self.current_epoch + (step / steps_per_epoch)
        elif epoch is not None:
            current_progress_epochs = float(epoch)
        else:
            current_progress_epochs = float(self.current_epoch)
            
        if current_progress_epochs < self.decay_epochs:
            progress = current_progress_epochs / float(self.decay_epochs)
            decayed = self.lambda_init - progress * (self.lambda_init - self.lambda_min)
            return max(self.lambda_min, float(decayed))
        return float(self.lambda_min)
        
    def step_epoch(self, epoch=None):
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1
        self.lambda_t = self.get_lambda()
        return self.lambda_t
        
    def step_batch(self, step, steps_per_epoch):
        self.lambda_t = self.get_lambda(step=step, steps_per_epoch=steps_per_epoch)
        return self.lambda_t
        
    def forward(self, seg_pred, seg_target, cls_pred, cls_target, alpha=None):
        """
        Args:
            seg_pred: Predicted segmentation mask tensor (B, 1, D, H, W) in [0, 1]
            seg_target: Ground truth mask tensor (B, 1, D, H, W)
            cls_pred: Classification logits (B, 2)
            cls_target: Ground truth class labels (B,) [0 or 1]
            alpha: Learned routing scalar tensor (B, 1) or None
        """
        # 1. Classification Loss
        l_cls = self.cls_criterion(cls_pred, cls_target)
        
        # 2. Segmentation Loss (BCE + Soft Dice)
        l_bce = self.bce(seg_pred, seg_target)
        l_dice = self.soft_dice(seg_pred, seg_target)
        l_seg = l_bce + l_dice
        
        # 3. Annealed Anti-Collapse Penalty
        if alpha is not None:
            penalty = torch.mean((alpha - 0.5) ** 2)
            anti_collapse = self.lambda_t * penalty
        else:
            penalty = torch.tensor(0.0, device=seg_pred.device)
            anti_collapse = torch.tensor(0.0, device=seg_pred.device)
            
        # 4. Total Multi-Task Loss
        l_total = l_cls + (self.lambda_seg * l_seg) + anti_collapse
        
        return l_total, l_cls, l_seg, l_bce, l_dice, penalty, self.lambda_t

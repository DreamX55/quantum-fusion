import torch
import torch.nn as nn

class AnnealedAntiCollapseLoss(nn.Module):
    """
    Phase 4: Annealed Anti-Collapse Loss Function.
    
    Addresses Pitfall 1 (Classical Collapse) in the Research Novelty Framework:
    L_total = BCE(y, y_hat) + lambda(t) * (alpha - 0.5)^2
    
    The penalty term (alpha - 0.5)^2 penalizes extreme routing (alpha -> 0 or alpha -> 1),
    enforcing equal exploration around alpha = 0.5 during early training.
    The coefficient lambda(t) decays over time, allowing the router to discover
    the task-optimal classical-quantum balance without premature collapse.
    """
    def __init__(self, lambda_init=1.0, lambda_min=0.01, decay_rate=0.9, schedule="exponential"):
        super().__init__()
        self.bce = nn.BCELoss()
        self.lambda_init = float(lambda_init)
        self.lambda_min = float(lambda_min)
        self.decay_rate = float(decay_rate)
        self.schedule = schedule
        self.current_epoch = 0
        self.current_step = 0
        self.total_steps = 100
        self.lambda_t = self.lambda_init
        
    def get_lambda(self, epoch=None, step=None, total_steps=None):
        """
        Computes the annealed lambda coefficient based on the chosen schedule.
        Supports both epoch-level and fine-grained intra-epoch batch scheduling.
        """
        if step is not None and total_steps is not None and total_steps > 0:
            progress = min(1.0, max(0.0, step / total_steps))
            if self.schedule == "linear":
                val = self.lambda_init - progress * (self.lambda_init - self.lambda_min)
            elif self.schedule == "cosine":
                import math
                val = self.lambda_min + 0.5 * (self.lambda_init - self.lambda_min) * (1.0 + math.cos(math.pi * progress))
            else: # exponential decay
                val = self.lambda_init * (self.decay_rate ** (progress * 10))
            return max(self.lambda_min, float(val))
            
        ep = epoch if epoch is not None else self.current_epoch
        if self.schedule == "linear":
            val = max(self.lambda_min, self.lambda_init - ep * self.decay_rate)
        else:
            val = max(self.lambda_min, self.lambda_init * (self.decay_rate ** ep))
        return float(val)
        
    def step_epoch(self):
        """Advances epoch counter and updates lambda."""
        self.current_epoch += 1
        self.lambda_t = self.get_lambda(epoch=self.current_epoch)
        return self.lambda_t

    def step_batch(self, step, total_steps):
        """Fine-grained intra-epoch step update for single-epoch exploration."""
        self.current_step = step
        self.total_steps = total_steps
        self.lambda_t = self.get_lambda(step=step, total_steps=total_steps)
        return self.lambda_t

    def forward(self, y_pred, y_true, alpha):
        """
        Args:
            y_pred: Model segmentation prediction (B, 1, D, H, W)
            y_true: Ground truth binary mask (B, 1, D, H, W)
            alpha: Router gating scalar tensor (B, 1) or scalar in [0, 1]
            
        Returns:
            loss_total: Combined annealed loss
            loss_bce: Segmentation cross-entropy term
            loss_penalty: Anti-collapse penalty term
            lambda_t: Current annealing coefficient
        """
        loss_bce = self.bce(y_pred, y_true)
        
        # (alpha - 0.5)^2 centered at 0.5 (balanced CQ routing)
        alpha_penalty = torch.mean((alpha - 0.5) ** 2)
        
        loss_penalty = self.lambda_t * alpha_penalty
        loss_total = loss_bce + loss_penalty
        
        return loss_total, loss_bce, loss_penalty, self.lambda_t

if __name__ == "__main__":
    print("--- Testing AnnealedAntiCollapseLoss ---")
    criterion = AnnealedAntiCollapseLoss(lambda_init=1.0, decay_rate=0.8)
    
    dummy_pred = torch.tensor([0.9, 0.1, 0.8], requires_grad=True)
    dummy_true = torch.tensor([1.0, 0.0, 1.0])
    
    # Test at alpha = 0.0 (collapsed to classical)
    alpha_0 = torch.tensor([0.0], requires_grad=True)
    l_tot_0, l_bce_0, l_pen_0, lam = criterion(dummy_pred, dummy_true, alpha_0)
    print(f"Alpha=0.0 | Total: {l_tot_0.item():.4f} | BCE: {l_bce_0.item():.4f} | Penalty: {l_pen_0.item():.4f} (Max penalty)")
    
    # Test at alpha = 0.5 (balanced)
    alpha_half = torch.tensor([0.5], requires_grad=True)
    l_tot_h, l_bce_h, l_pen_h, lam = criterion(dummy_pred, dummy_true, alpha_half)
    print(f"Alpha=0.5 | Total: {l_tot_h.item():.4f} | BCE: {l_bce_h.item():.4f} | Penalty: {l_pen_h.item():.4f} (Zero penalty)")
    
    assert l_pen_h.item() == 0.0, "Penalty at alpha=0.5 must be 0"
    assert l_pen_0.item() > 0.0, "Penalty at alpha=0.0 must be positive"
    print("[SUCCESS] AnnealedAntiCollapseLoss verified.")

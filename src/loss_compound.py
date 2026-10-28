import torch
import torch.nn as nn

class SoftDiceLoss(nn.Module):
    """
    Continuous Soft Dice Loss for 3D binary segmentation.
    L_Dice = 1 - (2 * sum(y_true * y_pred) + eps) / (sum(y_true^2) + sum(y_pred^2) + eps)
    """
    def __init__(self, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon
        
    def forward(self, y_pred, y_true):
        # Flatten spatial dimensions per batch sample
        y_pred_flat = y_pred.view(y_pred.size(0), -1)
        y_true_flat = y_true.view(y_true.size(0), -1)
        
        intersection = torch.sum(y_pred_flat * y_true_flat, dim=1)
        cardinality = torch.sum(y_pred_flat ** 2, dim=1) + torch.sum(y_true_flat ** 2, dim=1)
        
        dice = (2.0 * intersection + self.epsilon) / (cardinality + self.epsilon)
        loss_dice = 1.0 - dice
        return torch.mean(loss_dice)

class CompoundBCEDiceLoss(nn.Module):
    """
    Combined BCE + Soft Dice Loss with Annealed Anti-Collapse Penalty for Experiment A Overhaul.
    
    L_task = L_BCE(y, y_hat) + L_Dice(y, y_hat)
    L_total = L_task + lambda(t) * (alpha - 0.5)^2
    
    Annealing schedule:
    lambda(t) decays smoothly from lambda_init (1.0) to lambda_min (0.01) over the first 10 epochs,
    and remains at lambda_min (0.01) for epochs 11 to 30.
    """
    def __init__(self, lambda_init=1.0, lambda_min=0.01, decay_epochs=10, total_epochs=30, epsilon=1e-5):
        super().__init__()
        self.bce = nn.BCELoss()
        self.soft_dice = SoftDiceLoss(epsilon=epsilon)
        self.lambda_init = float(lambda_init)
        self.lambda_min = float(lambda_min)
        self.decay_epochs = int(decay_epochs)
        self.total_epochs = int(total_epochs)
        self.current_epoch = 0
        self.lambda_t = self.lambda_init
        
    def get_lambda(self, epoch=None, step=None, steps_per_epoch=None):
        """
        Computes lambda(t) based on the current progress:
        - Epoch 0 to decay_epochs (0 to 10): Linear decay from lambda_init to lambda_min
        - Epoch > decay_epochs (11 to 30): Constant lambda_min
        Supports fine-grained intra-epoch step progress.
        """
        if step is not None and steps_per_epoch is not None and steps_per_epoch > 0:
            current_progress_epochs = self.current_epoch + (step / steps_per_epoch)
        elif epoch is not None:
            current_progress_epochs = float(epoch)
        else:
            current_progress_epochs = float(self.current_epoch)
            
        if current_progress_epochs <= self.decay_epochs:
            # Linear decay from lambda_init to lambda_min
            fraction = current_progress_epochs / float(self.decay_epochs)
            lam = self.lambda_init - fraction * (self.lambda_init - self.lambda_min)
        else:
            # Plateaus at lambda_min for remaining epochs
            lam = self.lambda_min
            
        return max(self.lambda_min, float(lam))
        
    def step_epoch(self, epoch=None):
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1
        self.lambda_t = self.get_lambda(epoch=self.current_epoch)
        return self.lambda_t
        
    def step_batch(self, step, steps_per_epoch):
        self.lambda_t = self.get_lambda(step=step, steps_per_epoch=steps_per_epoch)
        return self.lambda_t

    def forward(self, y_pred, y_true, alpha=None):
        """
        Args:
            y_pred: (B, 1, D, H, W) continuous predictions in [0, 1]
            y_true: (B, 1, D, H, W) binary ground truth masks
            alpha: (B, 1) or scalar routing coefficient. If None or non-adaptive, penalty = 0.
            
        Returns:
            loss_total: Combined total loss
            loss_bce: BCE component
            loss_dice: Soft Dice component
            loss_penalty: Anti-collapse penalty
            lambda_t: Current lambda value
        """
        loss_bce = self.bce(y_pred, y_true)
        loss_dice = self.soft_dice(y_pred, y_true)
        loss_task = loss_bce + loss_dice
        
        if alpha is not None and isinstance(alpha, torch.Tensor) and alpha.requires_grad:
            alpha_penalty = torch.mean((alpha - 0.5) ** 2)
            loss_penalty = self.lambda_t * alpha_penalty
        else:
            loss_penalty = torch.tensor(0.0, device=y_pred.device)
            
        loss_total = loss_task + loss_penalty
        return loss_total, loss_bce, loss_dice, loss_penalty, self.lambda_t

if __name__ == "__main__":
    print("--- Testing CompoundBCEDiceLoss ---")
    criterion = CompoundBCEDiceLoss()
    
    pred = torch.tensor([0.9, 0.1, 0.8, 0.2]).view(1, 1, 2, 2, 1)
    true = torch.tensor([1.0, 0.0, 1.0, 0.0]).view(1, 1, 2, 2, 1)
    alpha = torch.tensor([[0.5]], requires_grad=True)
    
    l_tot, l_bce, l_dice, l_pen, lam = criterion(pred, true, alpha)
    print(f"Total: {l_tot.item():.4f} | BCE: {l_bce.item():.4f} | Dice: {l_dice.item():.4f} | Pen: {l_pen.item():.4f} | Lam: {lam:.4f}")
    assert l_pen.item() == 0.0, "Penalty at alpha=0.5 must be 0"
    print("[SUCCESS] CompoundBCEDiceLoss verified.")

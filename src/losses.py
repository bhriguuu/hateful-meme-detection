"""
Loss functions for Hateful Meme Detection.

Includes Focal Loss for handling class imbalance in the dataset
(64% not hateful vs 36% hateful).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    When gamma > 0, reduces the loss for well-classified examples,
    focusing training on hard, misclassified examples.
    
    Reference:
        Lin et al., "Focal Loss for Dense Object Detection" (ICCV 2017)
    
    Args:
        alpha: Weighting factor for positive class (handles imbalance)
               Higher alpha = more weight on positive (minority) class
        gamma: Focusing parameter (gamma=0 equals BCE)
               Higher gamma = more focus on hard examples
        reduction: 'mean', 'sum', or 'none'
    
    Example:
        >>> criterion = FocalLoss(alpha=0.6412, gamma=2.0)
        >>> loss = criterion(logits, labels)
    """
    
    def __init__(
        self, 
        alpha: float = 0.25, 
        gamma: float = 2.0, 
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(
        self, 
        inputs: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Logits [batch] or [batch, 1]
            targets: Binary labels [batch]
            
        Returns:
            Focal loss value
        """
        # Flatten inputs if needed
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # Compute BCE loss (without reduction)
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        
        # Get probabilities
        pt = torch.exp(-bce_loss)
        
        # Apply alpha weighting
        # alpha for positive class, (1-alpha) for negative class
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Apply focal weighting
        focal_weight = alpha_t * (1 - pt) ** self.gamma
        
        # Compute final loss
        loss = focal_weight * bce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class LabelSmoothingBCE(nn.Module):
    """
    Binary Cross-Entropy with Label Smoothing.
    
    Softens the target labels to prevent overconfident predictions
    and improve generalization.
    
    Args:
        smoothing: Label smoothing factor (0.0 to 0.5)
        pos_weight: Weight for positive class
    
    Example:
        >>> criterion = LabelSmoothingBCE(smoothing=0.1)
        >>> loss = criterion(logits, labels)
    """
    
    def __init__(
        self, 
        smoothing: float = 0.1,
        pos_weight: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.smoothing = smoothing
        self.pos_weight = pos_weight
    
    def forward(
        self, 
        inputs: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute label-smoothed BCE loss.
        
        Args:
            inputs: Logits [batch]
            targets: Binary labels [batch]
            
        Returns:
            Smoothed BCE loss
        """
        # Apply label smoothing
        # 0 -> smoothing, 1 -> 1-smoothing
        targets_smooth = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        
        return F.binary_cross_entropy_with_logits(
            inputs, targets_smooth, pos_weight=self.pos_weight
        )


class CombinedLoss(nn.Module):
    """
    Combination of multiple loss functions.
    
    Useful for training with both focal loss and auxiliary objectives.
    
    Args:
        focal_weight: Weight for focal loss
        bce_weight: Weight for BCE loss
        alpha: Focal loss alpha
        gamma: Focal loss gamma
    """
    
    def __init__(
        self,
        focal_weight: float = 0.7,
        bce_weight: float = 0.3,
        alpha: float = 0.25,
        gamma: float = 2.0
    ):
        super().__init__()
        self.focal_weight = focal_weight
        self.bce_weight = bce_weight
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(
        self, 
        inputs: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute combined loss."""
        focal = self.focal_loss(inputs, targets)
        bce = self.bce_loss(inputs, targets)
        return self.focal_weight * focal + self.bce_weight * bce


def get_loss_function(
    loss_type: str = 'focal',
    alpha: Optional[float] = None,
    gamma: float = 2.0,
    pos_weight: Optional[float] = None,
    smoothing: float = 0.0,
    device: str = 'cuda'
) -> nn.Module:
    """
    Factory function to create loss function.
    
    Args:
        loss_type: 'focal', 'bce', 'weighted_bce', 'smoothed_bce', 'combined'
        alpha: Focal loss alpha (if None, defaults to 0.5)
        gamma: Focal loss gamma
        pos_weight: Positive class weight for BCE
        smoothing: Label smoothing factor
        device: Device for tensors
        
    Returns:
        Loss function module
    """
    if loss_type == 'focal':
        alpha = alpha if alpha is not None else 0.5
        return FocalLoss(alpha=alpha, gamma=gamma)
    
    elif loss_type == 'bce':
        return nn.BCEWithLogitsLoss()
    
    elif loss_type == 'weighted_bce':
        weight = torch.tensor([pos_weight]).to(device) if pos_weight else None
        return nn.BCEWithLogitsLoss(pos_weight=weight)
    
    elif loss_type == 'smoothed_bce':
        weight = torch.tensor([pos_weight]).to(device) if pos_weight else None
        return LabelSmoothingBCE(smoothing=smoothing, pos_weight=weight)
    
    elif loss_type == 'combined':
        alpha = alpha if alpha is not None else 0.5
        return CombinedLoss(alpha=alpha, gamma=gamma)
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


if __name__ == "__main__":
    # Test loss functions
    batch_size = 8
    logits = torch.randn(batch_size)
    targets = torch.randint(0, 2, (batch_size,)).float()
    
    print("Loss Function Tests")
    print("=" * 50)
    
    # Test Focal Loss
    focal = FocalLoss(alpha=0.6412, gamma=2.0)
    focal_loss = focal(logits, targets)
    print(f"Focal Loss: {focal_loss.item():.4f}")
    
    # Test BCE Loss
    bce = nn.BCEWithLogitsLoss()
    bce_loss = bce(logits, targets)
    print(f"BCE Loss: {bce_loss.item():.4f}")
    
    # Test Label Smoothing BCE
    smooth_bce = LabelSmoothingBCE(smoothing=0.1)
    smooth_loss = smooth_bce(logits, targets)
    print(f"Smoothed BCE Loss: {smooth_loss.item():.4f}")
    
    # Test Combined Loss
    combined = CombinedLoss()
    combined_loss = combined(logits, targets)
    print(f"Combined Loss: {combined_loss.item():.4f}")

import torch
import torch.nn as nn
import torch.nn.functional as F


class OrdinalRegressionLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        learn_thresholds: bool=True,
        init_scale: float=2.0
    ) -> None:
        """
        Initialize the Ordinal Regression Loss.

        Args:
            num_classes (int): Number of ordinal classes (ranks)
            learn_thresholds (:obj:`bool`, optional): Whether to learn threshold parameters or use fixed ones, defaults to `True`
            init_scale (:obj:`float`, optional): Scale for initializing thresholds, defaults to `2.0`
        """
        super().__init__()

        num_thresholds = num_classes - 1

        # Initialize thresholds
        if learn_thresholds:
            # Learnable thresholds: initialize with uniform spacing
            self.thresholds = nn.Parameter(
                torch.linspace(- init_scale, init_scale, num_thresholds),
                requires_grad=True
            )
        else:
            # Fixed thresholds with uniform spacing
            self.register_buffer(
                'thresholds',
                torch.linspace(- init_scale, init_scale, num_thresholds)
            )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the ordinal regression loss.

        Args:
            logits (torch.Tensor): Raw predictions (batch_size, 1)
            targets (torch.Tensor): Target classes (batch_size,) with values in [0, num_classes - 1]

        Returns:
            torch.Tensor: Loss value (batch_size,)
        """
        # Compute binary decisions for each threshold
        differences = logits - self.thresholds.unsqueeze(0)
        # (batch_size, num_thresholds)

        # Convert target classes to binary labels
        target_labels = torch.arange(len(self.thresholds)).expand(
            targets.size(0), -1
        ).to(targets.device) # (batch_size, num_thresholds)

        binary_targets = (target_labels < targets.unsqueeze(1)).float()
        # (batch_size, num_thresholds)

        # Compute binary cross entropy loss for each threshold
        losses = F.binary_cross_entropy_with_logits(
            differences,
            binary_targets,
            reduction='none'
        )
        # (batch_size, num_thresholds)

        return losses.mean(dim=1) # (batch_size,)

    def predict_probas(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Convert logits to class probabilities.

        Args:
            logits (torch.Tensor): Raw predictions (batch_size, 1)

        Returns:
            torch.Tensor: Class probabilities (batch_size, num_classes)
        """
        differences = logits - self.thresholds.unsqueeze(0)

        # Compute cumulative probabilities using sigmoid
        cumulative_probas = torch.sigmoid(differences)
        # (batch_size, num_thresholds)

        # Add boundary probabilities (0 and 1)
        zeros = torch.zeros_like(cumulative_probas[:, :1]) # (batch_size, 1)

        ones = torch.ones_like(zeros) # (batch_size, 1)

        cumulative_probas = torch.cat([zeros, cumulative_probas, ones], dim=-1)
        # (batch_size, num_classes + 1)

        # Convert cumulative probabilities to class probabilities
        class_probas = cumulative_probas[:, 1:] - cumulative_probas[:, :-1]
        # (batch_size, num_classes)

        return class_probas

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

        Usage::

            criterion = OrdinalRegressionLoss(4)

            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()

            probas = criterion.predict_probas(logits)
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
            reduction='mean'
        )

        return losses # torch.Size([])

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


def trend_segment_weights(labels: torch.Tensor) -> torch.Tensor:
    """
    Compute the weight at each position from the label sequence.
    The weight equals the remaining length until the end of the current trend segment.

    Args:
        labels (torch.Tensor): a 1D tensor of shape (batch_size,), where elements 0 or 1 denote SELL or BUY.

    Returns:
        torch.Tensor: a tensor with the same shape as labels, where each element is the weight at the corresponding position.
    """

    seq_len = len(labels)
    weights = torch.ones(seq_len, dtype=torch.float32)
    # Traverse from back to front to determine the distance
    #   to the next reversal
    next_change = [0] * seq_len

    # The last point’s weight is set to 0 first (will add 1 later)
    next_change[-1] = 0

    # We use the next_change array to store:
    # "the length of the same trend continuing forward from the current
    #   position minus 1"
    for i in range(seq_len-2, -1, -1):
        if labels[i] == labels[i+1]:
            # If the next label is the same, the same trend continues
            next_change[i] = next_change[i+1] + 1
        else:
            # If the next label differs, the next point is a trend reversal
            next_change[i] = 0

    # next_change[i] stores the length from i to the end of this segment
    #   (excluding i itself),
    # so the actual weight is that value + 1
    weights = torch.tensor([nc + 1 for nc in next_change], dtype=torch.float32)
    return weights


class TrendAwareLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            inputs (torch.Tensor): model output logits, shape (batch_size, 2) or (1, batch_size, 2) for convenience
            labels (torch.Tensor): ground-truth labels, shape (batch_size,)
        """
        # Compute per-step cross-entropy loss
        # per-time-step loss vector
        ce_loss = F.cross_entropy(inputs, labels, reduction='none')

        # Compute weights
        weights = trend_segment_weights(labels)

        # Weighted loss
        weighted_loss = ce_loss * weights.to(inputs.device)

        return weighted_loss.mean()

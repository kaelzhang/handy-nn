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


def _to_index_labels(labels: torch.Tensor) -> torch.LongTensor:
    """
    Convert labels to index form.

    Args:
        labels (torch.Tensor): shape either of
        - (batch_size,) with integer class indices
        - (batch_size, num_classes) one-hot

    Returns:
        torch.LongTensor: shape (batch_size,), LongTensor of class indices
    """

    if labels.dim() == 1:
        return labels.long()
    elif labels.dim() == 2:
        return labels.argmax(dim=-1).long()
    else:
        raise ValueError(
            "labels must be (batch_size,) indices or (batch_size, num_classes) one-hot"
        )

def _segment_remaining_weights(idx_labels: torch.LongTensor) -> torch.Tensor:
    """
    Core weighting algorithm: for each position, compute the remaining length
    until the end of the current label segment (inclusive). Longer remaining
    runs inside a segment receive larger weights.

    Args:
        idx_labels (torch.LongTensor): shape (batch_size,)

    Returns:
        torch.Tensor: shape (batch_size,), FloatTensor of positive weights
    """

    if idx_labels.dim() != 1:
        raise ValueError("idx_labels must be 1-D with shape (batch_size,)")

    batch_size = idx_labels.shape[0]
    device = idx_labels.device
    weights = torch.ones(batch_size, dtype=torch.float32, device=device)

    # Right-to-left scan to measure remaining run-length within the segment.
    run_length = 0
    weights[batch_size - 1] = 1.0
    for i in range(batch_size - 2, -1, -1):
        if idx_labels[i] == idx_labels[i + 1]:
            run_length += 1
        else:
            run_length = 0
        weights[i] = float(run_length + 1)

    return weights


class TrendAwareLoss(nn.Module):
    """
    Trend-aware cross-entropy loss.

    Idea:
    - Penalize "too-early / too-late" misclassification inside a label segment more heavily by multiplying per-sample cross-entropy with the segment remaining-length weight.
    - Normalize by the sum of weights to keep a stable loss scale.

    Args:
        logits (torch.Tensor): shape (batch_size, num_classes)
        labels (torch.Tensor): shape (batch_size,) or (batch_size, num_classes) one-hot

    Note:
        The batch dimension is interpreted as an ordered sequence for the
        purpose of computing segment weights.
    """

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        if reduction not in ('mean', 'sum'):
            raise ValueError("reduction must be 'mean' or 'sum'")

        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        # Shape checks for logits
        if logits.dim() != 2:
            raise ValueError("logits must be 2-D with shape (batch_size, num_classes)")
        batch_size, num_classes = logits.shape

        # Convert labels to indices
        idx_labels = _to_index_labels(labels)
        if idx_labels.shape != (batch_size,):
            raise ValueError(
                f"labels shape {tuple(idx_labels.shape)} is not compatible with "
                f"logits {(batch_size, num_classes)}"
            )

        # Per-sample cross-entropy: (batch_size,)
        ce_per_sample = F.cross_entropy(
            logits, idx_labels, reduction='none'
        )

        # Segment weights along the batch (treated as ordered time)
        weights = _segment_remaining_weights(idx_labels).to(
            ce_per_sample.dtype
        )

        # Weighted aggregation
        weighted = ce_per_sample * weights
        if self.reduction == 'mean':
            loss = weighted.sum() / (weights.sum() + 1e-12)
        else:
            # 'sum'
            loss = weighted.sum()

        return loss

import torch
import torch.nn as nn
import torch.nn.functional as F


class OrdinalRegressionLoss(nn.Module):
    """
    Implements the immediate threshold loss with logistic link function as described in
    "On the Consistency of Ordinal Regression Methods" by Pedregosa et al.

    The loss uses the immediate threshold formulation which is proven to be consistent
    for the absolute error.

    Args:
        num_classes (int): Number of ordinal classes to partition the odds into (e.g., 5 for ratings 1-5)
        threshold_init (str): How to initialize thresholds ('uniform' or 'ordered')
    """

    def __init__(
        self,
        num_classes: int,
        threshold_init='uniform'
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        num_thresholds = num_classes - 1

        # Initialize thresholds θ_1 < θ_2 < ... < θ_{k-1}
        if threshold_init == 'uniform':
            thresholds = torch.linspace(0, 1, num_thresholds)
        else:  # 'ordered'
            thresholds = torch.randn(num_thresholds)
            thresholds = torch.sort(thresholds)[0]

        # Make thresholds learnable parameters
        self.thresholds = nn.Parameter(
            torch.log(thresholds / (1 - thresholds))
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the immediate threshold loss.

        Args:
            logits (torch.Tensor): Linear logits from model, shape (batch_size, 1)
            targets (torch.Tensor): Target classes, shape (batch_size, ), values in [0, num_classes - 1]

        Returns:
            torch.Tensor: Mean loss value
        """

        # Get sorted thresholds using sigmoid to ensure ordering
        thresholds = torch.sigmoid(self.thresholds)

        # Expand logits for comparison with all thresholds
        pred_expanded = logits.unsqueeze(2).expand(-1, -1, self.num_classes - 1)

        # Compute all binary decisions P(y > k) using logistic function
        probas = torch.sigmoid(pred_expanded - thresholds)

        # Convert ordinal targets to binary decisions
        targets_expanded = targets.unsqueeze(1).unsqueeze(2).expand(-1, logits.size(1), self.num_classes - 1)
        target_indices = torch.arange(self.num_classes - 1, device=logits.device).view(1, 1, -1)
        binary_targets = (targets_expanded > target_indices).float()

        # Compute binary cross entropy loss for each threshold
        losses = F.binary_cross_entropy(
            probas, binary_targets, reduction='none')

        return losses.mean()

    def logits_to_probas(
        self,
        logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute class probabilities from linear predictions.

        Returns:
            torch.Tensor: Class probabilities, shape (batch_size, num_features, num_classes)
        """
        thresholds = torch.sigmoid(self.thresholds)

        # Expand logits for comparison with all thresholds
        pred_expanded = logits.unsqueeze(2).expand(-1, -1, self.num_classes - 1)

        # Compute P(y > k) for all thresholds
        probas = torch.sigmoid(pred_expanded - thresholds)

        # Convert to class probabilities P(y = k)
        probs = torch.zeros(
            (logits.size(0), logits.size(1), self.num_classes),
            device=logits.device
        )

        # P(y = 0) = 1 - P(y > 0)
        probs[:, :, 0] = 1 - probas[:, :, 0]

        # P(y = k) = P(y > k-1) - P(y > k)
        probs[:, :, 1:-1] = probas[:, :, :-1] - probas[:, :, 1:]

        # P(y = K) = P(y > K-1)
        probs[:, :, -1] = probas[:, :, -1]

        return probs

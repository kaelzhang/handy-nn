from typing import (
    Tuple
)

import torch
import torch.nn as nn


EPSILON = 1e-15

# Ref:
# On the Consistency of Ordinal Regression Methods (Pedregosa et. al., 2017)
# https://arxiv.org/pdf/1408.2327
class OrdinalRegression(nn.Module):
    """
    Ordinal Regression layer that models the inherent order in BUY/HOLD/SELL signals
    """
    def __init__(
        self,
        num_classes: int,
        hidden_size: int = 768,
        scale: float = 20.0,
        train_thresholds: bool = False
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        num_thresholds = num_classes - 1

        self.classifiers = nn.ModuleList([
            nn.Linear(hidden_size, 1) for _ in range(num_thresholds)
        ])

        thresholds = torch.arange(num_thresholds).float() * scale
        thresholds = thresholds / (num_classes - 2) - scale / 2

        self.thresholds = nn.Parameter(thresholds)

        if not train_thresholds:
            self.thresholds.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Returns:
            Tuple[Tensor, Tensor]: (logits, probas)
            - logits: raw, unnormalized output
            - probas: probabilities transformed from logits
        """

        # Get the output of each classifier
        logits = torch.cat([classifier(x) for classifier in self.classifiers], dim=1)

        # Convert logits to probabilities
        probas = self._logits_to_probas(logits)
        return logits, probas

    def _logits_to_probas(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Converts the output of the binary classifiers to class probabilities

        For example::
        - p(class=0) = 1 - p(logit1)
        - p(class=1) = p(logit1) - p(logit2)
        - p(class=2) = p(logit2)
        """

        sigmoids = torch.sigmoid(self.cutpoints - logits)
        link_mat = sigmoids[:, 1:] - sigmoids[:, :-1]
        link_mat = torch.cat((
                sigmoids[:, [0]],
                link_mat,
                (1 - sigmoids[:, [-1]])
            ),
            dim=1
        )

        probas = torch.clamp(link_mat, EPSILON, 1 - EPSILON)

        return probas

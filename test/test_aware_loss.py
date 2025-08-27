import torch

from handy_nn import TrendAwareLoss

from torch.nn import CrossEntropyLoss


def test_aware_loss():
    # create a dummy 4-item logits tensor
    logits = torch.tensor([
        [0.1, 0.9],
        [0.4, 0.8],
        [0.7, 0.3]
    ])

    targets = torch.tensor([
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 0.0]
    ])

    aware_criterion = TrendAwareLoss()
    ce_criterion = CrossEntropyLoss()

    aware_loss = aware_criterion(logits, targets)
    ce_loss = ce_criterion(logits, targets)

    assert aware_loss > ce_loss

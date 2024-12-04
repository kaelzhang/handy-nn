import torch

from handy_nn import OrdinalRegressionLoss

def test_logits_to_probas():
    model = OrdinalRegressionLoss(4)

    assert model.thresholds.shape == (3,)

    # create a dummy 4-item logits tensor
    logits = torch.tensor([
        [0.1],
        [0.4],
    ])

    loss = model(logits, torch.tensor([1, 2]))

    assert loss.shape == (2,)

    probas = model.predict_probas(logits)

    assert probas.shape == (2, 4)




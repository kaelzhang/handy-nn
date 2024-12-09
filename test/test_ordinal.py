import torch

from handy_nn import OrdinalRegressionLoss

def do_test_for_model(model):
    assert model.thresholds.shape == (3,)

    # create a dummy 4-item logits tensor
    logits = torch.tensor([
        [0.1],
        [0.4],
    ])

    loss = model(logits, torch.tensor([1, 2]))

    assert loss.shape == torch.Size([])

    probas = model.predict_probas(logits)

    assert probas.shape == (2, 4)


def test_ordinal_regression_loss_train_threshold():
    model = OrdinalRegressionLoss(4)

    do_test_for_model(model)


def test_ordinal_regression_loss_fixed_threshold():
    model = OrdinalRegressionLoss(4, learn_thresholds=False)

    do_test_for_model(model)

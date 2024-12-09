[![](https://github.com/kaelzhang/handy-nn/actions/workflows/python.yml/badge.svg)](https://github.com/kaelzhang/handy-nn/actions/workflows/python.yml)
[![](https://codecov.io/gh/kaelzhang/handy-nn/branch/main/graph/badge.svg)](https://codecov.io/gh/kaelzhang/handy-nn)
[![](https://img.shields.io/pypi/v/handy-nn.svg)](https://pypi.org/project/handy-nn/)
[![](https://img.shields.io/pypi/l/handy-nn.svg)](https://github.com/kaelzhang/handy-nn)

# handy-nn

Delightful and useful neural networks models, including OrdinalRegressionLoss, etc.

## Install

```sh
$ pip install handy-nn
```

## Usage

```py
from handy_nn import OrdinalRegressionLoss

# Initialize the loss function
num_classes = 5
criterion = OrdinalRegressionLoss(num_classes)

# For training
logits = model(inputs)  # Shape: (batch_size, 1)
loss = criterion(logits, targets)
loss.backward()  # shape: torch.Size([])

# To get class probabilities
probas = criterion.predict_probas(logits)  # Shape: (batch_size, num_classes)
```

# APIs

## OrdinalRegressionLoss(num_classes, learn_thresholds, init_scale)

- **num_classes** `int`: Number of ordinal classes (ranks)
- **learn_thresholds** `bool=True`: Whether to learn threshold parameters or use fixed ones, defaults to `True`.
- **init_scale** `float=2.0`: Scale for initializing thresholds, defaults to `2.0`

Creates the loss function for ordinal regression.

The goal of [ordinal regression](https://en.wikipedia.org/wiki/Ordinal_regression) is to model the relationship between one or more independent variables and an ordinal dependent variable. It predicts the probability that an observation falls into a specific ordinal category or a category higher than a certain threshold. This is particularly useful in fields like social sciences, medicine, and customer surveys where outcomes are often ordinal.

## License

[MIT](LICENSE)

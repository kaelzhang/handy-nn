[![](https://github.com/kaelzhang/handy-nn/actions/workflows/python.yml/badge.svg)](https://github.com/kaelzhang/handy-nn/actions/workflows/python.yml)
[![](https://codecov.io/gh/kaelzhang/handy-nn/branch/master/graph/badge.svg)](https://codecov.io/gh/kaelzhang/handy-nn)
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
loss.backward()

# To get class probabilities
probas = criterion.predict_probas(logits)  # Shape: (batch_size, num_classes)
```

## License

[MIT](LICENSE)

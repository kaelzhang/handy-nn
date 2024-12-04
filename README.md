[![](https://github.com/kaelzhang/nn-models/actions/workflows/python.yml/badge.svg)](https://github.com/kaelzhang/nn-models/actions/workflows/python.yml)
[![](https://codecov.io/gh/kaelzhang/nn-models/branch/master/graph/badge.svg)](https://codecov.io/gh/kaelzhang/nn-models)
[![](https://img.shields.io/pypi/v/nn-models.svg)](https://pypi.org/project/nn-models/)
[![](https://img.shields.io/pypi/l/nn-models.svg)](https://github.com/kaelzhang/nn-models)

# nn-models

Delightful and useful neural networks models

## Install

```sh
$ pip install nn-models
```

## Usage

```py
from nn_models import OrdinalRegressionLoss

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

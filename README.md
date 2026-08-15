# dabbling with neural networks

with this (following Andrej Karpathy's "Neural Networks: Zero to Hero" series) I want to understand how neural networks
work from the ground up

I deliberately structured the repo and implementation so I can use it as a learning reference for myself later. this
includes my own notes woven throughout the notebooks, documenting the concepts as I worked through them

---

**progress**

so far I've built out the [autograd](src/autograd/engine.py) and [tanh-MLP](src/mlp/mlp_tanh.py) implementations shown
(via gradient descent), and used their concepts to build and train a model from scratch for
the [UCI ML Breast Cancer Wisconsin (diagnostic)](https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-wisconsin-diagnostic-dataset)
dataset ([trainer.py](src/mlp/trainer.py), [run.py](real_datasets/UCI_ML_Breast_Cancer/run.py))
---

- ["Neural Networks: Zero to Hero" series](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)
- [Andrej Karpathy's GitHub](https://github.com/karpathy)

---

```
# activate venv
source .venv/bin/activate
```

```
# install deps
python -m pip install -e .

# install dev deps
python -m pip install -e ".[dev]"
```

```
# run tests
pytest
```

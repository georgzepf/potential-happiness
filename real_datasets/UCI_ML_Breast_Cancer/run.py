from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from mlp import MLP
from mlp.loss import mse
from mlp.lr import linear_decay
from mlp.trainer import Trainer

# NOTE: see real_datasets/UCI_ML_Breast_Cancer/dataset_exploration.ipynb

dataset = load_breast_cancer()  # data: 569 samples, 30 features
x = dataset.data
y = dataset.target

# saving 20% of samples for testing the model
x_train_raw, x_test_raw, y_train_raw, y_test_raw = train_test_split(
    x, y, test_size=0.2, random_state=26
)

# preparing x

train_means = x_train_raw.mean(axis=0)
train_stds = x_train_raw.std(axis=0)

x_train_normalized = (x_train_raw - train_means) / train_stds
# uses means/stds form x_train_raw, not x_test_raw (in order not to leak test data to the model)
x_test_normalized = (x_test_raw - train_means) / train_stds

x_train = [list(map(float, sample)) for sample in x_train_normalized]
x_test = [list(map(float, sample)) for sample in x_test_normalized]

# preparing y

# target labels are 0/1, our model (using tanh) gives us -1/1
y_train = [1.0 if label == 1 else -1.0 for label in y_train_raw]
y_test = [1.0 if label == 1 else -1.0 for label in y_test_raw]

# training

model = MLP(30, [15, 15, 1])
trainer = Trainer(model, mse, x_train, y_train, "uci_ml_breast_cancer")

(trainer
 .load_parameters()
 .optimize(max_steps=500, target_loss=0.1, lr=linear_decay(start=0.05, end=0.005))
 .persist_parameters()
 .evaluate(x_test, y_test)
 )

import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.nn.utils.rnn import pad_sequence

from nucleus.models import NeuralNetwork
from nucleus.data_augment import (
    modify_approximated_landau,
    gauss_mod_left,
    emg_distribution,
    pearson_type_iv,
    electron_energy_spectrum,
)

# Load dữ liệu từ PDD.csv
df = pd.read_csv("PDD.csv")
PDD = df.iloc[:, 0].to_numpy()  # 48 giá trị độ sâu
matrix_R = df.iloc[:, 1:].to_numpy()  # Ma trận R (48 x 201)

nn_model = NeuralNetwork(
    in_features=128,
    out_features=4,
    mid_features=128,
    n_layers=5,
    device="cuda" if torch.cuda.is_availabel() else "cpu",
)

# Khởi tạo optimizer
learning_rate = 0.00001
optimizer = Adam(nn_model.parameters(), lr=learning_rate)

# Khởi tạo hàm loss
loss_fn = torch.nn.MSELoss()


def train(model, epoch): ...

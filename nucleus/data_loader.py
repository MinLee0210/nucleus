import pandas as pd

from torch.utils.data import Dataset

from .data_augment import generate_training_data


class ElectronSpectrum(Dataset):
    def __init__(
        self,
        data_dir,
        num_samples: int,
        E_values: float,
        parameters: dict,
        mode: str,
        noise_stddev: float = 0.03,
    ):
        super().__init__()

        try:
            if data_dir.endswith(".csv"):
                df = pd.read_csv(data_dir, index_col=False)
        except Exception as e:
            raise ValueError("data_dir is not in right format") from e

        self.PDD = df.iloc[:, 0].to_numpy()
        self.matrix_R = df.iloc[:, 1:].to_numpy()

        self.X_train, self.Y_train = generate_training_data(
            num_samples=num_samples,
            matrix_R=self.matrix_R,
            E_values=E_values,
            parameters=parameters,
            noise_stddev=noise_stddev,
        )

    def __len__(self):
        return len(self.X_train)

    def __getitem__(self, idx):
        return self.X_train[idx], self.Y_train[idx]

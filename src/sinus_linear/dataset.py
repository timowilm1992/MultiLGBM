import pandas as pd
import numpy as np
from lightgbm import Dataset

def read_data(dataset_path):
    train_df = pd.read_csv(f"{dataset_path}/train.csv", index_col=0)
    test_df = pd.read_csv(f"{dataset_path}/test.csv", index_col=0)
    return train_df, test_df


def build_train_dataset_and_test_df(dataset_path, dirichlet_param = [0.5, 0.5], repeat = 10):
    train_df, test_df = read_data(dataset_path=dataset_path)
    train_df = pd.concat([train_df] * repeat)
    dirichlet_samples = np.random.dirichlet(dirichlet_param, len(train_df))
    beta1 = dirichlet_samples[:,0] 
    beta2 = dirichlet_samples[:,1]

    train_df["beta1"] = beta1
    train_df["beta2"] = beta2

    train_df.drop(["sin", "linear"], axis=1, inplace=True)

    input_data = train_df.drop(["t_sin", "t_linear"], axis=1)

    label_data=train_df.drop("x", axis=1)

    train_data = Dataset(input_data, label_data["t_sin"].to_numpy())

    train_data.t_linear = label_data["t_linear"].to_numpy()
    train_data.beta1 = label_data["beta1"].to_numpy()
    train_data.beta2 = label_data["beta2"].to_numpy()

    return train_data, test_df
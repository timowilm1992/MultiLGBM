import lightgbm as lgb
from functools import partial
import argparse
from pathlib import Path
from src.sinus_linear.dataset import build_train_dataset_and_test_df
import json
from src.sinus_linear.loss import sinus_linear_loss_mse
from src.sinus_linear.visualize import visualize



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', default="sinus_linear")
    parser.add_argument('--config-name', default="sinus_linear/default_config")
    parser.add_argument('--plot', default=False, help='plot data')
    args = parser.parse_args()

    dataset_path = f"{Path(__file__).parent.parent}/datasets/{args.dataset_name}"

    config_path = f"{Path(__file__).parent.parent}/configs/{args.config_name}.json"

    with open(config_path, "r") as config_file:
        config = json.load(config_file)

    train_dataset, test_df = build_train_dataset_and_test_df(dataset_path=dataset_path, dirichlet_param=config["dirichlet_param"], repeat=config["repeat"])

    lgbm_params = config["lgbm_params"]

    if lgbm_params["objective"] == "sinus_linear_loss_mse":
        lgbm_params["objective"] = partial(sinus_linear_loss_mse, config["regularization"])
    else:
        raise AssertionError(lgbm_params["objective"] + " objective is not supported")

    model = lgb.train(lgbm_params, train_dataset, feval=partial(sinus_linear_loss_mse, config["regularization"]))

    visualize(model=model, test_df=test_df, num_images=200, plot=args.plot)

  


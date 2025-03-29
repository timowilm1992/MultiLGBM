import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os, argparse


data_directory = f"{Path(__file__).parent.parent.parent}/datasets/sinus_linear"


if not os.path.isdir(data_directory):
    os.makedirs(data_directory, exist_ok=True)

def sample_data(n):
    x = np.sort(np.random.uniform(0,7,n))

    y1= np.sin(x)
    y2= 1/7 * x 

    y1_noise = y1 + np.random.normal(0,.15,n)
    y2_noise = y2 +  np.random.normal(0,.15,n)
    return x, y1, y2, y1_noise, y2_noise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--number-train-examples', default=50000)
    parser.add_argument('--number-test-examples', default=1000)
    parser.add_argument('--plot', default=False, help='plot data')
    args = parser.parse_args()

    x, sin, linear, t_sin, t_linear = sample_data(int(args.number_train_examples))
    x_test, sin_test, linear_test, t_sin_test, t_linear_test = sample_data(int(args.number_test_examples))


    if args.plot:
        plt.scatter(x,t_sin, c="lightskyblue")
        plt.scatter(x,t_linear,c="lightcoral")

        plt.plot(x,sin, c="blue", label="sin_x")
        plt.plot(x,linear, c="red", label="linear_x")

        plt.title("Can LGBM learn which task it has to solve?")

        plt.xlabel("x")
        plt.ylabel("x/7 , sin(x)")

        plt.legend()

        plt.show()

    data = np.column_stack([x,sin,linear,t_sin,t_linear])
    test_data = np.column_stack([x_test,sin_test,linear_test,t_sin_test,t_linear_test])

    train_df = pd.DataFrame({"x": x, "sin": sin, "t_sin": t_sin, "linear": linear, "t_linear": t_linear})
    test_df = pd.DataFrame({"x": x_test, "sin": sin_test, "t_sin": t_sin_test, "linear": linear_test, "t_linear": t_linear_test})

    train_df.to_csv(f"{data_directory}/train.csv")
    test_df.to_csv(f"{data_directory}/test.csv")
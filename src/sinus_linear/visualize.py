import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import numpy as np
from functools import partial


def animate(axs, test_df, t_sin, t_linear, sin, linear, sin_losses, linear_losses, num_images, model, i):
    axs[0].clear()
    axs[1].clear()
    axs[0].scatter(test_df["x"],t_sin, c="lightskyblue")
    axs[0].scatter(test_df["x"],t_linear,c="lightcoral")

    axs[0].plot(test_df["x"],sin, c="blue", label="$sin(x)$")
    axs[0].plot(test_df["x"],linear, c="red", label="$x/7$")

    axs[0].title.set_text("Trade-Off Prediction")
    axs[0].set_xlabel("$x$")
    axs[0].set_ylabel("$x/7$ , $sin(x)$")

    axs[0].legend()

    axs[1].plot(sin_losses,linear_losses, c="lightskyblue")
    

    axs[1].title.set_text("Pareto Front")

    axs[1].set_xlabel("MSE($\hat{y}_{pred}, y_{sin}$)")
    axs[1].set_ylabel("MSE($\hat{y}_{pred}, y_{x/7}$)")
    test_df["beta1"] = 1000*[i/num_images]
    test_df["beta2"] = 1000*[1-i/num_images]
    y_pred = model.predict(test_df)
    x = test_df["x"]
    plot1 = axs[0].plot(x, y_pred, animated=True, color="black", label="$\hat{y}_{pred}$")
    plot2 = [axs[1].scatter([sin_losses[i]], [linear_losses[i]], c= "black", animated= True, zorder=2)]

    axs[0].legend(handles=[mpatches.Patch(color="blue", label="$sin(x)$"), mpatches.Patch(color="red", label="$x/7$"), mpatches.Patch(color="black", label="$\hat{y}_{pred}$")], loc="lower left")
    axs[1].legend(handles=[mpatches.Patch(color="black", label="MSE Trade-Off")],loc="lower left")

    return plot1 + plot2



def visualize(model, test_df, num_images=200, plot=True):
    t_sin = test_df["t_sin"].to_numpy()
    t_linear = test_df["t_linear"].to_numpy()
    sin = test_df["sin"].to_numpy()
    linear = test_df["linear"].to_numpy()
    test_df = test_df.drop(["t_sin", "t_linear","sin", "linear"], axis=1)

    sin_losses = []
    linear_losses = []

    for i in range(num_images+1):
        test_df["beta1"] = len(test_df)*[i/num_images]
        test_df["beta2"] = len(test_df)*[1-i/num_images]
        y_pred = model.predict(test_df)
        loss_sin = np.mean((y_pred - t_sin)**2)
        loss_linear = np.mean((y_pred - t_linear)**2)
        sin_losses.append(loss_sin)
        linear_losses.append(loss_linear)

    fig, axs = plt.subplots(1,2, figsize=(12,6))
    fig.suptitle('Pareto Front Approximation with MultiLGBM')
    axs[0].scatter(test_df["x"],t_sin, c="lightskyblue")
    axs[0].scatter(test_df["x"],t_linear,c="lightcoral")

    axs[0].plot(test_df["x"],sin, c="blue", label="$sin(x)")
    axs[0].plot(test_df["x"],linear, c="red", label="$x/7$")

    axs[0].title.set_text("Trade-Off Prediction")
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("$x/7$ , $sin(x)$")

    axs[0].legend()
    axs[1].plot(sin_losses,linear_losses, c="lightskyblue")
    axs[1].title.set_text("Loss Trade-Off")
    axs[1].set_xlabel("MSE($\hat{y}_{pred}, y_{sin}$)")
    axs[1].set_ylabel("MSE($\hat{y}_{pred}, y_{x/7}$)")


    ani_ = partial(animate, axs, test_df, t_sin, t_linear, sin, linear, sin_losses, linear_losses, num_images, model)

    ani =animation.FuncAnimation(fig, ani_,
                               frames=num_images+1, interval=25, blit=True, repeat=False)

    ani.save('simple.gif', writer='imagemagick', fps=30)
    
    if plot:
        plt.show()
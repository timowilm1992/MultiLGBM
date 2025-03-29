from functools import partial
from autograd import elementwise_grad as egrad
import autograd.numpy as np



def regression_l2_loss(beta1, t_sin, beta2, t_linear, reg, y_pred):
    sin_loss = 0.5 * beta1 * (y_pred - t_sin)**2 
    linear_loss = 0.5 * beta2 * (y_pred - t_linear)**2
    regularization = sin_loss/(sin_loss + linear_loss) * (np.log(sin_loss/(sin_loss + linear_loss)) - np.log(1/2)) + linear_loss/(sin_loss + linear_loss) * (np.log(linear_loss/(sin_loss + linear_loss)) - np.log(1/2))
    return np.mean(sin_loss + linear_loss + reg * regularization)



def sinus_linear_loss_mse(reg, y_pred, data):
    t_sin = data.get_label()
    t_linear = data.t_linear
    beta1 = data.beta1
    beta2 = data.beta2
    fn = partial(regression_l2_loss, beta1, t_sin, beta2, t_linear, reg)
    grad = egrad(fn)(y_pred)
    hess = egrad(egrad(fn))(y_pred)
    return grad, hess
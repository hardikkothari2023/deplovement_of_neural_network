import numpy as np
import pandas as pd

from src.config import config
import src.preprocessing.preprocessing as pp
from src.preprocessing.data_management import load_dataset, save_model, load_model
import pipeline as pl

z = [None] * len(config.layer_sizes)
h = [None] * len(config.layer_sizes)

del_fl_by_del_z = [None] * len(config.layer_sizes)
del_hl_by_del_theta0 = [None] * len(config.layer_sizes)
del_hl_by_del_theta = [None] * len(config.layer_sizes)
del_L_by_del_h = [None] * len(config.layer_sizes)
del_L_by_del_theta0 = [None] * len(config.layer_sizes)
del_L_by_del_theta = [None] * len(config.layer_sizes)

def layer_neurons_weighted_sum(previous_layer_neurons_outputs, current_layer_neurons_biases, current_layer_neurons_weights):
    return current_layer_neurons_biases + np.matmul(previous_layer_neurons_outputs, current_layer_neurons_weights)

def layer_neurons_output(current_layer_neurons_weighted_sums, current_layer_neurons_activation_function):
    if current_layer_neurons_activation_function == "linear":
        return current_layer_neurons_weighted_sums
    elif current_layer_neurons_activation_function == "sigmoid":
        return 1 / (1 + np.exp(-current_layer_neurons_weighted_sums))
    elif current_layer_neurons_activation_function == "tanh":
        return (np.exp(current_layer_neurons_weighted_sums) - np.exp(-current_layer_neurons_weighted_sums)) / \
               (np.exp(current_layer_neurons_weighted_sums) + np.exp(-current_layer_neurons_weighted_sums))
    elif current_layer_neurons_activation_function == "relu":
        return current_layer_neurons_weighted_sums * (current_layer_neurons_weighted_sums > 0)

def del_layer_neurons_outputs_wrt_weighted_sums(current_layer_neurons_activation_function, current_layer_neurons_weighted_sums):
    if current_layer_neurons_activation_function == "linear":
        return np.ones_like(current_layer_neurons_weighted_sums)
    elif current_layer_neurons_activation_function == "sigmoid":
        current_layer_neurons_outputs = 1 / (1 + np.exp(-current_layer_neurons_weighted_sums))
        return current_layer_neurons_outputs * (1 - current_layer_neurons_outputs)
    elif current_layer_neurons_activation_function == "tanh":
        return (2 / (np.exp(current_layer_neurons_weighted_sums) + np.exp(-current_layer_neurons_weighted_sums))) ** 2
    elif current_layer_neurons_activation_function == "relu":
        return (current_layer_neurons_weighted_sums > 0)

def del_layer_neurons_outputs_wrt_biases(current_layer_neurons_outputs_dels):
    return current_layer_neurons_outputs_dels

def del_layer_neurons_outputs_wrt_weights(previous_layer_neurons_outputs, current_layer_neurons_outputs_dels):
    return np.matmul(previous_layer_neurons_outputs.T, current_layer_neurons_outputs_dels)

def run_training(tol, epsilon, max_epochs):
    epoch_counter = 0
    mse = 1
    loss_per_epoch = list()
    loss_per_epoch.append(mse)

    training_data = load_dataset("train.csv")

    obj = pp.preprocess_data()
    obj.fit(training_data.iloc[:, 0:2], training_data.iloc[:, 2])
    X_train, Y_train = obj.transform(training_data.iloc[:, 0:2], training_data.iloc[:, 2])

    pl.initialize_parameters()

    batch_size = 2
    num_batches = X_train.shape[0] // batch_size

    while True:
        mse = 0

        for b in range(num_batches):
            start_idx = b * batch_size
            end_idx = start_idx + batch_size

            X_batch = X_train[start_idx:end_idx]
            Y_batch = Y_train[start_idx:end_idx].reshape(batch_size, -1)

            for i in range(batch_size):
                h[0] = X_batch[i].reshape(1, X_train.shape[1])

                # Forward pass
                for l in range(1, len(config.layer_sizes)):
                    z[l] = layer_neurons_weighted_sum(h[l - 1], pl.theta0[l], pl.theta[l])
                    h[l] = layer_neurons_output(z[l], config.f[l])
                    del_fl_by_del_z[l] = del_layer_neurons_outputs_wrt_weighted_sums(config.f[l], z[l])
                    del_hl_by_del_theta0[l] = del_layer_neurons_outputs_wrt_biases(del_fl_by_del_z[l])
                    del_hl_by_del_theta[l] = del_layer_neurons_outputs_wrt_weights(h[l - 1], del_fl_by_del_z[l])

                # Compute loss
                L = (1 / 2) * (Y_batch[i][0] - h[len(config.layer_sizes) - 1][0, 0]) ** 2
                mse += L

                # Backpropagation
                del_L_by_del_h[len(config.layer_sizes) - 1] = (h[len(config.layer_sizes) - 1] - Y_batch[i])
                for l in range(len(config.layer_sizes) - 2, 0, -1):
                    del_L_by_del_h[l] = np.matmul(del_L_by_del_h[l + 1],
                                                  (del_fl_by_del_z[l + 1] * pl.theta[l + 1]).T)

                # Gradient descent updates
                for l in range(1, len(config.layer_sizes)):
                    del_L_by_del_theta0[l] = del_L_by_del_h[l] * del_hl_by_del_theta0[l]
                    del_L_by_del_theta[l] = del_L_by_del_h[l] * del_hl_by_del_theta[l]

                    pl.theta0[l] = pl.theta0[l] - (epsilon * del_L_by_del_theta0[l])
                    pl.theta[l] = pl.theta[l] - (epsilon * del_L_by_del_theta[l])

        mse /= X_train.shape[0]
        epoch_counter += 1
        loss_per_epoch.append(mse)

        print("Epoch # {}, Loss = {}".format(epoch_counter, mse))

        if mse < tol or epoch_counter >= max_epochs:
            break

if __name__ == "__main__":
    run_training(1e-10, 0.009, 20000)
    save_model(pl.theta0, pl.theta)

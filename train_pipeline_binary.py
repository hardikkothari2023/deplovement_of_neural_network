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
cache_theta0 = [None] * len(config.layer_sizes)
cache_theta = [None] * len(config.layer_sizes)

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

def binary_cross_entropy_loss(y_true, y_pred):
    return - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def rmsprop_update(cache, gradient, decay_rate=0.9, learning_rate=0.01):
    epsilon = 1e-8
    if cache is None:
        cache = np.zeros_like(gradient)
    cache = decay_rate * cache + (1 - decay_rate) * gradient**2
    return learning_rate * gradient / (np.sqrt(cache) + epsilon), cache

def run_training(tol, initial_learning_rate, max_epochs, decay_rate=0.9, patience=5):
    epoch_counter = 0
    best_loss = float('inf')
    patience_counter = 0
    
    training_data = load_dataset("train.csv")
    obj = pp.preprocess_data()
    obj.fit(training_data.iloc[:, 0:2], training_data.iloc[:, 2])
    X_train, Y_train = obj.transform(training_data.iloc[:, 0:2], training_data.iloc[:, 2])
    
    pl.initialize_parameters()
    
    num_batches = X_train.shape[0] // batch_size
    
    while patience_counter < patience:
        total_loss = 0
        
        # Shuffle data for each epoch
        shuffle_index = np.random.permutation(X_train.shape[0])
        X_train_shuffled = X_train[shuffle_index]
        Y_train_shuffled = Y_train[shuffle_index]
        
        for b in range(num_batches):
            start_idx = b * batch_size
            end_idx = start_idx + batch_size
            
            X_batch = X_train_shuffled[start_idx:end_idx]
            Y_batch = Y_train_shuffled[start_idx:end_idx].reshape(batch_size, -1)
            
            batch_loss = 0
            
            for i in range(batch_size):
                h[0] = X_batch[i].reshape(1, X_train.shape[1])
                
                # Forward pass
                for l in range(1, len(config.layer_sizes)):
                    z[l] = layer_neurons_weighted_sum(h[l - 1], pl.theta0[l], pl.theta[l])
                    h[l] = layer_neurons_output(z[l], config.f[l])
                    del_fl_by_del_z[l] = del_layer_neurons_outputs_wrt_weighted_sums(config.f[l], z[l])
                    del_hl_by_del_theta0[l] = del_layer_neurons_outputs_wrt_biases(del_fl_by_del_z[l])
                    del_hl_by_del_theta[l] = del_layer_neurons_outputs_wrt_weights(h[l - 1], del_fl_by_del_z[l])
                
                # Compute loss (binary cross-entropy)
                y_pred = h[len(config.layer_sizes) - 1]
                L = binary_cross_entropy_loss(Y_batch[i], y_pred)
                batch_loss += np.mean(L)  # Average over the batch
                
                # Backpropagation
                del_L_by_del_h[len(config.layer_sizes) - 1] = (y_pred - Y_batch[i]) / (y_pred * (1 - y_pred))
                for l in range(len(config.layer_sizes) - 2, 0, -1):
                    del_L_by_del_h[l] = np.matmul(del_L_by_del_h[l + 1], (del_fl_by_del_z[l + 1] * pl.theta[l + 1]).T)
                
                # RMSprop update for biases
                for l in range(1, len(config.layer_sizes)):
                    del_L_by_del_theta0[l], cache_theta0[l] = rmsprop_update(cache_theta0[l], del_L_by_del_h[l] * del_hl_by_del_theta0[l], learning_rate=initial_learning_rate)
                    pl.theta0[l] = pl.theta0[l] - del_L_by_del_theta0[l]
                    
                    del_L_by_del_theta[l], cache_theta[l] = rmsprop_update(cache_theta[l], del_L_by_del_h[l] * del_hl_by_del_theta[l], learning_rate=initial_learning_rate)
                    pl.theta[l] = pl.theta[l] - del_L_by_del_theta[l]
            
            total_loss += batch_loss / batch_size  # Average over the batch
        
        mse = total_loss / num_batches  # Average over all batches
        epoch_counter += 1
        
        # Learning rate decay
        initial_learning_rate *= decay_rate
        
        print("Epoch # {}, Loss = {}".format(epoch_counter, mse))
        
        # Early stopping based on validation loss
        if mse < best_loss:
            best_loss = mse
            patience_counter = 0
            save_model(pl.theta0, pl.theta)  # Save the best model
        else:
            patience_counter += 1
        
        # Check termination conditions
        if patience_counter >= patience or epoch_counter >= max_epochs:
            break
    
if __name__ == "__main__":
    batch_size = 2
    run_training(1e-8 , 0.000001, 1000)

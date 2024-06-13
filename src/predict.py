import pandas as pd
import numpy as np
from src import trained_models
from src.config import config
import src.preprocessing.preprocessing as pp
from src.preprocessing.data_management import load_dataset, save_model, load_model
import pipeline as pl
import pickle
from src import trained_models
 

def layer_neurons_weighted_sum(previous_layer_neurons_outputs, current_layer_neurons_biases, current_layer_neurons_weights):
    return current_layer_neurons_biases + np.matmul(previous_layer_neurons_outputs, current_layer_neurons_weights)

def layer_neurons_output(current_layer_neurons_weighted_sums, current_layer_neurons_activation_function):
    if current_layer_neurons_activation_function == "linear":
        return current_layer_neurons_weighted_sums
    elif current_layer_neurons_activation_function == "sigmoid":
        return 1/(1 + np.exp(-current_layer_neurons_weighted_sums))
    elif current_layer_neurons_activation_function == "tanh":
        return (np.exp(current_layer_neurons_weighted_sums) - np.exp(-current_layer_neurons_weighted_sums)) / \
               (np.exp(current_layer_neurons_weighted_sums) + np.exp(-current_layer_neurons_weighted_sums))
    elif current_layer_neurons_activation_function == "relu":
        return current_layer_neurons_weighted_sums * (current_layer_neurons_weighted_sums > 0)

def predict(X, theta0, theta):
    h = [None]*config.NUM_LAYERS
    h[0] = X

    for l in range(1, config.NUM_LAYERS):
        z = layer_neurons_weighted_sum(h[l-1], theta0[l], theta[l])
        h[l] = layer_neurons_output(z, config.f[l])

    return h[config.NUM_LAYERS-1]

def evaluate_model(X, Y, theta0, theta):
    correct_predictions = 0
    total_predictions = X.shape[0]

    for i in range(total_predictions):
        X_sample = X[i].reshape(1, -1)
        prediction = predict(X_sample, theta0, theta)
        predicted_label = 1 if prediction >= 0.5 else 0
        if predicted_label == Y[i]:
            correct_predictions += 1

    accuracy = correct_predictions / total_predictions
    return accuracy

if __name__ == "__main__":
    # Load trained model parameters
    theta0, theta = load_model('two_input_xor_nn.pkl')

    # XOR test data
    X_test = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y_test = np.array([0, 1, 1, 0])

    # Evaluate the model
    accuracy = evaluate_model(X_test, Y_test, theta0, theta)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    # Predict individual samples and print the results
    for i in range(X_test.shape[0]):
        X_sample = X_test[i].reshape(1, -1)
        prediction = predict(X_sample, theta0, theta)
        predicted_label = 1 if prediction >= 0.5 else 0
        print(f"Input: {X_test[i]}, Predicted Output: {predicted_label}, True Output: {Y_test[i]}")

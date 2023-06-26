"""
This Python module implements a simple logistic regression model using a single neuron and mini-batch gradient descent.

The 'Neuron' class initializes with random weights and takes in a dataset for training. The neuron is trained to make predictions based on the input features. 
The model's weights are adjusted according to the computed gradient of the loss function.

The module also includes a 'test' function to verify the model's predictions on different datasets. 
This module serves as a basic demonstration of logistic regression and gradient descent in machine learning.
"""


import numpy as np


class Neuron:
    def __init__(self, dataset):
        np.random.seed(42)
        self.weights = np.random.normal(0, 1, 3 + 1)
        self.dataset = dataset
        self.perform_training()

    def perform_training(self, lr=0.01, mini_batch_size=10, n_epochs=200):
        for _ in range(n_epochs):
            for batch_start in range(0, len(self.dataset), mini_batch_size):
                batch = self.dataset[batch_start: batch_start +
                                     mini_batch_size]
                pred_and_label = [
                    {"pred": self.calculate_prediction(
                        data["features"]), "label": data["label"]}
                    for data in batch
                ]
                gradient = self._calculate_gradient(batch, pred_and_label)
                self.weights -= lr * np.array(gradient)

    def calculate_prediction(self, features):
        inputs = np.append(features, 1)
        linear_combination = np.dot(self.weights, inputs)
        return 1 / (1 + np.exp(-linear_combination))

    def _calculate_gradient(self, batch, pred_and_label):
        errors = [
            pl["pred"] - pl["label"] for pl in pred_and_label
        ]
        gradient = np.zeros(len(self.weights))
        for i, data in enumerate(batch):
            inputs = np.append(data["features"], 1)
            for j, input in enumerate(inputs):
                gradient[j] += errors[i] * input
        return gradient / len(batch)


def test():
    # Test Case 1: Simple case with single example
    dataset1 = [
        {"features": [0.1, 0.2, 0.3], "label": 1}
    ]
    neuron1 = Neuron(dataset1)
    print(round(neuron1.calculate_prediction([0.1, 0.2, 0.3]), 4))

    # Test Case 2: Case with multiple examples, but same label
    dataset2 = [
        {"features": [0.1, 0.2, 0.3], "label": 0},
        {"features": [0.4, 0.5, 0.6], "label": 0},
    ]
    neuron2 = Neuron(dataset2)
    print(round(neuron2.calculate_prediction([0.1, 0.2, 0.3]), 4))
    print(round(neuron2.calculate_prediction([0.4, 0.5, 0.6]), 4))

    # Test Case 3: Complex case with multiple examples and different labels
    dataset3 = [
        {"features": [0.1, 0.2, 0.3], "label": 0},
        {"features": [0.4, 0.5, 0.6], "label": 1},
        {"features": [0.7, 0.8, 0.9], "label": 0},
        {"features": [0.2, 0.3, 0.4], "label": 1},
        {"features": [0.5, 0.6, 0.7], "label": 0},
    ]
    neuron3 = Neuron(dataset3)
    print(round(neuron3.calculate_prediction([0.1, 0.2, 0.3]), 4))
    print(round(neuron3.calculate_prediction([0.4, 0.5, 0.6]), 4))
    print(round(neuron3.calculate_prediction([0.7, 0.8, 0.9]), 4))
    print(round(neuron3.calculate_prediction([0.2, 0.3, 0.4]), 4))
    print(round(neuron3.calculate_prediction([0.5, 0.6, 0.7]), 4))


test()

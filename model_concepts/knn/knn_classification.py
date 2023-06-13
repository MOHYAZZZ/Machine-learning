# K-Nearest Neighbors (KNN) Classification
# This code implements the K-Nearest Neighbors (KNN) algorithm. 
# To use this code, replace the example data in the `EXAMPLES` dictionary with your own data. 
# Then, provide the input features and the desired value of k to the `predict_label` function. The predicted label will be returned as the output.

import math

def euclidean_distance(point1, point2):
    # Calculate the Euclidean distance between two points
    squared_diff = [(a - b) ** 2 for a, b in zip(point1, point2)]
    return math.sqrt(sum(squared_diff))

def find_k_nearest_neighbors(examples, features, k):
    # Calculate the distances between the features and all examples
    distances = [(pid, euclidean_distance(features, example['features'])) for pid, example in examples.items()]
    # Sort the distances in ascending order
    distances.sort(key=lambda x: x[1])
    # Return the k nearest neighbors' pids
    return [pid for pid, _ in distances[:k]]

def predict_label(examples, features, k, label_key="is_intrusive"):
    # Find the k nearest neighbors
    nearest_neighbors = find_k_nearest_neighbors(examples, features, k)
    # Count the labels of the nearest neighbors
    label_counts = {0: 0, 1: 0}
    for pid in nearest_neighbors:
        label = examples[pid][label_key]
        label_counts[label] += 1
    # Predict the label with the majority vote
    predicted_label = max(label_counts, key=label_counts.get)
    return predicted_label


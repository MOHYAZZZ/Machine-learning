"""
This script implements the k-means clustering algorithm from scratch, without using any libraries that provide k-means functionality. It is used to find 'k' centroids in a given multi-dimensional feature space.

The algorithm uses the Manhattan distance as the distance metric and follows these steps:
1. Initialize 'k' centroids at random positions.
2. Assign each point in the feature space to the closest centroid.
3. Update the position of each centroid to be the mean of the points assigned to it.
4. Repeat steps 2 and 3 for a predetermined number of iterations.

The script includes a class 'Centroid' to manage the centroids and their associated points.

Three test cases are also provided to validate the functionality of the algorithm.
"""


import random


class Centroid:
    def __init__(self, location):
        self.location = location
        self.closest_users = set()


def get_k_means(user_feature_map, num_features_per_user, k):
    random.seed(42)
    # Initialize centroids
    initial_centroid_users = random.sample(
        sorted(list(user_feature_map.keys())), k)
    centroids = [Centroid(user_feature_map[user_id])
                 for user_id in initial_centroid_users]

    for _ in range(10):  # maximum of 10 iterations
        for centroid in centroids:  # reset closest users
            centroid.closest_users.clear()

        # Assign each user to the closest centroid
        for user_id, features in user_feature_map.items():
            closest_centroid = min(centroids, key=lambda centroid: sum(
                abs(a-b) for a, b in zip(centroid.location, features)))
            closest_centroid.closest_users.add(user_id)

        # Update centroid location by computing the average of the closest users' features
        for centroid in centroids:
            centroid.location = [sum(user_feature_map[user_id][i] for user_id in centroid.closest_users) / len(
                centroid.closest_users) for i in range(num_features_per_user)]

    return [centroid.location for centroid in centroids]


# Test case 1
user_feature_map1 = {
    "uid_1": [1, 2],
    "uid_2": [2, 3],
    "uid_3": [3, 4]
}
print(get_k_means(user_feature_map1, 2, 1))  # Expected output: [[2.0, 3.0]]

# Test case 2
user_feature_map2 = {
    "uid_1": [1, 1],
    "uid_2": [2, 2],
    "uid_3": [5, 5],
    "uid_4": [6, 6]
}
# Expected output (order may vary): [[1.5, 1.5], [5.5, 5.5]]
print(get_k_means(user_feature_map2, 2, 2))

# Test case 3
user_feature_map3 = {
    "uid_1": [1, 2, 3, 4, 5],
    "uid_2": [2, 3, 4, 5, 6],
    "uid_3": [3, 4, 5, 6, 7],
    "uid_4": [4, 5, 6, 7, 8],
    "uid_5": [10, 11, 12, 13, 14],
    "uid_6": [11, 12, 13, 14, 15],
    "uid_7": [12, 13, 14, 15, 16],
    "uid_8": [13, 14, 15, 16, 17],
    "uid_9": [20, 21, 22, 23, 24],
    "uid_10": [21, 22, 23, 24, 25]
}

print(get_k_means(user_feature_map3, 5, 3))
